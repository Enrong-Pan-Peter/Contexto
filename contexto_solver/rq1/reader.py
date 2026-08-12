"""Read solver traces and join self-reported individuals to realized ranks.

The reader is the shared foundation for every RQ1 script. It parses a raw
per-run solver trace (a JSON list of events, NOT an experiment ``--output``
summary), pulls the run-level configuration, recovers the hidden target for API
runs, and produces one :class:`Individual` per self-reported proposal with its
realized game rank joined back from the trace's ``GUESS`` events.

Facts respected here (verified against the solver code):

- ``predicted_closeness`` is the operator's estimated probability that its best
  proposed word ranks within the top-100 closest words; ``predicted_bucket`` is
  the categorical companion. Both are read via the normalized
  :func:`contexto_solver.self_report.read_self_report` defaults so old traces
  open unchanged.
- Realized ranks are stored under ``GUESS.details.rank`` for BOTH api and local
  traces (the api path normalizes the raw ``distance`` to ``distance + 1`` before
  logging, so ranks are 1-based in every trace).
- Self-reports are attached to different events per mode:
  ``OPERATOR_SAMPLED`` (self-adaptive / MAP-Elites mutation children, with
  ``sampled_op``/``parent_id``/``parent_rank``/``child_hypothesis_name``),
  ``CROSSOVER`` (crossover children, via the serialized ``child``), and ``GUESS``
  (``llm_only``, which carries the guessed ``word`` and ``rank`` in place).
- Every instrumented trace carries ``instrumentation_provenance_hash`` in its
  ``RUN_CONFIG`` event; callers compare it across a batch before pooling.

No network, no LLM, no solver, no writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..self_report import read_self_report

# Event names that carry a realized guess with a rank (or an invalid marker).
_GUESS_EVENTS = frozenset({"GUESS", "SKIP_INVALID_GUESS"})


@dataclass(frozen=True)
class RunConfig:
    """Run-level metadata pulled from the ``RUN_CONFIG`` event."""

    game: str | None
    method: str | None
    solver: str | None
    game_number: int | None
    config_target: str | None
    provenance_hash: str | None
    trace_schema_version: Any
    self_report: Any
    rationale_inheritance: Any
    sigma_mode: str | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Individual:
    """One self-reported proposal joined to its realized outcome.

    ``proposed_word``/``realized_rank`` are the *first evaluated* word of the
    child and its realized rank (the literal referent of the self-report, which
    asks about the operator's best proposed word; the operator lists best-first).
    ``child_best_word``/``child_best_rank`` are the child's realized best across
    every word it guessed. They coincide for single-guess modes (llm_only,
    MAP-Elites) and can differ for the multi-word EA operators.
    """

    trace_file: str
    game: str | None
    method: str | None
    game_number: int | None
    target: str | None
    target_source: str | None
    provenance_hash: str | None
    trace_schema_version: Any
    sigma_mode: str | None
    generation: Any
    source_event: str
    operator: str | None
    hypothesis_name: str | None
    parent_id: str | None
    parent_rank: int | None
    predicted_closeness: float | None
    predicted_closeness_clamped: bool
    predicted_bucket: str | None
    self_report_parse_failed: bool
    injected_rationale_hash: str | None
    rationale_inherited: bool
    rationale_truncated: bool
    basis_words_count: int
    proposed_word: str | None
    realized_rank: int | None
    proposed_word_invalid: bool
    child_best_word: str | None
    child_best_rank: int | None


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load a raw per-run solver trace (a JSON list of events).

    Raises ``ValueError`` with a helpful hint when handed an experiment
    ``--output`` summary (a dict with a ``runs`` list) instead of a trace.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        hint = ""
        if isinstance(data, dict) and isinstance(data.get("runs"), list) and data["runs"]:
            first = data["runs"][0] if isinstance(data["runs"][0], dict) else {}
            trace_path = first.get("trace_path") if isinstance(first, dict) else None
            hint = (
                f" This looks like an experiment summary (--output), not a solver trace."
                + (f" Pass the per-run file instead, e.g. {trace_path}" if trace_path else
                   " Pass the per-run traces listed under runs[].trace_path.")
            )
        raise ValueError(f"{path} is not a solver trace (expected a list of events).{hint}")
    return data


def run_config(events: list[dict[str, Any]]) -> RunConfig:
    """Extract the run-level configuration from the ``RUN_CONFIG`` event."""
    details: dict[str, Any] = {}
    for event in events:
        if event.get("event") == "RUN_CONFIG":
            details = event.get("details", {}) or {}
            break
    sigma_mode = details.get("self_adaptive_sigma_mode") or details.get("mapelites_sigma_mode")
    return RunConfig(
        game=details.get("game"),
        method=details.get("method"),
        solver=details.get("solver"),
        game_number=details.get("game_number"),
        config_target=details.get("target"),
        provenance_hash=details.get("instrumentation_provenance_hash"),
        trace_schema_version=details.get("trace_schema_version"),
        self_report=details.get("self_report"),
        rationale_inheritance=details.get("rationale_inheritance"),
        sigma_mode=sigma_mode,
        raw=details,
    )


def recover_target(events: list[dict[str, Any]], config: RunConfig | None = None) -> tuple[str | None, str | None]:
    """Recover the hidden target word and how it was recovered.

    Order of preference:
    1. ``RUN_CONFIG.target`` (present for local runs; ``None`` for api runs).
    2. The ``SOLVED`` event's ``answer`` (the word logged at rank 1).
    3. Any ``GUESS`` event whose ``rank == 1``.

    Returns ``(target, source)`` where ``source`` is one of ``"run_config"``,
    ``"solved"``, ``"rank1_guess"``, or ``None`` when the run never reached the
    target (unsolved api run -> unrecoverable).
    """
    config = config or run_config(events)
    if config.config_target:
        return config.config_target, "run_config"
    for event in events:
        if event.get("event") == "SOLVED":
            answer = (event.get("details", {}) or {}).get("answer")
            if answer:
                return str(answer).lower().strip(), "solved"
    for event in events:
        if event.get("event") == "GUESS":
            details = event.get("details", {}) or {}
            if details.get("rank") == 1 and details.get("word"):
                return str(details["word"]).lower().strip(), "rank1_guess"
    return None, None


def _named_guess_events(
    events: list[dict[str, Any]], child_name: str | None, generation: Any
) -> list[dict[str, Any]]:
    """All GUESS/SKIP events for a child, matched by hypothesis name + generation.

    A child's guesses are logged with ``details.hypothesis == child.category_name``
    in the same generation as its proposal event, so matching on name+generation
    groups exactly that child's evaluations. Names are LLM-generated category
    labels and are effectively unique within a generation; a rare intra-generation
    name collision would pool two children (documented limitation).
    """
    if not child_name:
        return []
    matched: list[dict[str, Any]] = []
    for event in events:
        if event.get("event") not in _GUESS_EVENTS:
            continue
        if event.get("generation") != generation:
            continue
        details = event.get("details", {}) or {}
        if details.get("hypothesis") == child_name:
            matched.append(event)
    return matched


def _first_valid(guess_events: list[dict[str, Any]]) -> tuple[str | None, int | None]:
    """Best (min-rank) valid word/rank across a child's guess events."""
    best_word: str | None = None
    best_rank: int | None = None
    for event in guess_events:
        if event.get("event") != "GUESS":
            continue
        details = event.get("details", {}) or {}
        rank = details.get("rank")
        word = details.get("word")
        if isinstance(rank, int) and rank > 0 and word:
            if best_rank is None or rank < best_rank:
                best_rank, best_word = rank, str(word)
    return best_word, best_rank


def _proposed_from_events(guess_events: list[dict[str, Any]]) -> tuple[str | None, int | None, bool]:
    """First-evaluated word for a child: (word, rank, invalid).

    ``invalid`` is True when the first evaluation was a ``SKIP_INVALID_GUESS``
    (rank is then ``None``). Guess events are already in trace (evaluation) order.
    """
    if not guess_events:
        return None, None, False
    first = guess_events[0]
    details = first.get("details", {}) or {}
    word = details.get("word")
    if first.get("event") == "SKIP_INVALID_GUESS":
        return (str(word) if word else None), None, True
    rank = details.get("rank")
    return (str(word) if word else None), (int(rank) if isinstance(rank, int) else None), False


def _basis_count(self_report: dict[str, Any]) -> int:
    rationale = self_report.get("rationale")
    if isinstance(rationale, dict) and isinstance(rationale.get("basis_words"), list):
        return len(rationale["basis_words"])
    return 0


def _make_individual(
    *,
    self_report: dict[str, Any],
    source_event: str,
    operator: str | None,
    hypothesis_name: str | None,
    parent_id: str | None,
    parent_rank: int | None,
    generation: Any,
    proposed_word: str | None,
    realized_rank: int | None,
    proposed_word_invalid: bool,
    child_best_word: str | None,
    child_best_rank: int | None,
    config: RunConfig,
    target: str | None,
    target_source: str | None,
    trace_file: str,
) -> Individual:
    report = read_self_report({"self_report": self_report})
    injected = report.get("injected_rationale_hash")
    return Individual(
        trace_file=trace_file,
        game=config.game,
        method=config.method,
        game_number=config.game_number,
        target=target,
        target_source=target_source,
        provenance_hash=config.provenance_hash,
        trace_schema_version=config.trace_schema_version,
        sigma_mode=config.sigma_mode,
        generation=generation,
        source_event=source_event,
        operator=operator,
        hypothesis_name=hypothesis_name,
        parent_id=parent_id,
        parent_rank=parent_rank,
        predicted_closeness=report.get("predicted_closeness"),
        predicted_closeness_clamped=bool(report.get("predicted_closeness_clamped")),
        predicted_bucket=report.get("predicted_bucket"),
        self_report_parse_failed=bool(report.get("self_report_parse_failed")),
        injected_rationale_hash=injected,
        rationale_inherited=injected is not None,
        rationale_truncated=bool(report.get("rationale_truncated")),
        basis_words_count=_basis_count(report),
        proposed_word=proposed_word,
        realized_rank=realized_rank,
        proposed_word_invalid=proposed_word_invalid,
        child_best_word=child_best_word,
        child_best_rank=child_best_rank,
    )


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None


def extract_individuals(
    events: list[dict[str, Any]], *, trace_file: str = ""
) -> list[Individual]:
    """One :class:`Individual` per self-reported proposal in a trace.

    Sources (deduplicated so ``MUTATE`` mirrors of ``OPERATOR_SAMPLED`` are not
    double-counted): ``OPERATOR_SAMPLED``, ``CROSSOVER`` (serialized child),
    ``GUESS`` (llm_only), and base-EA ``MUTATE`` (whose ``self_report`` rides in
    the event details and refers to the first child in ``children``).
    """
    config = run_config(events)
    target, target_source = recover_target(events, config)
    seen_op_sampled = False
    individuals: list[Individual] = []

    for index, event in enumerate(events):
        name = event.get("event")
        details = event.get("details", {}) or {}
        generation = event.get("generation")

        if name == "OPERATOR_SAMPLED" and isinstance(details.get("self_report"), dict):
            seen_op_sampled = True
            child_name = details.get("child_hypothesis_name")
            guesses = _named_guess_events(events, child_name, generation)
            proposed_word, realized_rank, invalid = _proposed_from_events(guesses)
            best_word, best_rank = _first_valid(guesses)
            individuals.append(
                _make_individual(
                    self_report=details["self_report"],
                    source_event="OPERATOR_SAMPLED",
                    operator=details.get("sampled_op"),
                    hypothesis_name=child_name,
                    parent_id=details.get("parent_id"),
                    parent_rank=_int_or_none(details.get("parent_rank")),
                    generation=generation,
                    proposed_word=proposed_word,
                    realized_rank=realized_rank,
                    proposed_word_invalid=invalid,
                    child_best_word=best_word,
                    child_best_rank=best_rank,
                    config=config,
                    target=target,
                    target_source=target_source,
                    trace_file=trace_file,
                )
            )
        elif name == "CROSSOVER":
            child = details.get("child")
            if isinstance(child, dict) and isinstance(child.get("self_report"), dict):
                child_name = child.get("category_name")
                guesses = _named_guess_events(events, child_name, generation)
                proposed_word, realized_rank, invalid = _proposed_from_events(guesses)
                best_word, best_rank = _first_valid(guesses)
                # Fall back to the serialized child when no named guesses matched.
                if best_word is None:
                    best_word = child.get("best_word")
                    best_rank = _int_or_none(child.get("best_rank"))
                parent_ids = details.get("parent_ids") or details.get("parents")
                parent_ranks = details.get("parent_ranks")
                individuals.append(
                    _make_individual(
                        self_report=child["self_report"],
                        source_event="CROSSOVER",
                        operator="crossover",
                        hypothesis_name=child_name,
                        parent_id=_join_ids(parent_ids),
                        parent_rank=_best_rank(parent_ranks),
                        generation=generation,
                        proposed_word=proposed_word,
                        realized_rank=realized_rank,
                        proposed_word_invalid=invalid,
                        child_best_word=best_word,
                        child_best_rank=best_rank,
                        config=config,
                        target=target,
                        target_source=target_source,
                        trace_file=trace_file,
                    )
                )
        elif name == "GUESS" and isinstance(details.get("self_report"), dict):
            # llm_only: the guessed word and its realized rank are on the event.
            word = details.get("word")
            rank = _int_or_none(details.get("rank"))
            individuals.append(
                _make_individual(
                    self_report=details["self_report"],
                    source_event="GUESS",
                    operator="next_guess",
                    hypothesis_name=None,
                    parent_id=None,
                    parent_rank=None,
                    generation=generation,
                    proposed_word=(str(word) if word else None),
                    realized_rank=rank,
                    proposed_word_invalid=False,
                    child_best_word=(str(word) if word else None),
                    child_best_rank=rank,
                    config=config,
                    target=target,
                    target_source=target_source,
                    trace_file=trace_file,
                )
            )
        elif name == "MUTATE" and isinstance(details.get("self_report"), dict) and not seen_op_sampled:
            # Base ea_llm mutation: self_report rides in the event and refers to
            # the first child. Skipped for self-adaptive/MAP-Elites, which log the
            # same record under OPERATOR_SAMPLED (guarded by seen_op_sampled).
            children = details.get("children")
            child_name = None
            if isinstance(children, list) and children:
                first = children[0]
                child_name = first if isinstance(first, str) else (first or {}).get("category_name")
            guesses = _named_guess_events(events, child_name, generation)
            proposed_word, realized_rank, invalid = _proposed_from_events(guesses)
            best_word, best_rank = _first_valid(guesses)
            individuals.append(
                _make_individual(
                    self_report=details["self_report"],
                    source_event="MUTATE",
                    operator=details.get("method") or "mutation",
                    hypothesis_name=child_name,
                    parent_id=details.get("parent"),
                    parent_rank=None,
                    generation=generation,
                    proposed_word=proposed_word,
                    realized_rank=realized_rank,
                    proposed_word_invalid=invalid,
                    child_best_word=best_word,
                    child_best_rank=best_rank,
                    config=config,
                    target=target,
                    target_source=target_source,
                    trace_file=trace_file,
                )
            )

    return individuals


def _join_ids(value: Any) -> str | None:
    if isinstance(value, list):
        parts = [str(item) for item in value if item]
        return ";".join(parts) if parts else None
    if value:
        return str(value)
    return None


def _best_rank(ranks: Any) -> int | None:
    if isinstance(ranks, list):
        valid = [int(rank) for rank in ranks if isinstance(rank, int) and rank > 0]
        return min(valid) if valid else None
    return _int_or_none(ranks)


def read_run(path: str | Path) -> tuple[RunConfig, list[Individual]]:
    """Convenience: load a trace and return its config plus extracted individuals."""
    events = load_trace(path)
    config = run_config(events)
    individuals = extract_individuals(events, trace_file=Path(path).name)
    return config, individuals
