"""Per-operator parent-child GloVe cosine distributions over solver traces (offline).

For each variation event in instrumented self-adaptive traces -- mutation
children from ``OPERATOR_SAMPLED`` and crossover children from ``CROSSOVER`` --
this computes the GloVe cosine similarity between the parent's best word at the
moment of variation and the child's word, and summarizes the distribution per
operator (``s_mutation``, ``m_mutation``, ``ml_mutation``, ``l_mutation``,
``crossover``). This is the semantic step size each operator actually takes.

Parent-word reconstruction (verified against the trace schema):

- Hypothesis ids map to names via the ``INIT`` hypothesis list, earlier
  ``OPERATOR_SAMPLED`` children (``child_id`` -> ``child_hypothesis_name``), and
  serialized ``CROSSOVER`` children (``hypothesis_id`` -> ``category_name``).
- The parent's word is its best-ranked (min rank) ``GUESS`` with
  ``details.hypothesis == parent_name`` occurring EARLIER IN THE TRACE than the
  variation event, so intra-generation ordering is respected exactly.
- Where the event logs a ``parent_rank`` (mutations) or ``parent_ranks``
  (crossover), the reconstruction is cross-checked; the summary reports the
  match rate and mismatching rows are flagged, not dropped.

Child words follow the RQ1 reader semantics: ``child_first`` is the first
evaluated word (the literal operator output; primary), ``child_best`` the
child's best realized word in its birth generation, both matched by hypothesis
name + generation. Crossover children yield one row per parent (``parent_slot``
0/1), so the crossover distribution counts child-parent pairs.

Rows where the parent is unresolvable or a word is missing from the GloVe
vocabulary are counted per reason and skipped. Analysis only: no network, no
LLM, no writes outside ``--output`` (the GloVe file is read-only).

Usage (PowerShell):

    python scripts/operator_parent_child_cosine.py traces/rq1_A1/ea_llm_self_adaptive_api_*.json `
        --output traces/operator_cosine_A1 [--glove-path data/glove.6B.300d.txt]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from contexto_solver import config
from contexto_solver.embeddings import EmbeddingModel
from contexto_solver.rq1.reader import (
    _first_valid,
    _named_guess_events,
    _proposed_from_events,
    load_trace,
    run_config,
)

SKIP_REASONS = (
    "parent_unresolved",
    "parent_no_prior_guess",
    "parent_word_oov",
    "child_no_word",
    "child_word_oov",
)


def _cosine(u: np.ndarray, v: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denominator == 0:
        return None
    return float((u @ v) / denominator)


def _id_to_name(events: list[dict[str, Any]]) -> dict[str, str]:
    """Hypothesis id -> hypothesis name, from INIT, OPERATOR_SAMPLED, CROSSOVER."""
    mapping: dict[str, str] = {}
    for event in events:
        details = event.get("details", {}) or {}
        name = event.get("event")
        if name == "INIT":
            for hypothesis in details.get("hypotheses", []) or []:
                if isinstance(hypothesis, dict) and hypothesis.get("hypothesis_id"):
                    mapping[str(hypothesis["hypothesis_id"])] = str(hypothesis.get("category_name") or "")
        elif name == "OPERATOR_SAMPLED":
            if details.get("child_id") and details.get("child_hypothesis_name"):
                mapping[str(details["child_id"])] = str(details["child_hypothesis_name"])
        elif name == "CROSSOVER":
            child = details.get("child")
            if isinstance(child, dict) and child.get("hypothesis_id"):
                mapping[str(child["hypothesis_id"])] = str(child.get("category_name") or "")
    return mapping


def _guess_index(events: list[dict[str, Any]]) -> list[tuple[int, str, str, int]]:
    """(event_index, hypothesis_name, word, rank) for every valid GUESS, trace order."""
    rows: list[tuple[int, str, str, int]] = []
    for index, event in enumerate(events):
        if event.get("event") != "GUESS":
            continue
        details = event.get("details", {}) or {}
        word = details.get("word")
        rank = details.get("rank")
        hypothesis = details.get("hypothesis")
        if word and hypothesis and isinstance(rank, int) and rank > 0:
            rows.append((index, str(hypothesis), str(word), rank))
    return rows


def _parent_best_before(
    guesses: list[tuple[int, str, str, int]], parent_name: str, before_index: int
) -> tuple[str, int] | None:
    """The parent's best (min-rank) word among its guesses before ``before_index``."""
    best: tuple[str, int] | None = None
    for index, hypothesis, word, rank in guesses:
        if index >= before_index:
            break
        if hypothesis != parent_name:
            continue
        if best is None or rank < best[1]:
            best = (word, rank)
    return best


def extract_pairs(
    events: list[dict[str, Any]], trace_file: str, model: EmbeddingModel
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """All parent-child cosine rows for one trace, plus skip-reason counts."""
    trace_config = run_config(events)
    id_to_name = _id_to_name(events)
    guesses = _guess_index(events)
    skips: dict[str, int] = {reason: 0 for reason in SKIP_REASONS}
    rows: list[dict[str, Any]] = []

    def make_row(
        *,
        event_index: int,
        generation: Any,
        operator: str,
        child_name: str | None,
        parent_id: Any,
        parent_name_hint: str | None,
        logged_parent_rank: Any,
        parent_slot: int | None,
    ) -> None:
        parent_name = id_to_name.get(str(parent_id)) if parent_id else None
        if not parent_name:
            parent_name = parent_name_hint
        if not parent_name:
            skips["parent_unresolved"] += 1
            return
        parent_best = _parent_best_before(guesses, parent_name, event_index)
        if parent_best is None:
            skips["parent_no_prior_guess"] += 1
            return
        parent_word, parent_rank_reconstructed = parent_best

        child_guesses = _named_guess_events(events, child_name, generation)
        child_first, child_first_rank, _invalid = _proposed_from_events(child_guesses)
        child_best, child_best_rank = _first_valid(child_guesses)
        if not child_first and not child_best:
            skips["child_no_word"] += 1
            return

        parent_vector = model.get_vector(parent_word)
        if parent_vector is None:
            skips["parent_word_oov"] += 1
            return

        cosine_first = None
        if child_first:
            child_vector = model.get_vector(child_first)
            cosine_first = _cosine(parent_vector, child_vector) if child_vector is not None else None
        cosine_best = None
        if child_best:
            child_vector = model.get_vector(child_best)
            cosine_best = _cosine(parent_vector, child_vector) if child_vector is not None else None
        if cosine_first is None and cosine_best is None:
            skips["child_word_oov"] += 1
            return

        rows.append(
            {
                "trace_file": trace_file,
                "game_number": trace_config.game_number,
                "generation": generation,
                "operator": operator,
                "parent_slot": parent_slot,
                "child_name": child_name,
                "parent_name": parent_name,
                "parent_word": parent_word,
                "parent_rank_reconstructed": parent_rank_reconstructed,
                "parent_rank_logged": logged_parent_rank if isinstance(logged_parent_rank, int) else None,
                "parent_rank_matches": (
                    parent_rank_reconstructed == logged_parent_rank
                    if isinstance(logged_parent_rank, int)
                    else None
                ),
                "child_first_word": child_first,
                "child_first_rank": child_first_rank,
                "cosine_first": cosine_first,
                "child_best_word": child_best,
                "child_best_rank": child_best_rank,
                "cosine_best": cosine_best,
            }
        )

    for event_index, event in enumerate(events):
        name = event.get("event")
        details = event.get("details", {}) or {}
        generation = event.get("generation")
        if name == "OPERATOR_SAMPLED":
            make_row(
                event_index=event_index,
                generation=generation,
                operator=str(details.get("sampled_op") or "unknown_mutation"),
                child_name=details.get("child_hypothesis_name"),
                parent_id=details.get("parent_id"),
                parent_name_hint=None,
                logged_parent_rank=details.get("parent_rank"),
                parent_slot=None,
            )
        elif name == "CROSSOVER":
            child = details.get("child")
            if not isinstance(child, dict):
                continue
            parent_ids = details.get("parent_ids") or []
            parent_names = details.get("parents") or []
            parent_ranks = details.get("parent_ranks") or []
            slots = max(len(parent_ids), len(parent_names))
            for slot in range(slots):
                make_row(
                    event_index=event_index,
                    generation=generation,
                    operator="crossover",
                    child_name=child.get("category_name"),
                    parent_id=parent_ids[slot] if slot < len(parent_ids) else None,
                    parent_name_hint=(
                        str(parent_names[slot]) if slot < len(parent_names) and parent_names[slot] else None
                    ),
                    logged_parent_rank=parent_ranks[slot] if slot < len(parent_ranks) else None,
                    parent_slot=slot,
                )
    return rows, skips


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else None,
        "min": float(np.min(array)),
        "q1": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "q3": float(np.percentile(array, 75)),
        "max": float(np.max(array)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-operator cosine distributions for both child-word choices."""
    by_operator: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        entry = by_operator.setdefault(row["operator"], {"cosine_first": [], "cosine_best": []})
        for key in ("cosine_first", "cosine_best"):
            if row[key] is not None:
                entry[key].append(row[key])
    return {
        operator: {key: _distribution(values) for key, values in entries.items()}
        for operator, entries in sorted(by_operator.items())
    }


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(sorted(matches))
        elif Path(pattern).exists():
            paths.append(pattern)
    return list(dict.fromkeys(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-operator parent-child GloVe cosine distributions (offline)."
    )
    parser.add_argument("traces", nargs="+", help="Raw trace JSON path(s) or glob(s).")
    parser.add_argument("--glove-path", default=config.GLOVE_PATH, help="Path to a GloVe text embedding file.")
    parser.add_argument("--output", required=True, help="Output directory for the pairs CSV + summary JSON.")
    args = parser.parse_args()

    paths = _expand_paths(args.traces)
    if not paths:
        raise SystemExit("No trace files matched the given path(s).")

    model = EmbeddingModel(args.glove_path)

    all_rows: list[dict[str, Any]] = []
    skips_total: dict[str, int] = {reason: 0 for reason in SKIP_REASONS}
    per_trace_meta: list[dict[str, Any]] = []
    for path in paths:
        name = Path(path).name
        events = load_trace(path)
        rows, skips = extract_pairs(events, name, model)
        all_rows.extend(rows)
        for reason, count in skips.items():
            skips_total[reason] += count
        per_trace_meta.append(
            {
                "trace_file": name,
                "game_number": run_config(events).game_number,
                "pairs": len(rows),
                "skips": {reason: count for reason, count in skips.items() if count},
            }
        )

    rank_checked = [row for row in all_rows if row["parent_rank_matches"] is not None]
    rank_matched = sum(1 for row in rank_checked if row["parent_rank_matches"])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "glove_path": str(args.glove_path),
        "traces": per_trace_meta,
        "total_pairs": len(all_rows),
        "skips_total": {reason: count for reason, count in skips_total.items() if count},
        "parent_rank_checked": len(rank_checked),
        "parent_rank_matched": rank_matched,
        "parent_rank_match_rate": (rank_matched / len(rank_checked)) if rank_checked else None,
        "per_operator": summarize(all_rows),
        "note": (
            "cosine_first pairs the parent's best pre-variation word with the child's "
            "first evaluated word (primary); cosine_best uses the child's best realized "
            "word. Crossover contributes one row per parent (parent_slot)."
        ),
    }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "operator_parent_child_pairs.csv"
    if all_rows:
        with pairs_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        pairs_path.write_text("", encoding="utf-8")
    summary_path = out_dir / "operator_parent_child_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Traces: {len(per_trace_meta)}  pairs: {len(all_rows)}  skips: {summary['skips_total'] or '{}'}")
    print(
        f"parent_rank cross-check: {rank_matched}/{len(rank_checked)} matched "
        f"({summary['parent_rank_match_rate']})"
    )
    for operator, entries in summary["per_operator"].items():
        first = entries["cosine_first"]
        print(
            f"  {operator}: n={first.get('n', 0)} median={first.get('median')} "
            f"q1={first.get('q1')} q3={first.get('q3')}"
        )
    print(f"Wrote {pairs_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
