"""Measure operator -> selection-survival and operator -> fitness coupling in
plain self-adaptive (`method=ea_llm_self_adaptive`) traces.

This is the self-adaptive counterpart to
`scripts/measure_sigma_fitness_coupling.py`. That script links a fired mutation
operator to a MAP-Elites archive outcome (cell win). Here the "win" is instead
whether the mutation child SURVIVED SELECTION, recovered from the logged `SELECT`
events. The fitness (delta) half is identical to the MAP-Elites version:
`delta = log(parent_rank) - log(child_rank)` on the log-rank scale.

Question it answers: does plain self-adaptive show the same
small-wins-but-sigma-drifts-large pattern, making the operator misdirection
general rather than MAP-Elites-specific?

How survival is recorded (verified against traces; see docs/architecture.md for
the event schema and the name-ambiguity caveat):
  * Within a generation the event order is
    `CANDIDATES -> SELECT -> OPERATOR_SAMPLED(xN) -> MUTATE -> CROSSOVER -> DEDUPLICATE`,
    so a mutation child created in generation G is NOT in G's `SELECT`; it is
    judged by the `SELECT` of generation G+1.
  * `SELECT.details` records `kept`, `discarded`, and `elite` as category NAMES
    only (no hypothesis IDs). "survived" is defined here as the child's
    `child_hypothesis_name` appearing in the next generation's `SELECT.kept`
    (or being its `elite`). This is the LOGGED selection step, which keeps the
    top `min(len//2, max_active_hypotheses)` plus the elite (top-`max_active`+
    elite, e.g. top-5). It is NOT the separate `mu`-cap (`SELF_ADAPTIVE_MU`,
    e.g. 15): `_cap_active_hypotheses` is unlogged, so this tool cannot and does
    not measure the mu cap.

Resolution rules (per the requested adjustments):
  * Child rank is the EVALUATED rank, read from the child's `GUESS` events (keyed
    by `category_name`), taking the min over guesses logged AFTER the child's
    `OPERATOR_SAMPLED`. It is NOT read from the creation-time serialized dict:
    a mutation child is serialized in `MUTATE` before it is guessed, so that
    `best_rank` is always the empty-hypothesis sentinel (1e9). This mirrors how
    `measure_sigma_fitness_coupling.py` reads child rank from the post-evaluation
    `ARCHIVE_*` events rather than a creation-time placeholder. Parent rank is the
    min `GUESS` rank under the parent's name at or before the child's creation
    (parent name resolved from `parent_id` via the serialized-dict id->name map).
  * Sentinel children (NO valid `GUESS` after creation, i.e. the operator
    produced no playable word) are EXCLUDED from delta and COUNTED AS CULLED.
  * A non-sentinel child is UNRESOLVABLE (dropped, reported) only when it is a
    last-generation child (no next `SELECT`) or its `child_hypothesis_name` is
    reused by another mutation child in the same trace (ambiguous name; its
    GUESS/SELECT linkage cannot be attributed). Otherwise it survived (in next
    `SELECT.kept`/`elite`) or was culled. Absence from the next `SELECT` means it
    was removed before selection (deduplication); this is counted as CULLED, not
    unresolvable (the `SELECT` kept+discarded lists cover all live hypotheses).
  * The per-operator truly-unresolvable fraction is reported as a bias diagnostic,
    and a SANITY GATE reports the per-operator sentinel fraction (expected
    single-digit %); a high fraction means the rank source is still wrong.

This tool reads existing traces only; it changes no solver, game, or
trace-schema code, consistent with the analysis-script invariant in
docs/architecture.md. It does not run the solver or any new games.

Usage:
    python scripts/measure_self_adaptive_selection_coupling.py
        [--traces-dir traces]
        [--targets herbaceous,notorious,superficial]
        [--max-generations 50] [--by-word] [--report-json PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

OPERATORS = ("s_mutation", "m_mutation", "ml_mutation", "l_mutation")
DEFAULT_TARGETS = ("herbaceous", "notorious", "superficial")

# Empty-hypothesis sentinel best rank (see Hypothesis): a child with this rank
# made no valid guess. Treated as "no valid rank" for both delta and survival.
SENTINEL_RANK = 1_000_000_000
JACKPOT = math.log(10.0)      # delta >= ln(10): >= 10x rank improvement
DISASTER = -math.log(10.0)    # delta <= -ln(10): >= 10x rank degradation


def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _is_real_rank(rank: Any) -> bool:
    return isinstance(rank, (int, float)) and 0 < rank < SENTINEL_RANK


# --------------------------------------------------------------------------- #
# Trace loading
# --------------------------------------------------------------------------- #
def _is_self_adaptive(events: list[dict[str, Any]]) -> bool:
    return any(e.get("event") == "OPERATOR_SAMPLED" for e in events)


def _trace_target(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("event") == "RUN_CONFIG":
            target = _details(event).get("target")
            return target if isinstance(target, str) else None
    return None


def _trace_max_generations(events: list[dict[str, Any]]) -> int | None:
    for event in events:
        if event.get("event") == "RUN_CONFIG":
            value = _details(event).get("max_generations")
            return value if isinstance(value, int) else None
    return None


def load_traces(
    traces_dir: Path,
    targets: list[str],
    max_generations: int | None,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Return [(trace_name, target, events)] for self-adaptive traces of the
    requested targets."""
    runs: list[tuple[str, str, list[dict[str, Any]]]] = []
    for target in targets:
        for path in sorted(traces_dir.glob(f"ea_llm_self_adaptive_local_{target}_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            events = [e for e in data if isinstance(e, dict)]
            if not _is_self_adaptive(events):
                continue
            if max_generations is not None and _trace_max_generations(events) != max_generations:
                continue
            run_target = _trace_target(events) or target
            runs.append((path.name, run_target, events))
    return runs


# --------------------------------------------------------------------------- #
# Per-trace indices
# --------------------------------------------------------------------------- #
def _iter_hypothesis_dicts(value: Any):
    """Yield serialized hypothesis dicts (carrying hypothesis_id + best_rank)."""
    if isinstance(value, dict):
        if "hypothesis_id" in value and "best_rank" in value:
            yield value
        for child in value.values():
            yield from _iter_hypothesis_dicts(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_hypothesis_dicts(item)


def build_id_to_name(events: list[dict[str, Any]]) -> dict[str, str]:
    """hypothesis_id -> category_name, from serialized hypothesis dicts plus
    OPERATOR_SAMPLED child records.

    Needed to resolve a child's `parent_id` to the category name that the
    parent's GUESS events are keyed by (GUESS events carry only the name).
    """
    id_to_name: dict[str, str] = {}
    for event in events:
        for record in _iter_hypothesis_dicts(_details(event)):
            hid = record.get("hypothesis_id")
            name = record.get("category_name")
            if isinstance(hid, str) and isinstance(name, str):
                id_to_name.setdefault(hid, name)
        if event.get("event") == "OPERATOR_SAMPLED":
            details = _details(event)
            child_id = details.get("child_id")
            name = details.get("child_hypothesis_name")
            if isinstance(child_id, str) and isinstance(name, str):
                id_to_name.setdefault(child_id, name)
    return id_to_name


def build_guesses_by_name(events: list[dict[str, Any]]) -> dict[str, list[tuple[int, int]]]:
    """category_name -> [(order, rank)] from GUESS events (valid ranks only).

    The EVALUATED rank of a hypothesis lives in its GUESS events, keyed by
    `category_name` (see `_guess_and_update` in methods/ea_core.py). It does NOT
    live in the creation-time serialized dict: a mutation child is serialized in
    the `MUTATE` event BEFORE it is guessed, so its `best_rank` there is the
    empty-hypothesis sentinel (1e9). Reading rank from this post-evaluation event
    mirrors how `measure_sigma_fitness_coupling.py` reads child rank from the
    post-evaluation `ARCHIVE_*` events for MAP-Elites.
    """
    by_name: dict[str, list[tuple[int, int]]] = {}
    for order, event in enumerate(events):
        if event.get("event") != "GUESS":
            continue
        details = _details(event)
        name = details.get("hypothesis")
        rank = details.get("rank")
        if isinstance(name, str) and _is_real_rank(rank):
            by_name.setdefault(name, []).append((order, int(rank)))
    return by_name


def child_best_rank(
    guesses_by_name: dict[str, list[tuple[int, int]]],
    name: str | None,
    child_order: int,
) -> int | None:
    """Child's best (min) evaluated rank from GUESS events under its name that
    occur AFTER its creation (`OPERATOR_SAMPLED` is logged before the child's
    guesses). None means the child made no valid guess (sentinel)."""
    if name is None:
        return None
    ranks = [rank for order, rank in guesses_by_name.get(name, []) if order > child_order]
    return min(ranks) if ranks else None


def parent_best_rank_at(
    guesses_by_name: dict[str, list[tuple[int, int]]],
    name: str | None,
    child_order: int,
) -> int | None:
    """Parent's best (min) evaluated rank as of the child's creation: GUESS events
    under the parent's name at or before the child's order."""
    if name is None:
        return None
    ranks = [rank for order, rank in guesses_by_name.get(name, []) if order <= child_order]
    return min(ranks) if ranks else None


def build_select_index(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """generation -> {kept: set, discarded: set, elite: str|None}."""
    index: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "SELECT":
            continue
        generation = event.get("generation")
        if not isinstance(generation, int):
            continue
        details = _details(event)
        index[generation] = {
            "kept": set(k for k in details.get("kept", []) if isinstance(k, str)),
            "discarded": set(d for d in details.get("discarded", []) if isinstance(d, str)),
            "elite": details.get("elite") if isinstance(details.get("elite"), str) else None,
        }
    return index


def ambiguous_child_names(events: list[dict[str, Any]]) -> set[str]:
    """Mutation-child names that are reused across OPERATOR_SAMPLED in this trace."""
    counts: dict[str, int] = {}
    for event in events:
        if event.get("event") != "OPERATOR_SAMPLED":
            continue
        name = _details(event).get("child_hypothesis_name")
        if isinstance(name, str):
            counts[name] = counts.get(name, 0) + 1
    return {name for name, count in counts.items() if count > 1}


# --------------------------------------------------------------------------- #
# Child record
# --------------------------------------------------------------------------- #
class ChildRecord:
    __slots__ = ("trace", "target", "operator", "survival", "unresolvable",
                 "is_sentinel", "delta")

    def __init__(self, trace, target, operator, survival, unresolvable, is_sentinel, delta):
        self.trace = trace
        self.target = target
        self.operator = operator
        self.survival = survival          # "survived" | "culled" | None
        self.unresolvable = unresolvable  # bool (last-gen or ambiguous name)
        self.is_sentinel = is_sentinel    # bool
        self.delta = delta                # float | None


def collect_records(
    runs: list[tuple[str, str, list[dict[str, Any]]]],
) -> list[ChildRecord]:
    records: list[ChildRecord] = []
    for trace_name, target, events in runs:
        guesses_by_name = build_guesses_by_name(events)
        id_to_name = build_id_to_name(events)
        select_index = build_select_index(events)
        ambiguous = ambiguous_child_names(events)
        select_gens = set(select_index)

        for order, event in enumerate(events):
            if event.get("event") != "OPERATOR_SAMPLED":
                continue
            d = _details(event)
            operator = d.get("sampled_op")
            child_id = d.get("child_id")
            parent_id = d.get("parent_id")
            name = d.get("child_hypothesis_name")
            generation = event.get("generation")
            if operator not in OPERATORS or not isinstance(child_id, str):
                continue

            # Reused category names can't be attributed to a single hypothesis,
            # so their GUESS/SELECT linkage is untrustworthy (kept unresolvable).
            is_ambiguous = isinstance(name, str) and name in ambiguous
            # Child's evaluated rank comes from its GUESS events (by name), not the
            # creation-time serialized dict (which is the empty sentinel).
            c_rank = (
                child_best_rank(guesses_by_name, name, order)
                if not is_ambiguous
                else None
            )
            # Genuine "no valid guess" child (excludes the ambiguous case).
            is_sentinel = (not is_ambiguous) and (c_rank is None)

            # delta-fitness: needs a real child rank and a real parent rank.
            delta = None
            parent_name = id_to_name.get(parent_id) if isinstance(parent_id, str) else None
            p_rank = parent_best_rank_at(guesses_by_name, parent_name, order)
            if c_rank is not None and p_rank is not None:
                delta = math.log(p_rank) - math.log(c_rank)

            # survival.
            survival: str | None
            unresolvable = False
            if is_ambiguous:
                # reused name: cannot map to a SELECT kept/discarded row.
                survival = None
                unresolvable = True
            elif is_sentinel:
                # no valid guess -> cannot be in top-max_active -> culled.
                survival = "culled"
            else:
                next_gen = generation + 1 if isinstance(generation, int) else None
                if next_gen is None or next_gen not in select_gens:
                    # last-generation child: no next SELECT to judge it.
                    survival = None
                    unresolvable = True
                else:
                    select = select_index[next_gen]
                    if name in select["kept"] or name == select["elite"]:
                        survival = "survived"
                    else:
                        # in discarded, or removed before selection (dedup) -> culled.
                        survival = "culled"

            records.append(
                ChildRecord(trace_name, target, operator, survival, unresolvable, is_sentinel, delta)
            )
    return records


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def dist_stats(values: list[float]) -> dict[str, Any]:
    values = sorted(v for v in values if v is not None)
    n = len(values)
    if n == 0:
        return {"n": 0}
    median = statistics.median(values)
    if n >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
    else:
        q1, q3 = values[0], values[-1]
    return {
        "n": n,
        "median": median,
        "q1": q1,
        "q3": q3,
        "min": values[0],
        "max": values[-1],
        "frac_improve": sum(1 for v in values if v > 0) / n,
        "frac_jackpot": sum(1 for v in values if v >= JACKPOT) / n,
        "frac_disaster": sum(1 for v in values if v <= DISASTER) / n,
    }


def survival_stats(records: list[ChildRecord], operator: str) -> dict[str, Any]:
    ops = [r for r in records if r.operator == operator]
    total = len(ops)
    unresolvable = sum(1 for r in ops if r.unresolvable)
    sentinel = sum(1 for r in ops if r.is_sentinel)
    resolved = [r for r in ops if not r.unresolvable]
    survived = sum(1 for r in resolved if r.survival == "survived")
    culled = sum(1 for r in resolved if r.survival == "culled")
    n_resolved = len(resolved)
    return {
        "total": total,
        "unresolvable": unresolvable,
        "unresolvable_rate": (unresolvable / total) if total else None,
        "sentinel_culled": sentinel,
        "resolved": n_resolved,
        "survived": survived,
        "culled": culled,
        "survival_rate": (survived / n_resolved) if n_resolved else None,
    }


# --------------------------------------------------------------------------- #
# Sanity gate
# --------------------------------------------------------------------------- #
SENTINEL_FRACTION_WARN = 0.10  # per-operator no-valid-guess fraction expected single-digit %


def sanity_gate(records: list[ChildRecord]) -> dict[str, Any]:
    """Per-operator sentinel (no-valid-guess) fraction. A correct rank source
    yields single-digit-percent sentinels; ~100% means the rank source is still
    reading the creation-time placeholder instead of the evaluated GUESS rank."""
    print("\n" + "=" * 92)
    print("SANITY GATE - per-operator sentinel (no-valid-guess) fraction")
    print("=" * 92)
    per_operator: dict[str, Any] = {}
    worst = 0.0
    for operator in OPERATORS:
        ops = [r for r in records if r.operator == operator]
        total = len(ops)
        sentinel = sum(1 for r in ops if r.is_sentinel)
        fraction = (sentinel / total) if total else None
        per_operator[operator] = {"total": total, "sentinel": sentinel, "fraction": fraction}
        if fraction is not None:
            worst = max(worst, fraction)
        flag = "" if (fraction is None or fraction <= SENTINEL_FRACTION_WARN) else "  <== TOO HIGH"
        print(f"  {operator:<12} sentinel {sentinel:>5}/{total:<5} ({_pct(fraction)}){flag}")
    passed = worst <= SENTINEL_FRACTION_WARN
    if passed:
        print(f"\nPASS: all operators <= {SENTINEL_FRACTION_WARN * 100:.0f}% sentinel "
              "(evaluated child rank is being read correctly).")
    else:
        print(f"\nWARNING: a sentinel fraction exceeds {SENTINEL_FRACTION_WARN * 100:.0f}% "
              "(worst {:.1f}%). The child-rank source is still wrong - it should read the "
              "evaluated GUESS rank, not the creation-time sentinel.".format(worst * 100))
    return {"per_operator": per_operator, "worst_fraction": worst, "passed": passed}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pct(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def report_survival(records: list[ChildRecord], scope: str) -> dict[str, Any]:
    print("\n" + "=" * 92)
    print(f"OPERATOR -> SELECTION SURVIVAL  [{scope}]")
    print("=" * 92)
    print("survived = child name in next-generation SELECT.kept/elite "
          "(top-max_active+elite; the mu cap is unlogged).")
    header = (f"{'operator':<12} {'N':>5} {'unresolv':>14} {'sentinel':>9} "
              f"{'resolved':>9} {'survived':>9} {'culled':>7} {'surv.rate':>10}")
    print(header)
    print("-" * len(header))
    out: dict[str, Any] = {}
    for operator in OPERATORS:
        s = survival_stats(records, operator)
        out[operator] = s
        print(f"{operator:<12} {s['total']:>5} "
              f"{s['unresolvable']:>4} ({_pct(s['unresolvable_rate'])}) "
              f"{s['sentinel_culled']:>9} {s['resolved']:>9} "
              f"{s['survived']:>9} {s['culled']:>7} {_pct(s['survival_rate']):>10}")
    print("\nunresolv = last-generation children + reused-name collisions (dropped from")
    print("the rate; fraction shown as a per-operator bias diagnostic). sentinel = no")
    print("valid guess; excluded from delta and counted within culled.")
    return out


def report_delta(records: list[ChildRecord], scope: str) -> dict[str, Any]:
    print("\n" + "=" * 92)
    print(f"OPERATOR -> FITNESS delta = log(parent_rank) - log(child_rank)  [{scope}]")
    print("=" * 92)
    print("positive delta = child better than parent (improvement). Log-rank scale.")
    header = (f"{'operator':<12} {'N':>5} {'median':>8} {'IQR':>16} {'min/max':>16} "
              f"{'%impr':>6} {'%jack':>6} {'%bad':>6}")
    print(header)
    print("-" * len(header))
    out: dict[str, Any] = {}
    excluded = 0
    for operator in OPERATORS:
        deltas = [r.delta for r in records if r.operator == operator and r.delta is not None]
        excluded += sum(1 for r in records if r.operator == operator and r.delta is None)
        s = dist_stats(deltas)
        out[operator] = s
        if s["n"] == 0:
            print(f"{operator:<12} {0:>5} {'--':>8}")
            continue
        print(f"{operator:<12} {s['n']:>5} {s['median']:>8.2f} "
              f"[{s['q1']:>6.2f},{s['q3']:>6.2f}] "
              f"[{s['min']:>6.2f},{s['max']:>6.2f}] "
              f"{s['frac_improve'] * 100:>5.0f}% {s['frac_jackpot'] * 100:>5.0f}% "
              f"{s['frac_disaster'] * 100:>5.0f}%")
    print("\n%impr = delta>0; %jack = delta>=ln(10) (>=10x better); "
          "%bad = delta<=-ln(10) (>=10x worse).")
    print(f"Excluded from delta (sentinel child or unresolved parent rank): {excluded}.")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure operator -> selection-survival and operator -> fitness "
                    "coupling in self-adaptive traces (analysis only; no solver runs).")
    parser.add_argument("--traces-dir", default="traces", help="Directory of trace JSON files.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS),
                        help="Comma-separated target words (default: herbaceous,notorious,superficial).")
    parser.add_argument("--max-generations", type=int,
                        help="If set, keep only traces whose RUN_CONFIG max_generations equals this.")
    parser.add_argument("--by-word", action="store_true",
                        help="Also print per-target survival and delta tables (consistency check).")
    parser.add_argument("--report-json", help="Optional path to dump the full report as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = [t.strip().lower() for t in args.targets.split(",") if t.strip()]
    runs = load_traces(Path(args.traces_dir), targets, args.max_generations)
    if not runs:
        print(f"No self-adaptive traces found in {args.traces_dir}/ for targets "
              f"{targets} (max_generations={args.max_generations}).", file=sys.stderr)
        return 1

    records = collect_records(runs)
    per_target_counts: dict[str, int] = {}
    for _, target, _ in runs:
        per_target_counts[target] = per_target_counts.get(target, 0) + 1

    print(f"Pooled {len(runs)} self-adaptive trace(s); {len(records)} mutation children "
          f"(crossover excluded).")
    print("Per-target trace counts: "
          + ", ".join(f"{t}={per_target_counts[t]}" for t in sorted(per_target_counts)))
    print("Traces:")
    for trace_name, target, _ in runs:
        print(f"  - {trace_name} [{target}]")

    gate = sanity_gate(records)

    report: dict[str, Any] = {
        "n_traces": len(runs),
        "n_children": len(records),
        "per_target_trace_counts": per_target_counts,
        "sanity_gate": gate,
        "pooled": {
            "survival": report_survival(records, "pooled: all targets"),
            "delta": report_delta(records, "pooled: all targets"),
        },
    }

    if args.by_word:
        report["by_word"] = {}
        for target in sorted({t for _, t, _ in runs}):
            target_records = [r for r in records if r.target == target]
            report["by_word"][target] = {
                "survival": report_survival(target_records, f"target={target}"),
                "delta": report_delta(target_records, f"target={target}"),
            }

    print("\nFirst-look pooled diagnostic, not a verdict: compare the per-operator")
    print("survival-rate spread and the delta ordering against the operator sigma drift")
    print("to judge whether the small-wins-but-sigma-drifts-large pattern is general.")

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON report to {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
