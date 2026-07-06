"""Measure the selection gradient on sigma (operator) in MAP-Elites runs.

Reads existing MAP-Elites traces (those with an AXIS_DEFINITION event), pools
across all of them, restricts to mutation children, and links:

    OPERATOR_SAMPLED.sampled_op
        -> child archive outcome (ARCHIVE_PLACE / REPLACE / REJECT) + child rank
        -> parent rank (via parent_id)

Reports (numbers only; changes no solver/config code):

  Measurement A - per-operator outcome rates: PLACE / REPLACE / REJECT counts and
    rates. PLACE (colonize empty cell) and REPLACE (beat a live incumbent) are
    kept separate. Counts printed beside every rate so power is visible.

  Measurement B - per-operator fitness-change distribution:
    delta = log(parent_rank) - log(child_rank)  (positive = improvement).
    Reports median, IQR, jackpot/much-worse tail fractions; split by run phase
    (early/growth = gen <= occupancy-freeze gen; late = post-freeze).

  Confounds: per-operator parent-rank distribution, plus delta stratified by
    parent-rank tercile. Log-rank used throughout. Per-operator event counts are
    stated explicitly; pooled PLACE/REPLACE counts may still be too small for
    strong claims - this is a first look, not a verdict.

Usage:
    python scripts/measure_sigma_fitness_coupling.py [--traces-dir traces]
        [--report-json PATH]

See docs/architecture.md for the trace event schema this parser relies on.
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
OUTCOMES = ("PLACE", "REPLACE", "REJECT")
JACKPOT = math.log(10.0)  # >= 10x rank improvement
DISASTER = -math.log(10.0)  # >= 10x rank degradation


def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _is_map_elites(events: list[Any]) -> bool:
    return any(isinstance(e, dict) and e.get("event") == "AXIS_DEFINITION" for e in events)


def load_map_elites_traces(traces_dir: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    runs: list[tuple[str, list[dict[str, Any]]]] = []
    for path in sorted(traces_dir.glob("*.json")):
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(events, list) and _is_map_elites(events):
            runs.append((path.name, [e for e in events if isinstance(e, dict)]))
    return runs


# --------------------------------------------------------------------------- #
# Per-run linkage
# --------------------------------------------------------------------------- #
class ChildRecord:
    __slots__ = ("run", "operator", "outcome", "child_rank", "parent_rank", "generation", "phase", "delta")

    def __init__(self, run, operator, outcome, child_rank, parent_rank, generation, phase):
        self.run = run
        self.operator = operator
        self.outcome = outcome
        self.child_rank = child_rank
        self.parent_rank = parent_rank
        self.generation = generation
        self.phase = phase
        self.delta = (
            math.log(parent_rank) - math.log(child_rank)
            if parent_rank and child_rank and parent_rank > 0 and child_rank > 0
            else None
        )


def index_run(events: list[dict[str, Any]]):
    """Build hypothesis_id -> rank and hypothesis_id -> (outcome, rank, gen)."""
    rank_by_hid: dict[str, int] = {}
    outcome_by_hid: dict[str, tuple[str, int, int | None]] = {}
    freeze_gen = 0
    for event in events:
        name = event.get("event")
        gen = event.get("generation")
        d = _details(event)
        if name == "ARCHIVE_PLACE":
            hid, rank = d.get("hypothesis_id"), d.get("rank")
            if hid is not None and rank is not None:
                rank_by_hid[hid] = rank
                outcome_by_hid[hid] = ("PLACE", rank, gen)
            if isinstance(gen, int) and gen > freeze_gen:
                freeze_gen = gen  # last colonization = occupancy freeze
        elif name == "ARCHIVE_REPLACE":
            hid, rank = d.get("new_hypothesis_id"), d.get("new_rank")
            if hid is not None and rank is not None:
                rank_by_hid[hid] = rank
                outcome_by_hid[hid] = ("REPLACE", rank, gen)
            old_hid, old_rank = d.get("old_hypothesis_id"), d.get("old_rank")
            if old_hid is not None and old_rank is not None:
                rank_by_hid.setdefault(old_hid, old_rank)
        elif name == "ARCHIVE_REJECT":
            hid, rank = d.get("child_hypothesis_id"), d.get("child_rank")
            if hid is not None and rank is not None:
                rank_by_hid.setdefault(hid, rank)
                outcome_by_hid[hid] = ("REJECT", rank, gen)
            inc_hid, inc_rank = d.get("incumbent_hypothesis_id"), d.get("incumbent_rank")
            if inc_hid is not None and inc_rank is not None:
                rank_by_hid.setdefault(inc_hid, inc_rank)
    return rank_by_hid, outcome_by_hid, freeze_gen


def collect_records(runs) -> list[ChildRecord]:
    records: list[ChildRecord] = []
    for run_name, events in runs:
        rank_by_hid, outcome_by_hid, freeze_gen = index_run(events)
        for event in events:
            if event.get("event") != "OPERATOR_SAMPLED":
                continue
            d = _details(event)
            operator = d.get("sampled_op")
            child_id = d.get("child_id")
            parent_id = d.get("parent_id")
            if operator not in OPERATORS or child_id is None:
                continue
            outcome_entry = outcome_by_hid.get(child_id)
            if outcome_entry is None:
                continue
            outcome, child_rank, _ = outcome_entry
            parent_rank = rank_by_hid.get(parent_id)
            gen = event.get("generation")
            phase = "late" if isinstance(gen, int) and gen > freeze_gen else "early"
            records.append(
                ChildRecord(run_name, operator, outcome, child_rank, parent_rank, gen, phase)
            )
    return records


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def _pct(num: int, den: int) -> str:
    return f"{(num / den * 100):5.1f}%" if den else "  n/a"


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
        "iqr": q3 - q1,
        "min": values[0],
        "max": values[-1],
        "frac_improve": sum(1 for v in values if v > 0) / n,
        "frac_jackpot": sum(1 for v in values if v >= JACKPOT) / n,
        "frac_disaster": sum(1 for v in values if v <= DISASTER) / n,
    }


def rank_dist(values: list[int]) -> dict[str, Any]:
    values = sorted(v for v in values if v is not None)
    n = len(values)
    if n == 0:
        return {"n": 0}
    if n >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
    else:
        q1, q3 = values[0], values[-1]
    return {"n": n, "median": statistics.median(values), "q1": q1, "q3": q3,
            "min": values[0], "max": values[-1]}


# --------------------------------------------------------------------------- #
# Measurement A
# --------------------------------------------------------------------------- #
def measurement_a(records: list[ChildRecord]) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("MEASUREMENT A - per-operator outcome rates (selection gradient on sigma)")
    print("=" * 78)
    header = f"{'operator':<12} {'N':>5} {'PLACE':>16} {'REPLACE':>16} {'REJECT':>16}"
    print(header)
    print("-" * len(header))
    out: dict[str, Any] = {}
    for op in OPERATORS:
        ops = [r for r in records if r.operator == op]
        n = len(ops)
        counts = {oc: sum(1 for r in ops if r.outcome == oc) for oc in OUTCOMES}
        print(f"{op:<12} {n:>5} "
              f"{counts['PLACE']:>5} ({_pct(counts['PLACE'], n)}) "
              f"{counts['REPLACE']:>5} ({_pct(counts['REPLACE'], n)}) "
              f"{counts['REJECT']:>5} ({_pct(counts['REJECT'], n)})")
        out[op] = {
            "n": n,
            "counts": counts,
            "place_rate": counts["PLACE"] / n if n else None,
            "replace_rate": counts["REPLACE"] / n if n else None,
            "reject_rate": counts["REJECT"] / n if n else None,
        }
    print("\nPLACE = colonize empty cell (trivial win, confounded by cell emptiness).")
    print("REPLACE = beat a live incumbent (the genuine fitness win).")
    print("Key comparison: spread in REPLACE-rate across operators => real gradient;")
    print("flat REPLACE-rate => sigma drifts. High PLACE + flat REPLACE for l_mutation")
    print("=> the archive sigma_l elevation is a colonization artifact, not fitness.")
    return out


# --------------------------------------------------------------------------- #
# Measurement B
# --------------------------------------------------------------------------- #
def _print_delta_block(title: str, records: list[ChildRecord]) -> dict[str, Any]:
    print(f"\n--- {title} ---")
    print(f"{'operator':<12} {'N':>4} {'median':>8} {'IQR':>16} {'min/max':>16} "
          f"{'%impr':>6} {'%jack':>6} {'%bad':>6}")
    block: dict[str, Any] = {}
    for op in OPERATORS:
        deltas = [r.delta for r in records if r.operator == op and r.delta is not None]
        s = dist_stats(deltas)
        block[op] = s
        if s["n"] == 0:
            print(f"{op:<12} {0:>4} {'--':>8}")
            continue
        print(f"{op:<12} {s['n']:>4} {s['median']:>8.2f} "
              f"[{s['q1']:>6.2f},{s['q3']:>6.2f}] "
              f"[{s['min']:>6.2f},{s['max']:>6.2f}] "
              f"{s['frac_improve']*100:>5.0f}% {s['frac_jackpot']*100:>5.0f}% "
              f"{s['frac_disaster']*100:>5.0f}%")
    return block


def measurement_b(records: list[ChildRecord]) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("MEASUREMENT B - per-operator fitness-change delta = log(parent) - log(child)")
    print("=" * 78)
    print("positive delta = child better than parent (improvement). Log-rank scale.")
    out = {
        "all": _print_delta_block("ALL PHASES POOLED", records),
        "early": _print_delta_block("EARLY / GROWTH (gen <= occupancy-freeze)",
                                    [r for r in records if r.phase == "early"]),
        "late": _print_delta_block("LATE (post-freeze, exhaustion-dominated)",
                                   [r for r in records if r.phase == "late"]),
    }
    print("\n%impr = fraction with delta>0; %jack = delta>=ln(10) (>=10x better);")
    print("%bad = delta<=-ln(10) (>=10x worse).")
    return out


# --------------------------------------------------------------------------- #
# Confounds
# --------------------------------------------------------------------------- #
def parent_rank_report(records: list[ChildRecord]) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("CONFOUND - per-operator parent-rank distribution (log-rank baseline)")
    print("=" * 78)
    print(f"{'operator':<12} {'N':>4} {'med rank':>10} {'Q1':>9} {'Q3':>9} {'min':>7} {'max':>9}")
    out: dict[str, Any] = {}
    for op in OPERATORS:
        ranks = [r.parent_rank for r in records if r.operator == op and r.parent_rank is not None]
        rd = rank_dist(ranks)
        out[op] = rd
        if rd["n"] == 0:
            print(f"{op:<12} {0:>4}")
            continue
        print(f"{op:<12} {rd['n']:>4} {rd['median']:>10.0f} {rd['q1']:>9.0f} "
              f"{rd['q3']:>9.0f} {rd['min']:>7.0f} {rd['max']:>9.0f}")
    print("\nl_mutation children may come from systematically different-ranked parents;")
    print("compare medians above before attributing delta differences to the operator.")
    return out


def delta_by_parent_tercile(records: list[ChildRecord]) -> dict[str, Any]:
    print("\n" + "=" * 78)
    print("CONFOUND CONTROL - median delta per operator, stratified by parent-rank tercile")
    print("=" * 78)
    usable = [r for r in records if r.delta is not None and r.parent_rank]
    parent_ranks = sorted(r.parent_rank for r in usable)
    out: dict[str, Any] = {}
    if len(parent_ranks) < 3:
        print("  insufficient data for terciles.")
        return out
    t1, t2 = statistics.quantiles(parent_ranks, n=3)
    print(f"tercile edges on parent rank: low<= {t1:.0f} < mid <= {t2:.0f} < high")
    print(f"{'operator':<12} {'low N/med':>16} {'mid N/med':>16} {'high N/med':>16}")
    for op in OPERATORS:
        row = {}
        cells = []
        for label, lo, hi in (("low", -1, t1), ("mid", t1, t2), ("high", t2, float("inf"))):
            deltas = [r.delta for r in usable
                      if r.operator == op and lo < r.parent_rank <= hi]
            if deltas:
                med = statistics.median(deltas)
                cells.append(f"{len(deltas):>3}/{med:>7.2f}")
                row[label] = {"n": len(deltas), "median": med}
            else:
                cells.append(f"{0:>3}/{'--':>7}")
                row[label] = {"n": 0}
        out[op] = row
        print(f"{op:<12} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure sigma-fitness coupling in MAP-Elites traces (reports numbers only).")
    parser.add_argument("--traces-dir", default="traces", help="Directory of trace JSON files.")
    parser.add_argument("--report-json", help="Optional path to dump the full report as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = load_map_elites_traces(Path(args.traces_dir))
    if not runs:
        print(f"No MAP-Elites traces (with AXIS_DEFINITION) found in {args.traces_dir}/.", file=sys.stderr)
        return 1
    records = collect_records(runs)

    print(f"Pooled {len(runs)} MAP-Elites run(s); {len(records)} linked mutation children.")
    print("Runs:")
    for run_name, _ in runs:
        print(f"  - {run_name}")
    unresolved_parent = sum(1 for r in records if r.parent_rank is None)
    if unresolved_parent:
        print(f"\n[note] {unresolved_parent} children had an unresolved parent rank "
              f"(excluded from delta stats).")

    report = {
        "n_runs": len(runs),
        "n_children": len(records),
        "measurement_a": measurement_a(records),
        "measurement_b": measurement_b(records),
        "parent_rank_distribution": parent_rank_report(records),
        "delta_by_parent_tercile": delta_by_parent_tercile(records),
    }

    print("\nPower caveat: PLACE/REPLACE events are rare even pooled across runs; small")
    print("per-operator counts mean these are first-look signals, not verdicts.")

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON report to {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
