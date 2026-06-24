"""Compare the MAP-Elites sigma-control arms.

This is the arm-comparison counterpart to ``measure_sigma_fitness_coupling.py``.
That script *pools* MAP-Elites runs to estimate an operator-level selection
gradient; it does not separate the sigma-mode arms. This script instead reads
the sigma-control batch produced by ``scripts/run_sigma_control.py``, groups
runs by ``MAPELITES_SIGMA_MODE``, and contrasts the arms head-to-head.

It reports, paired by ``(target, seed)`` so each arm is evaluated on the same
problems with the same seeds:

  * best_rank distribution per arm (median / IQR / min / max),
  * solve rate per arm (best_rank == 1),
  * archive occupancy per arm (occupied cells), and
  * the final per-operator archive sigma per arm, read from the LAST
    ``ARCHIVE_SNAPSHOT`` event of each run's trace (mean over occupied cells).

Three comparisons are highlighted for easy reading:
  * adaptive vs frozen_uniform
  * adaptive vs frozen_fixed
  * random   vs adaptive   (focus on the per-operator archive sigma: random
    redraws Dirichlet(1) each child, so its archive sigma should look flat,
    while adaptive should show operator structure if selection shapes sigma)

Input sources (see docs/architecture.md for the schemas):
  * Summaries (preferred): the per-arm experiment summary JSON files written by
    ``run_sigma_control.py`` as ``traces/sigma_control_<mode>.json``. Each row
    carries ``mapelites_sigma_mode``, ``target``, ``run_index``, ``solved``,
    ``best_rank``, ``archive_occupancy``, and ``trace_path``; the trace at
    ``trace_path`` supplies the last-snapshot per-operator sigma.
  * Traces (verification / fallback): individual MAP-Elites trace JSON files.
    Arm, target, seed, solved, best_rank, occupancy, and per-operator sigma are
    all derived directly from ``RUN_CONFIG`` and ``ARCHIVE_SNAPSHOT`` events.

This script reads traces and summaries only. It changes no solver, config, or
trace-schema code, consistent with the analysis-script invariant in
docs/architecture.md.

Usage:
    # Default: discover traces/sigma_control_*.json and compare arms.
    python scripts/compare_sigma_control_arms.py [--traces-dir traces]
        [--summaries traces/sigma_control_adaptive.json,...]
        [--report-json PATH]

    # Verification / direct-trace mode (e.g. before the batch lands):
    python scripts/compare_sigma_control_arms.py --traces "traces/ea_llm_map_elites_*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Operator order matches the sigma vector order [s, m, ml, l] used throughout
# the solver and the experiment summary's final_archive_sigma_{s,m,ml,l}.
OPERATORS = ("s_mutation", "m_mutation", "ml_mutation", "l_mutation")

# The three comparisons the task asks to make easy to read. Each is
# (left_arm, right_arm, note); order is preserved in the printed output.
COMPARISONS = (
    ("adaptive", "frozen_uniform", "does self-adaptation beat a pinned uniform prior?"),
    ("adaptive", "frozen_fixed", "does self-adaptation beat a pinned hand-set sigma?"),
    ("random", "adaptive",
     "compare the per-operator archive sigma: random should stay flat, "
     "adaptive should show operator structure if selection shapes sigma."),
)


# --------------------------------------------------------------------------- #
# Run record
# --------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    arm: str
    target: str | None
    seed: int | None
    run_index: int | None
    solved: bool | None
    best_rank: int | None
    archive_occupancy: int | None
    final_archive_sigma: list[float] | None  # mean per-operator sigma, last snapshot
    source: str

    @property
    def pair_key(self) -> tuple[str | None, int | None]:
        return (self.target, self.seed)


# --------------------------------------------------------------------------- #
# Trace parsing
# --------------------------------------------------------------------------- #
def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _load_trace_events(path: Path) -> list[dict[str, Any]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, list):
        return None
    return [event for event in data if isinstance(event, dict)]


def _run_config(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        if event.get("event") == "RUN_CONFIG":
            return _details(event)
    return {}


def _last_snapshot(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for event in events:
        if event.get("event") == "ARCHIVE_SNAPSHOT":
            last = _details(event)
    return last


def _snapshot_mean_sigma(snapshot: dict[str, Any] | None) -> list[float] | None:
    """Mean per-operator sigma over the incumbents in an ARCHIVE_SNAPSHOT."""
    if not snapshot:
        return None
    cells = snapshot.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    sums = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        sigma = cell.get("sigma")
        if not isinstance(sigma, list) or len(sigma) != 4:
            continue
        for index in range(4):
            sums[index] += float(sigma[index])
        count += 1
    if count == 0:
        return None
    return [total / count for total in sums]


def _trace_sigma(trace_path: str | None) -> list[float] | None:
    if not trace_path:
        return None
    path = Path(trace_path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / trace_path
    if not path.exists():
        return None
    events = _load_trace_events(path)
    if events is None:
        return None
    return _snapshot_mean_sigma(_last_snapshot(events))


def _is_map_elites_trace(events: list[dict[str, Any]]) -> bool:
    return any(event.get("event") == "AXIS_DEFINITION" for event in events)


def _trace_solved(events: list[dict[str, Any]], best_rank: int | None) -> bool | None:
    for event in events:
        if event.get("event") == "SOLVED":
            return True
        if event.get("event") == "FAILED":
            return False
    if best_rank is None:
        return None
    return best_rank == 1


# --------------------------------------------------------------------------- #
# Record builders
# --------------------------------------------------------------------------- #
def record_from_trace(path: Path, default_mode: str) -> RunRecord | None:
    events = _load_trace_events(path)
    if events is None or not _is_map_elites_trace(events):
        return None
    config = _run_config(events)
    snapshot = _last_snapshot(events)

    best_rank = None
    occupancy = None
    if snapshot is not None:
        raw_rank = snapshot.get("best_rank")
        if isinstance(raw_rank, (int, float)):
            best_rank = int(raw_rank)
        raw_occ = snapshot.get("occupied_cells")
        if isinstance(raw_occ, int):
            occupancy = raw_occ

    arm = config.get("mapelites_sigma_mode")
    if not isinstance(arm, str) or not arm:
        arm = default_mode
    seed = config.get("random_seed")
    run_index = config.get("run_index")

    solved = _trace_solved(events, best_rank)
    # SOLVED fires the generation AFTER the final ARCHIVE_SNAPSHOT and no snapshot is
    # written at/after the solve, so the snapshot records the pre-solve best_rank
    # (e.g. 4 or 2). A solved run's true best_rank is the rank-1 winning guess.
    if solved:
        best_rank = 1
    return RunRecord(
        arm=arm,
        target=config.get("target") if isinstance(config.get("target"), str) else None,
        seed=seed if isinstance(seed, int) else None,
        run_index=run_index if isinstance(run_index, int) else None,
        solved=solved,
        best_rank=best_rank,
        archive_occupancy=occupancy,
        final_archive_sigma=_snapshot_mean_sigma(snapshot),
        source=path.name,
    )


def records_from_summary(path: Path, default_mode: str) -> list[RunRecord]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, dict):
        return []
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    base_seed = metadata.get("random_seed")
    meta_mode = metadata.get("mapelites_sigma_mode")
    rows = data.get("runs")
    if not isinstance(rows, list):
        return []

    records: list[RunRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Only MAP-Elites rows carry a sigma mode; skip any foreign method rows.
        if row.get("method") not in (None, "ea_llm_map_elites"):
            continue
        arm = row.get("mapelites_sigma_mode") or meta_mode
        if not isinstance(arm, str) or not arm:
            arm = default_mode

        run_index = row.get("run_index")
        run_index = run_index if isinstance(run_index, int) else None
        seed = None
        if isinstance(base_seed, int) and run_index is not None:
            seed = base_seed + run_index
        elif run_index is not None:
            seed = run_index

        best_rank = row.get("best_rank")
        best_rank = int(best_rank) if isinstance(best_rank, (int, float)) else None
        occupancy = row.get("archive_occupancy")
        occupancy = occupancy if isinstance(occupancy, int) else None

        sigma = _trace_sigma(row.get("trace_path"))
        if sigma is None:
            # Fall back to the summary's own final-archive sigma columns.
            cols = [row.get(f"final_archive_sigma_{name}") for name in ("s", "m", "ml", "l")]
            if all(isinstance(value, (int, float)) for value in cols):
                sigma = [float(value) for value in cols]

        records.append(
            RunRecord(
                arm=arm,
                target=row.get("target") if isinstance(row.get("target"), str) else None,
                seed=seed,
                run_index=run_index,
                solved=bool(row["solved"]) if isinstance(row.get("solved"), bool) else None,
                best_rank=best_rank,
                archive_occupancy=occupancy,
                final_archive_sigma=sigma,
                source=path.name,
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Input resolution
# --------------------------------------------------------------------------- #
def _expand_globs(raw: str) -> list[Path]:
    paths: list[Path] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        matches = glob.glob(piece)
        if matches:
            paths.extend(Path(match) for match in sorted(matches))
        else:
            paths.append(Path(piece))
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def collect_records(args: argparse.Namespace) -> tuple[list[RunRecord], str]:
    if args.traces:
        records: list[RunRecord] = []
        for path in _expand_globs(args.traces):
            record = record_from_trace(path, args.default_mode)
            if record is not None:
                records.append(record)
        return records, "traces"

    if args.summaries:
        summary_paths = _expand_globs(args.summaries)
    else:
        summary_paths = sorted(Path(args.traces_dir).glob("sigma_control_*.json"))

    records = []
    for path in summary_paths:
        records.extend(records_from_summary(path, args.default_mode))
    return records, "summaries"


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def rank_dist(values: list[int]) -> dict[str, Any]:
    values = sorted(value for value in values if value is not None)
    n = len(values)
    if n == 0:
        return {"n": 0}
    if n >= 4:
        q1, _, q3 = statistics.quantiles(values, n=4)
    else:
        q1, q3 = values[0], values[-1]
    return {
        "n": n,
        "median": statistics.median(values),
        "q1": q1,
        "q3": q3,
        "min": values[0],
        "max": values[-1],
    }


def occupancy_dist(values: list[int]) -> dict[str, Any]:
    return rank_dist(values)


def mean_sigma(records: list[RunRecord]) -> list[float] | None:
    vectors = [r.final_archive_sigma for r in records if r.final_archive_sigma is not None]
    if not vectors:
        return None
    sums = [0.0, 0.0, 0.0, 0.0]
    for vector in vectors:
        for index in range(4):
            sums[index] += vector[index]
    count = len(vectors)
    return [total / count for total in sums]


def solve_rate(records: list[RunRecord]) -> tuple[int, int]:
    decided = [r for r in records if r.solved is not None]
    solved = sum(1 for r in records if r.solved)
    return solved, len(decided)


def _fmt_rank_dist(dist: dict[str, Any]) -> str:
    if dist["n"] == 0:
        return "n=0"
    return (f"n={dist['n']:>2} med={dist['median']:>8.0f} "
            f"[{dist['q1']:>8.0f},{dist['q3']:>8.0f}] "
            f"min={dist['min']:>7.0f} max={dist['max']:>8.0f}")


def _fmt_occ_dist(dist: dict[str, Any]) -> str:
    if dist["n"] == 0:
        return "n=0"
    return (f"med={dist['median']:>5.1f} "
            f"[{dist['min']:>2.0f},{dist['max']:>2.0f}]")


def _fmt_sigma(sigma: list[float] | None) -> str:
    if sigma is None:
        return "[   --  ,   --  ,   --  ,   --  ]"
    return "[" + ", ".join(f"{value:6.3f}" for value in sigma) + "]"


def _fmt_rate(num: int, den: int) -> str:
    if den == 0:
        return "  n/a"
    return f"{num / den * 100:5.1f}% ({num}/{den})"


def arm_summary(records: list[RunRecord]) -> dict[str, Any]:
    solved, decided = solve_rate(records)
    return {
        "n_runs": len(records),
        "solved": solved,
        "decided": decided,
        "solve_rate": (solved / decided) if decided else None,
        "best_rank": rank_dist([r.best_rank for r in records]),
        "archive_occupancy": occupancy_dist([r.archive_occupancy for r in records]),
        "mean_archive_sigma": mean_sigma(records),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def report_overview(by_arm: dict[str, list[RunRecord]]) -> dict[str, Any]:
    print("\n" + "=" * 92)
    print("PER-ARM OVERVIEW (all runs, grouped by MAPELITES_SIGMA_MODE)")
    print("=" * 92)
    print(f"{'arm':<16} {'runs':>5} {'solve_rate':>15}  {'best_rank (median [IQR] min/max)':<52}")
    print("-" * 92)
    out: dict[str, Any] = {}
    for arm in sorted(by_arm):
        summary = arm_summary(by_arm[arm])
        out[arm] = summary
        print(f"{arm:<16} {summary['n_runs']:>5} "
              f"{_fmt_rate(summary['solved'], summary['decided']):>15}  "
              f"{_fmt_rank_dist(summary['best_rank']):<52}")
    print("\nFinal per-operator archive sigma per arm (mean over runs of each run's")
    print("last ARCHIVE_SNAPSHOT mean sigma); order = [s, m, ml, l]:")
    print(f"{'arm':<16} {'occupancy':>16}  {'archive sigma [s, m, ml, l]':<40}")
    print("-" * 92)
    for arm in sorted(by_arm):
        summary = out[arm]
        print(f"{arm:<16} {_fmt_occ_dist(summary['archive_occupancy']):>16}  "
              f"{_fmt_sigma(summary['mean_archive_sigma']):<40}")
    return out


def _paired_subset(
    left: list[RunRecord],
    right: list[RunRecord],
) -> tuple[list[RunRecord], list[RunRecord], list[tuple[Any, Any]]]:
    """Return left/right records restricted to shared (target, seed) keys."""
    left_by_key = {r.pair_key: r for r in left if None not in r.pair_key}
    right_by_key = {r.pair_key: r for r in right if None not in r.pair_key}
    shared = sorted(set(left_by_key) & set(right_by_key))
    paired_left = [left_by_key[key] for key in shared]
    paired_right = [right_by_key[key] for key in shared]
    return paired_left, paired_right, shared


def _paired_rank_outcome(
    left: list[RunRecord],
    right: list[RunRecord],
) -> dict[str, Any]:
    """Per-pair best_rank head-to-head (lower rank is better)."""
    left_wins = right_wins = ties = comparable = 0
    deltas: list[float] = []
    for left_record, right_record in zip(left, right):
        if left_record.best_rank is None or right_record.best_rank is None:
            continue
        comparable += 1
        deltas.append(float(right_record.best_rank - left_record.best_rank))
        if left_record.best_rank < right_record.best_rank:
            left_wins += 1
        elif right_record.best_rank < left_record.best_rank:
            right_wins += 1
        else:
            ties += 1
    return {
        "comparable_pairs": comparable,
        "left_better": left_wins,
        "right_better": right_wins,
        "ties": ties,
        "median_rank_delta_right_minus_left": statistics.median(deltas) if deltas else None,
    }


def report_comparison(
    left_arm: str,
    right_arm: str,
    note: str,
    by_arm: dict[str, list[RunRecord]],
) -> dict[str, Any]:
    print("\n" + "=" * 92)
    print(f"COMPARISON: {left_arm}  vs  {right_arm}")
    print(f"  ({note})")
    print("=" * 92)

    left_all = by_arm.get(left_arm, [])
    right_all = by_arm.get(right_arm, [])
    if not left_all or not right_all:
        missing = [arm for arm, runs in ((left_arm, left_all), (right_arm, right_all)) if not runs]
        print(f"  arm(s) not present in input: {', '.join(missing)}; skipping comparison.")
        return {"available": False, "missing_arms": missing}

    paired_left, paired_right, shared = _paired_subset(left_all, right_all)
    print(f"  paired (target, seed) keys present in both arms: {len(shared)}")
    if not shared:
        print("  no shared (target, seed) pairs; cannot pair. Showing nothing further.")
        return {"available": True, "n_pairs": 0}

    left_sum = arm_summary(paired_left)
    right_sum = arm_summary(paired_right)

    label_width = max(len(left_arm), len(right_arm), 6)
    print(f"\n  {'metric':<26} {left_arm:<{label_width}}   {right_arm:<{label_width}}")
    print("  " + "-" * (26 + 2 * label_width + 6))
    print(f"  {'paired runs':<26} "
          f"{left_sum['n_runs']:<{label_width}}   {right_sum['n_runs']:<{label_width}}")
    print(f"  {'solve rate':<26} "
          f"{_fmt_rate(left_sum['solved'], left_sum['decided']):<{label_width}}   "
          f"{_fmt_rate(right_sum['solved'], right_sum['decided']):<{label_width}}")
    print(f"  {'best_rank median':<26} "
          f"{_fmt_median(left_sum['best_rank']):<{label_width}}   "
          f"{_fmt_median(right_sum['best_rank']):<{label_width}}")
    print(f"  {'best_rank IQR':<26} "
          f"{_fmt_iqr(left_sum['best_rank']):<{label_width}}   "
          f"{_fmt_iqr(right_sum['best_rank']):<{label_width}}")
    print(f"  {'archive occupancy median':<26} "
          f"{_fmt_occ_median(left_sum['archive_occupancy']):<{label_width}}   "
          f"{_fmt_occ_median(right_sum['archive_occupancy']):<{label_width}}")

    print(f"\n  per-operator archive sigma [s, m, ml, l] (mean over paired runs):")
    print(f"    {left_arm:<16} {_fmt_sigma(left_sum['mean_archive_sigma'])}")
    print(f"    {right_arm:<16} {_fmt_sigma(right_sum['mean_archive_sigma'])}")
    if left_sum["mean_archive_sigma"] and right_sum["mean_archive_sigma"]:
        diff = [l - r for l, r in
                zip(left_sum["mean_archive_sigma"], right_sum["mean_archive_sigma"])]
        print(f"    {'difference':<16} {_fmt_sigma(diff)}  ({left_arm} - {right_arm})")

    outcome = _paired_rank_outcome(paired_left, paired_right)
    print(f"\n  paired best_rank head-to-head (lower rank wins):")
    print(f"    {left_arm} better: {outcome['left_better']}   "
          f"{right_arm} better: {outcome['right_better']}   "
          f"ties: {outcome['ties']}   (comparable pairs: {outcome['comparable_pairs']})")
    if outcome["median_rank_delta_right_minus_left"] is not None:
        print(f"    median (rank_{right_arm} - rank_{left_arm}) = "
              f"{outcome['median_rank_delta_right_minus_left']:.1f} "
              f"(positive => {left_arm} reaches better ranks)")

    return {
        "available": True,
        "n_pairs": len(shared),
        "shared_keys": [list(key) for key in shared],
        "left": {"arm": left_arm, **left_sum},
        "right": {"arm": right_arm, **right_sum},
        "paired_best_rank_outcome": outcome,
    }


def _fmt_median(dist: dict[str, Any]) -> str:
    return "--" if dist["n"] == 0 else f"{dist['median']:.0f}"


def _fmt_iqr(dist: dict[str, Any]) -> str:
    return "--" if dist["n"] == 0 else f"[{dist['q1']:.0f}, {dist['q3']:.0f}]"


def _fmt_occ_median(dist: dict[str, Any]) -> str:
    return "--" if dist["n"] == 0 else f"{dist['median']:.1f}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MAP-Elites sigma-control arms (reads traces/summaries only).")
    parser.add_argument("--traces-dir", default="traces",
                        help="Directory scanned for sigma_control_*.json summaries when "
                             "--summaries is omitted.")
    parser.add_argument("--summaries",
                        help="Comma-separated summary JSON paths or globs (per-arm batch "
                             "outputs from run_sigma_control.py).")
    parser.add_argument("--traces",
                        help="Comma-separated MAP-Elites trace paths or globs. When set, runs "
                             "are built directly from traces (verification / fallback mode) "
                             "instead of from summaries.")
    parser.add_argument("--default-mode", default="unknown",
                        help="Arm label assigned to runs whose sigma mode is absent from the "
                             "trace/summary (older traces predate MAPELITES_SIGMA_MODE).")
    parser.add_argument("--report-json", help="Optional path to dump the full report as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, source = collect_records(args)
    if not records:
        print(f"No MAP-Elites runs found via {source}. "
              f"(traces-dir={args.traces_dir}, summaries={args.summaries}, traces={args.traces})",
              file=sys.stderr)
        return 1

    by_arm: dict[str, list[RunRecord]] = {}
    for record in records:
        by_arm.setdefault(record.arm, []).append(record)

    print(f"Loaded {len(records)} MAP-Elites run(s) from {source} across "
          f"{len(by_arm)} arm(s): {', '.join(sorted(by_arm))}.")
    paired_complete = sum(1 for r in records if None not in r.pair_key)
    if paired_complete < len(records):
        print(f"[note] {len(records) - paired_complete} run(s) lack a (target, seed) key and "
              f"cannot be paired (e.g. older traces without run_index/seed in RUN_CONFIG).")

    report: dict[str, Any] = {
        "source": source,
        "n_runs": len(records),
        "arms": sorted(by_arm),
        "overview": report_overview(by_arm),
        "comparisons": {},
    }
    for left_arm, right_arm, note in COMPARISONS:
        report["comparisons"][f"{left_arm}_vs_{right_arm}"] = report_comparison(
            left_arm, right_arm, note, by_arm
        )

    print("\nNotes: best_rank is the run's best archive rank (1 = solved). Pairing is by")
    print("(target, seed) so each arm is judged on identical problems. Per-operator archive")
    print("sigma is read from the LAST ARCHIVE_SNAPSHOT of each run's trace (mean over cells).")

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON report to {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
