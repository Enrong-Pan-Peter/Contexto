"""Cross-run RQ1 report: medians/IQR, two-arm tests, power gate, pgfplots CSVs.

Reads raw per-run solver traces, reduces each run to scalar calibration metrics,
and writes: a per-run metric CSV, a cross-run median/IQR summary CSV, a pooled
reliability-curve CSV (and optional PNG), and a JSON bundle with the two-arm
comparison and the power-gate variance table. Analysis only: no network, LLM, or
trace/cache writes beyond the report files it is told to produce.

Usage (PowerShell):

    python scripts/rq1_report.py traces/ea_llm_self_adaptive_api_*.json \
        --output-dir traces/rq1_report --which child_best \
        --arm-key sigma_mode --arm-a adaptive --arm-b frozen_uniform \
        --paired --effect 0.1 --reliability-image
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contexto_solver.rq1 import report as rq1_report
from contexto_solver.rq1.reader import Individual, extract_individuals, load_trace, run_config
from contexto_solver.rq1.records import provenance_hashes

_ARM_KEYS = {
    "mode": lambda config: config.method,
    "method": lambda config: config.method,
    "sigma_mode": lambda config: config.sigma_mode,
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
    parser = argparse.ArgumentParser(description="Cross-run RQ1 report and power gate.")
    parser.add_argument("traces", nargs="+", help="Raw trace JSON path(s) or glob(s).")
    parser.add_argument("--output-dir", required=True, help="Directory for the report CSVs and JSON.")
    parser.add_argument("--which", choices=["child_best", "first_proposed"], default="child_best")
    parser.add_argument("--arm-key", choices=sorted(_ARM_KEYS), help="Run attribute defining arms.")
    parser.add_argument("--arm-a", help="Arm A label (value of --arm-key).")
    parser.add_argument("--arm-b", help="Arm B label (value of --arm-key).")
    parser.add_argument("--paired", action="store_true", help="Pair runs by game_number for Wilcoxon.")
    parser.add_argument("--effect", type=float, help="Effect size for the power-gate run-count estimate.")
    parser.add_argument("--reliability-image", action="store_true", help="Also save a pooled reliability PNG.")
    args = parser.parse_args()

    paths = _expand_paths(args.traces)
    if not paths:
        raise SystemExit("No trace files matched the given path(s).")

    individuals_by_run: dict[str, list[Individual]] = {}
    run_config_by_run: dict[str, Any] = {}
    all_individuals: list[Individual] = []
    for path in paths:
        name = Path(path).name
        events = load_trace(path)
        config = run_config(events)
        individuals = extract_individuals(events, trace_file=name)
        individuals_by_run[name] = individuals
        run_config_by_run[name] = config
        all_individuals.extend(individuals)

    provenance = provenance_hashes(all_individuals)
    if provenance.mixed:
        print("WARNING: mixed instrumentation_provenance_hash across traces.", file=sys.stderr)

    scalars = rq1_report.scalars_by_run(individuals_by_run, args.which)
    summary = rq1_report.cross_run_summary(scalars)
    variance = rq1_report.variance_table(scalars, effect=args.effect)
    pooled_reliability = rq1_report.pooled_reliability(individuals_by_run, args.which)

    two_arm: dict[str, Any] | None = None
    if args.arm_key and args.arm_a and args.arm_b:
        arm_fn = _ARM_KEYS[args.arm_key]
        arm_of = {name: arm_fn(config) for name, config in run_config_by_run.items()}
        paired_key = None
        if args.paired:
            paired_key = {name: config.game_number for name, config in run_config_by_run.items()}
        two_arm = rq1_report.two_arm_comparison(
            scalars, arm_of, args.arm_a, args.arm_b, paired_key=paired_key
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rq1_report.write_per_run_csv(out_dir / "per_run_metrics.csv", scalars)
    rq1_report.write_cross_run_csv(out_dir / "cross_run_summary.csv", summary)
    rq1_report.write_reliability_csv(out_dir / "reliability_curve.csv", pooled_reliability)
    if args.reliability_image:
        rq1_report.save_reliability_image(
            pooled_reliability, out_dir / "reliability_curve.png", title="Pooled calibration"
        )

    bundle = {
        "traces": [Path(p).name for p in paths],
        "which_realized": args.which,
        "n_runs": len(individuals_by_run),
        "provenance_mixed": provenance.mixed,
        "provenance_hashes": {str(k): v for k, v in provenance.hashes.items()},
        "per_run_scalars": scalars,
        "cross_run_summary": summary,
        "variance_power_gate": variance,
        "two_arm_comparison": two_arm,
        "pooled_reliability": pooled_reliability,
    }
    (out_dir / "report.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"n_runs": len(individuals_by_run), "cross_run_summary": summary,
                      "two_arm_comparison": two_arm, "variance_power_gate": variance}, indent=2, ensure_ascii=False))
    print(f"\nWrote report to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
