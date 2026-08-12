"""Compute RQ1 calibration metrics per run and pooled, with splits.

Reads raw per-run solver traces, extracts self-reported individuals, and reports
Spearman rho, bucket accuracy / confusion / signed error / off-by-one, a binned
reliability curve with ECE, Brier score, and AUROC -- overall and split by
operator, generation, parent-rank bin, inheritance status, and mode. Output is
JSON (stdout and/or ``--output``). Analysis only: no network, LLM, or trace/cache
writes.

Usage (PowerShell):

    python scripts/rq1_metrics.py traces/ea_llm_self_adaptive_api_*.json \
        --output traces/rq1_metrics.json --which child_best

With ``--augmented-csv`` it also writes the tidy per-individual rows plus the
derived realized bucket / positive label / bucket-error columns (the agreement
writeback), so the exact rows behind every metric are inspectable.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contexto_solver.rq1 import metrics as rq1_metrics
from contexto_solver.rq1.reader import Individual, extract_individuals, load_trace, run_config
from contexto_solver.rq1.records import TIDY_COLUMNS, provenance_hashes, to_tidy_row


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(sorted(matches))
        elif Path(pattern).exists():
            paths.append(pattern)
    return list(dict.fromkeys(paths))


_AUGMENTED_EXTRA = [
    "realized_rank_used",
    "realized_bucket",
    "positive_top100",
    "predicted_bucket_ordinal",
    "realized_bucket_ordinal",
    "bucket_signed_error",
]


def _augmented_row(individual: Individual, which: str) -> dict[str, Any]:
    row = to_tidy_row(individual)
    rank = rq1_metrics.realized_rank(individual, which)
    realized_b = rq1_metrics.realized_bucket(rank)
    pred_ord = rq1_metrics.bucket_ordinal(individual.predicted_bucket)
    realized_ord = rq1_metrics.bucket_ordinal(realized_b)
    row["realized_rank_used"] = rank
    row["realized_bucket"] = realized_b
    row["positive_top100"] = (None if rank is None else int(rank <= rq1_metrics.POSITIVE_RANK_THRESHOLD))
    row["predicted_bucket_ordinal"] = pred_ord
    row["realized_bucket_ordinal"] = realized_ord
    row["bucket_signed_error"] = (
        (pred_ord - realized_ord) if (pred_ord is not None and realized_ord is not None) else None
    )
    return row


def _write_augmented_csv(path: str, individuals: list[Individual], which: str) -> None:
    import csv

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(TIDY_COLUMNS) + _AUGMENTED_EXTRA
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for individual in individuals:
            writer.writerow(_augmented_row(individual, which))
    print(f"Wrote augmented CSV: {out_path} ({len(individuals)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RQ1 calibration metrics.")
    parser.add_argument("traces", nargs="+", help="Raw trace JSON path(s) or glob(s).")
    parser.add_argument("--output", help="Path to write the metrics JSON (also printed to stdout).")
    parser.add_argument(
        "--which",
        choices=["child_best", "first_proposed"],
        default="child_best",
        help="Which realized rank to calibrate against.",
    )
    parser.add_argument("--augmented-csv", help="Optional path to write the per-individual augmented rows.")
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Exit non-zero if pooled traces carry mixed provenance hashes.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.traces)
    if not paths:
        raise SystemExit("No trace files matched the given path(s).")

    per_run: dict[str, Any] = {}
    all_individuals: list[Individual] = []
    for path in paths:
        events = load_trace(path)
        config = run_config(events)
        individuals = extract_individuals(events, trace_file=Path(path).name)
        all_individuals.extend(individuals)
        per_run[Path(path).name] = {
            "method": config.method,
            "game_number": config.game_number,
            "provenance_hash": config.provenance_hash,
            "metrics": rq1_metrics.metrics_summary(individuals, args.which),
        }

    provenance = provenance_hashes(all_individuals)
    if provenance.mixed:
        print(
            "WARNING: mixed instrumentation_provenance_hash across traces; pooled "
            "metrics mix substrates.",
            file=sys.stderr,
        )
        if args.strict_provenance:
            raise SystemExit("Refusing to pool under --strict-provenance with mixed provenance hashes.")

    report = {
        "traces": [Path(p).name for p in paths],
        "which_realized": args.which,
        "provenance_hashes": {str(k): v for k, v in provenance.hashes.items()},
        "provenance_mixed": provenance.mixed,
        "per_run": per_run,
        "pooled": rq1_metrics.metrics_with_splits(all_individuals, args.which),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nWrote metrics JSON: {args.output}", file=sys.stderr)
    if args.augmented_csv:
        _write_augmented_csv(args.augmented_csv, all_individuals, args.which)


if __name__ == "__main__":
    main()
