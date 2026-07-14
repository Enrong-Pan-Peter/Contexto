"""Build the RQ1 tidy per-individual table from raw solver traces.

Reads one or more raw per-run trace JSON files (NOT experiment ``--output``
summaries) and emits a CSV with one row per self-reported individual: run
metadata, the operator self-report (predicted_closeness / predicted_bucket), and
the realized outcome joined back from the trace (the first proposed word's rank
plus the child's realized best). Analysis only: no network, no LLM, no writes to
traces or caches.

Usage (PowerShell):

    python scripts/rq1_tidy_table.py traces/ea_llm_self_adaptive_api_*.json \
        --output traces/rq1_tidy.csv

Provenance guard: if the traces carry more than one
``instrumentation_provenance_hash`` the substrate differed between runs; the
script warns and, with ``--strict-provenance``, exits non-zero before writing.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contexto_solver.rq1.reader import Individual, extract_individuals, load_trace, run_config
from contexto_solver.rq1.records import provenance_hashes, write_tidy_csv


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(sorted(matches))
        elif Path(pattern).exists():
            paths.append(pattern)
    return list(dict.fromkeys(paths))


def _summarize(individuals: list[Individual]) -> dict[str, Any]:
    with_pred = [i for i in individuals if i.predicted_closeness is not None]
    parse_failed = [i for i in individuals if i.self_report_parse_failed]
    with_realized = [i for i in individuals if i.realized_rank is not None]
    invalid_first = [i for i in individuals if i.proposed_word_invalid]
    return {
        "individuals": len(individuals),
        "with_predicted_closeness": len(with_pred),
        "parse_failed": len(parse_failed),
        "with_realized_rank": len(with_realized),
        "invalid_first_word": len(invalid_first),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RQ1 tidy per-individual table.")
    parser.add_argument("traces", nargs="+", help="Raw trace JSON path(s) or glob(s).")
    parser.add_argument("--output", required=True, help="Path to the CSV to write.")
    parser.add_argument(
        "--strict-provenance",
        action="store_true",
        help="Exit non-zero (before writing) if traces carry mixed provenance hashes.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.traces)
    if not paths:
        raise SystemExit("No trace files matched the given path(s).")

    individuals: list[Individual] = []
    print(f"Trace files read: {len(paths)}")
    for path in paths:
        events = load_trace(path)
        config = run_config(events)
        extracted = extract_individuals(events, trace_file=Path(path).name)
        individuals.extend(extracted)
        target = extracted[0].target if extracted else None
        print(
            f"  - {Path(path).name}: method={config.method} game_number={config.game_number} "
            f"target={target} individuals={len(extracted)} provenance={config.provenance_hash}"
        )

    provenance = provenance_hashes(individuals)
    print()
    print(f"Provenance hashes: {provenance.hashes}")
    if provenance.mixed:
        print(
            "WARNING: mixed instrumentation_provenance_hash across traces. The prompt/"
            "instrumentation substrate differed; do NOT pool these records without "
            "accounting for it."
        )
        if args.strict_provenance:
            raise SystemExit("Refusing to write under --strict-provenance with mixed provenance hashes.")

    summary = _summarize(individuals)
    print()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    out_path = write_tidy_csv(args.output, individuals)
    print(f"\nWrote tidy table: {out_path} ({len(individuals)} rows)")


if __name__ == "__main__":
    main()
