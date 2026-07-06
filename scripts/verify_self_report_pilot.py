"""Verify the RQ1 self-report pilot from raw trace files.

This reads the raw per-run solver trace JSON directly (NOT any experiment
summary) and reports whether the self-report instrumentation produced usable,
internally-consistent records. It is the gate for calling the instrumentation
done; run it manually after the pilot completes.

Pilot launch (do NOT run as part of this script), PowerShell:

    $env:SELF_REPORT="1"; python -m contexto_solver.experiment \
        --method ea_llm_map_elites --provider ollama \
        --targets ivory --runs-per-target 1 --max-generations 10 \
        --random-seed 0 --output traces/pilot_self_report.json

    # ea_llm_self_adaptive is the alternate method; same SELF_REPORT flag.

Then verify (point it at the per-run trace file(s), e.g. traces/ea_llm_map_elites_*_ivory_run1_*.json):

    python scripts/verify_self_report_pilot.py traces/ea_llm_map_elites_*_ivory_run1_*.json

Analysis only: reads traces, computes metrics, prints a report. It never runs the
solver, calls an LLM, or modifies any file.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any

# Acceptance thresholds (Phase 4).
PARSE_FAILURE_HARD_MAX = 0.10
PARSE_FAILURE_DISCUSS_MAX = 0.02
BASIS_WORDS_NONEMPTY_MIN = 0.90


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a solver trace (expected a list of events).")
    return data


def _record_from_self_report(
    self_report: dict[str, Any],
    source_event: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_event": source_event,
        "generation": context.get("generation"),
        "operator": context.get("operator"),
        "word": context.get("word"),
        "parents": context.get("parents"),
        "predicted_closeness": self_report.get("predicted_closeness"),
        "predicted_closeness_clamped": bool(self_report.get("predicted_closeness_clamped")),
        "rationale": self_report.get("rationale"),
        "self_report_parse_failed": bool(self_report.get("self_report_parse_failed")),
        "self_report_raw": self_report.get("self_report_raw"),
        "self_report_prompt": self_report.get("self_report_prompt"),
    }


def extract_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect one self-report record per proposed individual.

    Sources: ``OPERATOR_SAMPLED`` (mutation children) and ``CROSSOVER`` (crossover
    children, via the serialized child). Self-adaptive ``MUTATE.children`` also
    carry the record but duplicate ``OPERATOR_SAMPLED``, so they are skipped.
    """
    records: list[dict[str, Any]] = []
    for event in events:
        name = event.get("event")
        details = event.get("details", {}) or {}
        generation = event.get("generation")
        if name == "OPERATOR_SAMPLED" and isinstance(details.get("self_report"), dict):
            records.append(
                _record_from_self_report(
                    details["self_report"],
                    "OPERATOR_SAMPLED",
                    {
                        "generation": generation,
                        "operator": details.get("sampled_op"),
                        "word": details.get("child_hypothesis_name"),
                        "parents": [details.get("parent_id")],
                    },
                )
            )
        elif name == "CROSSOVER":
            child = details.get("child")
            if isinstance(child, dict) and isinstance(child.get("self_report"), dict):
                records.append(
                    _record_from_self_report(
                        child["self_report"],
                        "CROSSOVER",
                        {
                            "generation": generation,
                            "operator": "crossover",
                            "word": child.get("best_word"),
                            "parents": details.get("parent_ids") or details.get("parents"),
                        },
                    )
                )
    return records


def _basis_words(record: dict[str, Any]) -> list[str]:
    rationale = record.get("rationale")
    if not isinstance(rationale, dict):
        return []
    words = rationale.get("basis_words")
    return words if isinstance(words, list) else []


def basis_words_in_prompt(record: dict[str, Any]) -> tuple[int, int]:
    """Return (count of basis_words present in the stored prompt, total basis_words)."""
    prompt = record.get("self_report_prompt") or ""
    words = _basis_words(record)
    present = sum(1 for word in words if isinstance(word, str) and word in prompt)
    return present, len(words)


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    parse_failed = [r for r in records if r["self_report_parse_failed"]]
    parsed = [r for r in records if not r["self_report_parse_failed"]]
    closeness_values = [r["predicted_closeness"] for r in records if r["predicted_closeness"] is not None]
    clamped = [r for r in records if r["predicted_closeness_clamped"]]
    empty_basis = [r for r in records if not _basis_words(r)]

    parsed_with_basis = sum(1 for r in parsed if _basis_words(r))
    basis_membership_violations = []
    for r in records:
        present, count = basis_words_in_prompt(r)
        if count and present < count:
            missing = [w for w in _basis_words(r) if w not in (r.get("self_report_prompt") or "")]
            basis_membership_violations.append({"word": r.get("word"), "missing_basis_words": missing})

    in_range = all(0.0 <= value <= 1.0 for value in closeness_values)

    return {
        "total_proposals": total,
        "parse_failure_count": len(parse_failed),
        "parse_failure_rate": (len(parse_failed) / total) if total else None,
        "predicted_closeness_present": len(closeness_values),
        "predicted_closeness_min": min(closeness_values) if closeness_values else None,
        "predicted_closeness_max": max(closeness_values) if closeness_values else None,
        "predicted_closeness_mean": (sum(closeness_values) / len(closeness_values)) if closeness_values else None,
        "predicted_closeness_all_in_range": in_range,
        "clamped_count": len(clamped),
        "empty_basis_words_count": len(empty_basis),
        "parsed_count": len(parsed),
        "parsed_with_nonempty_basis": parsed_with_basis,
        "parsed_basis_nonempty_rate": (parsed_with_basis / len(parsed)) if parsed else None,
        "basis_membership_violations": basis_membership_violations,
    }


def evaluate_thresholds(metrics: dict[str, Any]) -> dict[str, Any]:
    failure_rate = metrics["parse_failure_rate"]
    basis_rate = metrics["parsed_basis_nonempty_rate"]
    checks = {
        "parse_failure_within_hard_max": (failure_rate is not None and failure_rate <= PARSE_FAILURE_HARD_MAX),
        "parse_failure_needs_discussion": (failure_rate is not None and failure_rate > PARSE_FAILURE_DISCUSS_MAX),
        "predicted_closeness_all_in_range": metrics["predicted_closeness_all_in_range"],
        "basis_words_nonempty_ok": (basis_rate is not None and basis_rate >= BASIS_WORDS_NONEMPTY_MIN),
        "basis_membership_clean": not metrics["basis_membership_violations"],
    }
    checks["all_pass"] = (
        checks["parse_failure_within_hard_max"]
        and checks["predicted_closeness_all_in_range"]
        and checks["basis_words_nonempty_ok"]
        and checks["basis_membership_clean"]
    )
    return checks


def _format_metrics(metrics: dict[str, Any]) -> str:
    lines = [
        f"total proposals:            {metrics['total_proposals']}",
        f"parse failures:             {metrics['parse_failure_count']} "
        f"(rate={_fmt(metrics['parse_failure_rate'])})",
        f"predicted_closeness present:{metrics['predicted_closeness_present']}",
        f"  min/mean/max:             {_fmt(metrics['predicted_closeness_min'])} / "
        f"{_fmt(metrics['predicted_closeness_mean'])} / {_fmt(metrics['predicted_closeness_max'])}",
        f"  all in [0,1]:             {metrics['predicted_closeness_all_in_range']}",
        f"clamped values:             {metrics['clamped_count']}",
        f"empty basis_words:          {metrics['empty_basis_words_count']}",
        f"parsed w/ nonempty basis:   {metrics['parsed_with_nonempty_basis']}/{metrics['parsed_count']} "
        f"(rate={_fmt(metrics['parsed_basis_nonempty_rate'])})",
        f"basis-in-prompt violations: {len(metrics['basis_membership_violations'])}",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


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
    parser = argparse.ArgumentParser(description="Verify the RQ1 self-report pilot from raw trace files.")
    parser.add_argument("traces", nargs="+", help="Raw trace JSON file path(s) or glob(s).")
    parser.add_argument("--sample", type=int, default=5, help="Number of full records to print verbatim.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the record sample.")
    parser.add_argument("--report-json", help="Optional path to dump the full structured report.")
    args = parser.parse_args()

    paths = _expand_paths(args.traces)
    if not paths:
        raise SystemExit("No trace files matched the given path(s).")

    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(extract_records(load_trace(path)))

    metrics = compute_metrics(records)
    checks = evaluate_thresholds(metrics)

    print(f"Trace files read: {len(paths)}")
    for path in paths:
        print(f"  - {path}")
    print()
    print(_format_metrics(metrics))
    print()
    print("Acceptance checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    if checks["parse_failure_needs_discussion"] and checks["parse_failure_within_hard_max"]:
        print("  NOTE: parse-failure rate exceeds 2% (within the 10% hard limit); flag for discussion.")

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.sample, len(records))) if records else []
    print(f"\n=== {len(sample)} sampled full records ===")
    for record in sample:
        print(json.dumps(record, indent=2, ensure_ascii=False))

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(
                {"trace_files": paths, "metrics": metrics, "checks": checks, "records": records},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote report JSON: {args.report_json}")


if __name__ == "__main__":
    main()
