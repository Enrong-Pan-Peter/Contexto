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
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contexto_solver.self_report import read_self_report

# Acceptance thresholds (Phase 4).
PARSE_FAILURE_HARD_MAX = 0.10
PARSE_FAILURE_DISCUSS_MAX = 0.02
BASIS_WORDS_NONEMPTY_MIN = 0.90
# Degenerate-clustering flag: a self-report signal is unusable for calibration if
# a single closeness value covers more than half the parsed reports, or the
# spread is effectively zero (the model always emits the same number).
CLUSTER_DOMINANT_VALUE_FRACTION = 0.50
CLUSTER_NEAR_ZERO_STD = 0.01

# The canonical self-report record shape, defined by Hypothesis.self_report_dict()
# and resolve_self_report(). Every mode -- including the non-Hypothesis llm_only
# path -- must serialize exactly these keys for the trace schema to be identical.
CANONICAL_SELF_REPORT_KEYS = frozenset(
    {
        "predicted_closeness",
        "predicted_closeness_clamped",
        "predicted_bucket",
        "rationale",
        "self_report_parse_failed",
        "self_report_raw",
        "self_report_prompt",
        "injected_rationale_hash",
        "rationale_truncated",
    }
)


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        hint = ""
        if isinstance(data, dict) and isinstance(data.get("runs"), list) and data["runs"]:
            first = data["runs"][0] if isinstance(data["runs"][0], dict) else {}
            trace_path = first.get("trace_path") if isinstance(first, dict) else None
            if trace_path:
                hint = (
                    f" This looks like an experiment summary (--output), not a solver "
                    f"trace. Pass the per-run file instead, e.g. {trace_path}"
                )
            else:
                hint = (
                    " This looks like an experiment summary (--output), not a solver "
                    "trace. Pass the per-run traces listed under runs[].trace_path."
                )
        raise ValueError(
            f"{path} is not a solver trace (expected a list of events).{hint}"
        )
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
        # ``guess_word`` is the single lowercase token actually submitted to the
        # game (present for llm_only GUESS and crossover children). ``hypothesis_name``
        # is the human-readable category LABEL, which may be a multi-word phrase and
        # is NEVER guessed. Keep them separate so a phrase label is not mistaken for
        # a two-word guess.
        "guess_word": context.get("guess_word"),
        "hypothesis_name": context.get("hypothesis_name"),
        "parents": context.get("parents"),
        "predicted_closeness": self_report.get("predicted_closeness"),
        "predicted_closeness_clamped": bool(self_report.get("predicted_closeness_clamped")),
        "predicted_bucket": self_report.get("predicted_bucket"),
        "rationale": self_report.get("rationale"),
        "self_report_parse_failed": bool(self_report.get("self_report_parse_failed")),
        "self_report_raw": self_report.get("self_report_raw"),
        "self_report_prompt": self_report.get("self_report_prompt"),
        "injected_rationale_hash": self_report.get("injected_rationale_hash"),
        "rationale_truncated": bool(self_report.get("rationale_truncated")),
    }


def extract_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect one self-report record per proposed individual.

    Sources, one per proposed word across the live modes:
    - ``OPERATOR_SAMPLED`` (self-adaptive / MAP-Elites mutation children),
    - ``CROSSOVER`` (crossover children, via the serialized child),
    - ``GUESS`` (``llm_only`` per-guess records, written by the non-Hypothesis
      trace helper). Only ``llm_only`` attaches ``self_report`` to ``GUESS``; EA
      modes log guesses without it, so this branch does not double-count.

    Self-adaptive ``MUTATE.children`` also carry the record but duplicate
    ``OPERATOR_SAMPLED``, so they are skipped.
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
                        # child not yet guessed at this event; only the label exists
                        "guess_word": None,
                        "hypothesis_name": details.get("child_hypothesis_name"),
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
                            "guess_word": child.get("best_word"),
                            "hypothesis_name": child.get("category_name"),
                            "parents": details.get("parent_ids") or details.get("parents"),
                        },
                    )
                )
        elif name == "GUESS" and isinstance(details.get("self_report"), dict):
            records.append(
                _record_from_self_report(
                    details["self_report"],
                    "GUESS",
                    {
                        "generation": generation,
                        "operator": "next_guess",
                        "guess_word": details.get("word"),
                        "hypothesis_name": None,
                        "parents": None,
                    },
                )
            )
    return records


def schema_audit(paths_and_events: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Audit trace-schema identity across files, per source event.

    Guards against the schema divergence most likely in ``llm_only``, which uses
    a non-Hypothesis trace helper: verifies every raw ``self_report`` object
    carries exactly ``CANONICAL_SELF_REPORT_KEYS`` and that every file's
    ``RUN_CONFIG.trace_schema_version`` agrees.
    """
    per_file: list[dict[str, Any]] = []
    raw_key_variants: set[frozenset[str]] = set()
    normalized_key_variants: set[frozenset[str]] = set()
    schema_versions: set[Any] = set()
    for path, events in paths_and_events:
        version: Any = None
        source_keys: dict[str, set[frozenset[str]]] = {}
        for event in events:
            name = event.get("event")
            details = event.get("details", {}) or {}
            if name == "RUN_CONFIG":
                version = details.get("trace_schema_version")
            found: list[tuple[str, dict[str, Any]]] = []
            if isinstance(details.get("self_report"), dict):
                found.append((name, details["self_report"]))
            child = details.get("child")
            if isinstance(child, dict) and isinstance(child.get("self_report"), dict):
                found.append((name, child["self_report"]))
            for source, report in found:
                raw_keys = frozenset(report.keys())
                source_keys.setdefault(source, set()).add(raw_keys)
                raw_key_variants.add(raw_keys)
                # Canonical identity is judged after a normalized read so pre-fix
                # traces (whose optional keys were absent) still pass; the raw
                # variants above are retained purely as an informational line.
                # read_self_report fills defaults for absent canonical keys and
                # drops unknowns, so we re-union any stray/renamed raw keys to keep
                # genuine schema divergence detectable.
                normalized = set(read_self_report({"self_report": report}).keys())
                stray = raw_keys - normalized
                normalized_key_variants.add(frozenset(normalized | stray))
        if version is not None:
            schema_versions.add(version)
        per_file.append(
            {
                "path": path,
                "trace_schema_version": version,
                "self_report_key_sets": {
                    source: [sorted(k) for k in variants] for source, variants in sorted(source_keys.items())
                },
            }
        )
    canonical = set(CANONICAL_SELF_REPORT_KEYS)
    keys_all_canonical = all(set(k) == canonical for k in normalized_key_variants)
    return {
        "per_file": per_file,
        "distinct_key_sets_raw": [sorted(k) for k in raw_key_variants],
        "distinct_key_sets_normalized": [sorted(k) for k in normalized_key_variants],
        "all_key_sets_canonical": keys_all_canonical,
        "schema_versions_seen": sorted(str(v) for v in schema_versions),
        "schema_version_consistent": len(schema_versions) <= 1,
    }


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
    bucket_values = [r["predicted_bucket"] for r in records if r["predicted_bucket"] is not None]
    clamped = [r for r in records if r["predicted_closeness_clamped"]]
    empty_basis = [r for r in records if not _basis_words(r)]
    inheritance_injected = [r for r in records if r.get("injected_rationale_hash")]
    inheritance_truncated = [r for r in inheritance_injected if r.get("rationale_truncated")]
    missing_prompt = [r for r in records if not r.get("self_report_prompt")]

    parsed_with_basis = sum(1 for r in parsed if _basis_words(r))
    basis_membership_violations = []
    for r in records:
        present, count = basis_words_in_prompt(r)
        if count and present < count:
            missing = [w for w in _basis_words(r) if w not in (r.get("self_report_prompt") or "")]
            basis_membership_violations.append(
                {
                    "guess_word": r.get("guess_word"),
                    "hypothesis_name": r.get("hypothesis_name"),
                    "missing_basis_words": missing,
                }
            )

    in_range = all(0.0 <= value <= 1.0 for value in closeness_values)
    closeness_missing = total - len(closeness_values)
    bucket_missing = total - len(bucket_values)

    # Degenerate-clustering detection over parsed closeness values.
    dominant_value: float | None = None
    dominant_fraction: float | None = None
    closeness_std: float | None = None
    if closeness_values:
        value_counts: dict[float, int] = {}
        for value in closeness_values:
            value_counts[value] = value_counts.get(value, 0) + 1
        dominant_value, dominant_count = max(value_counts.items(), key=lambda item: item[1])
        dominant_fraction = dominant_count / len(closeness_values)
        closeness_std = statistics.pstdev(closeness_values) if len(closeness_values) > 1 else 0.0
    closeness_clustered = bool(
        closeness_values
        and (
            (dominant_fraction is not None and dominant_fraction > CLUSTER_DOMINANT_VALUE_FRACTION)
            or (closeness_std is not None and closeness_std < CLUSTER_NEAR_ZERO_STD)
        )
    )
    bucket_counts: dict[str, int] = {}
    for bucket in bucket_values:
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    histogram_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    histogram_labels = ["[0.0,0.2)", "[0.2,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,1.0]"]
    histogram = {label: 0 for label in histogram_labels}
    for value in closeness_values:
        for index in range(len(histogram_labels)):
            if histogram_bins[index] <= value < histogram_bins[index + 1]:
                histogram[histogram_labels[index]] += 1
                break

    return {
        "total_proposals": total,
        "parse_failure_count": len(parse_failed),
        "parse_failure_rate": (len(parse_failed) / total) if total else None,
        "predicted_closeness_present": len(closeness_values),
        "predicted_closeness_missing": closeness_missing,
        "predicted_closeness_parse_rate": (len(closeness_values) / total) if total else None,
        "predicted_bucket_present": len(bucket_values),
        "predicted_bucket_missing": bucket_missing,
        "predicted_bucket_parse_rate": (len(bucket_values) / total) if total else None,
        "predicted_bucket_counts": bucket_counts,
        "predicted_closeness_min": min(closeness_values) if closeness_values else None,
        "predicted_closeness_max": max(closeness_values) if closeness_values else None,
        "predicted_closeness_mean": (sum(closeness_values) / len(closeness_values)) if closeness_values else None,
        "predicted_closeness_histogram": histogram,
        "predicted_closeness_all_in_range": in_range,
        "predicted_closeness_std": closeness_std,
        "predicted_closeness_dominant_value": dominant_value,
        "predicted_closeness_dominant_fraction": dominant_fraction,
        "predicted_closeness_clustered": closeness_clustered,
        "clamped_count": len(clamped),
        "empty_basis_words_count": len(empty_basis),
        "parsed_count": len(parsed),
        "parsed_with_nonempty_basis": parsed_with_basis,
        "parsed_basis_nonempty_rate": (parsed_with_basis / len(parsed)) if parsed else None,
        "basis_membership_violations": basis_membership_violations,
        "inheritance_injected_count": len(inheritance_injected),
        "inheritance_truncated_count": len(inheritance_truncated),
        "self_report_prompt_missing_count": len(missing_prompt),
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
        # Flag (not a hard gate): degenerate closeness clustering makes the signal
        # unusable for calibration even when every other check passes.
        "closeness_degenerate_cluster": bool(metrics["predicted_closeness_clustered"]),
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
        f"predicted_closeness present:{metrics['predicted_closeness_present']} "
        f"(parse rate={_fmt(metrics['predicted_closeness_parse_rate'])})",
        f"predicted_bucket present:   {metrics['predicted_bucket_present']} "
        f"(parse rate={_fmt(metrics['predicted_bucket_parse_rate'])})",
        f"  bucket counts:            {metrics['predicted_bucket_counts']}",
        f"  min/mean/max:             {_fmt(metrics['predicted_closeness_min'])} / "
        f"{_fmt(metrics['predicted_closeness_mean'])} / {_fmt(metrics['predicted_closeness_max'])}",
        f"  histogram [0-1]:          {metrics['predicted_closeness_histogram']}",
        f"  std / dominant value:     {_fmt(metrics['predicted_closeness_std'])} / "
        f"{_fmt(metrics['predicted_closeness_dominant_value'])} "
        f"(fraction={_fmt(metrics['predicted_closeness_dominant_fraction'])})",
        f"  degenerate clustering:    {metrics['predicted_closeness_clustered']}",
        f"  all in [0,1]:             {metrics['predicted_closeness_all_in_range']}",
        f"clamped values:             {metrics['clamped_count']}",
        f"empty basis_words:          {metrics['empty_basis_words_count']}",
        f"parsed w/ nonempty basis:   {metrics['parsed_with_nonempty_basis']}/{metrics['parsed_count']} "
        f"(rate={_fmt(metrics['parsed_basis_nonempty_rate'])})",
        f"basis-in-prompt violations: {len(metrics['basis_membership_violations'])}",
        f"inheritance injected:       {metrics['inheritance_injected_count']} "
        f"(truncated={metrics['inheritance_truncated_count']})",
        f"missing self_report_prompt: {metrics['self_report_prompt_missing_count']}",
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
    paths_and_events: list[tuple[str, list[dict[str, Any]]]] = []
    for path in paths:
        events = load_trace(path)
        paths_and_events.append((path, events))
        records.extend(extract_records(events))

    metrics = compute_metrics(records)
    checks = evaluate_thresholds(metrics)
    schema = schema_audit(paths_and_events)

    print(f"Trace files read: {len(paths)}")
    for path in paths:
        print(f"  - {path}")
    print()
    if metrics["total_proposals"] == 0:
        print(
            "WARNING: no self-report records were extracted. If this trace is from a "
            "run with SELF_REPORT=1, the schema may have diverged (e.g. llm_only "
            "writing under an unexpected event) or the run produced no proposals."
        )
        print()
    print(_format_metrics(metrics))
    print()
    print("Schema identity audit:")
    for entry in schema["per_file"]:
        print(f"  {entry['path']}")
        print(f"    trace_schema_version: {entry['trace_schema_version']}")
        for source, keysets in entry["self_report_key_sets"].items():
            print(f"    {source} self_report keys: {keysets}")
    print(f"  distinct raw self_report key sets across files (informational): {schema['distinct_key_sets_raw']}")
    print(f"  distinct normalized self_report key sets across files: {schema['distinct_key_sets_normalized']}")
    print(f"  all normalized key sets canonical: {schema['all_key_sets_canonical']}")
    print(f"  schema versions seen: {schema['schema_versions_seen']}")
    print(f"  schema version consistent across files: {schema['schema_version_consistent']}")
    print()
    print("Acceptance checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print(f"  schema_keys_canonical: {schema['all_key_sets_canonical']}")
    print(f"  schema_version_consistent: {schema['schema_version_consistent']}")
    if checks["parse_failure_needs_discussion"] and checks["parse_failure_within_hard_max"]:
        print("  NOTE: parse-failure rate exceeds 2% (within the 10% hard limit); flag for discussion.")
    if checks["closeness_degenerate_cluster"]:
        print(
            "  NOTE: predicted_closeness is degenerately clustered (a single value "
            ">50% of parsed reports, or near-zero std); the signal may be unusable "
            "for calibration."
        )

    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.sample, len(records))) if records else []
    print(f"\n=== {len(sample)} sampled full records ===")
    for record in sample:
        print(json.dumps(record, indent=2, ensure_ascii=False))

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(
                {
                    "trace_files": paths,
                    "metrics": metrics,
                    "checks": checks,
                    "schema_audit": schema,
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote report JSON: {args.report_json}")


if __name__ == "__main__":
    main()
