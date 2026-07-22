"""RQ2 paired analysis over replay-harness outputs (fully offline).

Consumes one or more ``rq2_replay_records.json`` files produced by
``scripts/rq2_replay.py`` and computes, per intervention contrast
(genuine-wrong, genuine-filler, genuine-absent):

- within-event paired differences on realized rank (genuine minus variant;
  negative = genuine ranked closer);
- word-identity and word-overlap (Jaccard over proposed word lists) rates
  across arms - the insensitivity measure;
- per-run medians of the paired differences and a Wilcoxon signed-rank test on
  those per-run medians (events clustered by run), plus a pooled event-level
  Wilcoxon as a descriptive companion;
- parse-failure and invalid-word rates per arm.

No network, no LLM, no writes to traces or caches. Outputs pgfplots-friendly
CSVs plus a summary JSON.

Usage (PowerShell):

    python scripts/rq2_analysis.py traces/rq2_replay_batch1/rq2_replay_records.json `
        --output traces/rq2_analysis_batch1
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy import stats

CONTRASTS = ("wrong", "filler", "absent")
ARMS = ("genuine", "wrong", "filler", "absent")


# --- loading -------------------------------------------------------------------


def load_replay_records(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            raise ValueError(f"{path} is not a replay output (expected a dict with a records list).")
        records.extend(payload["records"])
    return records


def group_by_event(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """(trace_file, child_id) -> arm -> record."""
    events: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        key = (record["trace_file"], record["child_id"])
        events.setdefault(key, {})[record["arm"]] = record
    return events


# --- paired differences ----------------------------------------------------------


def paired_differences(events: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """One row per (event, contrast) where both arms have a realized rank."""
    rows: list[dict[str, Any]] = []
    for (trace_file, child_id), arms in sorted(events.items()):
        genuine = arms.get("genuine")
        if genuine is None:
            continue
        genuine_rank = genuine.get("realized_rank")
        for contrast in CONTRASTS:
            variant = arms.get(contrast)
            if variant is None:
                continue
            variant_rank = variant.get("realized_rank")
            if genuine_rank is None or variant_rank is None:
                continue
            rows.append(
                {
                    "trace_file": trace_file,
                    "child_id": child_id,
                    "generation": genuine.get("generation"),
                    "parent_rank": genuine.get("parent_rank"),
                    "contrast": contrast,
                    "rank_genuine": genuine_rank,
                    "rank_variant": variant_rank,
                    "diff_genuine_minus_variant": genuine_rank - variant_rank,
                }
            )
    return rows


def word_agreement(events: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Per contrast: word-identity rate (first word) and mean Jaccard overlap."""
    rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        identical = 0
        jaccards: list[float] = []
        n = 0
        for arms in events.values():
            genuine = arms.get("genuine")
            variant = arms.get(contrast)
            if genuine is None or variant is None:
                continue
            genuine_word = genuine.get("proposed_word")
            variant_word = variant.get("proposed_word")
            if genuine_word is None or variant_word is None:
                continue
            n += 1
            if genuine_word == variant_word:
                identical += 1
            genuine_set = set(genuine.get("proposed_words") or [])
            variant_set = set(variant.get("proposed_words") or [])
            union = genuine_set | variant_set
            if union:
                jaccards.append(len(genuine_set & variant_set) / len(union))
        rows.append(
            {
                "contrast": contrast,
                "n_events": n,
                "word_identity_rate": (identical / n) if n else None,
                "mean_jaccard_overlap": (sum(jaccards) / len(jaccards)) if jaccards else None,
            }
        )
    return rows


def per_run_medians(diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for row in diff_rows:
        grouped.setdefault((row["trace_file"], row["contrast"]), []).append(row["diff_genuine_minus_variant"])
    return [
        {
            "trace_file": trace_file,
            "contrast": contrast,
            "n_events": len(diffs),
            "median_diff": statistics.median(diffs),
            "mean_diff": sum(diffs) / len(diffs),
        }
        for (trace_file, contrast), diffs in sorted(grouped.items())
    ]


def _wilcoxon(values: list[float]) -> dict[str, Any]:
    """Wilcoxon signed-rank against zero; None-safe for degenerate inputs."""
    nonzero = [value for value in values if value != 0]
    if len(nonzero) < 1 or len(values) < 2:
        return {"n": len(values), "n_nonzero": len(nonzero), "statistic": None, "pvalue": None}
    try:
        result = stats.wilcoxon(values, zero_method="wilcox")
    except ValueError:
        return {"n": len(values), "n_nonzero": len(nonzero), "statistic": None, "pvalue": None}
    return {
        "n": len(values),
        "n_nonzero": len(nonzero),
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def contrast_tests(diff_rows: list[dict[str, Any]], medians: list[dict[str, Any]]) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    for contrast in CONTRASTS:
        event_diffs = [float(r["diff_genuine_minus_variant"]) for r in diff_rows if r["contrast"] == contrast]
        run_medians = [float(r["median_diff"]) for r in medians if r["contrast"] == contrast]
        tests[contrast] = {
            "clustered_by_run": _wilcoxon(run_medians),
            "pooled_events": _wilcoxon(event_diffs),
            "pooled_descriptives": {
                "n": len(event_diffs),
                "mean": (sum(event_diffs) / len(event_diffs)) if event_diffs else None,
                "median": statistics.median(event_diffs) if event_diffs else None,
                "min": min(event_diffs) if event_diffs else None,
                "max": max(event_diffs) if event_diffs else None,
            },
        }
    return tests


def per_arm_rates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_records = [r for r in records if r["arm"] == arm]
        n = len(arm_records)
        llm_failed = sum(1 for r in arm_records if r.get("llm_parse_failed"))
        sr_failed = sum(1 for r in arm_records if r.get("self_report_parse_failed"))
        no_word = sum(1 for r in arm_records if r.get("proposed_word") is None)
        invalid = sum(1 for r in arm_records if r.get("proposed_word_invalid"))
        graded = sum(1 for r in arm_records if r.get("realized_rank") is not None)
        rows.append(
            {
                "arm": arm,
                "n": n,
                "llm_parse_failure_rate": (llm_failed / n) if n else None,
                "self_report_parse_failure_rate": (sr_failed / n) if n else None,
                "no_word_rate": (no_word / n) if n else None,
                "invalid_word_rate": (invalid / n) if n else None,
                "graded_rate": (graded / n) if n else None,
            }
        )
    return rows


# --- outputs ---------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = group_by_event(records)
    diff_rows = paired_differences(events)
    medians = per_run_medians(diff_rows)
    return {
        "paired_differences": diff_rows,
        "word_agreement": word_agreement(events),
        "per_run_medians": medians,
        "tests": contrast_tests(diff_rows, medians),
        "per_arm_rates": per_arm_rates(records),
        "n_events": len(events),
        "n_records": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 paired analysis over replay outputs (offline).")
    parser.add_argument("replay_outputs", nargs="+", help="rq2_replay_records.json path(s) or glob(s).")
    parser.add_argument("--output", required=True, help="Output directory for CSVs + summary JSON.")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.replay_outputs:
        matches = glob.glob(pattern)
        paths.extend(sorted(matches) if matches else ([pattern] if Path(pattern).exists() else []))
    if not paths:
        raise SystemExit("No replay output files matched the given path(s).")

    records = load_replay_records(paths)
    result = analyze(records)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rq2_paired_differences.csv", result["paired_differences"])
    _write_csv(out_dir / "rq2_word_agreement.csv", result["word_agreement"])
    _write_csv(out_dir / "rq2_per_run_medians.csv", result["per_run_medians"])
    _write_csv(out_dir / "rq2_per_arm_rates.csv", result["per_arm_rates"])
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [str(p) for p in paths],
        "n_events": result["n_events"],
        "n_records": result["n_records"],
        "tests": result["tests"],
        "word_agreement": result["word_agreement"],
        "per_arm_rates": result["per_arm_rates"],
    }
    (out_dir / "rq2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Events: {result['n_events']}  records: {result['n_records']}")
    for contrast, test in result["tests"].items():
        clustered = test["clustered_by_run"]
        print(
            f"  genuine-{contrast}: median diff (pooled) = {test['pooled_descriptives']['median']} "
            f"| clustered Wilcoxon p = {clustered['pvalue']} (n runs = {clustered['n']})"
        )
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
