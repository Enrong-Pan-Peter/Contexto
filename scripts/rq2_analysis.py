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
- a cluster-aware percentile bootstrap 95% CI (``--bootstrap`` draws, seeded)
  for the pooled rank difference per contrast -- Hodges-Lehmann pseudo-median
  (headline), plain median, and mean (sensitivity) -- and for the
  word-identity rate: runs (trace files) are resampled with replacement and
  the pooled statistic recomputed per draw, so the CI respects the run-level
  clustering.
  This is what lets the paper bound a null ("any causal effect of the
  rationale is within +/- X ranks") instead of only failing to reject;
- parse-failure and invalid-word rates per arm.

No network, no LLM, no writes to traces or caches. Outputs pgfplots-friendly
CSVs plus a summary JSON.

Usage (PowerShell):

    python scripts/rq2_analysis.py traces/rq2_replay_batch1/rq2_replay_records.json `
        --output traces/rq2_analysis_batch1 [--bootstrap 10000] [--seed 0]
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

import numpy as np
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


# --- cluster bootstrap -------------------------------------------------------------


def _identity_counts_by_run(
    events: dict[tuple[str, str], dict[str, dict[str, Any]]], contrast: str
) -> dict[str, tuple[int, int]]:
    """trace_file -> (n identical first words, n eligible events) for a contrast.

    Eligibility mirrors :func:`word_agreement` exactly: both the genuine and the
    variant arm must be present with a non-null ``proposed_word``.
    """
    counts: dict[str, tuple[int, int]] = {}
    for (trace_file, _child_id), arms in events.items():
        genuine = arms.get("genuine")
        variant = arms.get(contrast)
        if genuine is None or variant is None:
            continue
        genuine_word = genuine.get("proposed_word")
        variant_word = variant.get("proposed_word")
        if genuine_word is None or variant_word is None:
            continue
        identical, n = counts.get(trace_file, (0, 0))
        counts[trace_file] = (identical + (1 if genuine_word == variant_word else 0), n + 1)
    return counts


def _percentile_ci95(draws: list[float]) -> tuple[float, float]:
    array = np.asarray(draws, dtype=np.float64)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def _hodges_lehmann(values: np.ndarray, triu_cache: dict[int, tuple[np.ndarray, np.ndarray]]) -> float:
    """One-sample Hodges-Lehmann estimate: median of Walsh averages over i <= j."""
    n = len(values)
    if n not in triu_cache:
        triu_cache[n] = np.triu_indices(n)
    rows, cols = triu_cache[n]
    walsh = (values[rows] + values[cols]) / 2.0
    return float(np.median(walsh))


def _bootstrap_pooled_diffs(
    clusters: dict[str, list[float]], draws: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Percentile CIs for the pooled median, Hodges-Lehmann pseudo-median, and
    mean, resampling clusters with replacement.

    All three statistics are computed from the same resampled pool per draw
    (one RNG consumption per draw, so adding a statistic never changes the
    others' CIs). The HL pseudo-median is the headline bound: it is not pinned
    by the large atom at exactly zero (identical proposed words) the way the
    plain median can be, and it is far less outlier-driven than the mean, which
    is kept as a sensitivity companion.
    """
    empty = {"observed": None, "ci95_low": None, "ci95_high": None, "n_clusters": 0, "draws": 0}
    keys = sorted(clusters)
    arrays = [np.asarray(clusters[key], dtype=np.float64) for key in keys]
    if not arrays:
        return {"median": dict(empty), "hl": dict(empty), "mean": dict(empty)}
    triu_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pooled = np.concatenate(arrays)
    medians: list[float] = []
    hls: list[float] = []
    means: list[float] = []
    for _draw in range(draws):
        indices = rng.integers(0, len(arrays), size=len(arrays))
        sample = np.concatenate([arrays[i] for i in indices])
        medians.append(float(np.median(sample)))
        hls.append(_hodges_lehmann(sample, triu_cache))
        means.append(float(np.mean(sample)))
    base = {"n_clusters": len(arrays), "draws": draws}

    def entry(observed: float, draw_values: list[float]) -> dict[str, Any]:
        low, high = _percentile_ci95(draw_values)
        return {"observed": observed, "ci95_low": low, "ci95_high": high, **base}

    return {
        "median": entry(float(np.median(pooled)), medians),
        "hl": entry(_hodges_lehmann(pooled, triu_cache), hls),
        "mean": entry(float(np.mean(pooled)), means),
    }


def _bootstrap_pooled_rate(
    counts: dict[str, tuple[int, int]], draws: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Percentile CI for a pooled numerator/denominator rate, resampling clusters."""
    keys = sorted(counts)
    if not keys:
        return {"observed": None, "ci95_low": None, "ci95_high": None, "n_clusters": 0, "draws": 0}
    numerators = np.asarray([counts[key][0] for key in keys], dtype=np.float64)
    denominators = np.asarray([counts[key][1] for key in keys], dtype=np.float64)
    observed = float(numerators.sum() / denominators.sum())
    index_matrix = rng.integers(0, len(keys), size=(draws, len(keys)))
    rates = numerators[index_matrix].sum(axis=1) / denominators[index_matrix].sum(axis=1)
    low, high = _percentile_ci95(list(rates))
    return {
        "observed": observed,
        "ci95_low": low,
        "ci95_high": high,
        "n_clusters": len(keys),
        "draws": draws,
    }


def bootstrap_cis(
    diff_rows: list[dict[str, Any]],
    events: dict[tuple[str, str], dict[str, dict[str, Any]]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Per contrast: cluster (run-level) bootstrap CIs for the pooled median rank
    difference and the word-identity rate.

    Each (contrast, statistic) pair gets its own child generator seeded from
    ``(seed, contrast index, statistic index)``, so every CI is individually
    reproducible regardless of evaluation order.
    """
    results: dict[str, Any] = {}
    for contrast_index, contrast in enumerate(CONTRASTS):
        diff_clusters: dict[str, list[float]] = {}
        for row in diff_rows:
            if row["contrast"] == contrast:
                diff_clusters.setdefault(row["trace_file"], []).append(
                    float(row["diff_genuine_minus_variant"])
                )
        diff_cis = _bootstrap_pooled_diffs(
            diff_clusters, draws, np.random.default_rng([seed, contrast_index, 0])
        )
        results[contrast] = {
            "median_diff": diff_cis["median"],
            "hl_diff": diff_cis["hl"],
            "mean_diff": diff_cis["mean"],
            "word_identity_rate": _bootstrap_pooled_rate(
                _identity_counts_by_run(events, contrast),
                draws,
                np.random.default_rng([seed, contrast_index, 1]),
            ),
        }
    return results


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


def analyze(
    records: list[dict[str, Any]], *, bootstrap_draws: int = 10000, bootstrap_seed: int = 0
) -> dict[str, Any]:
    events = group_by_event(records)
    diff_rows = paired_differences(events)
    medians = per_run_medians(diff_rows)
    tests = contrast_tests(diff_rows, medians)
    if bootstrap_draws > 0:
        cis = bootstrap_cis(diff_rows, events, draws=bootstrap_draws, seed=bootstrap_seed)
        for contrast in CONTRASTS:
            tests[contrast]["cluster_bootstrap"] = cis[contrast]
    return {
        "paired_differences": diff_rows,
        "word_agreement": word_agreement(events),
        "per_run_medians": medians,
        "tests": tests,
        "per_arm_rates": per_arm_rates(records),
        "n_events": len(events),
        "n_records": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 paired analysis over replay outputs (offline).")
    parser.add_argument("replay_outputs", nargs="+", help="rq2_replay_records.json path(s) or glob(s).")
    parser.add_argument("--output", required=True, help="Output directory for CSVs + summary JSON.")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Cluster-bootstrap draws for the 95%% CIs (default 10000; 0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default 0).")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.replay_outputs:
        matches = glob.glob(pattern)
        paths.extend(sorted(matches) if matches else ([pattern] if Path(pattern).exists() else []))
    if not paths:
        raise SystemExit("No replay output files matched the given path(s).")

    records = load_replay_records(paths)
    result = analyze(records, bootstrap_draws=args.bootstrap, bootstrap_seed=args.seed)

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
        "bootstrap": (
            {
                "draws": args.bootstrap,
                "seed": args.seed,
                "method": "percentile bootstrap resampling run-level clusters (trace files) with replacement",
            }
            if args.bootstrap > 0
            else None
        ),
        "tests": result["tests"],
        "word_agreement": result["word_agreement"],
        "per_arm_rates": result["per_arm_rates"],
    }
    (out_dir / "rq2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Events: {result['n_events']}  records: {result['n_records']}")
    for contrast, test in result["tests"].items():
        clustered = test["clustered_by_run"]
        line = (
            f"  genuine-{contrast}: median diff (pooled) = {test['pooled_descriptives']['median']} "
            f"| clustered Wilcoxon p = {clustered['pvalue']} (n runs = {clustered['n']})"
        )
        bootstrap = test.get("cluster_bootstrap")
        if bootstrap:
            median_ci = bootstrap["median_diff"]
            hl_ci = bootstrap["hl_diff"]
            mean_ci = bootstrap["mean_diff"]
            rate_ci = bootstrap["word_identity_rate"]
            line += (
                f"\n    bootstrap 95% CI: HL diff {hl_ci['observed']} [{hl_ci['ci95_low']}, {hl_ci['ci95_high']}] "
                f"| median diff [{median_ci['ci95_low']}, {median_ci['ci95_high']}] "
                f"| mean diff {mean_ci['observed']} [{mean_ci['ci95_low']}, {mean_ci['ci95_high']}] "
                f"| identity rate {rate_ci['observed']} [{rate_ci['ci95_low']}, {rate_ci['ci95_high']}]"
            )
        print(line)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
