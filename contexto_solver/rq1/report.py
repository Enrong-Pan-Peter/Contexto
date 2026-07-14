"""Cross-run RQ1 reporting: summaries, two-arm tests, and power-gate variance.

Consumes the per-run :class:`~contexto_solver.rq1.reader.Individual` groups,
reduces each run to a small set of scalar calibration metrics, and then:

- summarizes each metric across runs with median + IQR (robust to the small,
  skewed run counts typical here),
- compares two arms of runs with Mann-Whitney U (independent) or Wilcoxon signed
  rank (paired by target/game), and
- emits a per-metric variance table plus a normal-approximation estimate of the
  runs-per-arm needed to detect a given effect (the power gate).

It also exposes pgfplots-friendly CSV writers and an optional reliability-curve
image. Only numpy/scipy/matplotlib (all dependencies) are used. No writes here
except the explicit CSV/PNG writers the caller invokes; no network or LLM.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from .metrics import metrics_summary, reliability_curve
from .reader import Individual

# Scalar metrics carried per run into the cross-run tables. Order is stable.
SCALAR_METRICS = [
    "count",
    "spearman_rho",
    "bucket_accuracy",
    "mean_signed_bucket_error",
    "off_by_one_rate",
    "ece",
    "brier",
    "auroc",
    "positive_rate",
]


def run_scalars(individuals: Sequence[Individual], which: str = "child_best") -> dict[str, Any]:
    """Reduce one run's individuals to the scalar metrics in :data:`SCALAR_METRICS`."""
    summary = metrics_summary(individuals, which)
    return {
        "count": summary["count"],
        "spearman_rho": summary["spearman"]["rho"],
        "bucket_accuracy": summary["buckets"]["accuracy"],
        "mean_signed_bucket_error": summary["buckets"]["mean_signed_bucket_error"],
        "off_by_one_rate": summary["buckets"]["off_by_one_rate"],
        "ece": summary["reliability"]["ece"],
        "brier": summary["brier"]["brier"],
        "auroc": summary["auroc"]["auroc"],
        "positive_rate": summary["brier"]["positive_rate"],
    }


def scalars_by_run(
    individuals_by_run: dict[str, Sequence[Individual]], which: str = "child_best"
) -> dict[str, dict[str, Any]]:
    return {run: run_scalars(individuals, which) for run, individuals in individuals_by_run.items()}


def _clean(values: Sequence[Any]) -> list[float]:
    return [float(value) for value in values if value is not None]


def summarize_values(values: Sequence[Any]) -> dict[str, Any]:
    """Median + IQR + mean/std/var for one metric across runs (Nones dropped)."""
    cleaned = _clean(values)
    n = len(cleaned)
    if n == 0:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None,
                "mean": None, "std": None, "var": None, "min": None, "max": None}
    array = np.asarray(cleaned, dtype=float)
    q1, median, q3 = (float(x) for x in np.percentile(array, [25, 50, 75]))
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    var = float(np.var(array, ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "mean": float(array.mean()),
        "std": std,
        "var": var,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def cross_run_summary(
    scalars: dict[str, dict[str, Any]], metrics: Sequence[str] = SCALAR_METRICS
) -> dict[str, dict[str, Any]]:
    """Per-metric median/IQR/mean/std across all runs."""
    return {metric: summarize_values([run.get(metric) for run in scalars.values()]) for metric in metrics}


def two_arm_comparison(
    scalars: dict[str, dict[str, Any]],
    arm_of: dict[str, str],
    arm_a: str,
    arm_b: str,
    *,
    paired_key: dict[str, Any] | None = None,
    metrics: Sequence[str] = SCALAR_METRICS,
) -> dict[str, Any]:
    """Compare two arms of runs per metric.

    Uses the paired Wilcoxon signed-rank test when ``paired_key`` maps each run to
    a pairing key (e.g. target/game_number) and both arms cover the same keys;
    otherwise the independent Mann-Whitney U test. Runs with a ``None`` metric are
    dropped from that metric's test.
    """
    runs_a = [run for run, arm in arm_of.items() if arm == arm_a]
    runs_b = [run for run, arm in arm_of.items() if arm == arm_b]
    results: dict[str, Any] = {}
    for metric in metrics:
        a_vals = {run: scalars[run].get(metric) for run in runs_a}
        b_vals = {run: scalars[run].get(metric) for run in runs_b}
        entry = _compare_metric(metric, a_vals, b_vals, paired_key)
        results[metric] = entry
    return {"arm_a": arm_a, "arm_b": arm_b, "n_runs_a": len(runs_a), "n_runs_b": len(runs_b), "metrics": results}


def _compare_metric(
    metric: str,
    a_vals: dict[str, Any],
    b_vals: dict[str, Any],
    paired_key: dict[str, Any] | None,
) -> dict[str, Any]:
    a_clean = {run: float(value) for run, value in a_vals.items() if value is not None}
    b_clean = {run: float(value) for run, value in b_vals.items() if value is not None}
    a_list = list(a_clean.values())
    b_list = list(b_clean.values())
    base: dict[str, Any] = {
        "n_a": len(a_list),
        "n_b": len(b_list),
        "median_a": float(np.median(a_list)) if a_list else None,
        "median_b": float(np.median(b_list)) if b_list else None,
        "test": None,
        "statistic": None,
        "pvalue": None,
    }
    paired_pairs = _paired_values(a_clean, b_clean, paired_key) if paired_key else None
    if paired_pairs is not None and len(paired_pairs) >= 1:
        diffs = [x - y for x, y in paired_pairs]
        if any(d != 0 for d in diffs) and len(diffs) >= 1:
            try:
                statistic, pvalue = stats.wilcoxon([p[0] for p in paired_pairs], [p[1] for p in paired_pairs])
                base.update({"test": "wilcoxon", "statistic": float(statistic), "pvalue": float(pvalue),
                             "n_pairs": len(paired_pairs)})
                return base
            except ValueError:
                pass
    if len(a_list) >= 1 and len(b_list) >= 1 and (len(a_list) + len(b_list)) >= 3:
        try:
            statistic, pvalue = stats.mannwhitneyu(a_list, b_list, alternative="two-sided")
            base.update({"test": "mannwhitneyu", "statistic": float(statistic), "pvalue": float(pvalue)})
        except ValueError:
            pass
    return base


def _paired_values(
    a_clean: dict[str, float], b_clean: dict[str, float], paired_key: dict[str, Any]
) -> list[tuple[float, float]] | None:
    a_by_key: dict[Any, float] = {}
    b_by_key: dict[Any, float] = {}
    for run, value in a_clean.items():
        key = paired_key.get(run)
        if key is not None:
            a_by_key[key] = value
    for run, value in b_clean.items():
        key = paired_key.get(run)
        if key is not None:
            b_by_key[key] = value
    shared = sorted(set(a_by_key) & set(b_by_key), key=str)
    if not shared:
        return None
    return [(a_by_key[key], b_by_key[key]) for key in shared]


def runs_needed_per_arm(std: float, effect: float, *, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """Normal-approximation two-sample size per arm to detect ``effect`` at ``power``.

    ``n = 2 (z_{1-alpha/2} + z_{1-power})^2 sigma^2 / effect^2`` (rounded up). Returns
    ``None`` for a non-positive effect or std.
    """
    if effect <= 0 or std <= 0:
        return None
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = 2 * (z_alpha + z_beta) ** 2 * (std ** 2) / (effect ** 2)
    return int(math.ceil(n))


def variance_table(
    scalars: dict[str, dict[str, Any]],
    *,
    effect: float | None = None,
    metrics: Sequence[str] = SCALAR_METRICS,
) -> dict[str, Any]:
    """Per-metric variance and, when ``effect`` is given, the power-gate run count."""
    table: dict[str, Any] = {}
    for metric in metrics:
        summary = summarize_values([run.get(metric) for run in scalars.values()])
        std = summary["std"]
        mean = summary["mean"]
        entry = {
            "n_runs": summary["n"],
            "mean": mean,
            "std": std,
            "var": summary["var"],
            "cv": (std / abs(mean)) if (std is not None and mean not in (None, 0)) else None,
        }
        if effect is not None and std is not None:
            entry["runs_needed_per_arm"] = runs_needed_per_arm(std, effect)
            entry["effect"] = effect
        table[metric] = entry
    return table


# --- pgfplots-friendly writers -----------------------------------------------

def write_per_run_csv(path: str | Path, scalars: dict[str, dict[str, Any]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", *SCALAR_METRICS])
        for run, values in scalars.items():
            writer.writerow([run, *[values.get(metric) for metric in SCALAR_METRICS]])
    return out_path


def write_cross_run_csv(path: str | Path, summary: dict[str, dict[str, Any]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "n", "median", "q1", "q3", "iqr", "mean", "std", "var", "min", "max"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for metric, values in summary.items():
            writer.writerow([metric, *[values.get(field) for field in fields[1:]]])
    return out_path


def write_reliability_csv(path: str | Path, reliability: dict[str, Any]) -> Path:
    """Reliability-curve points for a pgfplots calibration plot."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_lower", "bin_upper", "bin_center", "count", "mean_predicted", "empirical_rate"])
        for entry in reliability["bins"]:
            center = (entry["bin_lower"] + entry["bin_upper"]) / 2
            writer.writerow([
                entry["bin_lower"], entry["bin_upper"], center,
                entry["count"], entry["mean_predicted"], entry["empirical_rate"],
            ])
    return out_path


def save_reliability_image(reliability: dict[str, Any], path: str | Path, *, title: str | None = None) -> Path:
    """Save a calibration reliability plot (predicted vs empirical) as PNG.

    Uses the non-interactive Agg backend so it works headless. Bins with no
    samples are omitted from the plotted curve.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [entry["mean_predicted"] for entry in reliability["bins"] if entry["count"]]
    ys = [entry["empirical_rate"] for entry in reliability["bins"] if entry["count"]]

    figure, axes = plt.subplots(figsize=(4, 4))
    axes.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
    axes.plot(xs, ys, marker="o", label="observed")
    axes.set_xlabel("mean predicted_closeness")
    axes.set_ylabel("empirical P(rank <= 100)")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    if title:
        axes.set_title(title)
    axes.legend(loc="best", fontsize=8)
    figure.tight_layout()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def pooled_reliability(
    individuals_by_run: dict[str, Sequence[Individual]], which: str = "child_best", n_bins: int = 10
) -> dict[str, Any]:
    """Reliability curve over all individuals pooled across runs."""
    pooled: list[Individual] = []
    for individuals in individuals_by_run.values():
        pooled.extend(individuals)
    return reliability_curve(pooled, which, n_bins)
