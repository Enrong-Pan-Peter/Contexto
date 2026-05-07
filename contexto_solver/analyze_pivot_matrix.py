"""Analyze paired pivot-on/off local LLM experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import wilcoxon


METRICS = ("total_guesses", "best_rank", "generations")


@dataclass(frozen=True)
class ConditionSummary:
    condition: str
    target: str
    runs: int
    solved: int
    solve_rate: float
    total_guesses_median: float | None
    total_guesses_iqr: float | None
    best_rank_median: float | None
    best_rank_iqr: float | None
    generations_median: float | None
    generations_iqr: float | None


@dataclass(frozen=True)
class PairedMetricSummary:
    target: str
    metric: str
    pairs: int
    off_median: float | None
    off_iqr: float | None
    on_median: float | None
    on_iqr: float | None
    median_difference_on_minus_off: float | None
    wilcoxon_statistic: float | None
    wilcoxon_p_value: float | None
    cliffs_delta_on_vs_off: float | None


def main() -> None:
    args = _parse_args()
    off_summary = _load_summary(Path(args.off))
    on_summary = _load_summary(Path(args.on))

    off_rows = _tag_rows(off_summary, "pivot_off")
    on_rows = _tag_rows(on_summary, "pivot_on")
    pairs = _pair_rows(off_rows, on_rows)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    combined_rows = off_rows + on_rows
    condition_stats = _condition_summaries(combined_rows)
    paired_stats = _paired_summaries(pairs)

    _write_combined_runs(prefix.with_name(f"{prefix.name}_combined_runs.csv"), combined_rows)
    _write_condition_stats(prefix.with_name(f"{prefix.name}_condition_stats.csv"), condition_stats)
    _write_paired_stats(prefix.with_name(f"{prefix.name}_paired_stats.csv"), paired_stats)
    _write_json(
        prefix.with_name(f"{prefix.name}_analysis.json"),
        {
            "metadata": {
                "pivot_off": str(args.off),
                "pivot_on": str(args.on),
                "paired_runs": len(pairs),
                "targets": sorted({target for target, _ in pairs}),
            },
            "condition_stats": [summary.__dict__ for summary in condition_stats],
            "paired_stats": [summary.__dict__ for summary in paired_stats],
        },
    )

    _print_summary(condition_stats, paired_stats, len(pairs))
    print()
    print(f"Wrote combined runs CSV: {prefix.with_name(f'{prefix.name}_combined_runs.csv')}")
    print(f"Wrote condition stats CSV: {prefix.with_name(f'{prefix.name}_condition_stats.csv')}")
    print(f"Wrote paired stats CSV: {prefix.with_name(f'{prefix.name}_paired_stats.csv')}")
    print(f"Wrote analysis JSON: {prefix.with_name(f'{prefix.name}_analysis.json')}")


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError(f"{path} does not look like a batch experiment summary.")
    return data


def _tag_rows(summary: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    enable_pivot = summary.get("metadata", {}).get("enable_pivot")
    rows: list[dict[str, Any]] = []
    for row in summary["runs"]:
        tagged = dict(row)
        tagged["condition"] = condition
        tagged["enable_pivot"] = enable_pivot
        rows.append(tagged)
    return rows


def _pair_rows(
    off_rows: list[dict[str, Any]], on_rows: list[dict[str, Any]]
) -> dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]]:
    off_by_key = _index_by_pair_key(off_rows, "pivot_off")
    on_by_key = _index_by_pair_key(on_rows, "pivot_on")
    if off_by_key.keys() != on_by_key.keys():
        missing_on = sorted(off_by_key.keys() - on_by_key.keys())
        missing_off = sorted(on_by_key.keys() - off_by_key.keys())
        raise ValueError(f"Unpaired runs. Missing on={missing_on}; missing off={missing_off}")
    return {key: (off_by_key[key], on_by_key[key]) for key in sorted(off_by_key)}


def _index_by_pair_key(rows: list[dict[str, Any]], condition: str) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["target"]), int(row["run_index"]))
        if key in indexed:
            raise ValueError(f"Duplicate {condition} row for {key}")
        indexed[key] = row
    return indexed


def _condition_summaries(rows: list[dict[str, Any]]) -> list[ConditionSummary]:
    summaries: list[ConditionSummary] = []
    targets = ["ALL"] + sorted({str(row["target"]) for row in rows})
    for target in targets:
        scoped = rows if target == "ALL" else [row for row in rows if row["target"] == target]
        for condition in ("pivot_off", "pivot_on"):
            condition_rows = [row for row in scoped if row["condition"] == condition]
            if not condition_rows:
                continue
            summaries.append(
                ConditionSummary(
                    condition=condition,
                    target=target,
                    runs=len(condition_rows),
                    solved=sum(1 for row in condition_rows if row["solved"]),
                    solve_rate=sum(1 for row in condition_rows if row["solved"]) / len(condition_rows),
                    total_guesses_median=_median(_values(condition_rows, "total_guesses")),
                    total_guesses_iqr=_iqr(_values(condition_rows, "total_guesses")),
                    best_rank_median=_median(_values(condition_rows, "best_rank")),
                    best_rank_iqr=_iqr(_values(condition_rows, "best_rank")),
                    generations_median=_median(_values(condition_rows, "generations")),
                    generations_iqr=_iqr(_values(condition_rows, "generations")),
                )
            )
    return summaries


def _paired_summaries(
    pairs: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]]
) -> list[PairedMetricSummary]:
    summaries: list[PairedMetricSummary] = []
    targets = ["ALL"] + sorted({target for target, _ in pairs})
    for target in targets:
        scoped_pairs = [
            pair for (pair_target, _), pair in pairs.items() if target == "ALL" or pair_target == target
        ]
        for metric in METRICS:
            off_values = _metric_values([off for off, _ in scoped_pairs], metric)
            on_values = _metric_values([on for _, on in scoped_pairs], metric)
            differences = [on - off for off, on in zip(off_values, on_values)]
            statistic, p_value = _wilcoxon(differences)
            summaries.append(
                PairedMetricSummary(
                    target=target,
                    metric=metric,
                    pairs=len(differences),
                    off_median=_median(off_values),
                    off_iqr=_iqr(off_values),
                    on_median=_median(on_values),
                    on_iqr=_iqr(on_values),
                    median_difference_on_minus_off=_median(differences),
                    wilcoxon_statistic=statistic,
                    wilcoxon_p_value=p_value,
                    cliffs_delta_on_vs_off=_cliffs_delta(on_values, off_values),
                )
            )
    return summaries


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    values = _values(rows, metric)
    if len(values) != len(rows):
        raise ValueError(f"Metric {metric} is missing values.")
    return values


def _values(rows: Iterable[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(metric)
        if value is not None:
            values.append(float(value))
    return values


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _iqr(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) < 2:
        return 0.0
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2 == 0:
        lower = sorted_values[:midpoint]
        upper = sorted_values[midpoint:]
    else:
        lower = sorted_values[:midpoint]
        upper = sorted_values[midpoint + 1 :]
    if not lower or not upper:
        return 0.0
    return statistics.median(upper) - statistics.median(lower)


def _wilcoxon(differences: list[float]) -> tuple[float | None, float | None]:
    if not differences or all(difference == 0 for difference in differences):
        return None, None
    result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _cliffs_delta(group_a: list[float], group_b: list[float]) -> float | None:
    if not group_a or not group_b:
        return None
    greater = 0
    lesser = 0
    for a_value in group_a:
        for b_value in group_b:
            if a_value > b_value:
                greater += 1
            elif a_value < b_value:
                lesser += 1
    return (greater - lesser) / (len(group_a) * len(group_b))


def _write_combined_runs(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "condition",
        "enable_pivot",
        "solver",
        "mode",
        "target",
        "run_index",
        "solved",
        "answer",
        "best_word",
        "best_rank",
        "total_guesses",
        "generations",
        "trace_path",
        "alignment",
    ]
    _write_csv(path, fieldnames, rows)


def _write_condition_stats(path: Path, summaries: list[ConditionSummary]) -> None:
    _write_csv(path, list(ConditionSummary.__dataclass_fields__), [summary.__dict__ for summary in summaries])


def _write_paired_stats(path: Path, summaries: list[PairedMetricSummary]) -> None:
    _write_csv(path, list(PairedMetricSummary.__dataclass_fields__), [summary.__dict__ for summary in summaries])


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _print_summary(
    condition_stats: list[ConditionSummary],
    paired_stats: list[PairedMetricSummary],
    paired_run_count: int,
) -> None:
    print(f"Paired runs: {paired_run_count}")
    print()
    print("Condition summary")
    _print_table(
        [
            [
                summary.target,
                summary.condition,
                summary.runs,
                summary.solved,
                _fmt(summary.solve_rate),
                _fmt(summary.total_guesses_median),
                _fmt(summary.total_guesses_iqr),
                _fmt(summary.best_rank_median),
                _fmt(summary.best_rank_iqr),
            ]
            for summary in condition_stats
        ],
        [
            "target",
            "condition",
            "runs",
            "solved",
            "solve_rate",
            "guess_med",
            "guess_iqr",
            "rank_med",
            "rank_iqr",
        ],
    )
    print()
    print("Paired statistical summary")
    _print_table(
        [
            [
                summary.target,
                summary.metric,
                summary.pairs,
                _fmt(summary.off_median),
                _fmt(summary.off_iqr),
                _fmt(summary.on_median),
                _fmt(summary.on_iqr),
                _fmt(summary.median_difference_on_minus_off),
                _fmt(summary.wilcoxon_p_value),
                _fmt(summary.cliffs_delta_on_vs_off),
            ]
            for summary in paired_stats
        ],
        [
            "target",
            "metric",
            "pairs",
            "off_med",
            "off_iqr",
            "on_med",
            "on_iqr",
            "med_diff",
            "wilcoxon_p",
            "cliffs_delta",
        ],
    )


def _print_table(rows: list[list[Any]], headers: list[str]) -> None:
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired pivot matrix experiment summaries.")
    parser.add_argument("--off", required=True, help="Batch summary JSON with ENABLE_PIVOT=false.")
    parser.add_argument("--on", required=True, help="Batch summary JSON with ENABLE_PIVOT=true.")
    parser.add_argument(
        "--output-prefix",
        default="traces/pivot_matrix",
        help="Output prefix for combined CSV, stats CSV, and JSON analysis files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
