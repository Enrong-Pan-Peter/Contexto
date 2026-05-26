"""Inspect self-adaptive Contexto traces for sigma and operator behavior."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


OPERATOR_SAMPLED = "OPERATOR_SAMPLED"
SIGMA_TRAJECTORY = "SIGMA_TRAJECTORY"
OPERATORS = ["s_mutation", "m_mutation", "ml_mutation", "l_mutation"]


@dataclass
class HypothesisRecord:
    hypothesis_id: str
    parent_id: str | None
    sigma: np.ndarray
    best_word: str | None
    best_rank: float | None
    origin: str | None
    name: str
    generation: int | None
    order: int


def main() -> int:
    args = _parse_args()
    trace_path = Path(args.trace_path)
    if not trace_path.exists():
        print(f"Missing trace file: {trace_path}", file=sys.stderr)
        return 1

    trace = _load_trace(trace_path)
    operator_events = [event for event in trace if event.get("event") == OPERATOR_SAMPLED]
    if not operator_events:
        print("trace is not from a self-adaptive run")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else trace_path.with_name(f"{trace_path.stem}_inspection")
    out_dir.mkdir(parents=True, exist_ok=True)

    index = _build_hypothesis_index(trace)
    failures: list[str] = []

    print(f"Trace: {trace_path}")
    print(f"Output directory: {out_dir}")
    print(f"Hypothesis records indexed: {len(index)}")

    perturbation_ok = _check_perturbation_magnitude(operator_events, index)
    if perturbation_ok is False:
        failures.append("perturbation magnitude")
    lineage_ok = _check_lineage_integrity(operator_events, index)
    if lineage_ok is False:
        failures.append("parent lineage")
    sigma_ok = _check_population_mean_sigma(trace, out_dir)
    if sigma_ok is False:
        failures.append("population mean sigma")
    if not _check_best_lineage(index, out_dir):
        failures.append("best lineage")
    if not _check_operator_usage(operator_events, trace, out_dir):
        failures.append("operator usage")

    if failures:
        print(f"FAIL: {', '.join(failures)}")
    else:
        print("PASS")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a self-adaptive Contexto trace.")
    parser.add_argument("trace_path")
    parser.add_argument("--out-dir")
    return parser.parse_args()


def _load_trace(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a trace event list.")
    return data


def _build_hypothesis_index(trace: list[dict[str, Any]]) -> dict[str, HypothesisRecord]:
    index: dict[str, HypothesisRecord] = {}
    for order, event in enumerate(trace):
        generation = event.get("generation")
        if not isinstance(generation, int):
            generation = None
        for candidate in _iter_hypothesis_records(event.get("details")):
            record = _coerce_hypothesis_record(candidate, generation, order)
            if record is not None and record.hypothesis_id not in index:
                index[record.hypothesis_id] = record
        operator_record = _coerce_operator_child_record(event, generation, order)
        if operator_record is not None and operator_record.hypothesis_id not in index:
            index[operator_record.hypothesis_id] = operator_record
    return index


def _iter_hypothesis_records(value: Any):
    if isinstance(value, dict):
        if "hypothesis_id" in value and "sigma" in value:
            yield value
        for child_value in value.values():
            yield from _iter_hypothesis_records(child_value)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_hypothesis_records(item)


def _coerce_hypothesis_record(value: dict[str, Any], generation: int | None, order: int) -> HypothesisRecord | None:
    hypothesis_id = value.get("hypothesis_id")
    sigma = value.get("sigma")
    if not isinstance(hypothesis_id, str) or not isinstance(sigma, list):
        return None
    sigma_array = np.asarray(sigma, dtype=float)
    if sigma_array.shape != (4,):
        return None
    best_rank = value.get("best_rank")
    if best_rank is not None:
        best_rank = float(best_rank)
    name = value.get("category_name") or value.get("name") or "<unnamed>"
    return HypothesisRecord(
        hypothesis_id=hypothesis_id,
        parent_id=value.get("parent_id") if isinstance(value.get("parent_id"), str) else None,
        sigma=sigma_array,
        best_word=value.get("best_word") if isinstance(value.get("best_word"), str) else None,
        best_rank=best_rank,
        origin=value.get("origin") if isinstance(value.get("origin"), str) else None,
        name=str(name),
        generation=generation,
        order=order,
    )


def _coerce_operator_child_record(
    event: dict[str, Any],
    generation: int | None,
    order: int,
) -> HypothesisRecord | None:
    if event.get("event") != OPERATOR_SAMPLED:
        return None
    details = _details(event)
    child_id = details.get("child_id")
    child_sigma = details.get("child_sigma")
    if not isinstance(child_id, str) or not isinstance(child_sigma, list):
        return None
    sigma_array = np.asarray(child_sigma, dtype=float)
    if sigma_array.shape != (4,):
        return None
    child_name = details.get("child_hypothesis_name")
    return HypothesisRecord(
        hypothesis_id=child_id,
        parent_id=details.get("parent_id") if isinstance(details.get("parent_id"), str) else None,
        sigma=sigma_array,
        best_word=None,
        best_rank=None,
        origin="mutation",
        name=str(child_name) if isinstance(child_name, str) else f"mutation child {child_id[:8]}",
        generation=generation,
        order=order,
    )


def _check_perturbation_magnitude(
    operator_events: list[dict[str, Any]],
    index: dict[str, HypothesisRecord],
) -> bool | None:
    print("\n=== Check 1: Perturbation magnitude ===")
    checked: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []
    unresolved: list[str] = []
    too_small: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []
    too_large: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []

    for event in operator_events:
        details = _details(event)
        child_id = details.get("child_id")
        sigma_snapshot = details.get("sigma_snapshot")
        sampled_op = str(details.get("sampled_op"))
        if not isinstance(child_id, str) or not isinstance(sigma_snapshot, list):
            unresolved.append(str(child_id))
            continue
        child_sigma = _child_sigma(details, index)
        if child_sigma is None:
            unresolved.append(child_id)
            continue
        parent_sigma = np.asarray(sigma_snapshot, dtype=float)
        l2 = float(np.linalg.norm(child_sigma - parent_sigma))
        item = (child_id, sampled_op, parent_sigma, child_sigma, l2)
        checked.append(item)
        if l2 < 0.04:
            too_small.append(item)
        elif l2 > 0.25:
            too_large.append(item)

    print(f"Total mutations checked: {len(checked)}")
    print(f"Unresolved child records: {len(unresolved)}")
    if unresolved:
        _print_id_sample(unresolved, "unresolved child_id")
    l2_values = [item[4] for item in checked]
    print(f"L2 mean: {_fmt_float(np.mean(l2_values)) if l2_values else 'NA'}")
    print(f"L2 std: {_fmt_float(np.std(l2_values)) if l2_values else 'NA'}")
    print(f"Too small (<0.04): {len(too_small)}")
    _print_outliers(too_small)
    print(f"Too large (>0.25): {len(too_large)}")
    _print_outliers(too_large)
    if not checked and unresolved:
        print(
            "Legacy trace note: mutation children do not include child_sigma or full child records, "
            "so perturbation magnitude cannot be reconstructed from this trace."
        )
        return None
    return not unresolved and not too_small and not too_large


def _print_outliers(items: list[tuple[str, str, np.ndarray, np.ndarray, float]]) -> None:
    for child_id, sampled_op, parent_sigma, child_sigma, l2 in items:
        print(
            f"  child_id={child_id} sampled_op={sampled_op} "
            f"sigma_snapshot={_fmt_array(parent_sigma)} child_sigma={_fmt_array(child_sigma)} L2={l2:.6f}"
        )


def _check_lineage_integrity(
    operator_events: list[dict[str, Any]],
    index: dict[str, HypothesisRecord],
) -> bool | None:
    print("\n=== Check 2: Parent_id lineage integrity ===")
    missing_parent: list[str] = []
    orphan_parent: list[str] = []
    unresolved_child: list[str] = []
    for event in operator_events:
        details = _details(event)
        child_id = str(details.get("child_id"))
        parent_id = details.get("parent_id")
        child = index.get(child_id)
        if child is None:
            unresolved_child.append(child_id)
            continue
        if child is not None and child.origin == "crossover":
            continue
        if not isinstance(parent_id, str) or not parent_id:
            missing_parent.append(child_id)
        elif parent_id not in index:
            orphan_parent.append(child_id)
    print(f"Unresolved child records: {len(unresolved_child)}")
    _print_id_sample(unresolved_child, "child_id")
    print(f"Missing parent_id: {len(missing_parent)}")
    _print_id_sample(missing_parent, "child_id")
    print(f"Orphan parent_id: {len(orphan_parent)}")
    _print_id_sample(orphan_parent, "child_id")
    if unresolved_child and not missing_parent and not orphan_parent:
        print(
            "Legacy trace note: mutation children are referenced by id but not stored as records, "
            "so child-side lineage cannot be fully checked from this trace."
        )
        return None
    return not missing_parent and not orphan_parent


def _child_sigma(details: dict[str, Any], index: dict[str, HypothesisRecord]) -> np.ndarray | None:
    child_sigma = details.get("child_sigma")
    if isinstance(child_sigma, list):
        sigma_array = np.asarray(child_sigma, dtype=float)
        if sigma_array.shape == (4,):
            return sigma_array
    child_id = details.get("child_id")
    if not isinstance(child_id, str):
        return None
    child = index.get(child_id)
    return child.sigma if child is not None else None


def _print_id_sample(ids: list[str], label: str, limit: int = 10) -> None:
    for item_id in ids[:limit]:
        print(f"  {label}={item_id}")
    remaining = len(ids) - limit
    if remaining > 0:
        print(f"  ... {remaining} more")


def _check_population_mean_sigma(trace: list[dict[str, Any]], out_dir: Path) -> bool | None:
    print("\n=== Check 3: Population mean sigma over generations ===")
    events = [event for event in trace if event.get("event") == SIGMA_TRAJECTORY]
    if not events:
        print("No SIGMA_TRAJECTORY events found; skipping check 3.")
        return None

    generations = []
    sigmas = []
    for event in events:
        sigma = _details(event).get("mean_sigma")
        if isinstance(event.get("generation"), int) and isinstance(sigma, list) and len(sigma) == 4:
            generations.append(int(event["generation"]))
            sigmas.append(np.asarray(sigma, dtype=float))
    sigma_matrix = np.vstack(sigmas)
    _plot_sigma_lines(generations, sigma_matrix, "Mean sigma over generations", out_dir / "mean_sigma_over_generations.png")
    print(f"Initial mean_sigma: {_fmt_array(sigma_matrix[0])}")
    print(f"Final mean_sigma: {_fmt_array(sigma_matrix[-1])}")
    print(f"Delta: {_fmt_array(sigma_matrix[-1] - sigma_matrix[0])}")

    chaotic = []
    for previous, current_generation, current in zip(sigma_matrix[:-1], generations[1:], sigma_matrix[1:]):
        delta = np.abs(current - previous)
        if np.any(delta > 0.10):
            chaotic.append((current_generation, delta))
    print(f"Chaotic jumps: {len(chaotic)}")
    for generation, delta in chaotic:
        print(f"  generation={generation} abs_delta={_fmt_array(delta)}")
    return not chaotic


def _check_best_lineage(index: dict[str, HypothesisRecord], out_dir: Path) -> bool:
    print("\n=== Check 4: Best-lineage sigma trajectory ===")
    ranked = [record for record in index.values() if record.best_rank is not None]
    if not ranked:
        print("No indexed hypothesis has best_rank; cannot compute best lineage.")
        return False
    best = min(ranked, key=lambda record: (record.best_rank, record.order))
    lineage = []
    seen: set[str] = set()
    current: HypothesisRecord | None = best
    terminated_at_crossover = False
    while current is not None:
        if current.hypothesis_id in seen:
            raise RuntimeError(f"Cycle detected while walking parent_id at {current.hypothesis_id}.")
        seen.add(current.hypothesis_id)
        lineage.append(current)
        if current.origin == "crossover":
            terminated_at_crossover = True
            break
        if current.parent_id is None:
            break
        current = index.get(current.parent_id)

    lineage = list(reversed(lineage))
    sigma_matrix = np.vstack([record.sigma for record in lineage])
    _plot_sigma_lines(
        list(range(len(lineage))),
        sigma_matrix,
        "Best-lineage sigma trajectory",
        out_dir / "best_lineage_sigma_trajectory.png",
        x_label="Depth from root",
    )

    print(f"Best hypothesis: {best.name}")
    print(f"Best word/rank: {best.best_word} / {best.best_rank:g}")
    print(f"Generation reached: {best.generation}")
    print(f"Lineage length: {len(lineage)}")
    if terminated_at_crossover:
        print("Lineage walk terminated at a crossover ancestor.")
    print("Lineage:")
    for record in lineage:
        print(
            f"  ({record.hypothesis_id[:8]}, {record.name}, {record.best_word}, "
            f"{record.best_rank}, {_fmt_array(record.sigma)})"
        )
    return True


def _check_operator_usage(operator_events: list[dict[str, Any]], trace: list[dict[str, Any]], out_dir: Path) -> bool:
    print("\n=== Check 5: Operator usage histogram ===")
    counts = {operator: 0 for operator in OPERATORS}
    for event in operator_events:
        sampled_op = _details(event).get("sampled_op")
        if sampled_op in counts:
            counts[sampled_op] += 1
    total = sum(counts.values())
    frequencies = {operator: (counts[operator] / total if total else 0.0) for operator in OPERATORS}
    final_sigma = _final_mean_sigma(trace)
    deltas = {
        operator: abs(frequencies[operator] - float(final_sigma[index]))
        for index, operator in enumerate(OPERATORS)
    }
    mismatches = {operator: delta for operator, delta in deltas.items() if delta > 0.10}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(OPERATORS, [counts[operator] for operator in OPERATORS], color="tab:blue")
    ax.set_ylabel("Count")
    ax.set_title("Operator usage histogram")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "operator_usage_histogram.png", dpi=200)
    plt.close(fig)

    print(f"Raw counts: {counts}")
    print(f"Empirical frequencies: { {operator: round(frequencies[operator], 6) for operator in OPERATORS} }")
    print(f"Final-generation mean_sigma: {_fmt_array(final_sigma)}")
    print(f"|empirical_freq - final_mean_sigma|: { {operator: round(deltas[operator], 6) for operator in OPERATORS} }")
    print(f"Sampling mismatches (>0.10): {len(mismatches)}")
    for operator, delta in mismatches.items():
        print(f"  operator={operator} delta={delta:.6f}")
    return not mismatches


def _plot_sigma_lines(
    x_values: list[int],
    sigma_matrix: np.ndarray,
    title: str,
    output_path: Path,
    x_label: str = "Generation",
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for index, operator in enumerate(OPERATORS):
        ax.plot(x_values, sigma_matrix[:, index], marker="o", label=operator)
    ax.axhline(0.25, color="0.4", linestyle="--", linewidth=1.2, label="uniform baseline")
    ax.set_ylim(0, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability mass")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _final_mean_sigma(trace: list[dict[str, Any]]) -> np.ndarray:
    for event in reversed(trace):
        if event.get("event") == SIGMA_TRAJECTORY:
            sigma = _details(event).get("mean_sigma")
            if isinstance(sigma, list) and len(sigma) == 4:
                return np.asarray(sigma, dtype=float)
    return np.full(4, np.nan)


def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _fmt_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def _fmt_float(value: float | np.floating) -> str:
    return f"{float(value):.6f}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
