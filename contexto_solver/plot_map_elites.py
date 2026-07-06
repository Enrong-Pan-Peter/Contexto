"""Visualize MAP-Elites archive traces as static PNG figures.

Standalone analysis script that consumes a MAP-Elites trace JSON (produced by
``method=ea_llm_map_elites``) and renders seven figures plus an optional
combined summary. Mirrors the structural conventions of
``contexto_solver.plot_trajectory``: matplotlib is imported lazily so solver
code never depends on it, and all data is reconstructed from already-emitted
trace events (no new events required).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FIGURE_DIR = Path("figures")
SIGMA_LABELS = ["sigma_s", "sigma_m", "sigma_ml", "sigma_l"]
ARCHIVE_EVENTS = ("ARCHIVE_PLACE", "ARCHIVE_REPLACE", "ARCHIVE_REJECT")
DEFAULT_SNAPSHOT_GENS = [10, 20, 30, 40]
ALL_PLOTS = ["occupancy", "growth", "hits", "scatter", "sigma_final", "sigma_snapshots", "lineage"]
PLOT_FILENAMES = {
    "occupancy": "cell_occupancy.png",
    "growth": "archive_growth.png",
    "hits": "cell_hit_count.png",
    "scatter": "continuous_scatter.png",
    "sigma_final": "sigma_per_component_final.png",
    "sigma_snapshots": "sigma_snapshots_over_time.png",
    "lineage": "winning_lineage_sigma.png",
}


# --- generic trace helpers ---------------------------------------------------


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{trace_path} must contain a JSON list of trace events.")
    return data


def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _extract_target(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        if event.get("event") == "RUN_CONFIG":
            target = _details(event).get("target")
            if isinstance(target, str) and target:
                return target
    return None


def _run_label(trace_path: str | Path) -> str:
    return Path(trace_path).stem


def extract_axis_definition(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pull anchor scales and grid resolution from the AXIS_DEFINITION event."""
    for event in events:
        if event.get("event") != "AXIS_DEFINITION":
            continue
        details = _details(event)
        resolution = details.get("grid_resolution")
        if not isinstance(resolution, int) or resolution < 1:
            resolution = 5
        return {
            "resolution": resolution,
            "concreteness": _parse_axis(details.get("concreteness"), "concreteness"),
            "specificity": _parse_axis(details.get("specificity"), "specificity"),
        }
    return None


def _parse_axis(value: Any, default_label: str) -> dict[str, Any]:
    label = default_label
    anchors: dict[float, str] = {}
    if isinstance(value, dict):
        if isinstance(value.get("label"), str):
            label = value["label"]
        raw_anchors = value.get("anchors")
        if isinstance(raw_anchors, dict):
            for key, word in raw_anchors.items():
                try:
                    anchors[float(key)] = str(word)
                except (TypeError, ValueError):
                    continue
    positions = sorted(anchors)
    return {
        "label": label,
        "positions": positions,
        "words": [anchors[position] for position in positions],
    }


# --- archive state reconstruction --------------------------------------------


def _normalize_cell_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    cell = raw.get("cell")
    if not isinstance(cell, (list, tuple)) or len(cell) != 2:
        return None
    sigma = raw.get("sigma")
    sigma_array = np.asarray(sigma, dtype=float) if isinstance(sigma, list) and len(sigma) == 4 else None
    coordinates = raw.get("coordinates")
    coords = (
        (float(coordinates[0]), float(coordinates[1]))
        if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2
        else None
    )
    return {
        "cell": (int(cell[0]), int(cell[1])),
        "hypothesis_id": raw.get("hypothesis_id"),
        "best_word": raw.get("best_word"),
        "best_rank": raw.get("best_rank"),
        "coordinates": coords,
        "sigma": sigma_array,
    }


def _snapshots(events: list[dict[str, Any]]) -> list[tuple[int, dict[tuple[int, int], dict[str, Any]]]]:
    snapshots: list[tuple[int, dict[tuple[int, int], dict[str, Any]]]] = []
    for event in events:
        if event.get("event") != "ARCHIVE_SNAPSHOT":
            continue
        generation = event.get("generation")
        cells_raw = _details(event).get("cells")
        if not isinstance(cells_raw, list):
            continue
        cells: dict[tuple[int, int], dict[str, Any]] = {}
        for raw in cells_raw:
            if not isinstance(raw, dict):
                continue
            record = _normalize_cell_record(raw)
            if record is not None:
                cells[record["cell"]] = record
        snapshots.append((int(generation) if isinstance(generation, int) else -1, cells))
    return snapshots


def _state_via_replay(events: list[dict[str, Any]], up_to_gen: int) -> dict[tuple[int, int], dict[str, Any]]:
    """Fallback used only before the first ARCHIVE_SNAPSHOT (e.g. init state)."""
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for event in events:
        name = event.get("event")
        generation = event.get("generation")
        if name not in {"ARCHIVE_PLACE", "ARCHIVE_REPLACE"}:
            continue
        if isinstance(generation, int) and generation > up_to_gen:
            continue
        details = _details(event)
        cell = details.get("cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            continue
        key = (int(cell[0]), int(cell[1]))
        if name == "ARCHIVE_PLACE":
            cells[key] = {
                "cell": key,
                "hypothesis_id": details.get("hypothesis_id"),
                "best_word": details.get("best_word"),
                "best_rank": details.get("rank"),
                "coordinates": None,
                "sigma": np.asarray(details["sigma"], dtype=float)
                if isinstance(details.get("sigma"), list)
                else None,
            }
        else:  # ARCHIVE_REPLACE has no best_word/coordinates
            prior = cells.get(key, {"cell": key, "best_word": None, "coordinates": None})
            prior.update(
                {
                    "hypothesis_id": details.get("new_hypothesis_id"),
                    "best_rank": details.get("new_rank"),
                    "sigma": np.asarray(details["new_sigma"], dtype=float)
                    if isinstance(details.get("new_sigma"), list)
                    else None,
                }
            )
            cells[key] = prior
    return cells


def snapshot_for_gen(
    events: list[dict[str, Any]],
    up_to_gen: int | None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Return the archive state (cell -> incumbent record) at a generation.

    Snapshots are complete, so this is a lookup rather than a replay: pick the
    latest ARCHIVE_SNAPSHOT with generation <= up_to_gen (or the last one when
    up_to_gen is None). Falls back to ARCHIVE_PLACE/REPLACE replay only when no
    snapshot exists at or before the requested generation.
    """
    snapshots = _snapshots(events)
    if up_to_gen is None:
        return snapshots[-1][1] if snapshots else _state_via_replay(events, 10**9)
    candidates = [cells for generation, cells in snapshots if generation <= up_to_gen]
    if candidates:
        return candidates[-1]
    return _state_via_replay(events, up_to_gen)


def final_generation(events: list[dict[str, Any]]) -> int:
    snapshots = _snapshots(events)
    if snapshots:
        return snapshots[-1][0]
    gens = [event.get("generation") for event in events if event.get("event") in ARCHIVE_EVENTS]
    valid = [generation for generation in gens if isinstance(generation, int)]
    return max(valid) if valid else 0


def occupancy_series(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Occupied-cell count and placement-attempt count per generation."""
    place_gen: dict[tuple[int, int], int] = {}
    attempts: dict[int, int] = defaultdict(int)
    all_gens: set[int] = set()
    for event in events:
        name = event.get("event")
        generation = event.get("generation")
        if name not in ARCHIVE_EVENTS or not isinstance(generation, int):
            continue
        all_gens.add(generation)
        attempts[generation] += 1
        if name == "ARCHIVE_PLACE":
            details = _details(event)
            cell = details.get("cell")
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                key = (int(cell[0]), int(cell[1]))
                place_gen.setdefault(key, generation)
    generations = sorted(all_gens)
    occupancy = [sum(1 for gen in place_gen.values() if gen <= boundary) for boundary in generations]
    attempt_counts = [attempts[generation] for generation in generations]
    return {"generations": generations, "occupancy": occupancy, "attempts": attempt_counts}


def hit_counts(events: list[dict[str, Any]], resolution: int) -> np.ndarray:
    """Total placement attempts (PLACE+REPLACE+REJECT) landing in each cell."""
    grid = np.zeros((resolution, resolution), dtype=float)
    for event in events:
        if event.get("event") not in ARCHIVE_EVENTS:
            continue
        cell = _details(event).get("cell")
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            continue
        i, j = int(cell[0]), int(cell[1])
        if 0 <= i < resolution and 0 <= j < resolution:
            grid[j, i] += 1
    return grid


def placement_points(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join each PLACEMENT with the immediately-following ARCHIVE_* for rank.

    Relies on the back-to-back emission order in the solver's
    ``_place_and_compete`` (PLACEMENT then exactly one ARCHIVE_* outcome).
    """
    points: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if event.get("event") != "PLACEMENT":
            continue
        details = _details(event)
        coordinates = details.get("coordinates")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            continue
        rank = None
        outcome = None
        if index + 1 < len(events) and events[index + 1].get("event") in ARCHIVE_EVENTS:
            outcome = events[index + 1].get("event")
            rank = _rank_from_archive_event(events[index + 1])
        points.append(
            {
                "x": float(coordinates[0]),
                "y": float(coordinates[1]),
                "rank": rank,
                "word": details.get("word"),
                "generation": event.get("generation"),
                "outcome": outcome,
            }
        )
    return points


def _rank_from_archive_event(event: dict[str, Any]) -> int | None:
    details = _details(event)
    name = event.get("event")
    if name == "ARCHIVE_PLACE":
        rank = details.get("rank")
    elif name == "ARCHIVE_REPLACE":
        rank = details.get("new_rank")
    elif name == "ARCHIVE_REJECT":
        rank = details.get("child_rank")
    else:
        rank = None
    return rank if isinstance(rank, int) else None


# --- lineage reconstruction (ported from inspect_self_adaptive_trace.py) -----


def _build_hypothesis_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for order, event in enumerate(events):
        for raw in _iter_hypothesis_records(_details(event)):
            record = _coerce_record(raw, order)
            if record is not None and record["hypothesis_id"] not in index:
                index[record["hypothesis_id"]] = record
        operator_record = _operator_child_record(event, order)
        if operator_record is not None and operator_record["hypothesis_id"] not in index:
            index[operator_record["hypothesis_id"]] = operator_record
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


def _coerce_record(raw: dict[str, Any], order: int) -> dict[str, Any] | None:
    hypothesis_id = raw.get("hypothesis_id")
    sigma = raw.get("sigma")
    if not isinstance(hypothesis_id, str) or not isinstance(sigma, list) or len(sigma) != 4:
        return None
    name = raw.get("category_name") or raw.get("name") or "<unnamed>"
    best_rank = raw.get("best_rank")
    return {
        "hypothesis_id": hypothesis_id,
        "parent_id": raw.get("parent_id") if isinstance(raw.get("parent_id"), str) else None,
        "sigma": np.asarray(sigma, dtype=float),
        "origin": raw.get("origin") if isinstance(raw.get("origin"), str) else None,
        "name": str(name),
        "best_rank": float(best_rank) if isinstance(best_rank, (int, float)) else None,
        "order": order,
    }


def _operator_child_record(event: dict[str, Any], order: int) -> dict[str, Any] | None:
    if event.get("event") != "OPERATOR_SAMPLED":
        return None
    details = _details(event)
    child_id = details.get("child_id")
    child_sigma = details.get("child_sigma")
    if not isinstance(child_id, str) or not isinstance(child_sigma, list) or len(child_sigma) != 4:
        return None
    name = details.get("child_hypothesis_name")
    return {
        "hypothesis_id": child_id,
        "parent_id": details.get("parent_id") if isinstance(details.get("parent_id"), str) else None,
        "sigma": np.asarray(child_sigma, dtype=float),
        "origin": "mutation",
        "name": str(name) if isinstance(name, str) else f"mutation {child_id[:8]}",
        "best_rank": None,
        "order": order,
    }


def _build_crossover_index(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for order, event in enumerate(events):
        if event.get("event") != "CROSSOVER":
            continue
        details = _details(event)
        child = details.get("child")
        if not isinstance(child, dict):
            continue
        child_id = child.get("hypothesis_id")
        if not isinstance(child_id, str):
            continue
        parents = details.get("parents") if isinstance(details.get("parents"), list) else []
        names = [item if isinstance(item, str) else None for item in (parents + [None, None])[:2]]
        index[child_id] = {
            "order": order,
            "parent_names": names,
            "parent_sigmas": (_coerce_sigma(details.get("parent_a_sigma")), _coerce_sigma(details.get("parent_b_sigma"))),
        }
    return index


def _coerce_sigma(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    return np.asarray(value, dtype=float)


def _records_by_name(index: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for record in index.values():
        by_name.setdefault(record["name"], []).append(record)
    for records in by_name.values():
        records.sort(key=lambda record: record["order"])
    return by_name


def winning_lineage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the sigma trajectory along the longest branch ending at the best cell."""
    index = _build_hypothesis_index(events)
    crossover_index = _build_crossover_index(events)
    name_index = _records_by_name(index)

    final_state = snapshot_for_gen(events, None)
    ranked = [record for record in final_state.values() if isinstance(record.get("best_rank"), (int, float))]
    if not ranked:
        return {"depths": [], "sigmas": np.empty((0, 4)), "crossover_depths": [], "best": None}
    best_cell = min(ranked, key=lambda record: record["best_rank"])
    best_id = best_cell.get("hypothesis_id")
    best_record = index.get(best_id) if isinstance(best_id, str) else None
    if best_record is None:
        return {"depths": [], "sigmas": np.empty((0, 4)), "crossover_depths": [], "best": best_cell}

    branch = _longest_branch(best_record, index, crossover_index, name_index, set())
    root_to_best = list(reversed(branch))
    sigmas = np.vstack([node["sigma"] for node in root_to_best]) if root_to_best else np.empty((0, 4))
    crossover_depths = [depth for depth, node in enumerate(root_to_best) if node["origin"] == "crossover"]
    return {
        "depths": list(range(len(root_to_best))),
        "sigmas": sigmas,
        "crossover_depths": crossover_depths,
        "best": best_cell,
        "branch": root_to_best,
    }


def _longest_branch(
    record: dict[str, Any],
    index: dict[str, dict[str, Any]],
    crossover_index: dict[str, dict[str, Any]],
    name_index: dict[str, list[dict[str, Any]]],
    seen: set[str],
) -> list[dict[str, Any]]:
    if record["hypothesis_id"] in seen:
        return [record]
    seen = set(seen)
    seen.add(record["hypothesis_id"])

    parents: list[dict[str, Any]] = []
    if record["origin"] == "crossover":
        crossover = crossover_index.get(record["hypothesis_id"])
        if crossover is not None:
            for branch_index in range(2):
                parent = _resolve_crossover_parent(crossover, branch_index, name_index)
                if parent is not None:
                    parents.append(parent)
    elif record["parent_id"] is not None:
        parent = index.get(record["parent_id"])
        if parent is not None:
            parents.append(parent)

    if not parents:
        return [record]
    longest = max((_longest_branch(parent, index, crossover_index, name_index, seen) for parent in parents), key=len)
    return [record] + longest


def _resolve_crossover_parent(
    crossover: dict[str, Any],
    parent_index: int,
    name_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    name = crossover["parent_names"][parent_index]
    sigma = crossover["parent_sigmas"][parent_index]
    candidates = [record for record in name_index.get(name, [])] if name else []
    candidates = [record for record in candidates if record["order"] < crossover["order"]]
    if sigma is not None:
        sigma_matches = [record for record in candidates if np.allclose(record["sigma"], sigma, atol=1e-6)]
        if sigma_matches:
            return max(sigma_matches, key=lambda record: record["order"])
    if candidates:
        return max(candidates, key=lambda record: record["order"])
    return None


# --- plotting ----------------------------------------------------------------


def _get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _state_to_grid(
    cells: dict[tuple[int, int], dict[str, Any]],
    resolution: int,
    value_fn,
) -> np.ndarray:
    grid = np.full((resolution, resolution), np.nan, dtype=float)
    for (i, j), record in cells.items():
        if 0 <= i < resolution and 0 <= j < resolution:
            value = value_fn(record)
            if value is not None:
                grid[j, i] = value
    return grid


def _log_inv_rank(rank: Any) -> float | None:
    """log(1/rank): higher (brighter) means a better, lower rank."""
    if not isinstance(rank, (int, float)) or rank < 1:
        return None
    return -math.log10(float(rank))


def _apply_anchor_ticks(ax, axis_def: dict[str, Any]) -> None:
    concreteness = axis_def["concreteness"]
    specificity = axis_def["specificity"]
    if concreteness["positions"]:
        ax.set_xticks(concreteness["positions"])
        ax.set_xticklabels(
            [f"{position:.2f}\n{word}" for position, word in zip(concreteness["positions"], concreteness["words"])],
            fontsize=7,
        )
    if specificity["positions"]:
        ax.set_yticks(specificity["positions"])
        ax.set_yticklabels(
            [f"{position:.2f} {word}" for position, word in zip(specificity["positions"], specificity["words"])],
            fontsize=7,
        )
    ax.set_xlabel("concreteness (0 concrete -> 1 abstract)")
    ax.set_ylabel("specificity (0 general -> 1 specific)")


def _colormap(name: str):
    import matplotlib

    try:
        colormap = matplotlib.colormaps[name].copy()
    except (AttributeError, KeyError):
        import matplotlib.cm as cm

        colormap = cm.get_cmap(name).copy()
    colormap.set_bad("white")
    return colormap


def _draw_grid_heatmap(plt, fig, ax, grid, axis_def, title, cmap, vmin, vmax, colorbar_label, text_grid=None):
    colormap = _colormap(cmap)
    masked = np.ma.masked_invalid(grid)
    image = ax.imshow(
        masked,
        origin="lower",
        extent=[0.0, 1.0, 0.0, 1.0],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    resolution = grid.shape[0]
    if text_grid is not None:
        for j in range(resolution):
            for i in range(resolution):
                label = text_grid[j][i]
                if label:
                    ax.text(
                        (i + 0.5) / resolution,
                        (j + 0.5) / resolution,
                        label,
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black",
                    )
    _apply_anchor_ticks(ax, axis_def)
    ax.set_title(title, fontsize=9)
    if colorbar_label:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    return image


def plot_cell_occupancy(events, axis_def, output_path, annotate=True) -> None:
    plt = _get_plt()
    resolution = axis_def["resolution"]
    cells = snapshot_for_gen(events, None)
    grid = _state_to_grid(cells, resolution, lambda record: _log_inv_rank(record.get("best_rank")))
    text_grid = None
    if annotate:
        text_grid = [["" for _ in range(resolution)] for _ in range(resolution)]
        for (i, j), record in cells.items():
            if 0 <= i < resolution and 0 <= j < resolution and isinstance(record.get("best_word"), str):
                text_grid[j][i] = record["best_word"]
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_grid_heatmap(
        plt, fig, ax, grid, axis_def,
        "Cell occupancy (final): color = log(1/best_rank)",
        "viridis", None, None, "log(1/best_rank)", text_grid,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_archive_growth(events, output_path) -> None:
    plt = _get_plt()
    series = occupancy_series(events)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(series["generations"], series["occupancy"], color="tab:blue", marker="o", linewidth=2.0, label="occupied cells")
    ax.set_xlabel("Generation (0 = init)")
    ax.set_ylabel("Occupied cells", color="tab:blue")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    secondary = ax.twinx()
    secondary.bar(series["generations"], series["attempts"], color="tab:orange", alpha=0.25, label="placement attempts")
    secondary.set_ylabel("Placement attempts per generation", color="tab:orange")
    secondary.set_ylim(bottom=0)
    ax.set_title("Archive growth over time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cell_hit_count(events, axis_def, output_path) -> None:
    plt = _get_plt()
    resolution = axis_def["resolution"]
    grid = hit_counts(events, resolution)
    grid_display = np.where(grid > 0, grid, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_grid_heatmap(
        plt, fig, ax, grid_display, axis_def,
        "Cell hit counts: total placement attempts per cell",
        "magma", 0, None, "attempts",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_continuous_scatter(events, axis_def, output_path) -> None:
    plt = _get_plt()
    resolution = axis_def["resolution"]
    points = [point for point in placement_points(events) if isinstance(point["rank"], int) and point["rank"] >= 1]
    fig, ax = plt.subplots(figsize=(8, 7))
    if points:
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        colors = [-math.log10(point["rank"]) for point in points]
        scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=28, alpha=0.85, edgecolors="none")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="log(1/rank)")
    for k in range(1, resolution):
        ax.axvline(k / resolution, color="0.6", alpha=0.5, linewidth=0.6)
        ax.axhline(k / resolution, color="0.6", alpha=0.5, linewidth=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _apply_anchor_ticks(ax, axis_def)
    ax.set_title("Continuous placement scatter (color = log(1/rank))", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sigma_per_component_final(events, axis_def, output_path) -> None:
    plt = _get_plt()
    resolution = axis_def["resolution"]
    cells = snapshot_for_gen(events, None)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    last_image = None
    for component, ax in enumerate(axes):
        grid = _state_to_grid(
            cells,
            resolution,
            lambda record, k=component: float(record["sigma"][k]) if record.get("sigma") is not None else None,
        )
        last_image = _draw_grid_heatmap(
            plt, fig, ax, grid, axis_def, SIGMA_LABELS[component], "viridis", 0.0, 1.0, None,
        )
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.02, label="sigma (0-1)")
    fig.suptitle("Per-component sigma of incumbents (final state)", fontsize=11)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sigma_snapshots_over_time(events, axis_def, output_path, snapshot_gens) -> None:
    plt = _get_plt()
    resolution = axis_def["resolution"]
    last_gen = final_generation(events)
    rows = _resolve_snapshot_rows(snapshot_gens, last_gen)
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.6 * len(rows)), squeeze=False)
    last_image = None
    for row_index, (label, gen) in enumerate(rows):
        cells = snapshot_for_gen(events, gen)
        for component in range(4):
            ax = axes[row_index][component]
            grid = _state_to_grid(
                cells,
                resolution,
                lambda record, k=component: float(record["sigma"][k]) if record.get("sigma") is not None else None,
            )
            title = f"{label} | {SIGMA_LABELS[component]}" if component == 0 else SIGMA_LABELS[component]
            last_image = _draw_grid_heatmap(plt, fig, ax, grid, axis_def, title, "viridis", 0.0, 1.0, None)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, fraction=0.015, pad=0.02, label="sigma (0-1)")
    fig.suptitle("Per-component sigma snapshots over time", fontsize=12)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _resolve_snapshot_rows(snapshot_gens: list[int], last_gen: int) -> list[tuple[str, int | None]]:
    rows: list[tuple[str, int | None]] = []
    seen: set[int] = set()
    for gen in snapshot_gens:
        if gen >= last_gen:
            label = f"final (gen={last_gen})"
            key = last_gen
        else:
            label = f"gen={gen}"
            key = gen
        if key in seen:
            continue
        seen.add(key)
        rows.append((label, None if key == last_gen else key))
    if last_gen not in seen:
        rows.append((f"final (gen={last_gen})", None))
    return rows


def plot_winning_lineage_sigma(events, output_path) -> None:
    plt = _get_plt()
    lineage = winning_lineage(events)
    fig, ax = plt.subplots(figsize=(9, 6))
    sigmas = lineage["sigmas"]
    if sigmas.shape[0] > 0:
        for component in range(4):
            ax.plot(lineage["depths"], sigmas[:, component], marker="o", linewidth=2.0, label=SIGMA_LABELS[component])
        for marker_index, depth in enumerate(lineage["crossover_depths"]):
            ax.axvline(depth, linestyle=":", color="0.35", linewidth=1.2, label="crossover" if marker_index == 0 else None)
        best = lineage.get("best") or {}
        title_suffix = f" (best: {best.get('best_word')} rank {best.get('best_rank')})" if best else ""
    else:
        title_suffix = " (no resolvable lineage)"
        ax.text(0.5, 0.5, "No winning lineage could be reconstructed", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.25, color="0.5", linestyle="--", linewidth=1.0, label="uniform baseline")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Lineage depth (root -> best)")
    ax.set_ylabel("Sigma probability mass")
    ax.set_title(f"Winning lineage sigma trajectory{title_suffix}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_combined_summary(output_dir: Path, generated: list[str], output_path: Path) -> None:
    plt = _get_plt()
    images = [(name, output_dir / PLOT_FILENAMES[name]) for name in generated if (output_dir / PLOT_FILENAMES[name]).exists()]
    if not images:
        return
    columns = 3
    rows = math.ceil(len(images) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    for index in range(rows * columns):
        ax = axes[index // columns][index % columns]
        ax.axis("off")
        if index < len(images):
            name, path = images[index]
            ax.imshow(plt.imread(path))
            ax.set_title(name, fontsize=9)
    fig.suptitle("MAP-Elites summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"Missing trace file: {trace_path}")
        return 1

    events = load_trace(trace_path)
    axis_def = extract_axis_definition(events)
    if axis_def is None:
        print(
            f"{trace_path} has no AXIS_DEFINITION event; this is not a MAP-Elites trace. "
            "Use a trace produced by --method ea_llm_map_elites."
        )
        return 0

    requested = _resolve_requested_plots(args.plots)
    snapshot_gens = _parse_snapshot_gens(args.snapshot_gens)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_FIGURE_DIR / _run_label(trace_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "occupancy": lambda path: plot_cell_occupancy(events, axis_def, path),
        "growth": lambda path: plot_archive_growth(events, path),
        "hits": lambda path: plot_cell_hit_count(events, axis_def, path),
        "scatter": lambda path: plot_continuous_scatter(events, axis_def, path),
        "sigma_final": lambda path: plot_sigma_per_component_final(events, axis_def, path),
        "sigma_snapshots": lambda path: plot_sigma_snapshots_over_time(events, axis_def, path, snapshot_gens),
        "lineage": lambda path: plot_winning_lineage_sigma(events, path),
    }

    target = _extract_target(events)
    print(f"Trace: {trace_path}")
    print(f"Target: {target}")
    print(f"Grid resolution: {axis_def['resolution']}")
    print(f"Output directory: {output_dir}")

    generated: list[str] = []
    for name in requested:
        output_path = output_dir / PLOT_FILENAMES[name]
        dispatch[name](output_path)
        generated.append(name)
        print(f"Wrote {output_path}")

    if args.combined:
        summary_path = output_dir / "map_elites_summary.png"
        plot_combined_summary(output_dir, generated, summary_path)
        print(f"Wrote {summary_path}")

    return 0


def _resolve_requested_plots(value: str) -> list[str]:
    if value == "all":
        return list(ALL_PLOTS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in ALL_PLOTS]
    if unknown:
        raise SystemExit(f"Unknown plot(s): {', '.join(unknown)}. Choose from: {', '.join(ALL_PLOTS)} or all.")
    return requested


def _parse_snapshot_gens(value: str) -> list[int]:
    gens: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            gens.append(int(item))
        except ValueError:
            raise SystemExit(f"Invalid --snapshot-gens value: {item!r}")
    return gens or list(DEFAULT_SNAPSHOT_GENS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MAP-Elites archive visualizations from a trace JSON.")
    parser.add_argument("--trace", required=True, help="Path to a MAP-Elites trace JSON.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to figures/<run_label>/.")
    parser.add_argument(
        "--plots",
        default="all",
        help=f"Comma-separated subset of {{{','.join(ALL_PLOTS)}}} or 'all'.",
    )
    parser.add_argument(
        "--snapshot-gens",
        default=",".join(str(gen) for gen in DEFAULT_SNAPSHOT_GENS),
        help="Comma-separated generations for the sigma snapshot grid (final is always appended).",
    )
    parser.add_argument("--combined", action="store_true", help="Also write a combined summary PNG.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
