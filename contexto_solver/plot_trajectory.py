"""Analyze and plot solver trajectories from trace JSON files."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import config
from .embeddings import EmbeddingModel


DEFAULT_FIGURE_DIR = Path("figures")
DEFAULT_NEIGHBORS = 50
DEFAULT_VARIANCE_COMPONENTS = 10


@dataclass(frozen=True)
class GenerationBest:
    generation: int
    word: str
    rank: int | None


@dataclass(frozen=True)
class GenerationCentroid:
    generation: int
    vector: np.ndarray
    valid_words: int
    skipped_words: int


@dataclass(frozen=True)
class HypothesisSnapshot:
    generation: int
    active_hypotheses: dict[str, "HypothesisState"]


@dataclass
class HypothesisState:
    words_tried: dict[str, int]
    status: str = "active"

    @property
    def best_word(self) -> str | None:
        if not self.words_tried:
            return None
        return min(self.words_tried, key=self.words_tried.get)

    @property
    def best_rank(self) -> int | None:
        best_word = self.best_word
        return self.words_tried[best_word] if best_word is not None else None


def check_explained_variance(
    embedding_model_path: str | Path,
    target_word: str,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    n_components: int = DEFAULT_VARIANCE_COMPONENTS,
) -> dict[str, Any]:
    """Fit PCA on a target neighborhood and report explained variance."""
    model = EmbeddingModel(embedding_model_path)
    neighborhood_words, neighborhood_embeddings = _target_neighborhood_embeddings(model, target_word, n_neighbors)
    projection = _fit_projection(neighborhood_embeddings, "pca", n_components)
    explained = [float(value) for value in projection.explained_variance_ratio_]
    cumulative = np.cumsum(explained)
    centered = neighborhood_embeddings - neighborhood_embeddings.mean(axis=0)
    total_variance = float(np.sum(np.var(centered, axis=0, ddof=1)))

    return {
        "embedding_model_path": str(embedding_model_path),
        "target_word": target_word,
        "n_neighbors_requested": n_neighbors,
        "n_neighbors_used": len(neighborhood_words),
        "n_components_requested": n_components,
        "n_components_used": len(explained),
        "explained_variance_ratio": explained,
        "cumulative_ratio": [float(value) for value in cumulative],
        "total_variance": total_variance,
    }


def plot_single_game(
    trace_path: str | Path,
    embedding_model_path: str | Path,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    n_components: int = 2,
    output_path: str | Path | None = None,
    projection: str = "pca",
    annotate_best: bool = True,
) -> None:
    """Plot one trace's guessed words, best-so-far path, and hypothesis centroid path."""
    if n_components != 2:
        raise ValueError("plot_single_game currently supports only n_components=2.")

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    trace_path = Path(trace_path)
    trace = _load_trace(trace_path)
    target = _extract_target(trace)
    guessed_words = _extract_guess_words(trace)
    best_points = _extract_generation_best_words(trace)
    for warning in _verify_cumulative_best_snapshots(trace):
        warnings.warn(f"{trace_path}: {warning}", RuntimeWarning, stacklevel=2)
    _warn_duplicate_hypothesis_names(trace, trace_path)

    model = EmbeddingModel(embedding_model_path)
    neighborhood_words, neighborhood_embeddings = _target_neighborhood_embeddings(model, target, n_neighbors)
    fitted_projection = _fit_projection(neighborhood_embeddings, projection, n_components)
    word_projection = _project_unique_words(
        model,
        [target, *neighborhood_words, *guessed_words, *(point.word for point in best_points)],
        fitted_projection,
    )
    target_key = _normalize_word(target)
    if target_key not in word_projection:
        raise ValueError(f"Target word {target!r} is missing from {model.path}.")
    target_xy = word_projection[target_key]

    guess_projection = _project_words_from_cache(guessed_words, word_projection)
    best_projection = _project_generation_best_from_cache(best_points, word_projection)
    centroid_points = _extract_generation_centroids(trace, model)
    centroid_projection = _project_generation_centroids(centroid_points, fitted_projection)
    if not centroid_points:
        print(f"{trace_path}: no hypothesis snapshots found; skipping centroid trajectory.")

    output = Path(output_path) if output_path is not None else _default_single_output_path(trace_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(guess_projection["points"]) > 0:
        ax.scatter(
            guess_projection["points"][:, 0],
            guess_projection["points"][:, 1],
            s=18,
            c="0.55",
            alpha=0.28,
            edgecolors="none",
            label="guessed words",
            zorder=1,
        )

    if len(best_projection["points"]) > 0:
        points = best_projection["points"]
        generations = np.asarray(best_projection["generations"], dtype=float)
        ax.scatter(points[:, 0], points[:, 1], c=generations, cmap="viridis", s=48, zorder=4)
        if annotate_best:
            _annotate_best_words(ax, points, best_projection["words"])
        if len(points) > 1:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            line_collection = LineCollection(segments, cmap="viridis", linewidth=2.4, zorder=3)
            line_collection.set_array(generations[1:])
            ax.add_collection(line_collection)
            _add_direction_arrows(ax, points, color="black", alpha=0.45)
            colorbar = fig.colorbar(line_collection, ax=ax)
            colorbar.set_label("Generation")
        ax.plot([], [], color="tab:blue", linewidth=2.4, label="best-so-far trajectory")

    if len(centroid_projection["points"]) > 0:
        centroid_xy = centroid_projection["points"]
        ax.plot(
            centroid_xy[:, 0],
            centroid_xy[:, 1],
            linestyle="--",
            linewidth=2.0,
            color="tab:orange",
            marker="o",
            markersize=4,
            label="active-hypothesis centroid",
            zorder=2,
        )
        _add_direction_arrows(ax, centroid_xy, color="tab:orange", alpha=0.7)

    ax.scatter([target_xy[0]], [target_xy[1]], s=180, c="red", edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate(target, (target_xy[0], target_xy[1]), xytext=(8, 8), textcoords="offset points", weight="bold")
    ax.set_title(f"Trajectory for {target} ({trace_path.name})")
    _label_projection_axes(ax, fitted_projection, projection)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Trace: {trace_path}")
    print(f"Target: {target}")
    print(f"Embedding: {embedding_model_path}")
    print(f"Neighborhood words used: {len(neighborhood_words)}")
    print(f"Guessed words loaded: {len(guessed_words)}")
    print(f"Guessed words skipped as out-of-vocabulary: {guess_projection['skipped']}")
    print(f"Best trajectory points plotted: {len(best_projection['points'])}")
    print(f"Best trajectory points skipped as out-of-vocabulary: {best_projection['skipped']}")
    print(f"Centroid trajectory points plotted: {len(centroid_projection['points'])}")
    print(f"Centroid generations skipped for no valid active best words: {max(0, len(best_points) - len(centroid_points))}")
    target_guess_delta = _target_guess_coordinate_delta(target, target_xy, guess_projection)
    if target_guess_delta is not None:
        print(f"Target/guess coordinate delta for {target}: {target_guess_delta:.6g}")
    print(_projection_diagnostic(fitted_projection, projection))
    print(f"Wrote figure: {output}")


def plot_multi_run(
    trace_paths: Iterable[str | Path],
    embedding_model_path: str | Path,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    n_components: int = 2,
    output_path: str | Path | None = None,
    projection: str = "pca",
) -> None:
    """Plot multiple traces' trajectories. Implemented after variance review."""
    raise NotImplementedError("Multi-run trajectory plotting will be implemented after the variance check.")


def plot_rank_trajectory(trace_path: str | Path, output_path: str | Path | None = None) -> None:
    """Plot best-rank and active-hypothesis rank trajectories from a trace."""
    import matplotlib.pyplot as plt

    trace_path = Path(trace_path)
    trace = _load_trace(trace_path)
    target = _extract_target(trace)
    best_points = [point for point in _extract_generation_best_words(trace) if point.rank is not None and point.rank >= 1]
    snapshots = _reconstruct_hypothesis_snapshots(trace)
    _warn_duplicate_hypothesis_names(trace, trace_path)

    output = Path(output_path) if output_path is not None else _default_rank_output_path(trace_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    hypothesis_series = _active_hypothesis_rank_series(snapshots)
    for series in hypothesis_series.values():
        if len(series) < 1:
            continue
        generations = [_display_generation(generation) for generation, _ in series]
        ranks = [rank for _, rank in series]
        ax.plot(generations, ranks, color="0.45", alpha=0.22, linewidth=1.0)

    mean_series = _population_mean_rank_series(snapshots)
    if mean_series:
        ax.plot(
            [_display_generation(generation) for generation, _ in mean_series],
            [rank for _, rank in mean_series],
            linestyle="--",
            color="tab:orange",
            linewidth=2.0,
            label="population mean active rank",
        )

    if best_points:
        ax.plot(
            [_display_generation(point.generation) for point in best_points],
            [point.rank for point in best_points],
            color="tab:blue",
            linewidth=3.0,
            label="best rank so far",
            zorder=3,
        )

    ax.plot([], [], color="0.45", alpha=0.35, linewidth=1.0, label="active hypothesis best rank")
    ax.axhline(1, color="red", linestyle=":", linewidth=1.5, label="target rank")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.9)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Rank (log scale)")
    ax.set_title(f"Rank trajectory for {target}\n{trace_path.name}")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Trace: {trace_path}")
    print(f"Target: {target}")
    print(f"Best-rank points plotted: {len(best_points)}")
    print(f"Active hypothesis trajectories plotted: {len(hypothesis_series)}")
    print(f"Population mean points plotted: {len(mean_series)}")
    print(f"Wrote figure: {output}")


def plot_distance_trajectory(
    trace_path: str | Path,
    embedding_model_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """Plot cosine distance-to-target trajectories from a trace."""
    import matplotlib.pyplot as plt

    trace_path = Path(trace_path)
    trace = _load_trace(trace_path)
    target = _extract_target(trace)
    snapshots = _reconstruct_hypothesis_snapshots(trace)
    _warn_duplicate_hypothesis_names(trace, trace_path)

    model = EmbeddingModel(embedding_model_path)
    target_vector = model.get_vector(target)
    if target_vector is None:
        raise ValueError(f"Target word {target!r} is missing from {model.path}.")

    best_series, guessed_skipped = _best_distance_series_from_guesses(trace, model, target_vector)
    hypothesis_series, hypothesis_skipped = _active_hypothesis_distance_series(snapshots, model, target_vector)
    mean_series, mean_skipped = _population_mean_distance_series(snapshots, model, target_vector)

    output = Path(output_path) if output_path is not None else _default_distance_output_path(trace_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for series in hypothesis_series.values():
        if len(series) < 1:
            continue
        generations = [_display_generation(generation) for generation, _ in series]
        distances = [distance for _, distance in series]
        ax.plot(generations, distances, color="0.45", alpha=0.22, linewidth=1.0)

    if mean_series:
        ax.plot(
            [_display_generation(generation) for generation, _ in mean_series],
            [distance for _, distance in mean_series],
            linestyle="--",
            color="tab:orange",
            linewidth=2.0,
            label="population mean active distance",
        )

    if best_series:
        ax.plot(
            [_display_generation(generation) for generation, _ in best_series],
            [distance for _, distance in best_series],
            color="tab:blue",
            linewidth=3.0,
            label="minimum distance so far",
            zorder=3,
        )

    ax.plot([], [], color="0.45", alpha=0.35, linewidth=1.0, label="active hypothesis distance")
    ax.axhline(0, color="red", linestyle=":", linewidth=1.5, label="target distance")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cosine distance to target")
    ax.set_title(f"Distance to target for {target}\n{trace_path.name}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Trace: {trace_path}")
    print(f"Target: {target}")
    print(f"Embedding: {embedding_model_path}")
    print(f"Best-distance points plotted: {len(best_series)}")
    print(f"Active hypothesis trajectories plotted: {len(hypothesis_series)}")
    print(f"Population mean points plotted: {len(mean_series)}")
    print(f"Guessed words skipped as out-of-vocabulary: {guessed_skipped}")
    print(f"Hypothesis best words skipped as out-of-vocabulary: {hypothesis_skipped + mean_skipped}")
    print(f"Wrote figure: {output}")


def main() -> None:
    args = _parse_args()
    plot_type = "variance" if args.variance else args.plot_type
    if plot_type == "single" and args.traces and not args.trace:
        plot_type = "multi"

    if plot_type == "variance":
        rows = []
        for embedding_path in args.embedding:
            try:
                result = check_explained_variance(
                    embedding_path,
                    args.target,
                    n_neighbors=args.n_neighbors,
                    n_components=args.n_components or DEFAULT_VARIANCE_COMPONENTS,
                )
            except (FileNotFoundError, ValueError, ImportError) as exc:
                result = {
                    "embedding_model_path": str(embedding_path),
                    "target_word": args.target,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(result)
        _print_variance_table(rows)
        return

    if plot_type == "rank":
        if not args.trace:
            raise SystemExit("--plot-type rank requires --trace.")
        plot_rank_trajectory(args.trace, output_path=args.output)
        return

    if plot_type == "distance":
        if not args.trace:
            raise SystemExit("--plot-type distance requires --trace.")
        plot_distance_trajectory(args.trace, args.embedding[0], output_path=args.output)
        return

    if plot_type == "single":
        if not args.trace:
            raise SystemExit("--plot-type single requires --trace.")
        plot_single_game(
            args.trace,
            args.embedding[0],
            n_neighbors=args.n_neighbors,
            n_components=args.n_components or 2,
            output_path=args.output,
            projection=args.projection,
            annotate_best=args.annotate_best,
        )
        return

    if plot_type == "multi":
        if not args.traces:
            raise SystemExit("--plot-type multi requires --traces.")
        plot_multi_run(
            args.traces,
            args.embedding[0],
            n_neighbors=args.n_neighbors,
            n_components=args.n_components or 2,
            output_path=args.output,
            projection=args.projection,
        )
        return

    raise SystemExit(f"Unknown plot type: {plot_type}")


def _load_trace(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{trace_path} must contain a JSON list of trace events.")
    return data


def _extract_target(trace: list[dict[str, Any]]) -> str:
    for event in trace:
        if event.get("event") == "RUN_CONFIG":
            target = _details(event).get("target")
            if isinstance(target, str) and target:
                return target
    raise ValueError("Trace does not contain RUN_CONFIG.details.target.")


def _extract_guess_words(trace: list[dict[str, Any]]) -> list[str]:
    words: list[str] = []
    for event in trace:
        if event.get("event") != "GUESS":
            continue
        word = _details(event).get("word")
        if isinstance(word, str) and word:
            words.append(word)
    return words


def _extract_generation_best_words(trace: list[dict[str, Any]]) -> list[GenerationBest]:
    """Return the final cumulative best-word snapshot for each generation."""
    by_generation: dict[int, GenerationBest] = {}
    for event in trace:
        generation = event.get("generation")
        if not isinstance(generation, int) or generation < 0:
            continue
        snapshot = _best_snapshot_from_event(event)
        if snapshot is None:
            continue
        by_generation[generation] = GenerationBest(
            generation=generation,
            word=snapshot.word,
            rank=snapshot.rank,
        )
    return [by_generation[generation] for generation in sorted(by_generation)]


def _extract_generation_centroids(trace: list[dict[str, Any]], model: EmbeddingModel) -> list[GenerationCentroid]:
    """Reconstruct active-hypothesis centroids from trace events.

    Hypotheses without a best word, and hypotheses whose best words are missing
    from the selected embedding model, are skipped. Generations with no valid
    active best words are omitted, leaving a gap instead of carrying values.
    """
    states: dict[str, HypothesisState] = {}
    by_generation: dict[int, GenerationCentroid] = {}
    if not _trace_has_hypotheses(trace):
        return []

    for event in trace:
        generation = event.get("generation")
        details = _details(event)
        event_name = event.get("event")

        if event_name == "INIT":
            for hypothesis in _coerce_hypotheses(details.get("hypotheses")):
                states[hypothesis["category_name"]] = HypothesisState(
                    words_tried=dict(hypothesis["words_tried"]),
                    status=hypothesis["status"],
                )
        elif event_name == "GUESS":
            _apply_guess_to_hypothesis(states, details)
        elif event_name == "SELECT":
            _apply_selection_to_hypotheses(states, details)
        elif event_name == "CROSSOVER":
            _upsert_hypothesis_state(states, details.get("child"))
        elif event_name == "PIVOT_TRIGGERED":
            _upsert_hypothesis_state(states, details.get("hypothesis"))
        elif event_name == "DEDUPLICATE":
            _apply_deduplication_to_hypotheses(states, details)

        if isinstance(generation, int) and generation >= 0:
            centroid = _centroid_for_active_hypotheses(states, model, generation)
            if centroid is not None:
                by_generation[generation] = centroid

    return [by_generation[generation] for generation in sorted(by_generation)]


def _reconstruct_hypothesis_snapshots(trace: list[dict[str, Any]]) -> list[HypothesisSnapshot]:
    """Reconstruct active hypothesis state at the end of each generation."""
    states: dict[str, HypothesisState] = {}
    by_generation: dict[int, HypothesisSnapshot] = {}
    if not _trace_has_hypotheses(trace):
        return []

    for event in trace:
        generation = event.get("generation")
        details = _details(event)
        event_name = event.get("event")

        if event_name == "INIT":
            for hypothesis in _coerce_hypotheses(details.get("hypotheses")):
                states[hypothesis["category_name"]] = HypothesisState(
                    words_tried=dict(hypothesis["words_tried"]),
                    status=hypothesis["status"],
                )
        elif event_name == "GUESS":
            _apply_guess_to_hypothesis(states, details)
        elif event_name == "SELECT":
            _apply_selection_to_hypotheses(states, details)
        elif event_name == "CROSSOVER":
            _upsert_hypothesis_state(states, details.get("child"))
        elif event_name == "PIVOT_TRIGGERED":
            _upsert_hypothesis_state(states, details.get("hypothesis"))
        elif event_name == "DEDUPLICATE":
            _apply_deduplication_to_hypotheses(states, details)

        if isinstance(generation, int) and generation >= 0:
            active = {
                name: HypothesisState(words_tried=dict(state.words_tried), status=state.status)
                for name, state in states.items()
                if state.status == "active"
            }
            by_generation[generation] = HypothesisSnapshot(generation=generation, active_hypotheses=active)

    return [by_generation[generation] for generation in sorted(by_generation)]


def _active_hypothesis_rank_series(
    snapshots: list[HypothesisSnapshot],
) -> dict[str, list[tuple[int, int]]]:
    series: dict[str, list[tuple[int, int]]] = {}
    for snapshot in snapshots:
        for name, state in snapshot.active_hypotheses.items():
            rank = state.best_rank
            if rank is None or rank < 1:
                continue
            series.setdefault(name, []).append((snapshot.generation, rank))
    return series


def _population_mean_rank_series(snapshots: list[HypothesisSnapshot]) -> list[tuple[int, float]]:
    means: list[tuple[int, float]] = []
    for snapshot in snapshots:
        ranks = [
            state.best_rank
            for state in snapshot.active_hypotheses.values()
            if state.best_rank is not None and state.best_rank >= 1
        ]
        if ranks:
            means.append((snapshot.generation, float(np.mean(ranks))))
    return means


def _best_distance_series_from_guesses(
    trace: list[dict[str, Any]],
    model: EmbeddingModel,
    target_vector: np.ndarray,
) -> tuple[list[tuple[int, float]], int]:
    by_generation: dict[int, float] = {}
    best_distance: float | None = None
    skipped = 0
    for event in trace:
        generation = event.get("generation")
        if event.get("event") != "GUESS" or not isinstance(generation, int) or generation < 0:
            continue
        word = _details(event).get("word")
        if not isinstance(word, str):
            continue
        vector = model.get_vector(word)
        if vector is None:
            skipped += 1
            continue
        distance = _cosine_distance(vector, target_vector)
        best_distance = distance if best_distance is None else min(best_distance, distance)
        by_generation[generation] = best_distance
    return ([(generation, by_generation[generation]) for generation in sorted(by_generation)], skipped)


def _active_hypothesis_distance_series(
    snapshots: list[HypothesisSnapshot],
    model: EmbeddingModel,
    target_vector: np.ndarray,
) -> tuple[dict[str, list[tuple[int, float]]], int]:
    series: dict[str, list[tuple[int, float]]] = {}
    skipped = 0
    for snapshot in snapshots:
        for name, state in snapshot.active_hypotheses.items():
            word = state.best_word
            if word is None:
                continue
            vector = model.get_vector(word)
            if vector is None:
                skipped += 1
                continue
            series.setdefault(name, []).append((snapshot.generation, _cosine_distance(vector, target_vector)))
    return series, skipped


def _population_mean_distance_series(
    snapshots: list[HypothesisSnapshot],
    model: EmbeddingModel,
    target_vector: np.ndarray,
) -> tuple[list[tuple[int, float]], int]:
    means: list[tuple[int, float]] = []
    skipped = 0
    for snapshot in snapshots:
        distances: list[float] = []
        for state in snapshot.active_hypotheses.values():
            word = state.best_word
            if word is None:
                continue
            vector = model.get_vector(word)
            if vector is None:
                skipped += 1
                continue
            distances.append(_cosine_distance(vector, target_vector))
        if distances:
            means.append((snapshot.generation, float(np.mean(distances))))
    return means, skipped


def _cosine_distance(vector: np.ndarray, target_vector: np.ndarray) -> float:
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(target_vector))
    if denominator == 0:
        return float("nan")
    return float(1.0 - ((vector @ target_vector) / denominator))


def _display_generation(generation: int) -> int:
    return generation + 1


def _coerce_hypotheses(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    hypotheses: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("category_name")
        words = item.get("words_tried")
        status = item.get("status", "active")
        if not isinstance(name, str) or not isinstance(words, dict):
            continue
        hypotheses.append(
            {
                "category_name": name,
                "words_tried": {str(word): int(rank) for word, rank in words.items() if isinstance(rank, int)},
                "status": status if isinstance(status, str) else "active",
            }
        )
    return hypotheses


def _apply_guess_to_hypothesis(states: dict[str, HypothesisState], details: dict[str, Any]) -> None:
    hypothesis_name = details.get("hypothesis")
    word = details.get("word")
    rank = details.get("rank")
    if not isinstance(hypothesis_name, str) or not isinstance(word, str) or not isinstance(rank, int):
        return
    state = states.setdefault(hypothesis_name, HypothesisState(words_tried={}))
    state.words_tried[word] = rank


def _apply_selection_to_hypotheses(states: dict[str, HypothesisState], details: dict[str, Any]) -> None:
    kept = {value for value in details.get("kept", []) if isinstance(value, str)}
    discarded = {value for value in details.get("discarded", []) if isinstance(value, str)}
    for name in kept:
        states.setdefault(name, HypothesisState(words_tried={})).status = "active"
    for name in discarded:
        states.setdefault(name, HypothesisState(words_tried={})).status = "dormant"


def _upsert_hypothesis_state(states: dict[str, HypothesisState], value: Any) -> None:
    for hypothesis in _coerce_hypotheses([value]):
        states[hypothesis["category_name"]] = HypothesisState(
            words_tried=dict(hypothesis["words_tried"]),
            status=hypothesis["status"],
        )


def _apply_deduplication_to_hypotheses(states: dict[str, HypothesisState], details: dict[str, Any]) -> None:
    merges = details.get("merged")
    if not isinstance(merges, list):
        return
    for merge in merges:
        if not isinstance(merge, dict):
            continue
        survivor = merge.get("survivor")
        discarded = merge.get("discarded")
        if not isinstance(survivor, str) or not isinstance(discarded, str) or survivor == discarded:
            continue
        survivor_state = states.setdefault(survivor, HypothesisState(words_tried={}))
        discarded_state = states.pop(discarded, None)
        if discarded_state is None:
            continue
        survivor_state.words_tried.update(discarded_state.words_tried)
        if discarded_state.status == "active":
            survivor_state.status = "active"


def _centroid_for_active_hypotheses(
    states: dict[str, HypothesisState],
    model: EmbeddingModel,
    generation: int,
) -> GenerationCentroid | None:
    vectors: list[np.ndarray] = []
    skipped = 0
    for state in states.values():
        if state.status != "active":
            continue
        best_word = state.best_word
        if best_word is None:
            skipped += 1
            continue
        vector = model.get_vector(best_word)
        if vector is None:
            skipped += 1
            continue
        vectors.append(vector)
    if not vectors:
        return None
    return GenerationCentroid(
        generation=generation,
        vector=np.mean(np.vstack(vectors), axis=0),
        valid_words=len(vectors),
        skipped_words=skipped,
    )


def _verify_cumulative_best_snapshots(trace: list[dict[str, Any]]) -> list[str]:
    """Check that best-rank snapshots never get worse within the trace."""
    warnings_found: list[str] = []
    previous_rank: int | None = None
    for event in trace:
        snapshot = _best_snapshot_from_event(event)
        if snapshot is None or snapshot.rank is None:
            continue
        if previous_rank is not None and snapshot.rank > previous_rank:
            warnings_found.append(
                f"best_rank worsened from {previous_rank} to {snapshot.rank} at generation {event.get('generation')}; "
                "best snapshots may not be cumulative."
            )
            break
        previous_rank = snapshot.rank
    return warnings_found


def _best_snapshot_from_event(event: dict[str, Any]) -> GenerationBest | None:
    details = _details(event)
    event_name = event.get("event")
    generation = event.get("generation")
    if not isinstance(generation, int):
        generation = -1

    if event_name == "PIVOT_TRIGGERED":
        word = details.get("best_word_after_pivot")
        rank = details.get("best_rank_after_pivot")
    elif event_name == "SOLVED":
        word = details.get("answer")
        rank = details.get("rank")
    else:
        word = details.get("best_word")
        rank = details.get("best_rank")

    if not isinstance(word, str) or not word:
        return None
    return GenerationBest(generation=generation, word=word, rank=rank if isinstance(rank, int) else None)


def _warn_duplicate_hypothesis_names(trace: list[dict[str, Any]], trace_path: str | Path) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in _iter_hypothesis_names(trace):
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        warnings.warn(
            f"{trace_path}: duplicate hypothesis category names detected ({sample}); "
            "centroid reconstruction may be approximate.",
            RuntimeWarning,
            stacklevel=2,
        )


def _trace_has_hypotheses(trace: list[dict[str, Any]]) -> bool:
    return any(_iter_hypothesis_names(trace))


def _iter_hypothesis_names(trace: list[dict[str, Any]]) -> Iterable[str]:
    for event in trace:
        details = _details(event)
        hypotheses = details.get("hypotheses")
        if isinstance(hypotheses, list):
            for hypothesis in hypotheses:
                name = _hypothesis_name(hypothesis)
                if name:
                    yield name

        child = details.get("child")
        name = _hypothesis_name(child)
        if name:
            yield name

        pivot_hypothesis = details.get("hypothesis")
        if isinstance(pivot_hypothesis, dict):
            name = _hypothesis_name(pivot_hypothesis)
            if name:
                yield name

        for collection_key in ("children",):
            values = details.get(collection_key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value:
                        yield value


def _hypothesis_name(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    name = value.get("category_name")
    return name if isinstance(name, str) and name else None


def _target_neighborhood_embeddings(
    model: EmbeddingModel,
    target_word: str,
    n_neighbors: int,
) -> tuple[list[str], np.ndarray]:
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if not model.has_word(target_word):
        raise ValueError(f"Target word {target_word!r} is missing from {model.path}.")

    neighbors = model.nearest_neighbors(target_word, n=n_neighbors)
    words = [word for word, _ in neighbors]
    vectors = [model.get_vector(word) for word in words]
    valid_vectors = [vector for vector in vectors if vector is not None]
    if len(valid_vectors) < 2:
        raise ValueError(f"Need at least 2 neighborhood vectors for {target_word!r}.")
    return words, np.vstack(valid_vectors)


def _fit_projection(embeddings: np.ndarray, method: str, n_components: int) -> Any:
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        max_components = min(embeddings.shape[0], embeddings.shape[1])
        if n_components < 1:
            raise ValueError("n_components must be at least 1.")
        effective_components = min(n_components, max_components)
        return PCA(n_components=effective_components).fit(embeddings)
    if method == "umap":
        import umap

        if n_components != 2:
            raise ValueError("UMAP projection currently uses n_components=2.")
        return umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        ).fit(embeddings)
    if method == "pacmap":
        import pacmap

        if n_components != 2:
            raise ValueError("PaCMAP projection currently uses n_components=2.")
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42,
        ).fit(embeddings)
        reducer._contexto_basis = embeddings
        return reducer
    raise ValueError(f"Unknown projection method: {method}")


def _label_projection_axes(ax: Any, projection_model: Any, projection_name: str) -> None:
    if projection_name == "pca":
        explained = [float(value) for value in projection_model.explained_variance_ratio_]
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
        return
    if projection_name == "pacmap":
        ax.set_xlabel("PaCMAP1")
        ax.set_ylabel("PaCMAP2")
        return
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")


def _projection_diagnostic(projection_model: Any, projection_name: str) -> str:
    if projection_name == "pca":
        explained = [float(value) for value in projection_model.explained_variance_ratio_]
        return f"Explained variance used: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}"
    if projection_name == "pacmap":
        return "Projection used: PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=42)"
    return "Projection used: UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)"


def _project_unique_words(model: EmbeddingModel, words: Iterable[str], projection: Any) -> dict[str, np.ndarray]:
    unique_words: dict[str, str] = {}
    for word in words:
        if not isinstance(word, str):
            continue
        normalized = _normalize_word(word)
        if normalized:
            unique_words.setdefault(normalized, word)

    vectors: list[np.ndarray] = []
    keys: list[str] = []
    for key, word in unique_words.items():
        vector = model.get_vector(word)
        if vector is None:
            continue
        keys.append(key)
        vectors.append(vector)

    if not vectors:
        return {}
    points = _transform_projection(projection, np.vstack(vectors))
    return {key: points[index] for index, key in enumerate(keys)}


def _project_words_from_cache(words: list[str], projected_words: dict[str, np.ndarray]) -> dict[str, Any]:
    points: list[np.ndarray] = []
    included_words: list[str] = []
    skipped = 0
    for word in words:
        point = projected_words.get(_normalize_word(word))
        if point is None:
            skipped += 1
            continue
        points.append(point)
        included_words.append(word)
    return {
        "words": included_words,
        "points": np.vstack(points) if points else np.empty((0, 2)),
        "skipped": skipped,
    }


def _project_words(model: EmbeddingModel, words: list[str], projection: Any) -> dict[str, Any]:
    vectors: list[np.ndarray] = []
    projected_words: list[str] = []
    skipped = 0
    for word in words:
        vector = model.get_vector(word)
        if vector is None:
            skipped += 1
            continue
        vectors.append(vector)
        projected_words.append(word)
    points = _transform_projection(projection, np.vstack(vectors)) if vectors else np.empty((0, 2))
    return {"words": projected_words, "points": points, "skipped": skipped}


def _project_generation_best_from_cache(
    best_points: list[GenerationBest],
    projected_words: dict[str, np.ndarray],
) -> dict[str, Any]:
    points: list[np.ndarray] = []
    words: list[str] = []
    generations: list[int] = []
    ranks: list[int | None] = []
    skipped = 0
    for point in best_points:
        projected = projected_words.get(_normalize_word(point.word))
        if projected is None:
            skipped += 1
            continue
        points.append(projected)
        words.append(point.word)
        generations.append(point.generation)
        ranks.append(point.rank)
    return {
        "words": words,
        "points": np.vstack(points) if points else np.empty((0, 2)),
        "generations": generations,
        "ranks": ranks,
        "skipped": skipped,
    }


def _project_generation_best(
    model: EmbeddingModel,
    best_points: list[GenerationBest],
    projection: Any,
) -> dict[str, Any]:
    vectors: list[np.ndarray] = []
    words: list[str] = []
    generations: list[int] = []
    ranks: list[int | None] = []
    skipped = 0
    for point in best_points:
        vector = model.get_vector(point.word)
        if vector is None:
            skipped += 1
            continue
        vectors.append(vector)
        words.append(point.word)
        generations.append(point.generation)
        ranks.append(point.rank)
    points = _transform_projection(projection, np.vstack(vectors)) if vectors else np.empty((0, 2))
    return {
        "words": words,
        "points": points,
        "generations": generations,
        "ranks": ranks,
        "skipped": skipped,
    }


def _project_generation_centroids(centroids: list[GenerationCentroid], projection: Any) -> dict[str, Any]:
    if not centroids:
        return {"points": np.empty((0, 2)), "generations": []}
    points = _transform_projection(projection, np.vstack([centroid.vector for centroid in centroids]))
    return {
        "points": points,
        "generations": [centroid.generation for centroid in centroids],
    }


def _transform_projection(projection: Any, embeddings: np.ndarray) -> np.ndarray:
    basis = getattr(projection, "_contexto_basis", None)
    if basis is not None:
        return projection.transform(embeddings, basis=basis)
    return projection.transform(embeddings)


def _target_guess_coordinate_delta(
    target: str,
    target_xy: np.ndarray,
    guess_projection: dict[str, Any],
) -> float | None:
    matching_points = [
        point
        for word, point in zip(guess_projection["words"], guess_projection["points"])
        if _normalize_word(word) == _normalize_word(target)
    ]
    if not matching_points:
        return None
    return min(float(np.linalg.norm(target_xy - point)) for point in matching_points)


def _normalize_word(word: str) -> str:
    return word.lower().strip()


def _add_direction_arrows(ax: Any, points: np.ndarray, color: str, alpha: float) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        if np.allclose(delta, 0):
            continue
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "alpha": alpha,
                "lw": 1.0,
                "shrinkA": 6,
                "shrinkB": 6,
            },
        )


def _annotate_best_words(ax: Any, points: np.ndarray, words: list[str]) -> None:
    seen: dict[str, int] = {}
    for index, (point, word) in enumerate(zip(points, words)):
        seen[word] = seen.get(word, 0) + 1
        label = word if seen[word] == 1 else f"{word} ({seen[word]})"
        offset = (6, 6) if index % 2 == 0 else (6, -10)
        ax.annotate(
            label,
            (point[0], point[1]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
            color="black",
            zorder=6,
        )


def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def _default_single_output_path(trace_path: str | Path) -> Path:
    trace_name = Path(trace_path).stem
    return DEFAULT_FIGURE_DIR / f"single_game_{trace_name}.png"


def _default_multi_output_path(target: str, trace_count: int) -> Path:
    return DEFAULT_FIGURE_DIR / f"multi_run_{target}_{trace_count}runs.png"


def _default_rank_output_path(trace_path: str | Path) -> Path:
    return DEFAULT_FIGURE_DIR / f"{Path(trace_path).stem}_rank.png"


def _default_distance_output_path(trace_path: str | Path) -> Path:
    return DEFAULT_FIGURE_DIR / f"{Path(trace_path).stem}_distance.png"


def _print_variance_table(rows: list[dict[str, Any]]) -> None:
    table_rows = []
    for row in rows:
        if "error" in row:
            table_rows.append(
                [
                    Path(row["embedding_model_path"]).name,
                    "ERR",
                    "ERR",
                    "ERR",
                    "ERR",
                    row["error"],
                ]
            )
            continue
        cumulative = row["cumulative_ratio"]
        table_rows.append(
            [
                Path(row["embedding_model_path"]).name,
                row["n_neighbors_used"],
                row["n_components_used"],
                _ratio_at(cumulative, 2),
                _ratio_at(cumulative, 3),
                f"{row['total_variance']:.6g}",
            ]
        )

    print(f"Target: {rows[0]['target_word'] if rows else 'NA'}")
    print()
    _print_table(
        table_rows,
        ["embedding", "neighbors", "components", "cum_2", "cum_3", "total_variance_or_error"],
    )


def _ratio_at(cumulative: list[float], index: int) -> str:
    if len(cumulative) < index:
        return "NA"
    return f"{cumulative[index - 1]:.4f}"


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Contexto solver trajectory traces.")
    parser.add_argument(
        "--plot-type",
        choices=["single", "multi", "rank", "distance", "variance"],
        default="single",
        help="Type of plot or analysis to run. Defaults to single.",
    )
    parser.add_argument("--variance", action="store_true", help="Run PCA explained-variance check.")
    parser.add_argument("--target", default=config.DEFAULT_TARGET, help="Target word for variance checks.")
    parser.add_argument(
        "--embedding",
        nargs="+",
        default=[config.GAME_EMBEDDING_PATH],
        help="Embedding path(s). Defaults to config.GAME_EMBEDDING_PATH.",
    )
    parser.add_argument("--trace", help="Trace JSON for a single-game plot.")
    parser.add_argument("--traces", nargs="+", help="Trace JSONs for a multi-run plot.")
    parser.add_argument("--output", help="Output image path.")
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_NEIGHBORS)
    parser.add_argument("--n-components", type=int, default=None)
    parser.add_argument("--projection", choices=["pca", "umap", "pacmap"], default="pca")
    parser.add_argument(
        "--no-annotate-best",
        dest="annotate_best",
        action="store_false",
        help="Disable word labels on best-rank trajectory points.",
    )
    parser.set_defaults(annotate_best=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
