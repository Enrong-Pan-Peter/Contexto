"""Batch experiment runner for local Contexto solver comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config
from .embeddings import EmbeddingModel
from .llm_client import LLMClient
from .local_game import LocalGame
from .logger import Logger
from .methods.ea_core import EALLMConfig
from .methods.ea_llm import EALLMMethod
from .methods.ea_llm_map_elites import EALLMMapElitesConfig, EALLMMapElitesMethod
from .methods.ea_llm_pivot import EALLMPivotConfig, EALLMPivotMethod
from .methods.ea_llm_self_adaptive import EALLMSelfAdaptiveConfig, EALLMSelfAdaptiveMethod
from .methods.embedding import EmbeddingConfig, EmbeddingMethod
from .methods.llm_only import LLMOnlyConfig, LLMOnlyMethod


def main() -> None:
    args = _parse_args()
    targets = _load_targets(args)
    if not targets:
        raise ValueError("Provide at least one target with --targets or --target-file.")

    game_embedding_path = args.game_embedding_path or args.glove_path or config.GAME_EMBEDDING_PATH
    solver_embedding_path = args.solver_embedding_path or args.glove_path or config.SOLVER_EMBEDDING_PATH
    llm_provider = args.provider or config.LLM_PROVIDER
    llm_model = _model_for_provider(llm_provider, args.model, args.ollama_model)
    method_family = _method_family(args.method)
    if args.mode == "aligned" and args.method == "embedding" and game_embedding_path != solver_embedding_path:
        raise ValueError("aligned embedding experiments require the same game and solver embedding path.")
    if args.mode == "non_aligned" and args.method == "embedding" and game_embedding_path == solver_embedding_path:
        raise ValueError("non_aligned embedding experiments require different game and solver embedding paths.")

    game_embedding_model = EmbeddingModel(game_embedding_path)
    solver_embedding_model = None
    if args.method == "embedding":
        solver_embedding_model = (
            game_embedding_model
            if solver_embedding_path == game_embedding_path
            else EmbeddingModel(solver_embedding_path)
        )

    output_path = Path(args.output or _default_output_path(args.method, args.mode))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = _load_existing_rows(output_path) if args.resume else []
    completed = {(row["target"], row["run_index"]) for row in rows}
    for run_index in range(args.runs_per_target):
        for target in targets:
            if (target, run_index) in completed:
                continue
            try:
                rows.append(
                    _run_local_target(
                        target=target,
                        run_index=run_index,
                        args=args,
                        game_embedding_model=game_embedding_model,
                        solver_embedding_model=solver_embedding_model,
                        game_embedding_path=game_embedding_path,
                        solver_embedding_path=solver_embedding_path,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                    )
                )
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                print(f"Run failed for target={target} run_index={run_index}: {error_message}")
                rows.append(
                    _failed_run_row(
                        target=target,
                        run_index=run_index,
                        args=args,
                        game_embedding_path=game_embedding_path,
                        solver_embedding_path=solver_embedding_path,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        error=error_message,
                    )
                )
            _write_outputs(output_path, args, targets, game_embedding_path, solver_embedding_path, rows)

    _write_outputs(output_path, args, targets, game_embedding_path, solver_embedding_path, rows)

    print(f"Wrote JSON summary: {output_path}")
    print(f"Wrote CSV summary: {output_path.with_suffix('.csv')}")
    print(json.dumps(_aggregate(rows), indent=2))


def _write_outputs(
    output_path: Path,
    args: argparse.Namespace,
    targets: list[str],
    game_embedding_path: str,
    solver_embedding_path: str,
    rows: list[dict[str, Any]],
) -> None:
    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "solver": _method_family(args.method),
            "method": args.method,
            "mode": args.mode,
            "targets": targets,
            "runs_per_target": args.runs_per_target,
            "game_embedding_path": game_embedding_path,
            "solver_embedding_path": solver_embedding_path,
            "max_generations": args.max_generations,
            "llm_provider": (args.provider or config.LLM_PROVIDER) if _method_family(args.method) == "llm" else None,
            "llm_model": (
                _model_for_provider(args.provider or config.LLM_PROVIDER, args.model, args.ollama_model)
                if _method_family(args.method) == "llm"
                else None
            ),
            "random_seed": args.random_seed,
            "trace_schema_version": config.TRACE_SCHEMA_VERSION,
            "self_report": config.SELF_REPORT if args.method in _SELF_REPORT_METHODS else None,
            "enable_pivot": _enable_pivot_metadata(args.method),
            "initial_categories": _ea_initial_categories(args.method) if args.method in _EA_METHODS else None,
            "max_active_hypotheses": (
                config.MAX_ACTIVE_HYPOTHESES if args.method in {"ea_llm", "ea_llm_pivot"} else None
            ),
            "self_adaptive_initial_categories": (
                config.SELF_ADAPTIVE_INITIAL_CATEGORIES if args.method == "ea_llm_self_adaptive" else None
            ),
            "self_adaptive_mu": config.SELF_ADAPTIVE_MU if args.method == "ea_llm_self_adaptive" else None,
            "self_adaptive_concentration": (
                config.SELF_ADAPTIVE_CONCENTRATION if args.method == "ea_llm_self_adaptive" else None
            ),
            "self_adaptive_sigma_floor": config.SELF_ADAPTIVE_SIGMA_FLOOR if args.method == "ea_llm_self_adaptive" else None,
            "mapelites_grid_resolution": config.MAPELITES_GRID_RESOLUTION if args.method == "ea_llm_map_elites" else None,
            "mapelites_mutations_per_gen": config.MAPELITES_MUTATIONS_PER_GEN if args.method == "ea_llm_map_elites" else None,
            "mapelites_crossovers_per_gen": config.MAPELITES_CROSSOVERS_PER_GEN if args.method == "ea_llm_map_elites" else None,
            "mapelites_initial_categories": config.MAPELITES_INITIAL_CATEGORIES if args.method == "ea_llm_map_elites" else None,
            "mapelites_concentration": config.SELF_ADAPTIVE_CONCENTRATION if args.method == "ea_llm_map_elites" else None,
            "mapelites_sigma_floor": config.SELF_ADAPTIVE_SIGMA_FLOOR if args.method == "ea_llm_map_elites" else None,
            "mapelites_sigma_mode": config.MAPELITES_SIGMA_MODE if args.method == "ea_llm_map_elites" else None,
            "mapelites_frozen_sigma": list(config.MAPELITES_FROZEN_SIGMA) if args.method == "ea_llm_map_elites" else None,
            "mapelites_ranked_context_k": config.MAPELITES_RANKED_CONTEXT_K if args.method == "ea_llm_map_elites" else None,
            "ea_llm_pivot_stall_no_improvement_generations": (
                config.EA_LLM_PIVOT_STALL_NO_IMPROVEMENT_GENERATIONS if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_stall_close_rank_threshold": (
                config.EA_LLM_PIVOT_STALL_CLOSE_RANK_THRESHOLD if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_stall_close_generations_limit": (
                config.EA_LLM_PIVOT_STALL_CLOSE_GENERATIONS_LIMIT if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_max_attempts_per_run": (
                config.EA_LLM_PIVOT_MAX_ATTEMPTS_PER_RUN if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_candidate_words_per_operator": (
                config.EA_LLM_PIVOT_CANDIDATE_WORDS_PER_OPERATOR if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_resolution_window": config.EA_LLM_PIVOT_RESOLUTION_WINDOW if args.method == "ea_llm_pivot" else None,
        },
        "aggregate": _aggregate(rows),
        "runs": rows,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output_path.with_suffix(".csv"), rows)


def _run_local_target(
    target: str,
    run_index: int,
    args: argparse.Namespace,
    game_embedding_model: EmbeddingModel,
    solver_embedding_model: EmbeddingModel | None,
    game_embedding_path: str,
    solver_embedding_path: str,
    llm_provider: str,
    llm_model: str,
) -> dict[str, Any]:
    game = LocalGame(game_embedding_model, target)
    logger = Logger()
    alignment = "aligned" if game_embedding_path == solver_embedding_path else "non_aligned"
    run_label = f"{args.method}_{alignment}_{target}_run{run_index + 1}"
    method_family = _method_family(args.method)
    logger.log(
        -1,
        "RUN_CONFIG",
        {
            "game": "local",
            "solver": method_family,
            "method": args.method,
            "mode": args.mode,
            "target": target,
            "run_index": run_index,
            "game_embedding_path": game_embedding_path,
            "solver_embedding_path": solver_embedding_path,
            "embedding_backend": Path(game_embedding_path).stem,
            "vocabulary_size": len(game_embedding_model.words),
            "alignment": alignment,
            "trace_schema_version": config.TRACE_SCHEMA_VERSION,
            "self_report": config.SELF_REPORT if args.method in _SELF_REPORT_METHODS else None,
            "max_generations": args.max_generations,
            "llm_provider": llm_provider if method_family == "llm" else None,
            "llm_model": llm_model if method_family == "llm" else None,
            "random_seed": _run_seed(args.random_seed, run_index),
            "enable_pivot": _enable_pivot_metadata(args.method),
            "initial_categories": _ea_initial_categories(args.method) if args.method in _EA_METHODS else None,
            "max_active_hypotheses": (
                config.MAX_ACTIVE_HYPOTHESES if args.method in {"ea_llm", "ea_llm_pivot"} else None
            ),
            "self_adaptive_initial_categories": (
                config.SELF_ADAPTIVE_INITIAL_CATEGORIES if args.method == "ea_llm_self_adaptive" else None
            ),
            "self_adaptive_mu": config.SELF_ADAPTIVE_MU if args.method == "ea_llm_self_adaptive" else None,
            "self_adaptive_concentration": (
                config.SELF_ADAPTIVE_CONCENTRATION if args.method == "ea_llm_self_adaptive" else None
            ),
            "self_adaptive_sigma_floor": config.SELF_ADAPTIVE_SIGMA_FLOOR if args.method == "ea_llm_self_adaptive" else None,
            "mapelites_grid_resolution": config.MAPELITES_GRID_RESOLUTION if args.method == "ea_llm_map_elites" else None,
            "mapelites_mutations_per_gen": config.MAPELITES_MUTATIONS_PER_GEN if args.method == "ea_llm_map_elites" else None,
            "mapelites_crossovers_per_gen": config.MAPELITES_CROSSOVERS_PER_GEN if args.method == "ea_llm_map_elites" else None,
            "mapelites_initial_categories": config.MAPELITES_INITIAL_CATEGORIES if args.method == "ea_llm_map_elites" else None,
            "mapelites_concentration": config.SELF_ADAPTIVE_CONCENTRATION if args.method == "ea_llm_map_elites" else None,
            "mapelites_sigma_floor": config.SELF_ADAPTIVE_SIGMA_FLOOR if args.method == "ea_llm_map_elites" else None,
            "mapelites_sigma_mode": config.MAPELITES_SIGMA_MODE if args.method == "ea_llm_map_elites" else None,
            "mapelites_frozen_sigma": list(config.MAPELITES_FROZEN_SIGMA) if args.method == "ea_llm_map_elites" else None,
            "mapelites_ranked_context_k": config.MAPELITES_RANKED_CONTEXT_K if args.method == "ea_llm_map_elites" else None,
            "ea_llm_pivot_stall_no_improvement_generations": (
                config.EA_LLM_PIVOT_STALL_NO_IMPROVEMENT_GENERATIONS if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_stall_close_rank_threshold": (
                config.EA_LLM_PIVOT_STALL_CLOSE_RANK_THRESHOLD if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_stall_close_generations_limit": (
                config.EA_LLM_PIVOT_STALL_CLOSE_GENERATIONS_LIMIT if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_max_attempts_per_run": (
                config.EA_LLM_PIVOT_MAX_ATTEMPTS_PER_RUN if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_candidate_words_per_operator": (
                config.EA_LLM_PIVOT_CANDIDATE_WORDS_PER_OPERATOR if args.method == "ea_llm_pivot" else None
            ),
            "ea_llm_pivot_resolution_window": config.EA_LLM_PIVOT_RESOLUTION_WINDOW if args.method == "ea_llm_pivot" else None,
        },
    )

    if args.method == "embedding":
        if solver_embedding_model is None:
            raise RuntimeError("Embedding solver requires a solver embedding model.")
        solver = EmbeddingMethod(
            game,
            solver_embedding_model,
            logger,
            EmbeddingConfig(
                max_generations=args.max_generations,
                trace_dir=config.TRACE_DIR,
                run_label=run_label,
                seed_count=args.seed_count,
                active_count=args.active_count,
                neighbors_per_word=args.neighbors_per_word,
                random_seed=_run_seed(args.random_seed, run_index),
            ),
        )
    else:
        llm_client = LLMClient(
            provider=llm_provider,
            api_key=args.api_key or _api_key_for_provider(llm_provider),
            model=llm_model,
        )
        solver = _build_llm_method(args.method, game, llm_client, logger, run_label, args)

    result = solver.solve()
    archive = getattr(solver, "archive", None)
    final_sigma = _archive_sigma_means(archive)
    is_mapelites = args.method == "ea_llm_map_elites"
    return {
        "solver": method_family,
        "method": args.method,
        "mode": args.mode,
        "target": target,
        "run_index": run_index,
        "solved": result["solved"],
        "answer": result["answer"],
        "best_word": result["best_word"],
        "best_rank": result["best_rank"],
        "total_guesses": result["total_guesses"],
        "generations": result["generations"],
        "trace_path": result["trace_path"],
        "archive_occupancy": len(archive) if archive is not None else None,
        "placement_cache_hit_rate": getattr(solver, "placement_cache_hit_rate", None),
        "self_report": config.SELF_REPORT if args.method in _SELF_REPORT_METHODS else None,
        "mapelites_sigma_mode": config.MAPELITES_SIGMA_MODE if is_mapelites else None,
        "mapelites_ranked_context_k": config.MAPELITES_RANKED_CONTEXT_K if is_mapelites else None,
        "final_archive_sigma_s": final_sigma[0],
        "final_archive_sigma_m": final_sigma[1],
        "final_archive_sigma_ml": final_sigma[2],
        "final_archive_sigma_l": final_sigma[3],
        "error": None,
        "llm_provider": llm_provider if method_family == "llm" else None,
        "llm_model": llm_model if method_family == "llm" else None,
        "game_embedding_path": game_embedding_path,
        "solver_embedding_path": solver_embedding_path,
        "alignment": alignment,
    }


def _failed_run_row(
    target: str,
    run_index: int,
    args: argparse.Namespace,
    game_embedding_path: str,
    solver_embedding_path: str,
    llm_provider: str,
    llm_model: str,
    error: str,
) -> dict[str, Any]:
    alignment = "aligned" if game_embedding_path == solver_embedding_path else "non_aligned"
    return {
        "solver": _method_family(args.method),
        "method": args.method,
        "mode": args.mode,
        "target": target,
        "run_index": run_index,
        "solved": False,
        "answer": target,
        "best_word": None,
        "best_rank": None,
        "total_guesses": None,
        "generations": None,
        "trace_path": None,
        "archive_occupancy": None,
        "placement_cache_hit_rate": None,
        "self_report": config.SELF_REPORT if args.method in _SELF_REPORT_METHODS else None,
        "mapelites_sigma_mode": config.MAPELITES_SIGMA_MODE if args.method == "ea_llm_map_elites" else None,
        "mapelites_ranked_context_k": config.MAPELITES_RANKED_CONTEXT_K if args.method == "ea_llm_map_elites" else None,
        "final_archive_sigma_s": None,
        "final_archive_sigma_m": None,
        "final_archive_sigma_ml": None,
        "final_archive_sigma_l": None,
        "error": error,
        "llm_provider": llm_provider if _method_family(args.method) == "llm" else None,
        "llm_model": llm_model if _method_family(args.method) == "llm" else None,
        "game_embedding_path": game_embedding_path,
        "solver_embedding_path": solver_embedding_path,
        "alignment": alignment,
    }


def _archive_sigma_means(archive: Any) -> list[float | None]:
    """Mean per-operator sigma over the final archive incumbents (or Nones)."""
    if not archive:
        return [None, None, None, None]
    incumbents = list(archive.values())
    sums = [0.0, 0.0, 0.0, 0.0]
    for hypothesis in incumbents:
        sigma = list(hypothesis.sigma)
        for index in range(4):
            sums[index] += float(sigma[index])
    count = len(incumbents)
    return [total / count for total in sums]


def _load_targets(args: argparse.Namespace) -> list[str]:
    targets: list[str] = []
    if args.targets:
        targets.extend(target.strip().lower() for target in args.targets.split(",") if target.strip())
    if args.target_file:
        target_path = Path(args.target_file)
        targets.extend(
            line.strip().lower()
            for line in target_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    return list(dict.fromkeys(targets))


def _load_existing_rows(output_path: Path) -> list[dict[str, Any]]:
    if not output_path.exists():
        return []
    data = json.loads(output_path.read_text(encoding="utf-8"))
    rows = data.get("runs", [])
    if not isinstance(rows, list):
        raise ValueError(f"Cannot resume from {output_path}: runs is not a list.")
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved_rows = [row for row in rows if row["solved"]]
    best_ranks = [row["best_rank"] for row in rows if row["best_rank"] is not None]
    generations = [row["generations"] for row in rows if row["generations"] is not None]
    return {
        "total_runs": len(rows),
        "solved_runs": len(solved_rows),
        "solve_rate": len(solved_rows) / len(rows) if rows else 0.0,
        "average_guesses_solved": (
            sum(row["total_guesses"] for row in solved_rows) / len(solved_rows)
            if solved_rows
            else None
        ),
        "average_best_rank": sum(best_ranks) / len(best_ranks) if best_ranks else None,
        "average_generations": (
            sum(generations) / len(generations)
            if generations
            else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "solver",
        "method",
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
        "archive_occupancy",
        "placement_cache_hit_rate",
        "self_report",
        "mapelites_sigma_mode",
        "mapelites_ranked_context_k",
        "final_archive_sigma_s",
        "final_archive_sigma_m",
        "final_archive_sigma_ml",
        "final_archive_sigma_l",
        "error",
        "llm_provider",
        "llm_model",
        "alignment",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _default_output_path(method: str, mode: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.TRACE_DIR}/experiment_{method}_{mode}_{timestamp}.json"


def _run_seed(seed: int | None, run_index: int) -> int | None:
    if seed is None:
        return None
    return seed + run_index


def _api_key_for_provider(provider: str) -> str:
    if provider == "ollama":
        return "ollama"
    if provider == "anthropic":
        return config.ANTHROPIC_API_KEY
    return config.LLM_API_KEY or config.OPENAI_API_KEY


def _model_for_provider(provider: str, cli_model: str | None, cli_ollama_model: str | None) -> str:
    if provider == "ollama":
        return cli_ollama_model or cli_model or config.OLLAMA_MODEL
    return cli_model or config.LLM_MODEL


def _build_llm_method(method: str, game, llm_client: LLMClient, logger: Logger, run_label: str, args: argparse.Namespace):
    if method == "llm_only":
        return LLMOnlyMethod(
            game,
            llm_client,
            logger,
            LLMOnlyConfig(max_generations=args.max_generations, trace_dir=config.TRACE_DIR, run_label=run_label),
        )

    ea_kwargs = {
        "max_generations": args.max_generations,
        "candidates_per_hypothesis": config.CANDIDATES_PER_HYPOTHESIS,
        "initial_categories": config.INITIAL_CATEGORIES,
        "starter_words_per_category": config.STARTER_WORDS_PER_CATEGORY,
        "mutations_per_generation": config.MUTATIONS_PER_GENERATION,
        "max_active_hypotheses": config.MAX_ACTIVE_HYPOTHESES,
        "trace_dir": config.TRACE_DIR,
        "run_label": run_label,
        "llm_workers": args.llm_workers,
        "local_search_rank_threshold": config.LOCAL_SEARCH_RANK_THRESHOLD,
        "self_report": config.SELF_REPORT,
    }
    if method == "ea_llm":
        return EALLMMethod(game, llm_client, logger, EALLMConfig(**ea_kwargs))
    if method == "ea_llm_self_adaptive":
        return EALLMSelfAdaptiveMethod(
            game,
            llm_client,
            logger,
            EALLMSelfAdaptiveConfig(
                **{**ea_kwargs, "initial_categories": config.SELF_ADAPTIVE_INITIAL_CATEGORIES},
                mu=config.SELF_ADAPTIVE_MU,
                concentration=config.SELF_ADAPTIVE_CONCENTRATION,
                sigma_floor=config.SELF_ADAPTIVE_SIGMA_FLOOR,
                random_seed=args.random_seed,
            ),
        )
    if method == "ea_llm_map_elites":
        return EALLMMapElitesMethod(
            game,
            llm_client,
            logger,
            EALLMMapElitesConfig(
                **{**ea_kwargs, "initial_categories": config.MAPELITES_INITIAL_CATEGORIES},
                mu=config.SELF_ADAPTIVE_MU,
                concentration=config.SELF_ADAPTIVE_CONCENTRATION,
                sigma_floor=config.SELF_ADAPTIVE_SIGMA_FLOOR,
                random_seed=args.random_seed,
                grid_resolution=config.MAPELITES_GRID_RESOLUTION,
                mutations_per_gen=config.MAPELITES_MUTATIONS_PER_GEN,
                crossovers_per_gen=config.MAPELITES_CROSSOVERS_PER_GEN,
                placement_cache_dir=config.MAPELITES_PLACEMENT_CACHE_DIR,
                anchors_concreteness=config.MAPELITES_ANCHORS_CONCRETENESS,
                anchors_specificity=config.MAPELITES_ANCHORS_SPECIFICITY,
                sigma_mode=config.MAPELITES_SIGMA_MODE,
                frozen_sigma=config.MAPELITES_FROZEN_SIGMA,
                ranked_context_k=config.MAPELITES_RANKED_CONTEXT_K,
            ),
        )
    if method == "ea_llm_pivot":
        return EALLMPivotMethod(
            game,
            llm_client,
            logger,
            EALLMPivotConfig(
                **ea_kwargs,
                stall_no_improvement_generations=config.EA_LLM_PIVOT_STALL_NO_IMPROVEMENT_GENERATIONS,
                stall_close_rank_threshold=config.EA_LLM_PIVOT_STALL_CLOSE_RANK_THRESHOLD,
                stall_close_generations_limit=config.EA_LLM_PIVOT_STALL_CLOSE_GENERATIONS_LIMIT,
                max_pivot_attempts_per_run=config.EA_LLM_PIVOT_MAX_ATTEMPTS_PER_RUN,
                pivot_candidate_words_per_operator=config.EA_LLM_PIVOT_CANDIDATE_WORDS_PER_OPERATOR,
                pivot_resolution_window=config.EA_LLM_PIVOT_RESOLUTION_WINDOW,
            ),
        )
    raise ValueError(f"Unknown LLM method: {method}")


def _method_family(method: str) -> str:
    return "embedding" if method == "embedding" else "llm"


def _enable_pivot_metadata(method: str) -> bool | None:
    if method == "ea_llm_pivot":
        return True
    if method in {"ea_llm", "ea_llm_self_adaptive", "ea_llm_map_elites"}:
        return False
    return None


def _ea_initial_categories(method: str) -> int:
    if method == "ea_llm_self_adaptive":
        return config.SELF_ADAPTIVE_INITIAL_CATEGORIES
    if method == "ea_llm_map_elites":
        return config.MAPELITES_INITIAL_CATEGORIES
    return config.INITIAL_CATEGORIES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Contexto solver experiments.")
    parser.add_argument("--targets", help="Comma-separated local target words.")
    parser.add_argument("--target-file", help="File containing one target word per line.")
    parser.add_argument("--mode", choices=["aligned", "non_aligned"], default="aligned")
    parser.add_argument(
        "--method",
        choices=["llm_only", "ea_llm", "ea_llm_pivot", "ea_llm_self_adaptive", "ea_llm_map_elites", "embedding"],
        default="embedding",
    )
    parser.add_argument("--glove-path", help="Shortcut path used for both game and solver embeddings.")
    parser.add_argument("--game-embedding-path", help="Embedding file used by LocalGame.")
    parser.add_argument("--solver-embedding-path", help="Embedding file used by the embedding solver.")
    parser.add_argument("--max-generations", type=int, default=config.MAX_GENERATIONS)
    parser.add_argument("--runs-per-target", type=int, default=1)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--seed-count", type=int, default=config.EMBEDDING_SEED_COUNT)
    parser.add_argument("--active-count", type=int, default=config.EMBEDDING_ACTIVE_COUNT)
    parser.add_argument("--neighbors-per-word", type=int, default=config.EMBEDDING_NEIGHBORS_PER_WORD)
    parser.add_argument("--llm-workers", type=int, default=config.LLM_WORKERS)
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model")
    parser.add_argument(
        "--ollama-model",
        help=(
            "Ollama model name. Defaults to OLLAMA_MODEL when --provider=ollama. "
            f"Supported local models: {', '.join(config.SUPPORTED_OLLAMA_MODELS)}."
        ),
    )
    parser.add_argument("--api-key")
    parser.add_argument("--output", help="Path to JSON summary output.")
    parser.add_argument("--resume", action="store_true", help="Skip runs already present in the output JSON.")
    return parser.parse_args()


_EA_METHODS = {"ea_llm", "ea_llm_pivot", "ea_llm_self_adaptive", "ea_llm_map_elites"}
_SELF_REPORT_METHODS = {"ea_llm_self_adaptive", "ea_llm_map_elites"}


if __name__ == "__main__":
    main()
