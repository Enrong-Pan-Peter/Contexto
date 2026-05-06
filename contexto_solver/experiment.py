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
from .solver_embedding import SolverEmbedding, SolverEmbeddingConfig
from .solver_llm import SolverLLM, SolverLLMConfig


def main() -> None:
    args = _parse_args()
    targets = _load_targets(args)
    if not targets:
        raise ValueError("Provide at least one target with --targets or --target-file.")

    game_embedding_path = args.game_embedding_path or args.glove_path or config.GAME_EMBEDDING_PATH
    solver_embedding_path = args.solver_embedding_path or args.glove_path or config.SOLVER_EMBEDDING_PATH
    if args.mode == "aligned" and args.solver == "embedding" and game_embedding_path != solver_embedding_path:
        raise ValueError("aligned embedding experiments require the same game and solver embedding path.")
    if args.mode == "non_aligned" and args.solver == "embedding" and game_embedding_path == solver_embedding_path:
        raise ValueError("non_aligned embedding experiments require different game and solver embedding paths.")

    game_embedding_model = EmbeddingModel(game_embedding_path)
    solver_embedding_model = None
    if args.solver == "embedding":
        solver_embedding_model = (
            game_embedding_model
            if solver_embedding_path == game_embedding_path
            else EmbeddingModel(solver_embedding_path)
        )

    output_path = Path(args.output or _default_output_path(args.solver, args.mode))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run_index in range(args.runs_per_target):
        for target in targets:
            rows.append(
                _run_local_target(
                    target=target,
                    run_index=run_index,
                    args=args,
                    game_embedding_model=game_embedding_model,
                    solver_embedding_model=solver_embedding_model,
                    game_embedding_path=game_embedding_path,
                    solver_embedding_path=solver_embedding_path,
                )
            )

    summary = {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "solver": args.solver,
            "mode": args.mode,
            "targets": targets,
            "runs_per_target": args.runs_per_target,
            "game_embedding_path": game_embedding_path,
            "solver_embedding_path": solver_embedding_path,
            "max_generations": args.max_generations,
            "random_seed": args.random_seed,
            "enable_pivot": config.ENABLE_PIVOT if args.solver == "llm" else None,
            "stall_no_improvement_generations": config.STALL_NO_IMPROVEMENT_GENERATIONS if args.solver == "llm" else None,
            "stall_close_rank_threshold": config.STALL_CLOSE_RANK_THRESHOLD if args.solver == "llm" else None,
            "stall_close_generations_limit": config.STALL_CLOSE_GENERATIONS_LIMIT if args.solver == "llm" else None,
            "max_pivot_attempts_per_run": config.MAX_PIVOT_ATTEMPTS_PER_RUN if args.solver == "llm" else None,
            "pivot_candidate_words_per_operator": config.PIVOT_CANDIDATE_WORDS_PER_OPERATOR if args.solver == "llm" else None,
            "pivot_resolution_window": config.PIVOT_RESOLUTION_WINDOW if args.solver == "llm" else None,
        },
        "aggregate": _aggregate(rows),
        "runs": rows,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(output_path.with_suffix(".csv"), rows)

    print(f"Wrote JSON summary: {output_path}")
    print(f"Wrote CSV summary: {output_path.with_suffix('.csv')}")
    print(json.dumps(summary["aggregate"], indent=2))


def _run_local_target(
    target: str,
    run_index: int,
    args: argparse.Namespace,
    game_embedding_model: EmbeddingModel,
    solver_embedding_model: EmbeddingModel | None,
    game_embedding_path: str,
    solver_embedding_path: str,
) -> dict[str, Any]:
    game = LocalGame(game_embedding_model, target)
    logger = Logger()
    alignment = "aligned" if game_embedding_path == solver_embedding_path else "non_aligned"
    run_label = f"{args.solver}_{alignment}_{target}_run{run_index + 1}"
    logger.log(
        -1,
        "RUN_CONFIG",
        {
            "game": "local",
            "solver": args.solver,
            "mode": args.mode,
            "target": target,
            "run_index": run_index,
            "game_embedding_path": game_embedding_path,
            "solver_embedding_path": solver_embedding_path,
            "alignment": alignment,
            "max_generations": args.max_generations,
            "random_seed": _run_seed(args.random_seed, run_index),
            "enable_pivot": config.ENABLE_PIVOT if args.solver == "llm" else None,
            "stall_no_improvement_generations": config.STALL_NO_IMPROVEMENT_GENERATIONS if args.solver == "llm" else None,
            "stall_close_rank_threshold": config.STALL_CLOSE_RANK_THRESHOLD if args.solver == "llm" else None,
            "stall_close_generations_limit": config.STALL_CLOSE_GENERATIONS_LIMIT if args.solver == "llm" else None,
            "max_pivot_attempts_per_run": config.MAX_PIVOT_ATTEMPTS_PER_RUN if args.solver == "llm" else None,
            "pivot_candidate_words_per_operator": config.PIVOT_CANDIDATE_WORDS_PER_OPERATOR if args.solver == "llm" else None,
            "pivot_resolution_window": config.PIVOT_RESOLUTION_WINDOW if args.solver == "llm" else None,
        },
    )

    if args.solver == "embedding":
        if solver_embedding_model is None:
            raise RuntimeError("Embedding solver requires a solver embedding model.")
        solver = SolverEmbedding(
            game,
            solver_embedding_model,
            logger,
            SolverEmbeddingConfig(
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
            provider=args.provider or config.LLM_PROVIDER,
            api_key=args.api_key or _api_key_for_provider(args.provider or config.LLM_PROVIDER),
            model=args.model or config.LLM_MODEL,
        )
        solver = SolverLLM(
            game,
            llm_client,
            logger,
            SolverLLMConfig(
                max_generations=args.max_generations,
                candidates_per_hypothesis=config.CANDIDATES_PER_HYPOTHESIS,
                initial_categories=config.INITIAL_CATEGORIES,
                starter_words_per_category=config.STARTER_WORDS_PER_CATEGORY,
                mutations_per_generation=config.MUTATIONS_PER_GENERATION,
                max_active_hypotheses=config.MAX_ACTIVE_HYPOTHESES,
                trace_dir=config.TRACE_DIR,
                run_label=run_label,
                llm_workers=args.llm_workers,
                local_search_rank_threshold=config.LOCAL_SEARCH_RANK_THRESHOLD,
                enable_pivot=config.ENABLE_PIVOT,
                stall_no_improvement_generations=config.STALL_NO_IMPROVEMENT_GENERATIONS,
                stall_close_rank_threshold=config.STALL_CLOSE_RANK_THRESHOLD,
                stall_close_generations_limit=config.STALL_CLOSE_GENERATIONS_LIMIT,
                max_pivot_attempts_per_run=config.MAX_PIVOT_ATTEMPTS_PER_RUN,
                pivot_candidate_words_per_operator=config.PIVOT_CANDIDATE_WORDS_PER_OPERATOR,
                pivot_resolution_window=config.PIVOT_RESOLUTION_WINDOW,
            ),
        )

    result = solver.solve()
    return {
        "solver": args.solver,
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
        "game_embedding_path": game_embedding_path,
        "solver_embedding_path": solver_embedding_path,
        "alignment": alignment,
    }


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


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    solved_rows = [row for row in rows if row["solved"]]
    best_ranks = [row["best_rank"] for row in rows if row["best_rank"] is not None]
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
            sum(row["generations"] for row in rows) / len(rows)
            if rows
            else None
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _default_output_path(solver: str, mode: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{config.TRACE_DIR}/experiment_{solver}_{mode}_{timestamp}.json"


def _run_seed(seed: int | None, run_index: int) -> int | None:
    if seed is None:
        return None
    return seed + run_index


def _api_key_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return config.ANTHROPIC_API_KEY
    return config.LLM_API_KEY or config.OPENAI_API_KEY


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Contexto solver experiments.")
    parser.add_argument("--targets", help="Comma-separated local target words.")
    parser.add_argument("--target-file", help="File containing one target word per line.")
    parser.add_argument("--mode", choices=["aligned", "non_aligned"], default="aligned")
    parser.add_argument("--solver", choices=["embedding", "llm"], default="embedding")
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
    parser.add_argument("--provider", choices=["openai", "anthropic"])
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    parser.add_argument("--output", help="Path to JSON summary output.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
