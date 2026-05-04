"""Command-line entry point for Contexto solvers."""

from __future__ import annotations

import argparse

from . import config
from .embeddings import EmbeddingModel
from .game_api import ContextoAPI
from .local_game import LocalGame
from .llm_client import LLMClient
from .logger import Logger
from .solver_embedding import SolverEmbedding, SolverEmbeddingConfig
from .solver_llm import SolverLLM, SolverLLMConfig


def main() -> None:
    args = _parse_args()
    embedding_model = None
    if args.game == "local" or args.solver == "embedding":
        embedding_model = EmbeddingModel(args.glove_path or config.GLOVE_PATH)

    if args.game == "local":
        target = args.target or config.DEFAULT_TARGET
        if embedding_model is None:
            raise RuntimeError("Embedding model is required for local games.")
        game = LocalGame(embedding_model, target)
        game_label = f"local_{target}"
    else:
        game_number = args.game_number or config.GAME_NUMBER
        game = ContextoAPI(
            game_number=game_number,
            base_url=config.API_BASE_URL,
            rate_limit=config.API_RATE_LIMIT,
        )
        game_label = f"api_{game_number}"

    logger = Logger()

    if args.solver == "llm":
        provider = args.provider or config.LLM_PROVIDER
        llm_client = LLMClient(
            provider=provider,
            api_key=args.api_key or _api_key_for_provider(provider),
            model=args.model or config.LLM_MODEL,
        )
        solver = SolverLLM(
            game,
            llm_client,
            logger,
            SolverLLMConfig(
                max_generations=args.max_generations or config.MAX_GENERATIONS,
                candidates_per_hypothesis=config.CANDIDATES_PER_HYPOTHESIS,
                initial_categories=config.INITIAL_CATEGORIES,
                starter_words_per_category=config.STARTER_WORDS_PER_CATEGORY,
                mutations_per_generation=config.MUTATIONS_PER_GENERATION,
                trace_dir=config.TRACE_DIR,
                run_label=f"llm_{game_label}",
                llm_workers=args.llm_workers or config.LLM_WORKERS,
            ),
        )
    else:
        if embedding_model is None:
            raise RuntimeError("Embedding model is required for embedding solver.")
        solver = SolverEmbedding(
            game,
            embedding_model,
            logger,
            SolverEmbeddingConfig(
                max_generations=args.max_generations or config.MAX_GENERATIONS,
                trace_dir=config.TRACE_DIR,
                run_label=f"embedding_{game_label}",
            ),
        )

    result = solver.solve()
    status = "SOLVED" if result["solved"] else "NOT SOLVED"
    print(f"Status: {status}")
    print(f"Best word: {result['best_word']}")
    print(f"Best rank: {result['best_rank']}")
    print(f"Total guesses: {result['total_guesses']}")
    print(f"Generations: {result['generations']}")
    print(f"Trace: {result['trace_path']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve Contexto with evolutionary search.")
    parser.add_argument("--game", choices=["local", "api"], default="api", help="Game backend.")
    parser.add_argument("--solver", choices=["llm", "embedding"], default="llm", help="Solver strategy.")
    parser.add_argument("--target", help="Target word for local game.")
    parser.add_argument("--game-number", type=int, help="Contexto game number to solve.")
    parser.add_argument("--max-generations", type=int, help="Maximum generations to run.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="LLM provider.")
    parser.add_argument("--model", help="LLM model name.")
    parser.add_argument("--api-key", help="LLM API key. Prefer using .env for local runs.")
    parser.add_argument("--glove-path", help="Path to a GloVe text embedding file.")
    parser.add_argument("--llm-workers", type=int, help="Number of parallel LLM generation calls.")
    return parser.parse_args()


def _api_key_for_provider(provider: str) -> str:
    if provider == "anthropic":
        return config.ANTHROPIC_API_KEY
    return config.LLM_API_KEY or config.OPENAI_API_KEY


if __name__ == "__main__":
    main()

