"""Command-line entry point for Contexto solvers."""

from __future__ import annotations

import argparse

from . import config
from .embeddings import EmbeddingModel
from .game_api import ContextoAPI
from .local_game import LocalGame
from .llm_client import LLMClient
from .logger import Logger
from .methods.ea_core import EALLMConfig
from .methods.ea_llm import EALLMMethod
from .methods.ea_llm_pivot import EALLMPivotConfig, EALLMPivotMethod
from .methods.ea_llm_self_adaptive import EALLMSelfAdaptiveConfig, EALLMSelfAdaptiveMethod
from .methods.embedding import EmbeddingConfig, EmbeddingMethod
from .methods.llm_only import LLMOnlyConfig, LLMOnlyMethod


def main() -> None:
    args = _parse_args()
    game_embedding_path = args.game_embedding_path or args.glove_path or config.GAME_EMBEDDING_PATH
    solver_embedding_path = args.solver_embedding_path or args.glove_path or config.SOLVER_EMBEDDING_PATH
    llm_provider = args.provider or config.LLM_PROVIDER
    llm_model = _model_for_provider(llm_provider, args.model, args.ollama_model)
    game_embedding_model = None
    solver_embedding_model = None

    if args.game == "local":
        game_embedding_model = EmbeddingModel(game_embedding_path)
    method_family = _method_family(args.method)

    if args.method == "embedding":
        if args.game == "local" and solver_embedding_path == game_embedding_path:
            solver_embedding_model = game_embedding_model
        else:
            solver_embedding_model = EmbeddingModel(solver_embedding_path)

    if args.game == "local":
        target = args.target or config.DEFAULT_TARGET
        game_number = None
        if game_embedding_model is None:
            raise RuntimeError("Embedding model is required for local games.")
        game = LocalGame(game_embedding_model, target)
        game_label = f"local_{target}"
    else:
        target = None
        game_number = args.game_number or config.GAME_NUMBER
        game = ContextoAPI(
            game_number=game_number,
            base_url=config.API_BASE_URL,
            rate_limit=config.API_RATE_LIMIT,
        )
        game_label = f"api_{game_number}"

    logger = Logger()
    logger.log(
        -1,
        "RUN_CONFIG",
        {
            "game": args.game,
            "solver": method_family,
            "method": args.method,
            "target": target,
            "game_number": game_number,
            "game_embedding_path": game_embedding_path if args.game == "local" else None,
            "solver_embedding_path": solver_embedding_path if args.method == "embedding" else None,
            "alignment": _alignment(args.game, args.method, game_embedding_path, solver_embedding_path),
            "max_generations": _default(args.max_generations, config.MAX_GENERATIONS),
            "llm_provider": llm_provider if method_family == "llm" else None,
            "llm_model": llm_model if method_family == "llm" else None,
            "llm_workers": _default(args.llm_workers, config.LLM_WORKERS) if args.method in _EA_METHODS else None,
            "local_search_rank_threshold": config.LOCAL_SEARCH_RANK_THRESHOLD if args.method in _EA_METHODS else None,
            "self_adaptive_mu": config.SELF_ADAPTIVE_MU if args.method == "ea_llm_self_adaptive" else None,
            "self_adaptive_concentration": (
                config.SELF_ADAPTIVE_CONCENTRATION if args.method == "ea_llm_self_adaptive" else None
            ),
            "self_adaptive_sigma_floor": config.SELF_ADAPTIVE_SIGMA_FLOOR if args.method == "ea_llm_self_adaptive" else None,
            "enable_pivot": _enable_pivot_metadata(args.method),
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
            "seed_count": _default(args.seed_count, config.EMBEDDING_SEED_COUNT) if args.method == "embedding" else None,
            "active_count": _default(args.active_count, config.EMBEDDING_ACTIVE_COUNT) if args.method == "embedding" else None,
            "neighbors_per_word": _default(args.neighbors_per_word, config.EMBEDDING_NEIGHBORS_PER_WORD) if args.method == "embedding" else None,
            "random_seed": _random_seed(args.random_seed),
        },
    )

    if method_family == "llm":
        llm_client = LLMClient(
            provider=llm_provider,
            api_key=args.api_key or _api_key_for_provider(llm_provider),
            model=llm_model,
        )
        solver = _build_llm_method(args.method, game, llm_client, logger, f"{args.method}_{game_label}", args)
    else:
        if solver_embedding_model is None:
            raise RuntimeError("Embedding model is required for embedding solver.")
        solver = EmbeddingMethod(
            game,
            solver_embedding_model,
            logger,
            EmbeddingConfig(
                max_generations=_default(args.max_generations, config.MAX_GENERATIONS),
                trace_dir=config.TRACE_DIR,
                run_label=f"embedding_{game_label}",
                seed_count=_default(args.seed_count, config.EMBEDDING_SEED_COUNT),
                active_count=_default(args.active_count, config.EMBEDDING_ACTIVE_COUNT),
                neighbors_per_word=_default(args.neighbors_per_word, config.EMBEDDING_NEIGHBORS_PER_WORD),
                random_seed=_random_seed(args.random_seed),
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
    parser.add_argument(
        "--method",
        choices=["llm_only", "ea_llm", "ea_llm_pivot", "ea_llm_self_adaptive", "embedding"],
        default="ea_llm_pivot",
        help="Solver method.",
    )
    parser.add_argument("--target", help="Target word for local game.")
    parser.add_argument("--game-number", type=int, help="Contexto game number to solve.")
    parser.add_argument("--max-generations", type=int, help="Maximum generations to run.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama"], help="LLM provider.")
    parser.add_argument("--model", help="LLM model name.")
    parser.add_argument("--ollama-model", help="Ollama model name. Defaults to OLLAMA_MODEL when --provider=ollama.")
    parser.add_argument("--api-key", help="LLM API key. Prefer using .env for local runs.")
    parser.add_argument("--glove-path", help="Path to a GloVe text embedding file.")
    parser.add_argument("--game-embedding-path", help="Embedding file used by the local game.")
    parser.add_argument("--solver-embedding-path", help="Embedding file used by the embedding solver.")
    parser.add_argument("--llm-workers", type=int, help="Number of parallel LLM generation calls.")
    parser.add_argument("--seed-count", type=int, help="Number of random seed words for embedding solver.")
    parser.add_argument("--active-count", type=int, help="Number of active words retained by embedding solver.")
    parser.add_argument("--neighbors-per-word", type=int, help="Nearest neighbors queried per active word.")
    parser.add_argument("--random-seed", type=int, help="Random seed for reproducible embedding solver runs.")
    return parser.parse_args()


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
            LLMOnlyConfig(
                max_generations=_default(args.max_generations, config.MAX_GENERATIONS),
                trace_dir=config.TRACE_DIR,
                run_label=run_label,
            ),
        )

    ea_kwargs = {
        "max_generations": _default(args.max_generations, config.MAX_GENERATIONS),
        "candidates_per_hypothesis": config.CANDIDATES_PER_HYPOTHESIS,
        "initial_categories": config.INITIAL_CATEGORIES,
        "starter_words_per_category": config.STARTER_WORDS_PER_CATEGORY,
        "mutations_per_generation": config.MUTATIONS_PER_GENERATION,
        "max_active_hypotheses": config.MAX_ACTIVE_HYPOTHESES,
        "trace_dir": config.TRACE_DIR,
        "run_label": run_label,
        "llm_workers": _default(args.llm_workers, config.LLM_WORKERS),
        "local_search_rank_threshold": config.LOCAL_SEARCH_RANK_THRESHOLD,
    }
    if method == "ea_llm":
        return EALLMMethod(game, llm_client, logger, EALLMConfig(**ea_kwargs))
    if method == "ea_llm_self_adaptive":
        return EALLMSelfAdaptiveMethod(
            game,
            llm_client,
            logger,
            EALLMSelfAdaptiveConfig(
                **ea_kwargs,
                mu=config.SELF_ADAPTIVE_MU,
                concentration=config.SELF_ADAPTIVE_CONCENTRATION,
                sigma_floor=config.SELF_ADAPTIVE_SIGMA_FLOOR,
                random_seed=_random_seed(args.random_seed),
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


def _alignment(game: str, method: str, game_embedding_path: str, solver_embedding_path: str) -> str:
    if game == "api":
        return "api_unknown"
    if method != "embedding":
        return "not_applicable"
    return "aligned" if game_embedding_path == solver_embedding_path else "non_aligned"


def _method_family(method: str) -> str:
    return "embedding" if method == "embedding" else "llm"


def _enable_pivot_metadata(method: str) -> bool | None:
    if method == "ea_llm_pivot":
        return True
    if method in {"ea_llm", "ea_llm_self_adaptive"}:
        return False
    return None


def _random_seed(cli_seed: int | None) -> int | None:
    if cli_seed is not None:
        return cli_seed
    if config.RANDOM_SEED in {None, ""}:
        return None
    return int(config.RANDOM_SEED)


def _default(value, default):
    return default if value is None else value


_EA_METHODS = {"ea_llm", "ea_llm_pivot", "ea_llm_self_adaptive"}


if __name__ == "__main__":
    main()

