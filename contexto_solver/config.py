"""Configuration helpers and defaults for the Contexto solver."""

from __future__ import annotations

import os
from pathlib import Path


def _env_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs into the process environment if absent."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()

# Paths
GLOVE_PATH = _env_value("GLOVE_PATH", "data/glove.6B.300d.txt")
EMBEDDING_CACHE_DIR = _env_value("EMBEDDING_CACHE_DIR", "data/embeddings")
MINILM_EMBEDDING_PATH = _env_value(
    "MINILM_EMBEDDING_PATH",
    f"{EMBEDDING_CACHE_DIR}/all-MiniLM-L6-v2.npz",
)
MPNET_EMBEDDING_PATH = _env_value(
    "MPNET_EMBEDDING_PATH",
    f"{EMBEDDING_CACHE_DIR}/all-mpnet-base-v2.npz",
)
DEFAULT_LOCAL_EMBEDDING_PATH = GLOVE_PATH
GAME_EMBEDDING_PATH = _env_value("GAME_EMBEDDING_PATH", DEFAULT_LOCAL_EMBEDDING_PATH)
SOLVER_EMBEDDING_PATH = _env_value("SOLVER_EMBEDDING_PATH", GAME_EMBEDDING_PATH)
TRACE_DIR = _env_value("TRACE_DIR", "traces")

# Real API
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.contexto.me/machado/en/game")
API_RATE_LIMIT = float(os.getenv("API_RATE_LIMIT", "0.5"))
GAME_NUMBER = int(os.getenv("GAME_NUMBER", "1314"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
SUPPORTED_OLLAMA_MODELS = ("qwen3:14b", "qwen3:30b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_REQUEST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "900"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY if LLM_PROVIDER == "openai" else ANTHROPIC_API_KEY)
LLM_WORKERS = int(os.getenv("LLM_WORKERS", "4"))

# Solver
MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "50"))
INITIAL_CATEGORIES = int(os.getenv("INITIAL_CATEGORIES", "6"))
CANDIDATES_PER_HYPOTHESIS = int(os.getenv("CANDIDATES_PER_HYPOTHESIS", "3"))
STARTER_WORDS_PER_CATEGORY = int(os.getenv("STARTER_WORDS_PER_CATEGORY", "3"))
MUTATIONS_PER_GENERATION = int(os.getenv("MUTATIONS_PER_GENERATION", "2"))
MAX_ACTIVE_HYPOTHESES = int(os.getenv("MAX_ACTIVE_HYPOTHESES", "5"))
LOCAL_SEARCH_RANK_THRESHOLD = int(os.getenv("LOCAL_SEARCH_RANK_THRESHOLD", "100"))
SELF_ADAPTIVE_MU = int(os.getenv("SELF_ADAPTIVE_MU", "15"))
SELF_ADAPTIVE_INITIAL_CATEGORIES = int(os.getenv("SELF_ADAPTIVE_INITIAL_CATEGORIES", "15"))
SELF_ADAPTIVE_CONCENTRATION = float(os.getenv("SELF_ADAPTIVE_CONCENTRATION", "50.0"))
SELF_ADAPTIVE_SIGMA_FLOOR = float(os.getenv("SELF_ADAPTIVE_SIGMA_FLOOR", "0.02"))
EA_LLM_PIVOT_STALL_NO_IMPROVEMENT_GENERATIONS = int(
    os.getenv("EA_LLM_PIVOT_STALL_NO_IMPROVEMENT_GENERATIONS", "3")
)
EA_LLM_PIVOT_STALL_CLOSE_RANK_THRESHOLD = int(os.getenv("EA_LLM_PIVOT_STALL_CLOSE_RANK_THRESHOLD", "30"))
EA_LLM_PIVOT_STALL_CLOSE_GENERATIONS_LIMIT = int(os.getenv("EA_LLM_PIVOT_STALL_CLOSE_GENERATIONS_LIMIT", "5"))
EA_LLM_PIVOT_MAX_ATTEMPTS_PER_RUN = int(os.getenv("EA_LLM_PIVOT_MAX_ATTEMPTS_PER_RUN", "5"))
EA_LLM_PIVOT_CANDIDATE_WORDS_PER_OPERATOR = int(
    os.getenv("EA_LLM_PIVOT_CANDIDATE_WORDS_PER_OPERATOR", "10")
)
EA_LLM_PIVOT_RESOLUTION_WINDOW = int(os.getenv("EA_LLM_PIVOT_RESOLUTION_WINDOW", "2"))
EMBEDDING_SEED_COUNT = int(os.getenv("EMBEDDING_SEED_COUNT", "12"))
EMBEDDING_ACTIVE_COUNT = int(os.getenv("EMBEDDING_ACTIVE_COUNT", "5"))
EMBEDDING_NEIGHBORS_PER_WORD = int(os.getenv("EMBEDDING_NEIGHBORS_PER_WORD", "10"))
RANDOM_SEED = os.getenv("RANDOM_SEED")

# MAP-Elites
MAPELITES_GRID_RESOLUTION = int(os.getenv("MAPELITES_GRID_RESOLUTION", "5"))
MAPELITES_MUTATIONS_PER_GEN = int(os.getenv("MAPELITES_MUTATIONS_PER_GEN", "15"))
MAPELITES_CROSSOVERS_PER_GEN = int(os.getenv("MAPELITES_CROSSOVERS_PER_GEN", "5"))
MAPELITES_INITIAL_CATEGORIES = int(os.getenv("MAPELITES_INITIAL_CATEGORIES", "15"))
MAPELITES_PLACEMENT_CACHE_DIR = _env_value("MAPELITES_PLACEMENT_CACHE_DIR", "data/placement_cache")
# Anchored placement scales. Concreteness: 0 = most concrete/physical, 1 = most
# abstract/conceptual. Specificity: 0 = most general, 1 = most specific.
MAPELITES_ANCHORS_CONCRETENESS = {
    0.0: "rock",
    0.25: "rain",
    0.5: "music",
    0.75: "fear",
    1.0: "freedom",
}
MAPELITES_ANCHORS_SPECIFICITY = {
    0.0: "thing",
    0.25: "animal",
    0.5: "bird",
    0.75: "songbird",
    1.0: "sparrow",
}

# Sigma-mode control for the MAP-Elites operator probabilities. ``adaptive`` is
# the current behavior (Dirichlet perturbation of the parent sigma). The frozen
# and random modes ignore the parent sigma so the operator-firing distribution is
# held fixed or randomized, used by the sigma-control experiment.
MAPELITES_SIGMA_MODE = _env_value("MAPELITES_SIGMA_MODE", "adaptive")


def _parse_sigma_vector(name: str, default: tuple[float, float, float, float]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    parts = [piece for piece in raw.replace(",", " ").split() if piece]
    try:
        values = tuple(float(piece) for piece in parts)
    except ValueError:
        return default
    if len(values) != 4:
        return default
    return values


# Fixed operator profile used only when MAPELITES_SIGMA_MODE == "frozen_fixed".
# Order is [s, m, ml, l]; default favors small mutation.
MAPELITES_FROZEN_SIGMA = _parse_sigma_vector("MAPELITES_FROZEN_SIGMA", (0.4, 0.3, 0.2, 0.1))

# Number of best-ranked guessed words injected into mutation prompts as context.
# 0 disables the feature (current behavior); 20 enables it.
MAPELITES_RANKED_CONTEXT_K = int(os.getenv("MAPELITES_RANKED_CONTEXT_K", "0"))

# Local game
DEFAULT_TARGET = os.getenv("DEFAULT_TARGET", "ivory")

# RQ1 operator self-report instrumentation (logged-only; never feeds selection).
# When on, the operator/crossover prompts request predicted_closeness + rationale
# fields and those are parsed and written to the trace. When off, prompts render
# byte-identical to the pre-instrumentation prompts.
SELF_REPORT = _env_bool("SELF_REPORT", False)

# Trace schema version so instrumented traces are distinguishable from older ones.
# Bumped to 2 when the self-report instrumentation and richer run metadata landed.
TRACE_SCHEMA_VERSION = 2

