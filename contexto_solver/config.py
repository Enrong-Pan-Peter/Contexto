"""Configuration helpers and defaults for the Contexto solver."""

from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower().strip() in {"1", "true", "yes", "on"}


def _env_value(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


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
DEFAULT_LOCAL_EMBEDDING_PATH = (
    MINILM_EMBEDDING_PATH if Path(MINILM_EMBEDDING_PATH).exists() else GLOVE_PATH
)
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_REQUEST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"))
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
STALL_NO_IMPROVEMENT_GENERATIONS = int(os.getenv("STALL_NO_IMPROVEMENT_GENERATIONS", "3"))
STALL_CLOSE_RANK_THRESHOLD = int(os.getenv("STALL_CLOSE_RANK_THRESHOLD", "30"))
STALL_CLOSE_GENERATIONS_LIMIT = int(os.getenv("STALL_CLOSE_GENERATIONS_LIMIT", "5"))
MAX_PIVOT_ATTEMPTS_PER_RUN = int(os.getenv("MAX_PIVOT_ATTEMPTS_PER_RUN", "5"))
PIVOT_CANDIDATE_WORDS_PER_OPERATOR = int(os.getenv("PIVOT_CANDIDATE_WORDS_PER_OPERATOR", "10"))
PIVOT_RESOLUTION_WINDOW = int(os.getenv("PIVOT_RESOLUTION_WINDOW", "2"))
ENABLE_PIVOT = _env_bool("ENABLE_PIVOT", True)
EMBEDDING_SEED_COUNT = int(os.getenv("EMBEDDING_SEED_COUNT", "12"))
EMBEDDING_ACTIVE_COUNT = int(os.getenv("EMBEDDING_ACTIVE_COUNT", "5"))
EMBEDDING_NEIGHBORS_PER_WORD = int(os.getenv("EMBEDDING_NEIGHBORS_PER_WORD", "10"))
RANDOM_SEED = os.getenv("RANDOM_SEED")

# Local game
DEFAULT_TARGET = os.getenv("DEFAULT_TARGET", "ivory")

