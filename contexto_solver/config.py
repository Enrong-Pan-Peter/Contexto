"""Configuration helpers and defaults for the Contexto solver."""

from __future__ import annotations

import os
from pathlib import Path


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
GLOVE_PATH = os.getenv("GLOVE_PATH", "data/glove.6B.300d.txt")
TRACE_DIR = os.getenv("TRACE_DIR", "traces")

# Real API
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.contexto.me/machado/en/game")
API_RATE_LIMIT = float(os.getenv("API_RATE_LIMIT", "0.5"))
GAME_NUMBER = int(os.getenv("GAME_NUMBER", "1314"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY if LLM_PROVIDER == "openai" else ANTHROPIC_API_KEY)
LLM_WORKERS = int(os.getenv("LLM_WORKERS", "4"))

# Solver
MAX_GENERATIONS = int(os.getenv("MAX_GENERATIONS", "15"))
INITIAL_CATEGORIES = int(os.getenv("INITIAL_CATEGORIES", "6"))
CANDIDATES_PER_HYPOTHESIS = int(os.getenv("CANDIDATES_PER_HYPOTHESIS", "3"))
STARTER_WORDS_PER_CATEGORY = int(os.getenv("STARTER_WORDS_PER_CATEGORY", "3"))
MUTATIONS_PER_GENERATION = int(os.getenv("MUTATIONS_PER_GENERATION", "2"))

# Local game
DEFAULT_TARGET = os.getenv("DEFAULT_TARGET", "ivory")

