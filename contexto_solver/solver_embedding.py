"""Evolutionary Contexto solver that uses embedding nearest neighbors."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .embeddings import EmbeddingModel
from .logger import Logger


class Game(Protocol):
    def guess(self, word: str) -> int: ...
    def total_guesses(self) -> int: ...
    def best_so_far(self) -> tuple[str | None, int | None]: ...
    def is_solved(self) -> bool: ...


@dataclass
class SolverEmbeddingConfig:
    max_generations: int
    trace_dir: str
    run_label: str
    seed_count: int = 12
    active_count: int = 5
    neighbors_per_word: int = 10
    random_seed: int | None = None


class SolverEmbedding:
    def __init__(self, game: Game, embedding_model: EmbeddingModel, logger: Logger, config: SolverEmbeddingConfig) -> None:
        self.game = game
        self.embedding_model = embedding_model
        self.logger = logger
        self.config = config
        self.active_words: list[str] = []
        self.generation = 0

    def initialize(self) -> bool:
        rng = random.Random(self.config.random_seed)
        seeds = rng.sample(self.embedding_model.vocabulary(), k=min(self.config.seed_count, len(self.embedding_model.words)))
        for word in seeds:
            self._guess(word, "INIT")
            if self.game.is_solved():
                self._log_solved()
                return True
        self._select_active()
        self.logger.log(
            self.generation,
            "INIT",
            {
                "active_words": self.active_words,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        return self.game.is_solved()

    def run_generation(self) -> bool:
        self.generation += 1
        candidates: list[str] = []
        guessed_words = set(getattr(self.game, "guesses", {}))

        for word in self.active_words:
            neighbors = self.embedding_model.nearest_neighbors(word, n=self.config.neighbors_per_word)
            candidates.extend(candidate for candidate, _ in neighbors if candidate not in guessed_words)

        best_word = self.best_word
        if best_word:
            wider_neighbors = self.embedding_model.nearest_neighbors(best_word, n=50)
            candidates.extend(candidate for candidate, _ in wider_neighbors[20:50] if candidate not in guessed_words)

        for word in dict.fromkeys(candidates):
            self._guess(word, "GUESS")
            if self.game.is_solved():
                self._log_solved()
                return True

        self._select_active()
        self.logger.log(
            self.generation,
            "SELECT",
            {
                "active_words": self.active_words,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        return False

    def solve(self, max_generations: int | None = None) -> dict[str, Any]:
        generation_limit = max_generations or self.config.max_generations
        solved = self.initialize()
        self._print_generation_summary()
        while not solved and self.generation < generation_limit:
            solved = self.run_generation()
            self._print_generation_summary()

        if not solved:
            self.logger.log(
                self.generation,
                "FAILED",
                {
                    "best_word": self.best_word,
                    "best_rank": self.best_rank,
                    "total_guesses": self.game.total_guesses(),
                },
            )

        trace_path = self._save_trace()
        return {
            "solved": solved,
            "answer": self.best_word if solved else None,
            "best_word": self.best_word,
            "best_rank": self.best_rank,
            "total_guesses": self.game.total_guesses(),
            "generations": self.generation,
            "trace_path": str(trace_path),
        }

    def _guess(self, word: str, event_type: str) -> int:
        rank = self.game.guess(word)
        if rank != -1:
            self.logger.log(
                self.generation,
                event_type,
                {
                    "word": word,
                    "rank": rank,
                    "best_word": self.best_word,
                    "best_rank": self.best_rank,
                    "total_guesses": self.game.total_guesses(),
                },
            )
        return rank

    def _select_active(self) -> None:
        guesses = getattr(self.game, "guesses", {})
        valid_guesses = [(word, rank) for word, rank in guesses.items() if rank > 0]
        valid_guesses.sort(key=lambda item: item[1])
        self.active_words = [word for word, _ in valid_guesses[: self.config.active_count]]

    def crossover_word(self, word_a: str, word_b: str) -> str | None:
        vector_a = self.embedding_model.get_vector(word_a)
        vector_b = self.embedding_model.get_vector(word_b)
        if vector_a is None or vector_b is None:
            return None
        neighbors = self.embedding_model.nearest_to_vector((vector_a + vector_b) / np.float32(2.0), n=1)
        return neighbors[0][0] if neighbors else None

    def _log_solved(self) -> None:
        self.logger.log(
            self.generation,
            "SOLVED",
            {
                "answer": self.best_word,
                "rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )

    def _save_trace(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.logger.save(Path(self.config.trace_dir) / f"{self.config.run_label}_{timestamp}.json")

    def _print_generation_summary(self) -> None:
        print(f"Generation {self.generation}: best word={self.best_word}, best rank={self.best_rank}")

    @property
    def best_word(self) -> str | None:
        return self.game.best_so_far()[0]

    @property
    def best_rank(self) -> int | None:
        return self.game.best_so_far()[1]
