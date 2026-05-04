"""Offline Contexto-style game backed by local embeddings."""

from __future__ import annotations

import numpy as np

from .embeddings import EmbeddingModel


class LocalGame:
    def __init__(self, embedding_model: EmbeddingModel, target_word: str) -> None:
        self.embedding_model = embedding_model
        self.target_word = target_word.lower().strip()
        target_vector = embedding_model.get_vector(self.target_word)
        if target_vector is None:
            raise ValueError(f"Target word '{self.target_word}' is not in the embedding vocabulary.")

        target_norm = np.linalg.norm(target_vector)
        similarities = embedding_model.vectors @ target_vector / (embedding_model.norms * target_norm)
        sorted_indices = np.argsort(-similarities)
        self.rankings = {
            embedding_model.words[index]: rank
            for rank, index in enumerate(sorted_indices, start=1)
        }
        self.guesses: dict[str, int] = {}

    def guess(self, word: str) -> int:
        cleaned_word = word.lower().strip()
        rank = self.rankings.get(cleaned_word, -1)
        if cleaned_word and cleaned_word not in self.guesses:
            self.guesses[cleaned_word] = rank
        return rank

    def total_guesses(self) -> int:
        return len(self.guesses)

    def best_so_far(self) -> tuple[str | None, int | None]:
        valid_guesses = {word: rank for word, rank in self.guesses.items() if rank > 0}
        if not valid_guesses:
            return None, None
        best_word = min(valid_guesses, key=valid_guesses.get)
        return best_word, valid_guesses[best_word]

    def is_solved(self) -> bool:
        return any(rank == 1 for rank in self.guesses.values())

    def get_target(self) -> str:
        return self.target_word
