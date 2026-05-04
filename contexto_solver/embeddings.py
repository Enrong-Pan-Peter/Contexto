"""Embedding loading and nearest-neighbor queries for the local game."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingModel:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.words: list[str] = []
        vectors: list[np.ndarray] = []

        print(f"Loading embeddings from {self.path}...")
        with self.path.open("r", encoding="utf-8") as embedding_file:
            for line_number, line in enumerate(embedding_file, start=1):
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                self.words.append(parts[0])
                vectors.append(np.asarray(parts[1:], dtype=np.float32))
                if line_number % 100_000 == 0:
                    print(f"Loaded {line_number:,} embeddings...")

        if not vectors:
            raise ValueError(f"No embeddings loaded from {self.path}")

        self.vectors = np.vstack(vectors)
        self.norms = np.linalg.norm(self.vectors, axis=1)
        self.word_to_index = {word: index for index, word in enumerate(self.words)}
        print(f"Loaded {len(self.words):,} embeddings.")

    def get_vector(self, word: str) -> np.ndarray | None:
        index = self.word_to_index.get(word.lower().strip())
        if index is None:
            return None
        return self.vectors[index]

    def has_word(self, word: str) -> bool:
        return word.lower().strip() in self.word_to_index

    def vocabulary(self) -> list[str]:
        return list(self.words)

    def nearest_neighbors(self, word: str, n: int = 10) -> list[tuple[str, float]]:
        vector = self.get_vector(word)
        if vector is None:
            return []
        return self.nearest_to_vector(vector, n=n, exclude={word.lower().strip()})

    def nearest_to_vector(self, vector: np.ndarray, n: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        target_norm = np.linalg.norm(vector)
        if target_norm == 0:
            return []

        similarities = self.vectors @ vector / (self.norms * target_norm)
        excluded = exclude or set()
        if excluded:
            for word in excluded:
                index = self.word_to_index.get(word)
                if index is not None:
                    similarities[index] = -np.inf

        count = min(n, len(self.words))
        candidate_indices = np.argpartition(-similarities, count - 1)[:count]
        sorted_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]
        return [(self.words[index], float(similarities[index])) for index in sorted_indices]
