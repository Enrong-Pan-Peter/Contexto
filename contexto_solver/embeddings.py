"""Embedding loading and nearest-neighbor queries for the local game."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingModel:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        print(f"Loading embeddings from {self.path}...")
        if self.path.suffix.lower() == ".npz":
            self.words, self.vectors, self.metadata = self._load_npz(self.path)
        else:
            self.words, self.vectors, self.metadata = self._load_text(self.path)

        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2:
            raise ValueError(f"Embedding matrix from {self.path} must be 2-dimensional.")
        if len(self.words) != self.vectors.shape[0]:
            raise ValueError(
                f"Word count ({len(self.words)}) does not match vector rows ({self.vectors.shape[0]})."
            )
        if not self.words:
            raise ValueError(f"No embeddings loaded from {self.path}")

        self.norms = np.linalg.norm(self.vectors, axis=1)
        self.word_to_index = {word: index for index, word in enumerate(self.words)}
        print(f"Loaded {len(self.words):,} embeddings with dimension {self.vectors.shape[1]}.")

    @staticmethod
    def _load_text(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
        words: list[str] = []
        vectors: list[np.ndarray] = []
        with path.open("r", encoding="utf-8") as embedding_file:
            for line_number, line in enumerate(embedding_file, start=1):
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    continue
                words.append(parts[0])
                vectors.append(np.asarray(parts[1:], dtype=np.float32))
                if line_number % 100_000 == 0:
                    print(f"Loaded {line_number:,} embeddings...")

        if not vectors:
            raise ValueError(f"No embeddings loaded from {path}")
        return words, np.vstack(vectors), {"format": "text", "path": str(path)}

    @staticmethod
    def _load_npz(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
        with np.load(path, allow_pickle=False) as data:
            if "words" not in data or "vectors" not in data:
                raise ValueError(f"{path} must contain 'words' and 'vectors' arrays.")
            words = [str(word) for word in data["words"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            metadata: dict[str, Any] = {"format": "npz", "path": str(path)}
            if "metadata_json" in data:
                metadata.update(json.loads(str(data["metadata_json"].item())))
        return words, vectors, metadata

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
