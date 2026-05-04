"""Hypothesis model for the evolutionary Contexto solver."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hypothesis:
    """A semantic category being explored by the solver."""

    category_name: str
    description: str
    words_tried: dict[str, int] = field(default_factory=dict)
    status: str = "active"
    parent: str | None = None
    origin: str = "init"

    @property
    def best_rank(self) -> int:
        if not self.words_tried:
            return 1_000_000_000
        return min(self.words_tried.values())

    @property
    def best_word(self) -> str | None:
        if not self.words_tried:
            return None
        return min(self.words_tried, key=self.words_tried.get)

    def update(self, word: str, rank: int) -> None:
        self.words_tried[word.lower().strip()] = rank

    def to_dict(self) -> dict:
        return {
            "category_name": self.category_name,
            "description": self.description,
            "words_tried": dict(sorted(self.words_tried.items(), key=lambda item: item[1])),
            "best_word": self.best_word,
            "best_rank": self.best_rank,
            "status": self.status,
            "parent": self.parent,
            "origin": self.origin,
        }

