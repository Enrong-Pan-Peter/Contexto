"""Wrapper around the public Contexto game API."""

from __future__ import annotations

import time
from urllib.parse import quote

import requests


class ContextoAPI:
    def __init__(self, game_number: int, base_url: str, rate_limit: float = 0.5) -> None:
        self.game_number = game_number
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.guesses: dict[str, int] = {}
        self.invalid_guesses: set[str] = set()

    def guess(self, word: str) -> int:
        cleaned_word = word.lower().strip()
        if not cleaned_word:
            return -1
        if cleaned_word in self.guesses:
            return self.guesses[cleaned_word]
        if cleaned_word in self.invalid_guesses:
            return -1

        time.sleep(self.rate_limit)
        url = f"{self.base_url}/{self.game_number}/{quote(cleaned_word)}"
        try:
            response = requests.get(url, timeout=15)
        except requests.RequestException:
            self.invalid_guesses.add(cleaned_word)
            return -1

        if response.status_code >= 400:
            self.invalid_guesses.add(cleaned_word)
            return -1

        try:
            rank = int(response.json()["distance"])
        except (KeyError, TypeError, ValueError):
            self.invalid_guesses.add(cleaned_word)
            return -1

        # The public API returns 0 for the answer. The shared interface uses 1.
        normalized_rank = rank + 1
        self.guesses[cleaned_word] = normalized_rank
        return normalized_rank

    def total_guesses(self) -> int:
        return len(self.guesses)

    def best_so_far(self) -> tuple[str | None, int | None]:
        if not self.guesses:
            return None, None
        best_word = min(self.guesses, key=self.guesses.get)
        return best_word, self.guesses[best_word]

    def is_solved(self) -> bool:
        return any(rank == 1 for rank in self.guesses.values())
