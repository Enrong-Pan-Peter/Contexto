"""Shared method interfaces."""

from __future__ import annotations

from typing import Any, Protocol


class Game(Protocol):
    def guess(self, word: str) -> int: ...
    def total_guesses(self) -> int: ...
    def best_so_far(self) -> tuple[str | None, int | None]: ...
    def is_solved(self) -> bool: ...


class SolverMethod(Protocol):
    def solve(self, max_generations: int | None = None) -> dict[str, Any]: ...

