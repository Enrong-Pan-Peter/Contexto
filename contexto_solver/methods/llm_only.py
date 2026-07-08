"""Pure LLM Contexto method with no evolutionary operators."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm_client import LLMClient
from ..logger import Logger
from ..self_report import resolve_self_report, self_report_block
from .base import Game
from .ea_core import _word_family, _word_families


@dataclass
class LLMOnlyConfig:
    max_generations: int
    trace_dir: str
    run_label: str
    # RQ1 operator self-report instrumentation (logged-only; see self_report.py).
    self_report: bool = False


class LLMOnlyMethod:
    def __init__(self, game: Game, llm_client: LLMClient, logger: Logger, config: LLMOnlyConfig) -> None:
        self.game = game
        self.llm_client = llm_client
        self.logger = logger
        self.config = config
        self.invalid_guesses: set[str] = set()
        self.generation = 0

    def solve(self, max_generations: int | None = None) -> dict[str, Any]:
        generation_limit = max_generations or self.config.max_generations
        self.logger.log(
            self.generation,
            "INIT",
            {
                "history": self._valid_history(),
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )

        solved = self.game.is_solved()
        while not solved and self.generation < generation_limit:
            self.generation += 1
            word, self_report_record = self._next_clean_guess()
            if not word:
                self.logger.log(
                    self.generation,
                    "SKIP_INVALID_GUESS",
                    {"word": "", "reason": "llm returned no valid single word"},
                )
                continue
            rank = self.game.guess(word)
            if rank == -1:
                self.invalid_guesses.add(word)
                self.logger.log(self.generation, "SKIP_INVALID_GUESS", {"word": word})
                continue

            guess_details: dict[str, Any] = {
                "word": word,
                "rank": rank,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            }
            if self_report_record is not None:
                guess_details["self_report"] = self_report_record
            self.logger.log(self.generation, "GUESS", guess_details)
            solved = self.game.is_solved()
            if solved:
                self._log_solved()
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

    def _next_clean_guess(self) -> tuple[str, dict[str, Any] | None]:
        known_words = set(self.invalid_guesses)
        known_words.update(self._valid_history())
        known_word_families = _word_families(known_words)
        block = self_report_block(self.config.self_report)
        for _ in range(3):
            if self.config.self_report:
                raw_word, response, raw = self.llm_client.next_guess(
                    self._valid_history(),
                    self.invalid_guesses,
                    self_report_block=block,
                    return_raw=True,
                )
            else:
                raw_word, response, raw = (
                    self.llm_client.next_guess(self._valid_history(), self.invalid_guesses),
                    None,
                    None,
                )
            word = _clean_word(raw_word)
            if not word:
                continue
            if word in known_words or _word_family(word) in known_word_families:
                continue
            record = None
            if self.config.self_report:
                record = resolve_self_report(
                    self.llm_client,
                    source=response,
                    raw=raw,
                    context=self._self_report_context(),
                    proposed_word=word,
                    rendered_prompt=None,
                )
            return word, record
        return "", None

    def _self_report_context(self) -> str:
        known = set(self.invalid_guesses) | set(self._valid_history())
        return json.dumps(sorted(known))

    def _valid_history(self) -> dict[str, int]:
        guesses = getattr(self.game, "guesses", {})
        return {word: rank for word, rank in guesses.items() if rank > 0}

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


def _clean_word(word: Any) -> str:
    cleaned_word = str(word).lower().strip()
    if not re.fullmatch(r"[a-z]+", cleaned_word):
        return ""
    return cleaned_word

