"""Evolutionary search loop for solving Contexto puzzles."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .contexto_api import ContextoAPI
from .hypothesis import Hypothesis
from .llm_client import LLMClient
from .logger import Logger


@dataclass
class SolverConfig:
    max_generations: int
    candidates_per_hypothesis: int
    initial_categories: int
    starter_words_per_category: int
    mutations_per_generation: int
    trace_dir: str


@dataclass
class SolverResult:
    solved: bool
    best_word: str | None
    best_rank: int | None
    total_guesses: int
    trace_path: Path


class Solver:
    def __init__(
        self,
        contexto_api: ContextoAPI,
        llm_client: LLMClient,
        logger: Logger,
        config: SolverConfig,
    ) -> None:
        self.contexto_api = contexto_api
        self.llm_client = llm_client
        self.logger = logger
        self.config = config
        self.hypotheses: list[Hypothesis] = []
        self.all_guesses: dict[str, int] = {}
        self.invalid_guesses: set[str] = set()
        self.generation = 0

    def initialize(self) -> bool:
        categories = self.llm_client.generate_initial_categories(
            n=self.config.initial_categories,
            starter_words=self.config.starter_words_per_category,
        )

        for category in categories:
            hypothesis = self._hypothesis_from_category(category)
            self.hypotheses.append(hypothesis)

            for word in _words_from_category(category):
                self._guess_and_update(word, hypothesis)
                if self._is_solved():
                    self._log_solved()
                    return True

        self.logger.log(
            self.generation,
            "INIT",
            {
                "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
                "best_word": self.best_word,
                "best_rank": self.best_rank,
            },
        )
        self._log_progress("initialization complete")
        return self._is_solved()

    def run_generation(self) -> bool:
        self.generation += 1
        candidates = self._generate_candidates()
        self._evaluate_candidates(candidates)

        if self._is_solved():
            self._log_solved()
            return True

        self._select()
        self._mutate()

        if self._is_solved():
            self._log_solved()
            return True

        self._log_progress("generation complete")
        return False

    def solve(self, max_generations: int | None = None) -> SolverResult:
        generation_limit = max_generations or self.config.max_generations
        solved = self.initialize()

        while not solved and self.generation < generation_limit:
            solved = self.run_generation()

        if not solved:
            self.logger.log(
                self.generation,
                "FAILED",
                {
                    "best_word": self.best_word,
                    "best_rank": self.best_rank,
                    "total_guesses": len(self.all_guesses),
                },
            )

        trace_path = self._save_trace()
        return SolverResult(
            solved=solved,
            best_word=self.best_word,
            best_rank=self.best_rank,
            total_guesses=len(self.all_guesses),
            trace_path=trace_path,
        )

    def _generate_candidates(self) -> list[tuple[Hypothesis, str]]:
        candidates: list[tuple[Hypothesis, str]] = []
        planned_words = set(self.all_guesses) | self.invalid_guesses

        for hypothesis in self._active_hypotheses():
            words = self.llm_client.propose_words(
                hypothesis,
                self.all_guesses,
                invalid_guesses=self.invalid_guesses,
                n=self.config.candidates_per_hypothesis,
            )

            for word in words:
                cleaned_word = _clean_word(word)
                if not cleaned_word or cleaned_word in planned_words:
                    continue
                planned_words.add(cleaned_word)
                candidates.append((hypothesis, cleaned_word))

        self.logger.log(
            self.generation,
            "CANDIDATES",
            {
                "count": len(candidates),
                "candidates": [
                    {"hypothesis": hypothesis.category_name, "word": word}
                    for hypothesis, word in candidates
                ],
            },
        )
        return candidates

    def _evaluate_candidates(self, candidates: list[tuple[Hypothesis, str]]) -> None:
        for hypothesis, word in candidates:
            self._guess_and_update(word, hypothesis)

    def _guess_and_update(self, word: str, hypothesis: Hypothesis) -> int | None:
        previous_best_word = self.best_word
        previous_best_rank = self.best_rank
        cleaned_word = _clean_word(word)
        rank = self.contexto_api.guess(word)
        if rank is None:
            if cleaned_word:
                self.invalid_guesses.add(cleaned_word)
            #region agent log
            _agent_debug_log(
                "contexto_solver/solver.py:_guess_and_update",
                "skipping guess without valid rank",
                {
                    "word": word,
                    "invalidGuesses": len(self.invalid_guesses),
                    "hypothesis": hypothesis.category_name,
                    "generation": self.generation,
                },
                "H8,H9,H10",
            )
            #endregion
            self.logger.log(
                self.generation,
                "SKIP_INVALID_GUESS",
                {
                    "word": word,
                    "hypothesis": hypothesis.category_name,
                },
            )
            return None
        self.all_guesses[cleaned_word] = rank
        hypothesis.update(cleaned_word, rank)
        best_improved = previous_best_rank is None or rank < previous_best_rank

        self.logger.log(
            self.generation,
            "GUESS",
            {
                "word": cleaned_word,
                "rank": rank,
                "hypothesis": hypothesis.category_name,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
            },
        )
        if best_improved:
            #region agent log
            _agent_debug_log(
                "contexto_solver/solver.py:_guess_and_update",
                "best rank improved",
                {
                    "generation": self.generation,
                    "word": cleaned_word,
                    "rank": rank,
                    "hypothesis": hypothesis.category_name,
                    "previousBestWord": previous_best_word,
                    "previousBestRank": previous_best_rank,
                    "totalGuesses": len(self.all_guesses),
                },
                "H11,H12",
            )
            #endregion
        return rank

    def _select(self) -> None:
        ranked = sorted(self.hypotheses, key=lambda hypothesis: hypothesis.best_rank)
        keep_count = max(1, len(ranked) // 2)
        kept = set(id(hypothesis) for hypothesis in ranked[:keep_count])

        for hypothesis in self.hypotheses:
            hypothesis.status = "active" if id(hypothesis) in kept else "dormant"

        self.logger.log(
            self.generation,
            "SELECT",
            {
                "kept": [hypothesis.category_name for hypothesis in ranked[:keep_count]],
                "discarded": [hypothesis.category_name for hypothesis in ranked[keep_count:]],
                "best_word": self.best_word,
                "best_rank": self.best_rank,
            },
        )

    def _mutate(self) -> None:
        top_hypotheses = sorted(
            self._active_hypotheses(),
            key=lambda hypothesis: hypothesis.best_rank,
        )[:2]

        for parent in top_hypotheses:
            subcategories = self.llm_client.specialize(
                parent,
                self.all_guesses,
                invalid_guesses=self.invalid_guesses,
                n=self.config.mutations_per_generation,
            )

            for category in subcategories:
                child = self._hypothesis_from_category(category)
                self.hypotheses.append(child)

                starter_scores = {}
                for word in _words_from_category(category):
                    cleaned_word = _clean_word(word)
                    if not cleaned_word or cleaned_word in self.all_guesses or cleaned_word in self.invalid_guesses:
                        continue
                    rank = self._guess_and_update(cleaned_word, child)
                    if rank is None:
                        continue
                    starter_scores[cleaned_word] = rank

                self.logger.log(
                    self.generation,
                    "MUTATE",
                    {
                        "parent": parent.category_name,
                        "child": child.to_dict(),
                        "starter_scores": starter_scores,
                    },
                )

    def _active_hypotheses(self) -> list[Hypothesis]:
        return [hypothesis for hypothesis in self.hypotheses if hypothesis.status == "active"]

    def _log_progress(self, message: str) -> None:
        top_hypotheses = sorted(self.hypotheses, key=lambda hypothesis: hypothesis.best_rank)[:5]
        #region agent log
        _agent_debug_log(
            "contexto_solver/solver.py:_log_progress",
            message,
            {
                "generation": self.generation,
                "bestWord": self.best_word,
                "bestRank": self.best_rank,
                "totalGuesses": len(self.all_guesses),
                "activeHypotheses": len(self._active_hypotheses()),
                "totalHypotheses": len(self.hypotheses),
                "topHypotheses": [
                    {
                        "name": hypothesis.category_name,
                        "bestWord": hypothesis.best_word,
                        "bestRank": hypothesis.best_rank,
                        "status": hypothesis.status,
                    }
                    for hypothesis in top_hypotheses
                ],
            },
            "H11,H12",
        )
        #endregion

    def _hypothesis_from_category(self, category: dict[str, Any]) -> Hypothesis:
        return Hypothesis(
            category_name=str(category.get("name", "unnamed category")),
            description=str(category.get("description", "")),
        )

    def _is_solved(self) -> bool:
        return self.best_rank == 0

    def _log_solved(self) -> None:
        #region agent log
        _agent_debug_log(
            "contexto_solver/solver.py:_log_solved",
            "solver detected answer",
            {
                "generation": self.generation,
                "answer": self.best_word,
                "rank": self.best_rank,
                "totalGuesses": len(self.all_guesses),
            },
            "H14",
        )
        #endregion
        self.logger.log(
            self.generation,
            "SOLVED",
            {
                "answer": self.best_word,
                "rank": self.best_rank,
                "total_guesses": len(self.all_guesses),
            },
        )

    def _save_trace(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{self.contexto_api.game_number}_{timestamp}.json"
        return self.logger.save(Path(self.config.trace_dir) / filename)

    @property
    def best_word(self) -> str | None:
        if not self.all_guesses:
            return None
        return min(self.all_guesses, key=self.all_guesses.get)

    @property
    def best_rank(self) -> int | None:
        if not self.all_guesses:
            return None
        return self.all_guesses[self.best_word]


def _words_from_category(category: dict[str, Any]) -> list[str]:
    words = category.get("words", [])
    if not isinstance(words, list):
        return []
    return [_clean_word(word) for word in words if _clean_word(word)]


def _clean_word(word: Any) -> str:
    cleaned_word = str(word).lower().strip()
    if not re.fullmatch(r"[a-z]+", cleaned_word):
        #region agent log
        _agent_debug_log(
            "contexto_solver/solver.py:_clean_word",
            "rejected invalid contexto guess",
            {
                "word": cleaned_word,
                "hasWhitespace": any(char.isspace() for char in cleaned_word),
            },
            "H5",
        )
        #endregion
        return ""
    return cleaned_word


def _agent_debug_log(location: str, message: str, data: dict[str, object], hypothesis_id: str) -> None:
    try:
        payload = {
            "sessionId": "0eedb7",
            "runId": "post-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with Path("debug-0eedb7.log").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass

