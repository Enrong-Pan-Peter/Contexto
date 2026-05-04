"""Evolutionary Contexto solver that uses an LLM for word generation."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .hypothesis import Hypothesis
from .llm_client import LLMClient
from .logger import Logger


class Game(Protocol):
    def guess(self, word: str) -> int: ...
    def total_guesses(self) -> int: ...
    def best_so_far(self) -> tuple[str | None, int | None]: ...
    def is_solved(self) -> bool: ...


@dataclass
class SolverLLMConfig:
    max_generations: int
    candidates_per_hypothesis: int
    initial_categories: int
    starter_words_per_category: int
    mutations_per_generation: int
    trace_dir: str
    run_label: str
    llm_workers: int = 4


class SolverLLM:
    def __init__(self, game: Game, llm_client: LLMClient, logger: Logger, config: SolverLLMConfig) -> None:
        self.game = game
        self.llm_client = llm_client
        self.logger = logger
        self.config = config
        self.hypotheses: list[Hypothesis] = []
        self.invalid_guesses: set[str] = set()
        self.generation = 0

    def initialize(self) -> bool:
        categories = self.llm_client.generate_initial_categories(
            n=self.config.initial_categories,
            starter_words=self.config.starter_words_per_category,
        )
        for category in categories:
            hypothesis = self._hypothesis_from_category(category, origin="init")
            self.hypotheses.append(hypothesis)
            for word in _words_from_category(category):
                self._guess_and_update(word, hypothesis)
                if self.game.is_solved():
                    self._log_solved()
                    return True

        self.logger.log(
            self.generation,
            "INIT",
            {
                "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        return self.game.is_solved()

    def run_generation(self) -> bool:
        self.generation += 1
        self._evaluate_candidates(self._generate_candidates())
        if self.game.is_solved():
            self._log_solved()
            return True
        self._local_search()
        if self.game.is_solved():
            self._log_solved()
            return True

        self._select()
        self._mutate()
        if self.game.is_solved():
            self._log_solved()
            return True
        self._crossover()
        if self.game.is_solved():
            self._log_solved()
            return True

        return False

    def solve(self, max_generations: int | None = None) -> dict[str, Any]:
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

    def _generate_candidates(self) -> list[tuple[Hypothesis, str]]:
        candidates: list[tuple[Hypothesis, str]] = []
        planned_words = set(self.game.guesses) | self.invalid_guesses if hasattr(self.game, "guesses") else set(self.invalid_guesses)
        active_hypotheses = self._active_hypotheses()
        max_workers = min(max(1, self.config.llm_workers), max(1, len(active_hypotheses)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (
                    hypothesis,
                    executor.submit(
                        self.llm_client.propose_words,
                        hypothesis,
                        hypothesis.words_tried,
                        self.invalid_guesses,
                        self.config.candidates_per_hypothesis,
                    ),
                )
                for hypothesis in active_hypotheses
            ]

            proposed_words = [(hypothesis, future.result()) for hypothesis, future in futures]

        for hypothesis, words in proposed_words:
            for word in words:
                cleaned_word = _clean_word(word)
                if not cleaned_word or cleaned_word in planned_words:
                    continue
                planned_words.add(cleaned_word)
                candidates.append((hypothesis, cleaned_word))

        self.logger.log(
            self.generation,
            "CANDIDATES",
            {"count": len(candidates), "candidates": [{"hypothesis": h.category_name, "word": w} for h, w in candidates]},
        )
        return candidates

    def _evaluate_candidates(self, candidates: list[tuple[Hypothesis, str]]) -> None:
        for hypothesis, word in candidates:
            self._guess_and_update(word, hypothesis)
            if self.game.is_solved():
                return

    def _guess_and_update(self, word: str, hypothesis: Hypothesis) -> int:
        rank = self.game.guess(word)
        if rank == -1:
            self.invalid_guesses.add(word)
            self.logger.log(self.generation, "SKIP_INVALID_GUESS", {"word": word, "hypothesis": hypothesis.category_name})
            return rank

        hypothesis.update(word, rank)
        best_word, best_rank = self.game.best_so_far()
        self.logger.log(
            self.generation,
            "GUESS",
            {
                "word": word,
                "rank": rank,
                "hypothesis": hypothesis.category_name,
                "best_word": best_word,
                "best_rank": best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        return rank

    def _select(self) -> None:
        ranked = sorted(self.hypotheses, key=lambda hypothesis: hypothesis.best_rank)
        keep_count = max(1, len(ranked) // 2)
        kept = set(id(hypothesis) for hypothesis in ranked[:keep_count])
        elite = ranked[0] if ranked else None
        if elite is not None:
            kept.add(id(elite))
        for hypothesis in self.hypotheses:
            hypothesis.status = "active" if id(hypothesis) in kept else "dormant"

        self.logger.log(
            self.generation,
            "SELECT",
            {
                "kept": [hypothesis.category_name for hypothesis in ranked[:keep_count]],
                "discarded": [hypothesis.category_name for hypothesis in ranked[keep_count:]],
                "elite": elite.category_name if elite else None,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )

    def _mutate(self) -> None:
        top_hypotheses = sorted(self._active_hypotheses(), key=lambda hypothesis: hypothesis.best_rank)[:2]
        max_workers = min(max(1, self.config.llm_workers), max(1, len(top_hypotheses)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                (
                    parent,
                    executor.submit(
                        self.llm_client.specialize,
                        parent,
                        parent.words_tried,
                        self.invalid_guesses,
                        self.config.mutations_per_generation,
                    ),
                )
                for parent in top_hypotheses
            ]
            specialization_results = [(parent, future.result()) for parent, future in futures]

        for parent, subcategories in specialization_results:
            children = []
            for category in subcategories:
                child = self._hypothesis_from_category(category, parent=parent.category_name, origin="mutation")
                self.hypotheses.append(child)
                children.append(child.category_name)
                for word in _words_from_category(category):
                    if word in self.invalid_guesses:
                        continue
                    self._guess_and_update(word, child)
                    if self.game.is_solved():
                        break

            self.logger.log(
                self.generation,
                "MUTATE",
                {
                    "parent": parent.category_name,
                    "children": children,
                    "best_word": self.best_word,
                    "best_rank": self.best_rank,
                    "total_guesses": self.game.total_guesses(),
                },
            )
            if self.game.is_solved():
                return

    def _crossover(self) -> None:
        active = sorted(self._active_hypotheses(), key=lambda hypothesis: hypothesis.best_rank)
        if len(active) < 2:
            return

        parent_a, parent_b = active[0], active[1]
        category = self.llm_client.crossover(
            parent_a.category_name,
            parent_b.category_name,
            parent_a.words_tried,
            parent_b.words_tried,
        )
        child = self._hypothesis_from_category(
            category,
            parent=f"{parent_a.category_name}+{parent_b.category_name}",
            origin="crossover",
        )
        self.hypotheses.append(child)
        for word in _words_from_category(category):
            if word in self.invalid_guesses:
                continue
            self._guess_and_update(word, child)
            if self.game.is_solved():
                break

        self.logger.log(
            self.generation,
            "CROSSOVER",
            {
                "parents": [parent_a.category_name, parent_b.category_name],
                "child": child.to_dict(),
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )

    def _local_search(self) -> None:
        best_word, best_rank = self.game.best_so_far()
        if best_word is None or best_rank is None or best_rank >= 50:
            return

        words = self.llm_client.local_search(best_word, best_rank, n=5)
        guessed = []
        local_hypothesis = Hypothesis(
            category_name=f"local search around {best_word}",
            description=f"Fine-grained guesses near {best_word}",
            parent=best_word,
            origin="local_search",
        )
        self.hypotheses.append(local_hypothesis)
        for word in words:
            cleaned_word = _clean_word(word)
            if not cleaned_word or cleaned_word in self.invalid_guesses:
                continue
            rank = self._guess_and_update(cleaned_word, local_hypothesis)
            if rank != -1:
                guessed.append({"word": cleaned_word, "rank": rank})
            if self.game.is_solved():
                break

        self.logger.log(
            self.generation,
            "LOCAL_SEARCH",
            {
                "center_word": best_word,
                "center_rank": best_rank,
                "guesses": guessed,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )

    def _active_hypotheses(self) -> list[Hypothesis]:
        return [hypothesis for hypothesis in self.hypotheses if hypothesis.status == "active"]

    def _hypothesis_from_category(
        self,
        category: dict[str, Any],
        parent: str | None = None,
        origin: str = "init",
    ) -> Hypothesis:
        return Hypothesis(
            category_name=str(category.get("name", "unnamed category")),
            description=str(category.get("description", "")),
            parent=parent,
            origin=origin,
        )

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

    @property
    def best_word(self) -> str | None:
        return self.game.best_so_far()[0]

    @property
    def best_rank(self) -> int | None:
        return self.game.best_so_far()[1]


def _words_from_category(category: dict[str, Any]) -> list[str]:
    words = category.get("words", [])
    if not isinstance(words, list):
        return []
    return [cleaned for word in words if (cleaned := _clean_word(word))]


def _clean_word(word: Any) -> str:
    cleaned_word = str(word).lower().strip()
    if not re.fullmatch(r"[a-z]+", cleaned_word):
        return ""
    return cleaned_word
