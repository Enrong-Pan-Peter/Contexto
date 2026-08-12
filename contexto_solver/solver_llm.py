"""Evolutionary Contexto solver that uses an LLM for word generation."""

from __future__ import annotations

import re
import json
import time
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
    max_active_hypotheses: int
    trace_dir: str
    run_label: str
    llm_workers: int = 4
    local_search_rank_threshold: int = 100
    enable_pivot: bool = True
    stall_no_improvement_generations: int = 3
    stall_close_rank_threshold: int = 30
    stall_close_generations_limit: int = 5
    max_pivot_attempts_per_run: int = 5
    pivot_candidate_words_per_operator: int = 10
    pivot_resolution_window: int = 2


class StallDetector:
    def __init__(
        self,
        no_improvement_generations: int,
        close_rank_threshold: int,
        close_generations_limit: int,
        resolution_window: int,
    ) -> None:
        self.no_improvement_generations = max(1, no_improvement_generations)
        self.close_rank_threshold = close_rank_threshold
        self.close_generations_limit = max(1, close_generations_limit)
        self.resolution_window = max(1, resolution_window)
        self.history: list[dict[str, int | None]] = []
        self.best_rank: int | None = None
        self.close_crossed_generation: int | None = None
        self.pending_pivots: list[dict[str, Any]] = []

    def record_generation(self, generation: int, best_rank: int | None) -> dict[str, Any]:
        previous_best = self.best_rank
        improved = best_rank is not None and (previous_best is None or best_rank < previous_best)
        if improved:
            self.best_rank = best_rank
        if best_rank is not None and best_rank < self.close_rank_threshold and self.close_crossed_generation is None:
            self.close_crossed_generation = generation

        self.history.append({"generation": generation, "best_rank": best_rank})
        resolutions = self._resolve_pending_pivots(generation, best_rank)
        return {"improved": improved, "resolutions": resolutions}

    def stall_trigger(self) -> dict[str, Any] | None:
        if not self.history:
            return None
        current = self.history[-1]
        current_generation = current["generation"]
        current_rank = current["best_rank"]
        if current_rank is None or current_rank <= 1:
            return None

        if (
            self.close_crossed_generation is not None
            and current_generation - self.close_crossed_generation >= self.close_generations_limit
        ):
            return {
                "condition": "close_rank_timeout",
                "detail": (
                    f"best rank stayed below {self.close_rank_threshold} for "
                    f"{current_generation - self.close_crossed_generation} generations without solving"
                ),
                "close_crossed_generation": self.close_crossed_generation,
                "current_generation": current_generation,
                "current_rank": current_rank,
            }

        lookback = self.no_improvement_generations
        if len(self.history) > lookback:
            comparison = self.history[-lookback - 1]
            comparison_rank = comparison["best_rank"]
            if comparison_rank is not None and current_rank >= comparison_rank:
                return {
                    "condition": "no_improvement",
                    "detail": f"best rank has not improved for {lookback} generations",
                    "comparison_generation": comparison["generation"],
                    "current_generation": current_generation,
                    "current_rank": current_rank,
                }
        return None

    def add_pending_pivot(self, pivot: dict[str, Any]) -> None:
        self.pending_pivots.append(pivot)

    def _resolve_pending_pivots(self, generation: int, best_rank: int | None) -> list[dict[str, Any]]:
        resolutions: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for pivot in self.pending_pivots:
            age = generation - pivot["trigger_generation"]
            trigger_best_rank = pivot["trigger_best_rank"]
            if best_rank is not None and best_rank < trigger_best_rank:
                resolutions.append(
                    {
                        "pivot": pivot,
                        "resolved": True,
                        "generation": generation,
                        "new_best_rank": best_rank,
                        "detail": "best rank improved after pivot",
                    }
                )
            elif age >= self.resolution_window:
                resolutions.append(
                    {
                        "pivot": pivot,
                        "resolved": False,
                        "generation": generation,
                        "new_best_rank": best_rank,
                        "detail": f"no improvement within {self.resolution_window} generations",
                    }
                )
            else:
                remaining.append(pivot)
        self.pending_pivots = remaining
        return resolutions


class SolverLLM:
    def __init__(self, game: Game, llm_client: LLMClient, logger: Logger, config: SolverLLMConfig) -> None:
        self.game = game
        self.llm_client = llm_client
        self.logger = logger
        self.config = config
        self.hypotheses: list[Hypothesis] = []
        self.invalid_guesses: set[str] = set()
        self.generation = 0
        self.stall_detector = StallDetector(
            no_improvement_generations=config.stall_no_improvement_generations,
            close_rank_threshold=config.stall_close_rank_threshold,
            close_generations_limit=config.stall_close_generations_limit,
            resolution_window=config.pivot_resolution_window,
        )
        self.pivot_attempts = 0
        self.consecutive_stall_fires = 0
        self.use_fresh_diversity_pivot = False

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

        if not self.game.is_solved() and not self.hypotheses:
            raise RuntimeError(
                "LLM solver initialization produced an empty hypothesis population: "
                f"none of the {len(categories)} initial categories were usable. "
                "Refusing to run empty generations."
            )

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
        self._cap_active_hypotheses()
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
        self._deduplicate_hypotheses()
        self._cap_active_hypotheses()

        if self._handle_stall_pivot():
            return True
        return False

    def solve(self, max_generations: int | None = None) -> dict[str, Any]:
        generation_limit = max_generations or self.config.max_generations
        #region agent log
        _agent_debug_log(
            "contexto_solver/solver_llm.py:solve",
            "solve started",
            {
                "requestedMaxGenerations": max_generations,
                "configuredMaxGenerations": self.config.max_generations,
                "effectiveGenerationLimit": generation_limit,
                "localSearchRankThreshold": self.config.local_search_rank_threshold,
                "enablePivot": self.config.enable_pivot,
                "initialHypotheses": len(self.hypotheses),
            },
            "H1,H5",
        )
        #endregion
        solved = self.initialize()
        self.stall_detector.record_generation(self.generation, self.best_rank)
        self._print_generation_summary()
        while not solved and self.generation < generation_limit:
            solved = self.run_generation()
            self._print_generation_summary()

        if not solved:
            #region agent log
            _agent_debug_log(
                "contexto_solver/solver_llm.py:solve",
                "solve failed at generation limit",
                {
                    "generation": self.generation,
                    "generationLimit": generation_limit,
                    "bestWord": self.best_word,
                    "bestRank": self.best_rank,
                    "totalGuesses": self.game.total_guesses(),
                    "activeHypotheses": _hypothesis_summary(self._active_hypotheses()),
                },
                "H1,H2,H3,H5",
            )
            #endregion
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
        planned_words = self._known_words()
        planned_word_families = _word_families(planned_words)
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
                        self.game.guesses if hasattr(self.game, "guesses") else None,
                    ),
                )
                for hypothesis in active_hypotheses
            ]

            proposed_words = [(hypothesis, future.result()) for hypothesis, future in futures]

        raw_count = 0
        rejected_invalid = 0
        rejected_duplicate = 0
        for hypothesis, words in proposed_words:
            for word in words:
                raw_count += 1
                cleaned_word = _clean_word(word)
                if not cleaned_word:
                    rejected_invalid += 1
                    continue
                if cleaned_word in planned_words or _word_family(cleaned_word) in planned_word_families:
                    rejected_duplicate += 1
                    continue
                planned_words.add(cleaned_word)
                planned_word_families.add(_word_family(cleaned_word))
                candidates.append((hypothesis, cleaned_word))

        #region agent log
        _agent_debug_log(
            "contexto_solver/solver_llm.py:_generate_candidates",
            "candidate filtering summary",
            {
                "generation": self.generation,
                "activeHypotheses": _hypothesis_summary(active_hypotheses),
                "rawCount": raw_count,
                "acceptedCount": len(candidates),
                "rejectedInvalid": rejected_invalid,
                "rejectedDuplicate": rejected_duplicate,
                "bestWord": self.best_word,
                "bestRank": self.best_rank,
            },
            "H2,H3,H4",
        )
        #endregion
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
        keep_count = min(max(1, len(ranked) // 2), self.config.max_active_hypotheses)
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
                "max_active_hypotheses": self.config.max_active_hypotheses,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        #region agent log
        _agent_debug_log(
            "contexto_solver/solver_llm.py:_select",
            "selection summary",
            {
                "generation": self.generation,
                "keepCount": keep_count,
                "maxActiveHypotheses": self.config.max_active_hypotheses,
                "kept": _hypothesis_summary(ranked[:keep_count]),
                "bestDiscarded": _hypothesis_summary(ranked[keep_count : keep_count + 5]),
                "bestWord": self.best_word,
                "bestRank": self.best_rank,
            },
            "H3",
        )
        #endregion

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

    def _deduplicate_hypotheses(self) -> None:
        kept: list[Hypothesis] = []
        merged_events: list[dict[str, Any]] = []

        for hypothesis in sorted(self.hypotheses, key=lambda item: item.best_rank):
            duplicate = next((existing for existing in kept if _are_duplicate_hypotheses(existing, hypothesis)), None)
            if duplicate is None:
                kept.append(hypothesis)
                continue

            survivor, discarded = (
                (duplicate, hypothesis)
                if duplicate.best_rank <= hypothesis.best_rank
                else (hypothesis, duplicate)
            )
            survivor.words_tried.update(discarded.words_tried)
            survivor.status = "active" if survivor.status == "active" or discarded.status == "active" else "dormant"
            if survivor is hypothesis:
                kept.remove(duplicate)
                kept.append(survivor)

            merged_events.append(
                {
                    "survivor": survivor.category_name,
                    "discarded": discarded.category_name,
                    "survivor_best_rank": survivor.best_rank,
                    "discarded_best_rank": discarded.best_rank,
                }
            )

        self.hypotheses = kept
        if merged_events:
            self._cap_active_hypotheses()
            self.logger.log(
                self.generation,
                "DEDUPLICATE",
                {
                    "merged": merged_events,
                    "remaining_hypotheses": len(self.hypotheses),
                    "active_hypotheses": len(self._active_hypotheses()),
                    "best_word": self.best_word,
                    "best_rank": self.best_rank,
                    "total_guesses": self.game.total_guesses(),
                },
            )

    def _cap_active_hypotheses(self) -> None:
        active = sorted(self._active_hypotheses(), key=lambda hypothesis: hypothesis.best_rank)
        allowed = set(id(hypothesis) for hypothesis in active[: self.config.max_active_hypotheses])
        for hypothesis in active:
            if id(hypothesis) not in allowed:
                hypothesis.status = "dormant"

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
        if best_word is None or best_rank is None or best_rank >= self.config.local_search_rank_threshold:
            #region agent log
            _agent_debug_log(
                "contexto_solver/solver_llm.py:_local_search",
                "local search skipped",
                {
                    "generation": self.generation,
                    "bestWord": best_word,
                    "bestRank": best_rank,
                    "threshold": self.config.local_search_rank_threshold,
                },
                "H2",
            )
            #endregion
            return

        planned_words = self._known_words()
        planned_word_families = _word_families(planned_words)
        guessed = []
        requested_attempts = []
        skipped_already_tried = 0
        local_hypothesis = Hypothesis(
            category_name=f"local search around {best_word}",
            description=f"Fine-grained guesses near {best_word}",
            parent=best_word,
            origin="local_search",
        )
        self.hypotheses.append(local_hypothesis)
        for attempt in range(2):
            words = self.llm_client.local_search(best_word, best_rank, n=5, all_guesses=planned_words)
            requested_attempts.append(words)
            attempt_guessed = 0
            for word in words:
                cleaned_word = _clean_word(word)
                if not cleaned_word or cleaned_word in self.invalid_guesses:
                    continue
                if cleaned_word in planned_words or _word_family(cleaned_word) in planned_word_families:
                    skipped_already_tried += 1
                    continue
                planned_words.add(cleaned_word)
                planned_word_families.add(_word_family(cleaned_word))
                rank = self._guess_and_update(cleaned_word, local_hypothesis)
                if rank != -1:
                    attempt_guessed += 1
                    guessed.append({"word": cleaned_word, "rank": rank})
                if self.game.is_solved():
                    break
            if attempt_guessed > 0 or self.game.is_solved():
                break

        self.logger.log(
            self.generation,
            "LOCAL_SEARCH",
            {
                "center_word": best_word,
                "center_rank": best_rank,
                "guesses": guessed,
                "attempts": requested_attempts,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        #region agent log
        _agent_debug_log(
            "contexto_solver/solver_llm.py:_local_search",
            "local search completed",
            {
                "generation": self.generation,
                "centerWord": best_word,
                "centerRank": best_rank,
                "requestedAttempts": requested_attempts,
                "guesses": guessed,
                "skippedAlreadyTried": skipped_already_tried,
                "bestWord": self.best_word,
                "bestRank": self.best_rank,
                "totalGuesses": self.game.total_guesses(),
            },
            "H2,H4",
        )
        #endregion

    def _handle_stall_pivot(self) -> bool:
        if not self.config.enable_pivot:
            return False

        detector_update = self.stall_detector.record_generation(self.generation, self.best_rank)
        for resolution in detector_update["resolutions"]:
            self._log_pivot_resolution(resolution)

        if detector_update["improved"]:
            self.consecutive_stall_fires = 0
            return False

        trigger = self.stall_detector.stall_trigger()
        if trigger is None or self.pivot_attempts >= self.config.max_pivot_attempts_per_run:
            return False

        self.consecutive_stall_fires += 1
        solved = self._run_pivot(trigger)
        if solved:
            self._log_solved()
        return solved

    def _run_pivot(self, trigger: dict[str, Any]) -> bool:
        best_word, best_rank = self.game.best_so_far()
        if best_word is None or best_rank is None:
            return False

        self.pivot_attempts += 1
        operator = self._select_pivot_operator()
        pivot_id = f"{self.generation}:{self.pivot_attempts}:{operator['name']}"
        before_rank = best_rank
        before_word = best_word

        try:
            pivot_hypothesis, generated_words = self._build_pivot_hypothesis(operator, best_word, best_rank)
        except Exception as exc:
            self.logger.log(
                self.generation,
                "PIVOT_FAILED",
                {
                    "pivot_id": pivot_id,
                    "stall_condition": trigger,
                    "operator": operator,
                    "best_word": before_word,
                    "best_rank": before_rank,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False

        accepted_words = self._new_clean_words(generated_words)
        ranks: list[dict[str, int]] = []
        if accepted_words:
            self.hypotheses.append(pivot_hypothesis)
            for word in accepted_words:
                rank = self._guess_and_update(word, pivot_hypothesis)
                if rank != -1:
                    ranks.append({"word": word, "rank": rank})
                if self.game.is_solved():
                    break

        after_word, after_rank = self.game.best_so_far()
        self.logger.log(
            self.generation,
            "PIVOT_TRIGGERED",
            {
                "pivot_id": pivot_id,
                "stall_condition": trigger,
                "operator": operator,
                "best_word": before_word,
                "best_rank": before_rank,
                "candidate_words": generated_words,
                "accepted_words": accepted_words,
                "ranks": ranks,
                "hypothesis": pivot_hypothesis.to_dict(),
                "origin": pivot_hypothesis.origin,
                "best_word_after_pivot": after_word,
                "best_rank_after_pivot": after_rank,
            },
        )

        if after_rank is not None and after_rank < before_rank:
            self._log_pivot_resolution(
                {
                    "pivot": {
                        "pivot_id": pivot_id,
                        "trigger_generation": self.generation,
                        "trigger_best_word": before_word,
                        "trigger_best_rank": before_rank,
                        "operator": operator,
                    },
                    "resolved": True,
                    "generation": self.generation,
                    "new_best_rank": after_rank,
                    "detail": "best rank improved during pivot generation",
                }
            )
            self.consecutive_stall_fires = 0
        else:
            self.stall_detector.add_pending_pivot(
                {
                    "pivot_id": pivot_id,
                    "trigger_generation": self.generation,
                    "trigger_best_word": before_word,
                    "trigger_best_rank": before_rank,
                    "operator": operator,
                }
            )

        if operator["name"] == "adjacent_category" and not (after_rank is not None and after_rank < before_rank):
            self.use_fresh_diversity_pivot = True
            self.consecutive_stall_fires = 0

        return self.game.is_solved()

    def _select_pivot_operator(self) -> dict[str, str]:
        if self.use_fresh_diversity_pivot:
            self.use_fresh_diversity_pivot = False
            return {
                "name": "adjacent_category",
                "label": "C",
                "reason": "fresh diversity after a full pivot cycle",
                "origin": "pivot_adjacent_category",
                "fresh": "true",
            }

        position = (self.consecutive_stall_fires - 1) % 3
        if position == 0:
            return {
                "name": "morphology",
                "label": "A",
                "reason": "first stall fire; try lexical and morphological variants",
                "origin": "pivot_morphology",
                "fresh": "false",
            }
        if position == 1:
            return {
                "name": "register_shift",
                "label": "B",
                "reason": "second consecutive stall fire; try alternate lexical registers",
                "origin": "pivot_register_shift",
                "fresh": "false",
            }
        return {
            "name": "adjacent_category",
            "label": "C",
            "reason": "third consecutive stall fire; try adjacent category jump",
            "origin": "pivot_adjacent_category",
            "fresh": "false",
        }

    def _build_pivot_hypothesis(self, operator: dict[str, str], best_word: str, best_rank: int) -> tuple[Hypothesis, list[str]]:
        all_guesses = self._known_words()
        n = self.config.pivot_candidate_words_per_operator
        if operator["name"] == "morphology":
            words = self.llm_client.pivot_morphology(best_word, best_rank, all_guesses, n=n)
            return (
                Hypothesis(
                    category_name=f"pivot morphology around {best_word}",
                    description=f"Lexical and morphological variants near {best_word}",
                    parent=best_word,
                    origin=operator["origin"],
                ),
                _coerce_word_list(words),
            )
        if operator["name"] == "register_shift":
            words = self.llm_client.pivot_register_shift(best_word, best_rank, all_guesses, n=n)
            return (
                Hypothesis(
                    category_name=f"pivot register shift around {best_word}",
                    description=f"Different lexical registers near {best_word}",
                    parent=best_word,
                    origin=operator["origin"],
                ),
                _coerce_word_list(words),
            )

        best_hypothesis = self._best_hypothesis()
        if operator.get("fresh") == "true":
            category = self.llm_client.pivot_fresh_adjacent_category(
                best_word,
                best_rank,
                [hypothesis.category_name for hypothesis in self._active_hypotheses()],
                all_guesses,
                n=n,
            )
        else:
            category = self.llm_client.pivot_adjacent_category(
                best_word,
                best_rank,
                best_hypothesis.category_name if best_hypothesis else "unknown category",
                best_hypothesis.description if best_hypothesis else "",
                best_hypothesis.words_tried if best_hypothesis else {},
                all_guesses,
                n=n,
            )
        hypothesis = self._hypothesis_from_category(
            category,
            parent=best_word,
            origin=operator["origin"],
        )
        return hypothesis, _words_from_category(category)

    def _new_clean_words(self, words: list[str]) -> list[str]:
        known_words = self._known_words()
        known_word_families = _word_families(known_words)
        accepted: list[str] = []
        for word in words:
            cleaned_word = _clean_word(word)
            if not cleaned_word or cleaned_word in known_words or _word_family(cleaned_word) in known_word_families:
                continue
            known_words.add(cleaned_word)
            known_word_families.add(_word_family(cleaned_word))
            accepted.append(cleaned_word)
        return accepted

    def _known_words(self) -> set[str]:
        known = set(self.invalid_guesses)
        if hasattr(self.game, "guesses"):
            known.update(self.game.guesses)
        return known

    def _best_hypothesis(self) -> Hypothesis | None:
        if not self.hypotheses:
            return None
        return min(self.hypotheses, key=lambda hypothesis: hypothesis.best_rank)

    def _log_pivot_resolution(self, resolution: dict[str, Any]) -> None:
        pivot = resolution["pivot"]
        self.logger.log(
            self.generation,
            "PIVOT_RESOLUTION",
            {
                "pivot_id": pivot["pivot_id"],
                "trigger_generation": pivot["trigger_generation"],
                "operator": pivot["operator"],
                "trigger_best_word": pivot["trigger_best_word"],
                "trigger_best_rank": pivot["trigger_best_rank"],
                "resolved": resolution["resolved"],
                "resolution_generation": resolution["generation"],
                "new_best_rank": resolution["new_best_rank"],
                "detail": resolution["detail"],
                "best_word": self.best_word,
                "best_rank": self.best_rank,
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

    def _print_generation_summary(self) -> None:
        print(f"Generation {self.generation}: best word={self.best_word}, best rank={self.best_rank}")

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


def _coerce_word_list(words: Any) -> list[str]:
    if not isinstance(words, list):
        return []
    return [str(word) for word in words]


def _clean_word(word: Any) -> str:
    cleaned_word = str(word).lower().strip()
    if not re.fullmatch(r"[a-z]+", cleaned_word):
        return ""
    return cleaned_word


def _word_families(words: set[str]) -> set[str]:
    return {_word_family(word) for word in words}


def _word_family(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 3 and word.endswith("es") and word[-3] in {"s", "x", "z"}:
        return word[:-2]
    if len(word) > 4 and word.endswith("ches"):
        return word[:-2]
    if len(word) > 4 and word.endswith("shes"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _are_duplicate_hypotheses(a: Hypothesis, b: Hypothesis) -> bool:
    shared_words = set(a.words_tried) & set(b.words_tried)
    if len(shared_words) >= 2:
        return True

    a_name = a.category_name.lower().strip()
    b_name = b.category_name.lower().strip()
    if not a_name or not b_name:
        return False
    if a_name in b_name or b_name in a_name:
        return True

    a_words = set(a_name.split())
    b_words = set(b_name.split())
    if not a_words or not b_words:
        return False
    jaccard = len(a_words & b_words) / len(a_words | b_words)
    if jaccard > 0.6:
        return True
    return len(a_words ^ b_words) <= 1


def _hypothesis_summary(hypotheses: list[Hypothesis]) -> list[dict[str, Any]]:
    return [
        {
            "name": hypothesis.category_name,
            "bestWord": hypothesis.best_word,
            "bestRank": hypothesis.best_rank,
            "status": hypothesis.status,
            "origin": hypothesis.origin,
        }
        for hypothesis in hypotheses
    ]


def _agent_debug_log(location: str, message: str, data: dict[str, object], hypothesis_id: str) -> None:
    try:
        payload = {
            "sessionId": "0eedb7",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-0eedb7.log", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass
