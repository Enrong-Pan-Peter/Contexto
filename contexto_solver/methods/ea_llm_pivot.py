"""EA+LLM Contexto method with stall-pivot operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..hypothesis import Hypothesis
from .ea_core import BaseEALLMMethod, EALLMConfig, _coerce_word_list, _words_from_category


@dataclass
class EALLMPivotConfig(EALLMConfig):
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


class EALLMPivotMethod(BaseEALLMMethod):
    config: EALLMPivotConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.stall_detector = StallDetector(
            no_improvement_generations=self.config.stall_no_improvement_generations,
            close_rank_threshold=self.config.stall_close_rank_threshold,
            close_generations_limit=self.config.stall_close_generations_limit,
            resolution_window=self.config.pivot_resolution_window,
        )
        self.pivot_attempts = 0
        self.consecutive_stall_fires = 0
        self.use_fresh_diversity_pivot = False

    def _after_initialize(self) -> None:
        self.stall_detector.record_generation(self.generation, self.best_rank)

    def _after_generation_update(self) -> bool:
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

    def _maybe_attach_pivot_self_report(
        self,
        hypothesis: Hypothesis,
        source: Any,
        raw: str | None,
        rendered_prompt: str | None,
        words: list[str],
    ) -> None:
        """Attach the logged-only self-report to a pivot hypothesis (flag-gated)."""
        if not self.config.self_report:
            return
        self._attach_self_report(
            hypothesis, source, raw, rendered_prompt, words[0] if words else None
        )

    def _build_pivot_hypothesis(self, operator: dict[str, str], best_word: str, best_rank: int) -> tuple[Hypothesis, list[str]]:
        all_guesses = self._known_words()
        n = self.config.pivot_candidate_words_per_operator
        block = self._self_report_block()
        want_raw = self.config.self_report

        if operator["name"] == "morphology":
            if want_raw:
                words, raw, rendered_prompt = self.llm_client.pivot_morphology(
                    best_word, best_rank, all_guesses, n=n, self_report_block=block, return_raw=True
                )
            else:
                words, raw, rendered_prompt = (
                    self.llm_client.pivot_morphology(best_word, best_rank, all_guesses, n=n),
                    None,
                    None,
                )
            hypothesis = Hypothesis(
                category_name=f"pivot morphology around {best_word}",
                description=f"Lexical and morphological variants near {best_word}",
                parent=best_word,
                origin=operator["origin"],
            )
            pivot_words = _coerce_word_list(words)
            # Word-list pivots carry the self-report alongside "words" at the top
            # level, so parse from the raw response.
            self._maybe_attach_pivot_self_report(hypothesis, raw, raw, rendered_prompt, pivot_words)
            return hypothesis, pivot_words
        if operator["name"] == "register_shift":
            if want_raw:
                words, raw, rendered_prompt = self.llm_client.pivot_register_shift(
                    best_word, best_rank, all_guesses, n=n, self_report_block=block, return_raw=True
                )
            else:
                words, raw, rendered_prompt = (
                    self.llm_client.pivot_register_shift(best_word, best_rank, all_guesses, n=n),
                    None,
                    None,
                )
            hypothesis = Hypothesis(
                category_name=f"pivot register shift around {best_word}",
                description=f"Different lexical registers near {best_word}",
                parent=best_word,
                origin=operator["origin"],
            )
            pivot_words = _coerce_word_list(words)
            self._maybe_attach_pivot_self_report(hypothesis, raw, raw, rendered_prompt, pivot_words)
            return hypothesis, pivot_words

        best_hypothesis = self._best_hypothesis()
        if operator.get("fresh") == "true":
            if want_raw:
                category, raw, rendered_prompt = self.llm_client.pivot_fresh_adjacent_category(
                    best_word,
                    best_rank,
                    [hypothesis.category_name for hypothesis in self._active_hypotheses()],
                    all_guesses,
                    n=n,
                    self_report_block=block,
                    return_raw=True,
                )
            else:
                category, raw, rendered_prompt = self.llm_client.pivot_fresh_adjacent_category(
                    best_word,
                    best_rank,
                    [hypothesis.category_name for hypothesis in self._active_hypotheses()],
                    all_guesses,
                    n=n,
                ), None, None
        else:
            if want_raw:
                category, raw, rendered_prompt = self.llm_client.pivot_adjacent_category(
                    best_word,
                    best_rank,
                    best_hypothesis.category_name if best_hypothesis else "unknown category",
                    best_hypothesis.description if best_hypothesis else "",
                    best_hypothesis.words_tried if best_hypothesis else {},
                    all_guesses,
                    n=n,
                    self_report_block=block,
                    return_raw=True,
                )
            else:
                category, raw, rendered_prompt = self.llm_client.pivot_adjacent_category(
                    best_word,
                    best_rank,
                    best_hypothesis.category_name if best_hypothesis else "unknown category",
                    best_hypothesis.description if best_hypothesis else "",
                    best_hypothesis.words_tried if best_hypothesis else {},
                    all_guesses,
                    n=n,
                ), None, None
        hypothesis = self._hypothesis_from_category(
            category,
            parent=best_word,
            origin=operator["origin"],
        )
        pivot_words = _words_from_category(category)
        self._maybe_attach_pivot_self_report(hypothesis, category, raw, rendered_prompt, pivot_words)
        return hypothesis, pivot_words

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


SolverLLMPivot = EALLMPivotMethod
SolverLLMPivotConfig = EALLMPivotConfig

