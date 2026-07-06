"""EA+LLM Contexto method with self-adaptive mutation operators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hypothesis import Hypothesis
from ..operators import OPERATOR_PROMPTS, assert_prompt_has_no_sigma_leak, perturb_sigma, sample_operator
from ..self_report import (
    SELF_REPORT_BLOCK,
    SELF_REPORT_FOLLOWUP_PROMPT,
    parse_self_report,
)
from .ea_core import BaseEALLMMethod, EALLMConfig, _words_from_category


@dataclass
class EALLMSelfAdaptiveConfig(EALLMConfig):
    mu: int = 15
    concentration: float = 50.0
    sigma_floor: float = 0.02
    random_seed: int | None = None
    disable_local_search: bool = True


class EALLMSelfAdaptiveMethod(BaseEALLMMethod):
    config: EALLMSelfAdaptiveConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(self.config.random_seed)
        self._local_search_disabled_logged = False

    def _after_initialize(self) -> None:
        self._log_sigma_trajectory()

    # --- RQ1 self-report instrumentation (logged-only) --------------------------

    def _self_report_block(self) -> str:
        """The prompt block requesting self-report fields, or "" when off."""
        return SELF_REPORT_BLOCK if self.config.self_report else ""

    def _self_report_context(self) -> str:
        """Words the operator saw (guessed + invalid), for the follow-up prompt.

        These are exactly the words interpolated into the operator prompt's
        ``all_guesses`` slot, so any ``basis_words`` the model cites from here are
        present in the stored operator prompt.
        """
        return json.dumps(sorted(self._known_words()))

    def _attach_self_report(
        self,
        child: Hypothesis,
        category: Any,
        raw: str | None,
        rendered_prompt: str | None,
        proposed_word: str | None,
    ) -> None:
        """Parse and attach the logged-only self-report to a child hypothesis.

        Never raises and never affects word acceptance. On a missing/failed
        report, issues one targeted follow-up about ``proposed_word``; on repeated
        failure stores nulls with ``self_report_parse_failed`` set and keeps the
        raw text for offline re-parsing.
        """
        report = parse_self_report(category)
        final_raw = raw
        if report["predicted_closeness"] is None and proposed_word:
            followup_prompt = SELF_REPORT_FOLLOWUP_PROMPT.format(
                word=proposed_word,
                context=self._self_report_context(),
            )
            try:
                parsed, followup_raw = self.llm_client.complete_json_prompt_with_raw(followup_prompt)
            except Exception:
                parsed, followup_raw = None, None
            if parsed is not None:
                followup = parse_self_report(parsed)
                final_raw = followup_raw if followup_raw is not None else raw
                if followup["predicted_closeness"] is not None:
                    report["predicted_closeness"] = followup["predicted_closeness"]
                    report["predicted_closeness_clamped"] = followup["predicted_closeness_clamped"]
                if followup["rationale"] and (
                    report["rationale"] is None or not report["rationale"]["basis_words"]
                ):
                    report["rationale"] = followup["rationale"]

        child.self_report_prompt = rendered_prompt
        child.self_report_raw = final_raw
        child.predicted_closeness = report["predicted_closeness"]
        child.predicted_closeness_clamped = report["predicted_closeness_clamped"]
        child.rationale = report["rationale"]
        rationale = report["rationale"]
        child.self_report_parse_failed = report["predicted_closeness"] is None and (
            rationale is None or (not rationale["basis_words"] and not rationale["reason"])
        )

    def _mutate(self) -> None:
        parents = sorted(self._active_hypotheses(), key=lambda hypothesis: hypothesis.best_rank)[: self.config.mu]
        children: list[dict[str, Any]] = []

        for parent in parents:
            operator = sample_operator(parent.sigma, self.rng)
            prompt_template = OPERATOR_PROMPTS[operator]
            parent_sigma = parent.sigma.copy()
            prompt = self.llm_client.build_operator_mutation_prompt(
                prompt_template,
                parent,
                self._known_words(),
                self.invalid_guesses,
                n=self.config.starter_words_per_category,
                active_categories=[hypothesis.category_name for hypothesis in self._active_hypotheses()],
                self_report_block=self._self_report_block(),
            )
            assert_prompt_has_no_sigma_leak(prompt, parent_sigma, operator)
            if self.config.self_report:
                category, raw = self.llm_client.complete_json_prompt_with_raw(prompt)
            else:
                category, raw = self.llm_client.complete_json_prompt(prompt), None
            if not isinstance(category, dict):
                continue

            child_sigma = perturb_sigma(
                parent_sigma,
                concentration=self.config.concentration,
                sigma_floor=self.config.sigma_floor,
                rng=self.rng,
            )
            child = self._hypothesis_from_category(
                category,
                parent=parent.category_name,
                origin=f"self_adaptive_{operator.value}",
                parent_id=parent.hypothesis_id,
                sigma=child_sigma,
            )
            self.hypotheses.append(child)
            if self.config.self_report:
                proposed = _words_from_category(category)
                self._attach_self_report(
                    child, category, raw, prompt, proposed[0] if proposed else None
                )
            children.append(child.to_dict())
            operator_details: dict[str, Any] = {
                "parent_id": parent.hypothesis_id,
                "child_id": child.hypothesis_id,
                "parent_rank": parent.best_rank,
                "sigma_snapshot": [float(value) for value in parent_sigma],
                "child_sigma": [float(value) for value in child.sigma],
                "child_hypothesis_name": child.category_name,
                "sampled_op": operator.value,
                "method": "self_adaptive",
            }
            if self.config.self_report:
                operator_details["self_report"] = child.self_report_dict()
            self.logger.log(self.generation, "OPERATOR_SAMPLED", operator_details)

            for word in _words_from_category(category):
                if word in self.invalid_guesses:
                    continue
                self._guess_and_update(word, child)
                if self.game.is_solved():
                    break
            if self.game.is_solved():
                break

        self.logger.log(
            self.generation,
            "MUTATE",
            {
                "method": "self_adaptive",
                "children": children,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )

    def _crossover(self) -> None:
        active = sorted(self._active_hypotheses(), key=lambda hypothesis: hypothesis.best_rank)
        if len(active) < 2:
            return

        parent_a, parent_b = active[0], active[1]
        block = self._self_report_block()
        if self.config.self_report:
            rendered_prompt = self.llm_client.build_crossover_prompt(
                parent_a.category_name,
                parent_b.category_name,
                parent_a.words_tried,
                parent_b.words_tried,
                block,
            )
            category, raw = self.llm_client.crossover(
                parent_a.category_name,
                parent_b.category_name,
                parent_a.words_tried,
                parent_b.words_tried,
                self_report_block=block,
                return_raw=True,
            )
        else:
            rendered_prompt, raw = None, None
            category = self.llm_client.crossover(
                parent_a.category_name,
                parent_b.category_name,
                parent_a.words_tried,
                parent_b.words_tried,
            )
        blended_sigma = 0.5 * (parent_a.sigma + parent_b.sigma)
        child_sigma = perturb_sigma(
            blended_sigma,
            concentration=self.config.concentration,
            sigma_floor=self.config.sigma_floor,
            rng=self.rng,
        )
        child = self._hypothesis_from_category(
            category,
            parent=f"{parent_a.category_name}+{parent_b.category_name}",
            origin="crossover",
            sigma=child_sigma,
        )
        self.hypotheses.append(child)
        if self.config.self_report and isinstance(category, dict):
            proposed = _words_from_category(category)
            self._attach_self_report(
                child, category, raw, rendered_prompt, proposed[0] if proposed else None
            )
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
                "parent_ids": [parent_a.hypothesis_id, parent_b.hypothesis_id],
                "parent_ranks": [parent_a.best_rank, parent_b.best_rank],
                "parent_a_sigma": [float(value) for value in parent_a.sigma],
                "parent_b_sigma": [float(value) for value in parent_b.sigma],
                "child_sigma_pre_perturbation": [float(value) for value in blended_sigma],
                "child": child.to_dict(),
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )

    def _local_search(self) -> None:
        if not self.config.disable_local_search:
            super()._local_search()
            return

        best_word, best_rank = self.game.best_so_far()
        if (
            self._local_search_disabled_logged
            or best_word is None
            or best_rank is None
            or best_rank >= self.config.local_search_rank_threshold
        ):
            return

        self._local_search_disabled_logged = True
        self.logger.log(
            self.generation,
            "LOCAL_SEARCH_DISABLED",
            {
                "center_word": best_word,
                "center_rank": best_rank,
                "threshold": self.config.local_search_rank_threshold,
                "reason": "self_adaptive_disable_local_search",
            },
        )

    def _after_generation_update(self) -> bool:
        self._log_sigma_trajectory()
        return False

    def _cap_active_hypotheses(self) -> None:
        active = sorted(self._active_hypotheses(), key=lambda hypothesis: hypothesis.best_rank)
        allowed = set(id(hypothesis) for hypothesis in active[: self.config.mu])
        for hypothesis in active:
            if id(hypothesis) not in allowed:
                hypothesis.status = "dormant"

    def _log_sigma_trajectory(self) -> None:
        active = self._active_hypotheses()
        if not active:
            return
        mean_sigma = np.mean(np.vstack([hypothesis.sigma for hypothesis in active]), axis=0)
        self.logger.log_sigma_trajectory(
            self.generation,
            mean_sigma=[float(value) for value in mean_sigma],
            population_size=len(active),
        )


SolverLLMSelfAdaptive = EALLMSelfAdaptiveMethod
SolverLLMSelfAdaptiveConfig = EALLMSelfAdaptiveConfig
