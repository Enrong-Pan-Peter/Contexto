"""EA+LLM Contexto method with self-adaptive mutation operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..hypothesis import Hypothesis
from ..operators import OPERATOR_PROMPTS, assert_prompt_has_no_sigma_leak, perturb_sigma, sample_operator
from .ea_core import BaseEALLMMethod, EALLMConfig, _words_from_category


@dataclass
class EALLMSelfAdaptiveConfig(EALLMConfig):
    mu: int = 5
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
            )
            assert_prompt_has_no_sigma_leak(prompt, parent_sigma, operator)
            category = self.llm_client.complete_json_prompt(prompt)
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
            children.append(child.to_dict())
            self.logger.log(
                self.generation,
                "OPERATOR_SAMPLED",
                {
                    "parent_id": parent.hypothesis_id,
                    "child_id": child.hypothesis_id,
                    "sigma_snapshot": [float(value) for value in parent_sigma],
                    "child_sigma": [float(value) for value in child.sigma],
                    "child_hypothesis_name": child.category_name,
                    "sampled_op": operator.value,
                    "method": "self_adaptive",
                },
            )

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
