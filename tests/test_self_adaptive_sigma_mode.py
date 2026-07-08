"""Tests for ea_llm_self_adaptive sigma-mode control."""

from __future__ import annotations

import unittest

import numpy as np

from contexto_solver.methods.ea_llm_self_adaptive import EALLMSelfAdaptiveConfig, EALLMSelfAdaptiveMethod
from contexto_solver.operators import initial_sigma


class SelfAdaptiveSigmaModeTests(unittest.TestCase):
    def test_frozen_uniform_resets_child_sigma(self) -> None:
        method = EALLMSelfAdaptiveMethod.__new__(EALLMSelfAdaptiveMethod)
        method.config = EALLMSelfAdaptiveConfig(
            max_generations=1,
            candidates_per_hypothesis=1,
            initial_categories=1,
            starter_words_per_category=1,
            mutations_per_generation=1,
            max_active_hypotheses=1,
            trace_dir="traces",
            run_label="test",
            sigma_mode="frozen_uniform",
        )
        parent_sigma = np.asarray([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
        child_sigma = method._mode_sigma(parent_sigma)
        np.testing.assert_allclose(child_sigma, initial_sigma())

    def test_adaptive_perturbs_parent_sigma(self) -> None:
        method = EALLMSelfAdaptiveMethod.__new__(EALLMSelfAdaptiveMethod)
        method.config = EALLMSelfAdaptiveConfig(
            max_generations=1,
            candidates_per_hypothesis=1,
            initial_categories=1,
            starter_words_per_category=1,
            mutations_per_generation=1,
            max_active_hypotheses=1,
            trace_dir="traces",
            run_label="test",
            sigma_mode="adaptive",
            random_seed=0,
        )
        method.rng = np.random.default_rng(0)
        parent_sigma = np.asarray([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
        child_sigma = method._mode_sigma(parent_sigma)
        self.assertFalse(np.allclose(child_sigma, initial_sigma()))
        self.assertFalse(np.allclose(child_sigma, parent_sigma))


if __name__ == "__main__":
    unittest.main()
