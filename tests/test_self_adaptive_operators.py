from __future__ import annotations

import unittest

import numpy as np

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.llm_client import LLMClient
from contexto_solver.operators import (
    N_OPERATORS,
    OPERATOR_DISTINGUISHING_PHRASES,
    OPERATOR_PROMPTS,
    OPERATORS,
    Operator,
    assert_prompt_has_no_sigma_leak,
    initial_sigma,
    perturb_sigma,
    sample_operator,
)


class SelfAdaptiveOperatorTests(unittest.TestCase):
    def test_operator_ids_are_exact(self) -> None:
        self.assertEqual(
            [operator.value for operator in OPERATORS],
            ["s_mutation", "m_mutation", "ml_mutation", "l_mutation"],
        )
        self.assertEqual(len(Operator), N_OPERATORS)

    def test_sample_operator_matches_sigma_empirically(self) -> None:
        rng = np.random.default_rng(123)
        sigma = np.asarray([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
        samples = [sample_operator(sigma, rng) for _ in range(10_000)]
        s_count = samples.count(Operator.S_MUTATION)
        self.assertGreaterEqual(s_count, 6_800)
        self.assertLessEqual(s_count, 7_200)

    def test_perturb_sigma_shape_floor_sum_and_mean(self) -> None:
        rng = np.random.default_rng(456)
        parent = np.asarray([0.7, 0.1, 0.1, 0.1], dtype=np.float64)
        draws = np.vstack([perturb_sigma(parent, 50.0, 0.02, rng) for _ in range(10_000)])
        self.assertEqual(draws.shape, (10_000, 4))
        self.assertTrue(np.all(draws >= 0.02))
        self.assertTrue(np.allclose(draws.sum(axis=1), 1.0, atol=1e-9))
        self.assertTrue(np.allclose(draws.mean(axis=0), parent, atol=0.01))

    def test_initial_sigma(self) -> None:
        sigma = initial_sigma()
        self.assertEqual(sigma.shape, (4,))
        self.assertTrue(np.allclose(sigma, np.full(4, 0.25)))
        self.assertTrue(np.isclose(sigma.sum(), 1.0, atol=1e-9))

    def test_prompt_leakage_for_all_operator_prompts(self) -> None:
        client = LLMClient(provider="ollama", api_key="ollama", model="test-model")
        parent = Hypothesis(
            category_name="types of plants",
            description="plant taxonomy",
            words_tried={"shrub": 50, "tree": 400},
        )
        sigma = np.asarray([0.31, 0.22, 0.18, 0.29], dtype=np.float64)
        parent.set_sigma(sigma)

        for operator in OPERATORS:
            prompt = client.build_operator_mutation_prompt(
                OPERATOR_PROMPTS[operator],
                parent,
                all_guesses={"shrub", "tree", "bush"},
                invalid_guesses={"sourcream"},
                n=3,
                active_categories=["types of plants", "foods"],
            )
            self.assertIn(OPERATOR_DISTINGUISHING_PHRASES[operator], prompt)
            assert_prompt_has_no_sigma_leak(prompt, sigma, operator)


if __name__ == "__main__":
    unittest.main()
