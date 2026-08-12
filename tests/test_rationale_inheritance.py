"""Tests for RQ1 parent-rationale inheritance (logged-only)."""

from __future__ import annotations

import unittest

import numpy as np

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.llm_client import LLMClient
from contexto_solver.methods.ea_llm_self_adaptive import EALLMSelfAdaptiveConfig, EALLMSelfAdaptiveMethod
from contexto_solver.operators import OPERATOR_PROMPTS, OPERATORS
from contexto_solver.self_report import (
    RATIONALE_INHERITANCE_MAX_REASON_CHARS,
    rationale_inheritance_block,
)


class RationaleInheritanceBlockTests(unittest.TestCase):
    def test_empty_when_parent_has_no_rationale(self) -> None:
        block, meta = rationale_inheritance_block(None)
        self.assertEqual(block, "")
        self.assertEqual(meta, {})

    def test_includes_parent_reason_and_basis(self) -> None:
        block, meta = rationale_inheritance_block(
            {"basis_words": ["shrub", "bush"], "reason": "Close semantic neighbor."}
        )
        self.assertIn("shrub", block)
        self.assertIn("Close semantic neighbor.", block)
        self.assertIn("hash", meta)
        self.assertFalse(meta["truncated"])

    def test_replacement_hook_for_tests(self) -> None:
        block, meta = rationale_inheritance_block(None, replacement="INJECTED")
        self.assertEqual(block, "INJECTED")
        self.assertIn("hash", meta)

    def test_truncates_long_reason(self) -> None:
        long_reason = "x" * (RATIONALE_INHERITANCE_MAX_REASON_CHARS + 20)
        block, meta = rationale_inheritance_block({"basis_words": [], "reason": long_reason})
        self.assertTrue(meta["truncated"])
        self.assertIn("...", block)
        self.assertNotIn(long_reason, block)


class RationaleInheritancePromptTests(unittest.TestCase):
    def test_operator_prompt_appends_inheritance_before_self_report(self) -> None:
        client = LLMClient(provider="ollama", api_key="ollama", model="test-model")
        parent = Hypothesis(
            category_name="plants",
            description="green things",
            words_tried={"shrub": 42},
            rationale={"basis_words": ["shrub"], "reason": "parent reason"},
        )
        inheritance, _meta = rationale_inheritance_block(parent.rationale)
        prompt = client.build_operator_mutation_prompt(
            OPERATOR_PROMPTS[OPERATORS[0]],
            parent,
            all_guesses={"shrub"},
            rationale_inheritance_block=inheritance,
            self_report_block="SELF_REPORT_TAIL",
        )
        self.assertLess(prompt.index("parent reason"), prompt.index("SELF_REPORT_TAIL"))

    def test_crossover_prompt_unchanged_without_inheritance_arg(self) -> None:
        client = LLMClient(provider="ollama", api_key="ollama", model="test-model")
        prompt = client.build_crossover_prompt(
            "a",
            "b",
            {"x": 1},
            {"y": 2},
            self_report_block="TAIL",
        )
        self.assertNotIn("prior rationale", prompt)


class RationaleInheritanceMethodTests(unittest.TestCase):
    def test_helper_skips_when_flag_off(self) -> None:
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
            rationale_inheritance=False,
        )
        parent = Hypothesis(
            category_name="plants",
            description="green things",
            rationale={"basis_words": ["shrub"], "reason": "parent reason"},
        )
        block, meta = method._rationale_inheritance_for_parent(parent)
        self.assertEqual(block, "")
        self.assertEqual(meta, {})


if __name__ == "__main__":
    unittest.main()
