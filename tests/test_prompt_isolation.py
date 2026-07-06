"""Phase 3: information-isolation audit of the operator/crossover prompts.

Interpolated-variable note
--------------------------
Every value the solver interpolates into an operator or crossover prompt is
listed below, with why each is leak-safe (cannot reveal the hidden target, an
unguessed word's rank, or the operator's strategy parameters):

Mutation prompts (``build_operator_mutation_prompt``):
- ``name`` / ``description``: the parent hypothesis's category text (LLM-authored;
  no game state).
- ``words_tried``: a JSON map of GUESSED words -> their game ranks. Only guessed
  words carry ranks.
- ``best_word`` / ``best_rank``: the best (word, rank) among the parent's guessed
  words. Guessed word only.
- ``all_guesses``: a sorted JSON list of GUESSED (and invalid) words, no ranks.
- ``invalid_guesses``: a sorted JSON list of rejected words, no ranks.
- ``n``: the number of starter words requested (a constant).
- ``active_categories``: category names of live hypotheses (LLM-authored text).
- ``ranked_context``: MAP-Elites only; top-K GUESSED words with their game ranks.
- ``self_report_block``: a fixed instruction string (RQ1); contains no game state
  and no numbers.

Crossover prompt (``build_crossover_prompt``):
- ``a_name`` / ``b_name``: parent category names.
- ``a_words`` / ``b_words``: JSON maps of the two parents' GUESSED words -> ranks.
- ``self_report_block``: same fixed instruction string.

The target word and the sigma vector are never passed to any prompt builder, and
ranks only ever originate from ``words_tried`` / ``best_rank`` / ``ranked_context``,
all of which are guessed words. The sigma-leakage assertion below double-checks
that no sigma literal, "sigma"/"probability" substring, or missing distinguishing
phrase slips in.
"""

from __future__ import annotations

import unittest

import numpy as np

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.llm_client import LLMClient
from contexto_solver.operators import OPERATOR_PROMPTS, OPERATORS, assert_prompt_has_no_sigma_leak
from contexto_solver.self_report import SELF_REPORT_BLOCK

# A realistic mid-run population of guessed words and their ranks.
GUESSED_RANKS = {
    "pearl": 300,
    "amber": 120,
    "tusk": 45,
    "horn": 210,
    "bone": 88,
}
# The hidden target and a distinctive rank value that no guessed word has, so a
# leak of either would be detectable as a literal substring.
TARGET_WORD = "elephant"
TARGET_RANK_TOKEN = "747474"


def _population() -> list[Hypothesis]:
    h1 = Hypothesis(category_name="gemstones", description="precious stones", words_tried={"pearl": 300, "amber": 120})
    h1.set_sigma(np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64))
    h2 = Hypothesis(category_name="animal tusks", description="tusks and horns", words_tried={"tusk": 45, "horn": 210})
    h2.set_sigma(np.asarray([0.1, 0.5, 0.25, 0.15], dtype=np.float64))
    h3 = Hypothesis(category_name="hard materials", description="rigid substances", words_tried={"bone": 88})
    h3.set_sigma(np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64))
    return [h1, h2, h3]


class PromptIsolationTests(unittest.TestCase):
    def _all_prompts(self, self_report_block: str) -> dict[str, str]:
        client = LLMClient(provider="ollama", api_key="ollama", model="test-model")
        population = _population()
        parent = population[1]
        all_guesses = set(GUESSED_RANKS)
        active_categories = [h.category_name for h in population]
        prompts: dict[str, str] = {}
        for operator in OPERATORS:
            prompts[operator.value] = client.build_operator_mutation_prompt(
                OPERATOR_PROMPTS[operator],
                parent,
                all_guesses=all_guesses,
                invalid_guesses={"gemstone"},
                n=3,
                active_categories=active_categories,
                ranked_context="",
                self_report_block=self_report_block,
            )
        prompts["crossover"] = client.build_crossover_prompt(
            population[0].category_name,
            population[1].category_name,
            population[0].words_tried,
            population[1].words_tried,
            self_report_block,
        )
        return prompts

    def _assert_isolated(self, prompts: dict[str, str]) -> None:
        for name, prompt in prompts.items():
            with self.subTest(operator=name):
                # (2) the target word never appears.
                self.assertNotIn(TARGET_WORD, prompt, f"{name} leaked the target word")
                # (1) no rank for an unguessed word (probe: the target's rank).
                self.assertNotIn(TARGET_RANK_TOKEN, prompt, f"{name} leaked an unguessed-word rank")

    def test_isolation_flag_off(self) -> None:
        self._assert_isolated(self._all_prompts(self_report_block=""))

    def test_isolation_flag_on(self) -> None:
        self._assert_isolated(self._all_prompts(self_report_block=SELF_REPORT_BLOCK))

    def test_no_sigma_leak_in_mutation_prompts(self) -> None:
        client = LLMClient(provider="ollama", api_key="ollama", model="test-model")
        parent = _population()[1]
        for block in ("", SELF_REPORT_BLOCK):
            for operator in OPERATORS:
                prompt = client.build_operator_mutation_prompt(
                    OPERATOR_PROMPTS[operator],
                    parent,
                    all_guesses=set(GUESSED_RANKS),
                    invalid_guesses={"gemstone"},
                    n=3,
                    active_categories=["gemstones", "animal tusks"],
                    self_report_block=block,
                )
                assert_prompt_has_no_sigma_leak(prompt, parent.sigma, operator)

    def test_ranked_context_only_contains_guessed_words(self) -> None:
        """MAP-Elites ranked context injects only guessed words with their ranks.

        Rendered from a ``_word_ranks`` tracker of guessed words, so the target
        and its rank cannot appear even when the context feature is enabled.
        """
        from types import SimpleNamespace

        from contexto_solver.methods.ea_llm_map_elites import EALLMMapElitesMethod

        stub = SimpleNamespace(
            config=SimpleNamespace(ranked_context_k=5),
            _word_ranks=dict(GUESSED_RANKS),
        )
        text = EALLMMapElitesMethod._render_ranked_context(stub)  # type: ignore[arg-type]
        self.assertNotIn(TARGET_WORD, text)
        self.assertNotIn(TARGET_RANK_TOKEN, text)
        for word in GUESSED_RANKS:
            self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
