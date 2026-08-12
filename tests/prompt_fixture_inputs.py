"""Canonical inputs for operator/crossover prompt snapshot fixtures.

Both the fixture generator and the byte-identical snapshot test import this so
they render prompts from identical inputs. ``build_prompts`` uses the real code
paths (``build_operator_mutation_prompt`` and ``crossover``) so the snapshot
compares exactly what the solver would send.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.llm_client import LLMClient
from contexto_solver.operators import OPERATOR_PROMPTS, OPERATORS


def make_client() -> LLMClient:
    return LLMClient(provider="ollama", api_key="ollama", model="test-model")


def make_parent() -> Hypothesis:
    parent = Hypothesis(
        category_name="types of plants",
        description="plant taxonomy and related greenery",
        words_tried={"shrub": 50, "tree": 400, "fern": 120},
    )
    parent.set_sigma(np.asarray([0.31, 0.22, 0.18, 0.29], dtype=np.float64))
    return parent


ALL_GUESSES = {"shrub", "tree", "fern", "bush", "hedge"}
INVALID_GUESSES = {"sourcream", "wildanimal"}
ACTIVE_CATEGORIES = ["types of plants", "outdoor environments", "botanical terms"]
N_STARTER = 3


def build_prompts(client: LLMClient, self_report_block: str = "") -> dict[str, str]:
    """Render every operator prompt plus the crossover prompt.

    Passing ``self_report_block=""`` (the default) must reproduce the
    pre-instrumentation prompts byte-for-byte.
    """
    parent = make_parent()
    prompts: dict[str, str] = {}
    for operator in OPERATORS:
        prompts[operator.value] = client.build_operator_mutation_prompt(
            OPERATOR_PROMPTS[operator],
            parent,
            all_guesses=ALL_GUESSES,
            invalid_guesses=INVALID_GUESSES,
            n=N_STARTER,
            active_categories=ACTIVE_CATEGORIES,
            self_report_block=self_report_block,
        )

    captured: dict[str, str] = {}
    original = client._json_request_with_retry

    def _capture(prompt: str) -> Any:
        captured["prompt"] = prompt
        return {}

    client._json_request_with_retry = _capture  # type: ignore[method-assign]
    try:
        parent_b = Hypothesis(
            category_name="outdoor environments",
            description="places and settings outdoors",
            words_tried={"forest": 80, "meadow": 300},
        )
        client.crossover(
            parent.category_name,
            parent_b.category_name,
            parent.words_tried,
            parent_b.words_tried,
            self_report_block=self_report_block,
        )
    finally:
        client._json_request_with_retry = original  # type: ignore[method-assign]
    prompts["crossover"] = captured["prompt"]
    return prompts
