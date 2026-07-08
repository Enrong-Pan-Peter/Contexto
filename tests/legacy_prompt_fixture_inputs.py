"""Canonical inputs for the LEGACY variation-proposal prompt snapshots.

These cover the proposal call types of the modes that are NOT built on the
s/m/ml/l operator prompts:

- ``llm_only``          -> ``next_guess``
- ``ea_llm``            -> ``specialize`` (crossover is shared with the operator
                          family and already captured under ``crossover.txt``)
- ``ea_llm_pivot``      -> ``specialize`` + ``pivot_morphology`` /
                          ``pivot_register_shift`` / ``pivot_adjacent_category`` /
                          ``pivot_fresh_adjacent_category``

Both the fixture generator and the snapshot test import this so they render from
identical inputs, using the real ``LLMClient`` code paths. The constructed
prompt is captured by stubbing ``_json_request_with_retry`` (every legacy
proposal method funnels its prompt through it, directly or via
``_request_json_list``).
"""

from __future__ import annotations

from typing import Any

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.llm_client import LLMClient

WORD = "shrub"
RANK = 50
ALL_GUESSES = {"shrub", "tree", "fern", "bush", "hedge"}
INVALID_GUESSES = {"sourcream", "wildanimal"}
ACTIVE_CATEGORIES = ["types of plants", "outdoor environments", "botanical terms"]
HISTORY = {"shrub": 50, "tree": 400}
WORDS_TRIED = {"shrub": 50, "tree": 400, "fern": 120}
CATEGORY_NAME = "types of plants"
CATEGORY_DESCRIPTION = "plant taxonomy and related greenery"
N_STARTER = 3
N_PIVOT = 10


def make_client() -> LLMClient:
    return LLMClient(provider="ollama", api_key="ollama", model="test-model")


def _make_hypothesis() -> Hypothesis:
    return Hypothesis(
        category_name=CATEGORY_NAME,
        description=CATEGORY_DESCRIPTION,
        words_tried=dict(WORDS_TRIED),
    )


# Legacy proposal call types that accept the self-report request block (i.e. are
# instrumented). ``propose_words`` and ``local_search`` are NOT instrumented and
# take no block, so a flag-on run leaves them byte-identical to flag-off.
LEGACY_INSTRUMENTED_PROMPT_NAMES = [
    "next_guess",
    "specialize",
    "pivot_morphology",
    "pivot_register_shift",
    "pivot_adjacent_category",
    "pivot_fresh_adjacent_category",
]
LEGACY_UNINSTRUMENTED_PROMPT_NAMES = ["propose_words", "local_search"]


def build_legacy_prompts(client: LLMClient, self_report_block: str = "") -> dict[str, str]:
    """Render every legacy variation-proposal prompt via the real call paths.

    ``self_report_block`` is threaded only into the instrumented methods (those
    that accept it); the uninstrumented ``propose_words`` / ``local_search`` never
    receive it, matching how the solver runs them.
    """
    captured: dict[str, str] = {}
    original = client._json_request_with_retry_and_raw

    def _capture(prompt: str) -> Any:
        captured["prompt"] = prompt
        # Every proposal method funnels its prompt through
        # ``_json_request_with_retry_and_raw`` (directly, via
        # ``_json_request_with_retry``, or via ``_request_json_list``). A
        # category-like dict doubles as a {"words": [...]} / {"specializations":
        # [...]} wrapper, so list- and dict-returning methods are all satisfied.
        return {"name": "x", "description": "d", "words": ["a"]}, "{}"

    prompts: dict[str, str] = {}
    client._json_request_with_retry_and_raw = _capture  # type: ignore[method-assign]
    try:
        hypothesis = _make_hypothesis()

        client.next_guess(HISTORY, INVALID_GUESSES, self_report_block=self_report_block)
        prompts["next_guess"] = captured["prompt"]

        client.propose_words(hypothesis, dict(HISTORY), invalid_guesses=INVALID_GUESSES, n=N_STARTER)
        prompts["propose_words"] = captured["prompt"]

        client.specialize(
            hypothesis,
            dict(HISTORY),
            invalid_guesses=INVALID_GUESSES,
            n=N_STARTER,
            self_report_block=self_report_block,
        )
        prompts["specialize"] = captured["prompt"]

        client.local_search(WORD, RANK, n=N_STARTER, all_guesses=set(ALL_GUESSES))
        prompts["local_search"] = captured["prompt"]

        client.pivot_morphology(
            WORD, RANK, set(ALL_GUESSES), n=N_PIVOT, self_report_block=self_report_block
        )
        prompts["pivot_morphology"] = captured["prompt"]

        client.pivot_register_shift(
            WORD, RANK, set(ALL_GUESSES), n=N_PIVOT, self_report_block=self_report_block
        )
        prompts["pivot_register_shift"] = captured["prompt"]

        client.pivot_adjacent_category(
            WORD,
            RANK,
            CATEGORY_NAME,
            CATEGORY_DESCRIPTION,
            dict(WORDS_TRIED),
            set(ALL_GUESSES),
            n=N_PIVOT,
            self_report_block=self_report_block,
        )
        prompts["pivot_adjacent_category"] = captured["prompt"]

        client.pivot_fresh_adjacent_category(
            WORD,
            RANK,
            list(ACTIVE_CATEGORIES),
            set(ALL_GUESSES),
            n=N_PIVOT,
            self_report_block=self_report_block,
        )
        prompts["pivot_fresh_adjacent_category"] = captured["prompt"]
    finally:
        client._json_request_with_retry_and_raw = original  # type: ignore[method-assign]
    return prompts


LEGACY_PROMPT_NAMES = [
    "next_guess",
    "propose_words",
    "specialize",
    "local_search",
    "pivot_morphology",
    "pivot_register_shift",
    "pivot_adjacent_category",
    "pivot_fresh_adjacent_category",
]
