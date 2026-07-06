"""One-off generator for the pre-instrumentation prompt fixtures.

Run BEFORE editing the prompt templates so the captured fixtures represent the
current (flag-off equivalent) prompts. The snapshot test then proves the edited
code still renders these byte-for-byte when the self_report flag is off.

    python -m tests._generate_baseline_fixtures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.llm_client import LLMClient
from contexto_solver.operators import OPERATOR_PROMPTS, OPERATORS

from tests.prompt_fixture_inputs import (
    ACTIVE_CATEGORIES,
    ALL_GUESSES,
    INVALID_GUESSES,
    N_STARTER,
    make_parent,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prompts_baseline"


def build_current_prompts(client: LLMClient) -> dict[str, str]:
    """Render prompts using the CURRENT signatures (no self_report_block)."""
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
        )
    finally:
        client._json_request_with_retry = original  # type: ignore[method-assign]
    prompts["crossover"] = captured["prompt"]
    return prompts


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    client = LLMClient(provider="ollama", api_key="ollama", model="test-model")
    for name, prompt in build_current_prompts(client).items():
        path = FIXTURE_DIR / f"{name}.txt"
        path.write_text(prompt, encoding="utf-8", newline="")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
