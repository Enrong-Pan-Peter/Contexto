"""Phase 1: byte-identity snapshots for legacy variation-proposal prompts.

Locks the flag-off prompts of the non-operator modes (llm_only, ea_llm,
ea_llm_pivot) before the shared self-report layer is introduced, so the
refactor can be proven to leave them byte-identical.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from contexto_solver.self_report import SELF_REPORT_BLOCK
from tests.legacy_prompt_fixture_inputs import (
    LEGACY_INSTRUMENTED_PROMPT_NAMES,
    LEGACY_PROMPT_NAMES,
    LEGACY_UNINSTRUMENTED_PROMPT_NAMES,
    build_legacy_prompts,
    make_client,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prompts_baseline_legacy"


class LegacyPromptSnapshotTests(unittest.TestCase):
    def test_flag_off_legacy_prompts_are_byte_identical(self) -> None:
        client = make_client()
        prompts = build_legacy_prompts(client)
        for name in LEGACY_PROMPT_NAMES:
            with self.subTest(prompt=name):
                expected = (FIXTURE_DIR / f"{name}.txt").read_text(encoding="utf-8")
                self.assertEqual(prompts[name], expected, f"{name} prompt drifted from baseline")

    def test_flag_on_appends_only_the_block_for_instrumented_prompts(self) -> None:
        """Phase 3: with the flag on, instrumented legacy prompts equal the
        baseline plus exactly the shared block; uninstrumented ones are unchanged."""
        client = make_client()
        prompts = build_legacy_prompts(client, self_report_block=SELF_REPORT_BLOCK)
        for name in LEGACY_INSTRUMENTED_PROMPT_NAMES:
            with self.subTest(prompt=name):
                expected = (FIXTURE_DIR / f"{name}.txt").read_text(encoding="utf-8")
                self.assertEqual(prompts[name], expected + SELF_REPORT_BLOCK)
        for name in LEGACY_UNINSTRUMENTED_PROMPT_NAMES:
            with self.subTest(prompt=name):
                expected = (FIXTURE_DIR / f"{name}.txt").read_text(encoding="utf-8")
                self.assertEqual(prompts[name], expected, f"{name} must be uninstrumented")


if __name__ == "__main__":
    unittest.main()
