"""Checkpoint 2: prompt byte-identity, parser behavior, and live smoke."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import requests

from contexto_solver import config
from contexto_solver.llm_client import LLMClient
from contexto_solver.self_report import (
    SELF_REPORT_BLOCK,
    parse_self_report,
)

from tests.prompt_fixture_inputs import build_prompts, make_client

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "prompts_baseline"
OPERATOR_NAMES = ["s_mutation", "m_mutation", "ml_mutation", "l_mutation", "crossover"]


class PromptSnapshotTests(unittest.TestCase):
    def test_flag_off_prompts_are_byte_identical(self) -> None:
        client = make_client()
        prompts = build_prompts(client, self_report_block="")
        for name in OPERATOR_NAMES:
            expected = (FIXTURE_DIR / f"{name}.txt").read_text(encoding="utf-8")
            self.assertEqual(prompts[name], expected, f"{name} prompt drifted from baseline")

    def test_flag_on_appends_only_the_block(self) -> None:
        client = make_client()
        prompts = build_prompts(client, self_report_block=SELF_REPORT_BLOCK)
        for name in OPERATOR_NAMES:
            expected = (FIXTURE_DIR / f"{name}.txt").read_text(encoding="utf-8")
            self.assertEqual(prompts[name], expected + SELF_REPORT_BLOCK)


class ParserTests(unittest.TestCase):
    def test_valid_json_dict(self) -> None:
        report = parse_self_report(
            {"predicted_closeness": 0.6, "basis_words": ["shrub"], "reason": "close neighbor."}
        )
        self.assertEqual(report["predicted_closeness"], 0.6)
        self.assertFalse(report["predicted_closeness_clamped"])
        self.assertEqual(report["rationale"], {"basis_words": ["shrub"], "reason": "close neighbor."})
        self.assertFalse(report["self_report_parse_failed"])

    def test_json_embedded_in_prose(self) -> None:
        raw = 'Sure! {"predicted_closeness": 0.4, "basis_words": ["bush"], "reason": "similar."} done.'
        report = parse_self_report(raw)
        self.assertEqual(report["predicted_closeness"], 0.4)
        self.assertEqual(report["rationale"]["basis_words"], ["bush"])
        self.assertFalse(report["self_report_parse_failed"])

    def test_missing_predicted_closeness(self) -> None:
        report = parse_self_report({"basis_words": ["bush"], "reason": "similar."})
        self.assertIsNone(report["predicted_closeness"])
        self.assertEqual(report["rationale"]["basis_words"], ["bush"])
        self.assertFalse(report["self_report_parse_failed"])

    def test_out_of_range_is_clamped(self) -> None:
        report = parse_self_report({"predicted_closeness": 1.7, "basis_words": [], "reason": ""})
        self.assertEqual(report["predicted_closeness"], 1.0)
        self.assertTrue(report["predicted_closeness_clamped"])
        self.assertFalse(report["self_report_parse_failed"])

    def test_negative_is_clamped(self) -> None:
        report = parse_self_report({"predicted_closeness": -3, "basis_words": [], "reason": ""})
        self.assertEqual(report["predicted_closeness"], 0.0)
        self.assertTrue(report["predicted_closeness_clamped"])

    def test_malformed_json_text(self) -> None:
        report = parse_self_report("{not valid json")
        self.assertIsNone(report["predicted_closeness"])
        self.assertIsNone(report["rationale"])
        self.assertTrue(report["self_report_parse_failed"])

    def test_empty_response(self) -> None:
        report = parse_self_report("")
        self.assertIsNone(report["predicted_closeness"])
        self.assertTrue(report["self_report_parse_failed"])

    def test_basis_words_non_list_becomes_empty(self) -> None:
        report = parse_self_report({"predicted_closeness": 0.5, "basis_words": "shrub", "reason": 12})
        self.assertEqual(report["rationale"]["basis_words"], [])
        self.assertEqual(report["rationale"]["reason"], "")
        self.assertEqual(report["predicted_closeness"], 0.5)


def _ollama_available() -> bool:
    if os.getenv("RUN_OLLAMA_SMOKE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    base = config.OLLAMA_BASE_URL.rstrip("/")
    try:
        response = requests.get(f"{base}/models", timeout=5)
        return response.status_code < 500
    except requests.RequestException:
        return False


@unittest.skipUnless(_ollama_available(), "Set RUN_OLLAMA_SMOKE=1 with a reachable Ollama server to run live smoke")
class LiveSmokeTests(unittest.TestCase):
    def test_each_operator_returns_parseable_self_report(self) -> None:
        client = LLMClient(provider="ollama", api_key="ollama", model=config.OLLAMA_MODEL)
        prompts = build_prompts(client, self_report_block=SELF_REPORT_BLOCK)
        for name in OPERATOR_NAMES:
            with self.subTest(operator=name):
                parsed, raw = client.complete_json_prompt_with_raw(prompts[name])
                report = parse_self_report(parsed)
                print(f"\n=== {name} raw response ===\n{raw}\n")
                self.assertIsInstance(parsed, dict)
                # A real response should carry a usable closeness or at least parse.
                self.assertFalse(report["self_report_parse_failed"])


if __name__ == "__main__":
    unittest.main()
