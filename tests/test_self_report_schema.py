"""Checkpoint 1: self-report schema and data-model tests."""

from __future__ import annotations

import json
import unittest

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.self_report import (
    clamp_predicted_closeness,
    read_self_report,
)


class SelfReportSchemaTests(unittest.TestCase):
    def test_fresh_hypothesis_omits_self_report(self) -> None:
        """A non-instrumented hypothesis serializes without a self_report key."""
        hypothesis = Hypothesis(category_name="plants", description="green things")
        payload = hypothesis.to_dict()
        self.assertNotIn("self_report", payload)
        self.assertNotIn("coordinates", payload)

    def test_to_dict_round_trip_with_self_report(self) -> None:
        hypothesis = Hypothesis(category_name="plants", description="green things")
        hypothesis.update("shrub", 42)
        hypothesis.predicted_closeness = 0.7
        hypothesis.predicted_closeness_clamped = False
        hypothesis.predicted_bucket = "top100"
        hypothesis.rationale = {"basis_words": ["shrub", "bush"], "reason": "close neighbor."}
        hypothesis.self_report_parse_failed = False
        hypothesis.self_report_raw = '{"predicted_closeness": 0.7}'
        hypothesis.self_report_prompt = "operator prompt text"

        payload = hypothesis.to_dict()
        self.assertIn("self_report", payload)

        # JSON round-trip must preserve the record exactly.
        restored = json.loads(json.dumps(payload))
        report = restored["self_report"]
        report.setdefault("injected_rationale_hash", None)
        report.setdefault("rationale_truncated", False)
        self.assertEqual(report["predicted_closeness"], 0.7)
        self.assertFalse(report["predicted_closeness_clamped"])
        self.assertEqual(report["predicted_bucket"], "top100")
        self.assertEqual(report["rationale"], {"basis_words": ["shrub", "bush"], "reason": "close neighbor."})
        self.assertFalse(report["self_report_parse_failed"])
        self.assertEqual(report["self_report_raw"], '{"predicted_closeness": 0.7}')
        self.assertEqual(report["self_report_prompt"], "operator prompt text")
        self.assertIsNone(report.get("injected_rationale_hash"))
        self.assertFalse(report.get("rationale_truncated"))

        # The tolerant reader recovers the same record (including null defaults).
        self.assertEqual(read_self_report(restored), report)

    def test_parse_failed_record_is_emitted(self) -> None:
        hypothesis = Hypothesis(category_name="plants", description="green things")
        hypothesis.self_report_parse_failed = True
        hypothesis.self_report_raw = "not json at all"
        hypothesis.self_report_prompt = "operator prompt text"
        payload = hypothesis.to_dict()
        self.assertIn("self_report", payload)
        self.assertTrue(payload["self_report"]["self_report_parse_failed"])
        self.assertIsNone(payload["self_report"]["predicted_closeness"])

    def test_read_old_format_trace_returns_nulls(self) -> None:
        """A serialized child from an old trace (no self_report) reads cleanly."""
        old_child = {
            "hypothesis_id": "abc",
            "category_name": "plants",
            "description": "green things",
            "words_tried": {"shrub": 42},
            "best_word": "shrub",
            "best_rank": 42,
        }
        report = read_self_report(old_child)
        self.assertIsNone(report["predicted_closeness"])
        self.assertFalse(report["predicted_closeness_clamped"])
        self.assertIsNone(report["predicted_bucket"])
        self.assertIsNone(report["rationale"])
        self.assertFalse(report["self_report_parse_failed"])
        self.assertIsNone(report["self_report_raw"])
        self.assertIsNone(report["self_report_prompt"])

    def test_read_self_report_handles_non_dicts(self) -> None:
        self.assertIsNone(read_self_report({})["predicted_closeness"])
        self.assertIsNone(read_self_report("nonsense")["predicted_closeness"])  # type: ignore[arg-type]

    def test_clamp_in_range_is_untouched(self) -> None:
        value, clamped = clamp_predicted_closeness(0.5)
        self.assertEqual(value, 0.5)
        self.assertFalse(clamped)

    def test_clamp_above_range(self) -> None:
        value, clamped = clamp_predicted_closeness(1.5)
        self.assertEqual(value, 1.0)
        self.assertTrue(clamped)

    def test_clamp_below_range(self) -> None:
        value, clamped = clamp_predicted_closeness(-0.2)
        self.assertEqual(value, 0.0)
        self.assertTrue(clamped)

    def test_clamp_boundaries_not_flagged(self) -> None:
        self.assertEqual(clamp_predicted_closeness(0.0), (0.0, False))
        self.assertEqual(clamp_predicted_closeness(1.0), (1.0, False))

    def test_clamp_non_numeric_never_crashes(self) -> None:
        for bad in ["abc", None, [], {}, "nan", float("nan"), float("inf")]:
            value, clamped = clamp_predicted_closeness(bad)
            self.assertIsNone(value)
            self.assertFalse(clamped)

    def test_clamp_numeric_string(self) -> None:
        value, clamped = clamp_predicted_closeness("0.8")
        self.assertEqual(value, 0.8)
        self.assertFalse(clamped)


if __name__ == "__main__":
    unittest.main()
