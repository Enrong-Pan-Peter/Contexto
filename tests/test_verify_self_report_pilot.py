"""Checkpoint 4: verification-script metrics over a synthetic trace fixture."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_self_report_pilot import (
    compute_metrics,
    evaluate_thresholds,
    extract_records,
    load_trace,
)


def _op_event(generation, op, name, self_report):
    return {
        "generation": generation,
        "event": "OPERATOR_SAMPLED",
        "details": {"sampled_op": op, "child_hypothesis_name": name, "parent_id": "p", "self_report": self_report},
    }


SYNTHETIC_EVENTS = [
    {"generation": -1, "event": "RUN_CONFIG", "details": {"self_report": True, "trace_schema_version": 2}},
    # good record
    _op_event(1, "s_mutation", "woody plants", {
        "predicted_closeness": 0.7,
        "predicted_closeness_clamped": False,
        "rationale": {"basis_words": ["shrub"], "reason": "close neighbor."},
        "self_report_parse_failed": False,
        "self_report_raw": '{"predicted_closeness": 0.7}',
        "self_report_prompt": "context includes shrub and others",
    }),
    # clamped record
    _op_event(1, "m_mutation", "garden uses", {
        "predicted_closeness": 1.0,
        "predicted_closeness_clamped": True,
        "rationale": {"basis_words": ["tree"], "reason": "guess."},
        "self_report_parse_failed": False,
        "self_report_raw": '{"predicted_closeness": 1.7}',
        "self_report_prompt": "context includes tree",
    }),
    # parse-failed record
    _op_event(2, "ml_mutation", "growth habits", {
        "predicted_closeness": None,
        "predicted_closeness_clamped": False,
        "rationale": None,
        "self_report_parse_failed": True,
        "self_report_raw": "not json",
        "self_report_prompt": "context includes clump",
    }),
    # parsed but empty basis_words
    _op_event(2, "l_mutation", "terrain", {
        "predicted_closeness": 0.5,
        "predicted_closeness_clamped": False,
        "rationale": {"basis_words": [], "reason": "unsure."},
        "self_report_parse_failed": False,
        "self_report_raw": '{"predicted_closeness": 0.5}',
        "self_report_prompt": "context includes soil",
    }),
    # a MUTATE mirror of the good record -> must NOT be double counted
    {
        "generation": 1,
        "event": "MUTATE",
        "details": {
            "method": "self_adaptive",
            "children": [
                {"category_name": "woody plants", "self_report": {"predicted_closeness": 0.7}},
            ],
        },
    },
    # crossover record with a basis word missing from the stored prompt
    {
        "generation": 3,
        "event": "CROSSOVER",
        "details": {
            "parents": ["gemstones", "tusks"],
            "parent_ids": ["a", "b"],
            "child": {
                "best_word": "grove",
                "self_report": {
                    "predicted_closeness": 0.8,
                    "predicted_closeness_clamped": False,
                    "rationale": {"basis_words": ["forest", "meadow"], "reason": "blend."},
                    "self_report_parse_failed": False,
                    "self_report_raw": "{}",
                    "self_report_prompt": "context includes forest only",
                },
            },
        },
    },
    # old-style OPERATOR_SAMPLED without a self_report -> ignored
    {"generation": 4, "event": "OPERATOR_SAMPLED", "details": {"sampled_op": "s_mutation", "child_id": "z"}},
    {"generation": 4, "event": "GUESS", "details": {"word": "grove", "rank": 20}},
]


class VerifyPilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = extract_records(SYNTHETIC_EVENTS)

    def test_record_extraction_counts_only_proposals(self) -> None:
        # 4 OPERATOR_SAMPLED (with self_report) + 1 CROSSOVER = 5; MUTATE mirror and
        # the old-style OPERATOR_SAMPLED are excluded.
        self.assertEqual(len(self.records), 5)
        sources = sorted(r["source_event"] for r in self.records)
        self.assertEqual(sources, ["CROSSOVER", "OPERATOR_SAMPLED", "OPERATOR_SAMPLED", "OPERATOR_SAMPLED", "OPERATOR_SAMPLED"])

    def test_metrics(self) -> None:
        metrics = compute_metrics(self.records)
        self.assertEqual(metrics["total_proposals"], 5)
        self.assertEqual(metrics["parse_failure_count"], 1)
        self.assertAlmostEqual(metrics["parse_failure_rate"], 0.2)
        self.assertEqual(metrics["predicted_closeness_present"], 4)
        self.assertEqual(metrics["predicted_closeness_min"], 0.5)
        self.assertEqual(metrics["predicted_closeness_max"], 1.0)
        self.assertAlmostEqual(metrics["predicted_closeness_mean"], 0.75)
        self.assertTrue(metrics["predicted_closeness_all_in_range"])
        self.assertEqual(metrics["clamped_count"], 1)
        self.assertEqual(metrics["empty_basis_words_count"], 2)
        self.assertEqual(metrics["parsed_count"], 4)
        self.assertEqual(metrics["parsed_with_nonempty_basis"], 3)
        self.assertAlmostEqual(metrics["parsed_basis_nonempty_rate"], 0.75)
        self.assertEqual(len(metrics["basis_membership_violations"]), 1)
        self.assertEqual(metrics["basis_membership_violations"][0]["missing_basis_words"], ["meadow"])

    def test_thresholds(self) -> None:
        checks = evaluate_thresholds(compute_metrics(self.records))
        # 20% parse failure fails the hard max and needs discussion.
        self.assertFalse(checks["parse_failure_within_hard_max"])
        self.assertTrue(checks["parse_failure_needs_discussion"])
        self.assertFalse(checks["basis_membership_clean"])
        self.assertFalse(checks["all_pass"])

    def test_load_trace_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(SYNTHETIC_EVENTS), encoding="utf-8")
            events = load_trace(path)
            self.assertEqual(len(extract_records(events)), 5)

    def test_clean_pilot_passes_thresholds(self) -> None:
        clean = [
            _op_event(1, "s_mutation", "a", {
                "predicted_closeness": 0.6,
                "predicted_closeness_clamped": False,
                "rationale": {"basis_words": ["shrub"], "reason": "x"},
                "self_report_parse_failed": False,
                "self_report_raw": "{}",
                "self_report_prompt": "has shrub",
            })
            for _ in range(20)
        ]
        checks = evaluate_thresholds(compute_metrics(extract_records(clean)))
        self.assertTrue(checks["all_pass"])
        self.assertFalse(checks["parse_failure_needs_discussion"])


if __name__ == "__main__":
    unittest.main()
