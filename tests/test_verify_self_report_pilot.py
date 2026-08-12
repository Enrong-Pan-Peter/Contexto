"""Checkpoint 4: verification-script metrics over a synthetic trace fixture."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_self_report_pilot import (
    CANONICAL_SELF_REPORT_KEYS,
    compute_metrics,
    evaluate_thresholds,
    extract_records,
    load_trace,
    schema_audit,
)


def _canonical_report(**overrides):
    record = {
        "predicted_closeness": 0.6,
        "predicted_closeness_clamped": False,
        "predicted_bucket": "top100",
        "rationale": {"basis_words": ["shrub"], "reason": "x"},
        "self_report_parse_failed": False,
        "self_report_raw": "{}",
        "self_report_prompt": "has shrub",
    }
    record.update(overrides)
    return record


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

    def test_degenerate_clustering_flag_on_single_value(self) -> None:
        # 20 parsed reports all with the same closeness -> dominant fraction 1.0
        # and zero std, so the clustering flag fires while other checks pass.
        clustered = [
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
        metrics = compute_metrics(extract_records(clustered))
        self.assertTrue(metrics["predicted_closeness_clustered"])
        self.assertEqual(metrics["predicted_closeness_dominant_value"], 0.6)
        self.assertAlmostEqual(metrics["predicted_closeness_dominant_fraction"], 1.0)
        self.assertEqual(metrics["predicted_closeness_std"], 0.0)
        checks = evaluate_thresholds(metrics)
        self.assertTrue(checks["closeness_degenerate_cluster"])

    def test_varied_closeness_does_not_trigger_clustering(self) -> None:
        varied = [
            _op_event(1, "s_mutation", f"h{i}", {
                "predicted_closeness": value,
                "predicted_closeness_clamped": False,
                "rationale": {"basis_words": ["shrub"], "reason": "x"},
                "self_report_parse_failed": False,
                "self_report_raw": "{}",
                "self_report_prompt": "has shrub",
            })
            for i, value in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.35])
        ]
        metrics = compute_metrics(extract_records(varied))
        self.assertFalse(metrics["predicted_closeness_clustered"])
        self.assertFalse(evaluate_thresholds(metrics)["closeness_degenerate_cluster"])

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


class LlmOnlyExtractionTests(unittest.TestCase):
    """llm_only writes self_report under GUESS events (non-Hypothesis helper)."""

    def _llm_only_trace(self):
        return [
            {"generation": -1, "event": "RUN_CONFIG", "details": {"self_report": True, "trace_schema_version": 2}},
            {"generation": 1, "event": "GUESS", "details": {"word": "bush", "rank": 40, "self_report": _canonical_report()}},
            {"generation": 2, "event": "GUESS", "details": {"word": "hedge", "rank": 30, "self_report": _canonical_report(predicted_closeness=0.5)}},
            # a guess without self_report (e.g. flag-off style) must be ignored
            {"generation": 3, "event": "GUESS", "details": {"word": "fern", "rank": 90}},
        ]

    def test_guess_records_are_extracted(self) -> None:
        records = extract_records(self._llm_only_trace())
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["source_event"] == "GUESS" for r in records))
        self.assertEqual(sorted(r["guess_word"] for r in records), ["bush", "hedge"])
        self.assertTrue(all(r["hypothesis_name"] is None for r in records))
        self.assertTrue(all(r["operator"] == "next_guess" for r in records))

    def test_llm_only_metrics_and_thresholds(self) -> None:
        checks = evaluate_thresholds(compute_metrics(extract_records(self._llm_only_trace())))
        self.assertTrue(checks["all_pass"])


class SchemaAuditTests(unittest.TestCase):
    def test_canonical_keys_and_consistent_version_pass(self) -> None:
        ea = [
            {"generation": -1, "event": "RUN_CONFIG", "details": {"trace_schema_version": 2}},
            _op_event(1, "s_mutation", "a", _canonical_report()),
        ]
        llm_only = [
            {"generation": -1, "event": "RUN_CONFIG", "details": {"trace_schema_version": 2}},
            {"generation": 1, "event": "GUESS", "details": {"word": "bush", "self_report": _canonical_report()}},
        ]
        audit = schema_audit([("ea.json", ea), ("llm_only.json", llm_only)])
        # Pre-fix traces (missing the optional injected_rationale_hash /
        # rationale_truncated keys) pass because the check runs after a
        # normalized read that fills defaults.
        self.assertTrue(audit["all_key_sets_canonical"])
        self.assertTrue(audit["schema_version_consistent"])
        self.assertEqual(audit["schema_versions_seen"], ["2"])
        # Raw variants remain informational (7 legacy keys); normalized is canonical.
        self.assertEqual(audit["distinct_key_sets_normalized"], [sorted(CANONICAL_SELF_REPORT_KEYS)])

    def test_divergent_field_names_flagged(self) -> None:
        bad = [
            {"generation": -1, "event": "RUN_CONFIG", "details": {"trace_schema_version": 2}},
            # llm_only serializing a stray/renamed key surfaces after normalization
            # because unknown keys are preserved (not silently dropped).
            {"generation": 1, "event": "GUESS", "details": {"word": "bush", "self_report": _canonical_report(stray_key=0.6)}},
        ]
        audit = schema_audit([("bad.json", bad)])
        self.assertFalse(audit["all_key_sets_canonical"])

    def test_schema_version_mismatch_flagged(self) -> None:
        v2 = [
            {"generation": -1, "event": "RUN_CONFIG", "details": {"trace_schema_version": 2}},
            _op_event(1, "s_mutation", "a", _canonical_report()),
        ]
        v1 = [
            {"generation": -1, "event": "RUN_CONFIG", "details": {"trace_schema_version": 1}},
            {"generation": 1, "event": "GUESS", "details": {"word": "bush", "self_report": _canonical_report()}},
        ]
        audit = schema_audit([("v2.json", v2), ("v1.json", v1)])
        self.assertFalse(audit["schema_version_consistent"])
        self.assertEqual(audit["schema_versions_seen"], ["1", "2"])


if __name__ == "__main__":
    unittest.main()
