"""Tests for the RQ3 lineage analysis with hand-computed synthetic fixtures.

Fixture A exercises every label path (survivor, culled, resurrection, dedup
exclusion, final-generation exclusion, name-collision exclusion) with
hand-computed agreement errors and matched gaps. Two permutation fixtures pin
the null behavior: one where the null is true (identical predictions -> gap 0,
p = 1.0 exactly) and one where it is false (predictions perfectly aligned with
survival -> extreme gap, small p).
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from scripts.rq3_lineage import (
    EXCLUSION_COLLISION,
    EXCLUSION_DEDUP,
    EXCLUSION_FINAL_GEN,
    extract_lineage,
    lineage_pairs,
    matched_gap,
    permutation_gap_test,
    transmission_test,
)


def _run_config_event() -> dict[str, Any]:
    return {
        "generation": -1,
        "event": "RUN_CONFIG",
        "details": {
            "game": "api",
            "game_number": 7,
            "method": "ea_llm_self_adaptive",
            "self_report": True,
            "rationale_inheritance": True,
            "instrumentation_provenance_hash": "testhash",
        },
    }


def _child(
    generation: int,
    child_id: str,
    name: str,
    closeness: float | None,
    bucket: str | None,
    parent_rank: int = 50,
    parent_id: str | None = None,
) -> dict[str, Any]:
    return {
        "generation": generation,
        "event": "OPERATOR_SAMPLED",
        "details": {
            "parent_id": parent_id or f"parent-of-{child_id}",
            "child_id": child_id,
            "parent_rank": parent_rank,
            "sigma_snapshot": [0.25, 0.25, 0.25, 0.25],
            "child_sigma": [0.25, 0.25, 0.25, 0.25],
            "child_hypothesis_name": name,
            "sampled_op": "s_mutation",
            "method": "self_adaptive",
            "self_report": {
                "predicted_closeness": closeness,
                "predicted_closeness_clamped": False,
                "predicted_bucket": bucket,
                "rationale": {"basis_words": ["x"], "reason": "r"},
                "self_report_parse_failed": False,
                "self_report_raw": "{}",
                "self_report_prompt": None,
                "injected_rationale_hash": None,
                "rationale_truncated": False,
            },
        },
    }


def _guess(generation: int, name: str, word: str, rank: int) -> dict[str, Any]:
    return {
        "generation": generation,
        "event": "GUESS",
        "details": {"word": word, "rank": rank, "hypothesis": name, "best_word": word, "best_rank": rank, "total_guesses": 1},
    }


def _select(generation: int, kept: list[str], discarded: list[str]) -> dict[str, Any]:
    return {
        "generation": generation,
        "event": "SELECT",
        "details": {"kept": kept, "discarded": discarded, "elite": kept[0] if kept else None},
    }


def _dedup(generation: int, merged: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "generation": generation,
        "event": "DEDUPLICATE",
        "details": {"merged": [{"survivor": survivor, "discarded": discarded} for survivor, discarded in merged]},
    }


def _fixture_a() -> list[dict[str, Any]]:
    """Hand-checkable lineage covering every label path.

    Generation 1 births (parent_rank 50 -> bin "11-100"):
      A: closeness 0.9, bucket top100, realized 50  -> binary_error 0.1, bucket_distance 0
      B: closeness 0.2, bucket top500, realized 500 -> binary_error 0.2, bucket_distance 0
      C: closeness 0.6, bucket top100, realized 800 -> binary_error 0.6, bucket_distance 2
      D: dedup-merged away in generation 1 -> excluded
      F/F: two same-name births -> both excluded (collision)
    SELECT gen 2: kept [A], discarded [B, C] -> A survivor; B, C culled.
    SELECT gen 3: kept [B] -> B resurrected (label stays culled).
    E born generation 3 (no SELECT gen 4) -> excluded final-generation.
    """
    return [
        _run_config_event(),
        _child(1, "a", "alpha", 0.9, "top100"),
        _guess(1, "alpha", "worda", 50),
        _child(1, "b", "beta", 0.2, "top500"),
        _guess(1, "beta", "wordb", 500),
        _child(1, "c", "gamma", 0.6, "top100"),
        _guess(1, "gamma", "wordc", 800),
        _child(1, "d", "delta", 0.5, "top100"),
        _guess(1, "delta", "wordd", 90),
        _child(1, "f1", "fern", 0.5, "top100"),
        _child(1, "f2", "fern", 0.5, "top100"),
        _dedup(1, [("alpha", "delta")]),
        _select(2, ["alpha"], ["beta", "gamma"]),
        _select(3, ["beta"], ["alpha"]),
        _child(3, "e", "epsilon", 0.5, "top100"),
        _guess(3, "epsilon", "worde", 60),
    ]


class LabelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.individuals, self.meta = extract_lineage(_fixture_a(), "fixture_a.json")
        self.by_id = {individual.child_id: individual for individual in self.individuals}

    def test_counts(self) -> None:
        counts = self.meta["counts"]
        self.assertEqual(counts["individuals"], 7)
        self.assertEqual(counts["survivor"], 1)
        self.assertEqual(counts["culled"], 2)
        self.assertEqual(counts[EXCLUSION_DEDUP], 1)
        self.assertEqual(counts[EXCLUSION_FINAL_GEN], 1)
        self.assertEqual(counts[EXCLUSION_COLLISION], 2)
        self.assertEqual(counts["resurrections"], 1)

    def test_labels_decided_by_first_select_only(self) -> None:
        self.assertEqual(self.by_id["a"].label, "survivor")
        self.assertEqual(self.by_id["b"].label, "culled")
        self.assertEqual(self.by_id["c"].label, "culled")
        # B reappears in SELECT gen 3 kept: resurrected, but the label is unchanged.
        self.assertTrue(self.by_id["b"].resurrected)
        self.assertFalse(self.by_id["a"].resurrected)
        self.assertFalse(self.by_id["c"].resurrected)

    def test_exclusion_reasons(self) -> None:
        self.assertEqual(self.by_id["d"].exclusion_reason, EXCLUSION_DEDUP)
        self.assertEqual(self.by_id["e"].exclusion_reason, EXCLUSION_FINAL_GEN)
        self.assertEqual(self.by_id["f1"].exclusion_reason, EXCLUSION_COLLISION)
        self.assertEqual(self.by_id["f2"].exclusion_reason, EXCLUSION_COLLISION)

    def test_hand_computed_agreement_errors(self) -> None:
        self.assertAlmostEqual(self.by_id["a"].binary_error, 0.1)
        self.assertAlmostEqual(self.by_id["b"].binary_error, 0.2)
        self.assertAlmostEqual(self.by_id["c"].binary_error, 0.6)
        self.assertEqual(self.by_id["a"].bucket_distance, 0.0)
        self.assertEqual(self.by_id["b"].bucket_distance, 0.0)
        self.assertEqual(self.by_id["c"].bucket_distance, 2.0)


class MatchedGapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.individuals, _meta = extract_lineage(_fixture_a(), "fixture_a.json")

    def test_binary_gap_hand_computed(self) -> None:
        # survivor mean 0.1; culled mean (0.2 + 0.6) / 2 = 0.4; gap = -0.3.
        gap, rows = matched_gap(self.individuals, "binary_error")
        self.assertAlmostEqual(gap, -0.3)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["parent_rank_bin"], "11-100")
        self.assertEqual(row["n_survivor"], 1)
        self.assertEqual(row["n_culled"], 2)
        self.assertAlmostEqual(row["mean_survivor"], 0.1)
        self.assertAlmostEqual(row["mean_culled"], 0.4)

    def test_bucket_gap_hand_computed(self) -> None:
        # survivor mean 0; culled mean (0 + 2) / 2 = 1; gap = -1.0.
        gap, _rows = matched_gap(self.individuals, "bucket_distance")
        self.assertAlmostEqual(gap, -1.0)


def _permutation_fixture(survivor_closeness: float, culled_closeness: float) -> list[dict[str, Any]]:
    """4 survivors + 4 culled in one (generation, bin) cell, all realized rank 50."""
    events: list[dict[str, Any]] = [_run_config_event()]
    survivors = [f"s{i}" for i in range(4)]
    culled = [f"c{i}" for i in range(4)]
    for name in survivors:
        events.append(_child(1, name, name, survivor_closeness, "top100"))
        events.append(_guess(1, name, "word" + name, 50))
    for name in culled:
        events.append(_child(1, name, name, culled_closeness, "top100"))
        events.append(_guess(1, name, "word" + name, 50))
    events.append(_select(2, survivors, culled))
    return events


class PermutationNullTests(unittest.TestCase):
    def test_null_true_gives_zero_gap_and_p_one(self) -> None:
        # Identical predictions everywhere: observed gap 0; every permutation
        # reproduces gap 0, so |perm| >= |obs| always and p = 1.0 exactly.
        individuals, _meta = extract_lineage(_permutation_fixture(0.5, 0.5), "null_true.json")
        result = permutation_gap_test(individuals, "binary_error", permutations=200, rng=np.random.default_rng(0))
        self.assertEqual(result["observed_gap"], 0.0)
        self.assertEqual(result["pvalue"], 1.0)
        self.assertEqual(result["n_pool"], 8)

    def test_null_false_gives_extreme_gap_and_small_p(self) -> None:
        # Survivors perfectly right (error 0), culled perfectly wrong (error 1):
        # observed gap -1.0. Under within-cell permutation |gap| = 1 only when
        # the four 1.0 predictions land exactly on one side: 2 / C(8,4) ~ 0.029.
        individuals, _meta = extract_lineage(_permutation_fixture(1.0, 0.0), "null_false.json")
        result = permutation_gap_test(individuals, "binary_error", permutations=1000, rng=np.random.default_rng(0))
        self.assertEqual(result["observed_gap"], -1.0)
        self.assertLess(result["pvalue"], 0.05)


class TransmissionTests(unittest.TestCase):
    def _fixture(self) -> list[dict[str, Any]]:
        # Parent errors 0.1/0.3/0.5 and child errors 0.2/0.4/0.6 (all rank 50,
        # y = 1) are co-monotone along lineage links -> Spearman rho = 1.0.
        events: list[dict[str, Any]] = [_run_config_event()]
        parent_closeness = {"p1": 0.9, "p2": 0.7, "p3": 0.5}
        child_closeness = {"q1": 0.8, "q2": 0.6, "q3": 0.4}
        for parent_id, closeness in parent_closeness.items():
            events.append(_child(1, parent_id, "name" + parent_id, closeness, "top100"))
            events.append(_guess(1, "name" + parent_id, "word" + parent_id, 50))
        for index, (child_id, closeness) in enumerate(child_closeness.items(), start=1):
            events.append(_child(2, child_id, "name" + child_id, closeness, "top100", parent_id=f"p{index}"))
            events.append(_guess(2, "name" + child_id, "word" + child_id, 50))
        return events

    def test_pairs_follow_parent_id_links(self) -> None:
        individuals, _meta = extract_lineage(self._fixture(), "trans.json")
        pairs = lineage_pairs(individuals)
        self.assertEqual(
            sorted((parent.child_id, child.child_id) for parent, child in pairs),
            [("p1", "q1"), ("p2", "q2"), ("p3", "q3")],
        )

    def test_observed_rho_hand_computed(self) -> None:
        individuals, _meta = extract_lineage(self._fixture(), "trans.json")
        result = transmission_test(individuals, "binary_error", permutations=10, rng=np.random.default_rng(0))
        self.assertEqual(result["n_pairs"], 3)
        self.assertAlmostEqual(result["observed_rho"], 1.0)


if __name__ == "__main__":
    unittest.main()
