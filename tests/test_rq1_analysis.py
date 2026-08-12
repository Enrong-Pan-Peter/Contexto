"""RQ1 analysis tests: reader join, metrics, mediator, and report oracles.

All fixtures are synthetic traces / hand-built individuals with numbers chosen so
every metric has a hand-computed oracle. No network, LLM, solver, or real trace
files are touched.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contexto_solver.rq1 import metrics as M
from contexto_solver.rq1 import mediator as MED
from contexto_solver.rq1 import report as R
from contexto_solver.rq1.reader import (
    Individual,
    extract_individuals,
    load_trace,
    read_run,
    recover_target,
    run_config,
)
from contexto_solver.rq1.records import provenance_hashes, to_tidy_row, write_tidy_csv, TIDY_COLUMNS


def _sr(pc, bucket, *, parse_failed=False, basis=("a", "b"), injected=None, truncated=False):
    return {
        "predicted_closeness": pc,
        "predicted_closeness_clamped": False,
        "predicted_bucket": bucket,
        "rationale": {"basis_words": list(basis), "reason": "r"},
        "self_report_parse_failed": parse_failed,
        "self_report_raw": "{}",
        "self_report_prompt": "p",
        "injected_rationale_hash": injected,
        "rationale_truncated": truncated,
    }


# A self-adaptive api trace: 2 mutation children + 1 crossover child, solved.
SELF_ADAPTIVE_TRACE = [
    {"generation": -1, "event": "RUN_CONFIG", "details": {
        "game": "api", "method": "ea_llm_self_adaptive", "target": None, "game_number": 999,
        "instrumentation_provenance_hash": "hashA", "trace_schema_version": 3,
        "self_adaptive_sigma_mode": "adaptive"}},
    {"generation": 1, "event": "OPERATOR_SAMPLED", "details": {
        "sampled_op": "s_mutation", "child_hypothesis_name": "alpha",
        "parent_id": "p1", "parent_rank": 5, "self_report": _sr(0.9, "top10")}},
    {"generation": 1, "event": "GUESS", "details": {"word": "ax", "rank": 8, "hypothesis": "alpha"}},
    {"generation": 1, "event": "GUESS", "details": {"word": "axe", "rank": 50, "hypothesis": "alpha"}},
    {"generation": 1, "event": "OPERATOR_SAMPLED", "details": {
        "sampled_op": "m_mutation", "child_hypothesis_name": "beta",
        "parent_id": "p2", "parent_rank": 200, "self_report": _sr(0.2, "beyond", injected="inh1")}},
    {"generation": 1, "event": "SKIP_INVALID_GUESS", "details": {"word": "zzz", "hypothesis": "beta"}},
    {"generation": 1, "event": "GUESS", "details": {"word": "bee", "rank": 400, "hypothesis": "beta"}},
    # crossover: child guesses precede the CROSSOVER event
    {"generation": 1, "event": "GUESS", "details": {"word": "cee", "rank": 90, "hypothesis": "gamma"}},
    {"generation": 1, "event": "GUESS", "details": {"word": "cee2", "rank": 300, "hypothesis": "gamma"}},
    {"generation": 1, "event": "CROSSOVER", "details": {
        "parents": ["alpha", "beta"], "parent_ids": ["p1", "p2"], "parent_ranks": [5, 200],
        "child": {"category_name": "gamma", "best_word": "cee", "best_rank": 90,
                  "words_tried": {"cee": 90, "cee2": 300}, "self_report": _sr(0.6, "top100")}}},
    # MUTATE mirror must NOT double-count (OPERATOR_SAMPLED already seen)
    {"generation": 1, "event": "MUTATE", "details": {
        "method": "self_adaptive", "children": [{"category_name": "alpha"}],
        "self_report": _sr(0.9, "top10")}},
    {"generation": 1, "event": "SOLVED", "details": {"answer": "omega", "rank": 1, "total_guesses": 6}},
]


class ReaderJoinTests(unittest.TestCase):
    def setUp(self):
        self.inds = extract_individuals(SELF_ADAPTIVE_TRACE, trace_file="t.json")
        self.by_name = {i.hypothesis_name: i for i in self.inds}

    def test_counts_and_no_double_count(self):
        # 2 OPERATOR_SAMPLED + 1 CROSSOVER; MUTATE mirror excluded.
        self.assertEqual(len(self.inds), 3)
        self.assertEqual(sorted(i.source_event for i in self.inds),
                         ["CROSSOVER", "OPERATOR_SAMPLED", "OPERATOR_SAMPLED"])

    def test_first_proposed_and_child_best(self):
        alpha = self.by_name["alpha"]
        self.assertEqual((alpha.proposed_word, alpha.realized_rank), ("ax", 8))
        self.assertEqual((alpha.child_best_word, alpha.child_best_rank), ("ax", 8))
        beta = self.by_name["beta"]
        # first evaluated event is an invalid skip -> no realized rank, flagged
        self.assertTrue(beta.proposed_word_invalid)
        self.assertEqual(beta.proposed_word, "zzz")
        self.assertIsNone(beta.realized_rank)
        self.assertEqual((beta.child_best_word, beta.child_best_rank), ("bee", 400))

    def test_crossover_join(self):
        gamma = self.by_name["gamma"]
        self.assertEqual((gamma.proposed_word, gamma.realized_rank), ("cee", 90))
        self.assertEqual(gamma.child_best_rank, 90)
        self.assertEqual(gamma.parent_id, "p1;p2")
        self.assertEqual(gamma.parent_rank, 5)  # min of parent_ranks

    def test_metadata_and_target(self):
        alpha = self.by_name["alpha"]
        self.assertEqual(alpha.target, "omega")
        self.assertEqual(alpha.target_source, "solved")
        self.assertEqual(alpha.operator, "s_mutation")
        self.assertEqual(alpha.parent_rank, 5)
        self.assertFalse(alpha.rationale_inherited)
        self.assertTrue(self.by_name["beta"].rationale_inherited)


class TargetRecoveryTests(unittest.TestCase):
    def test_run_config_target(self):
        events = [{"generation": -1, "event": "RUN_CONFIG", "details": {"target": "ivory"}}]
        self.assertEqual(recover_target(events), ("ivory", "run_config"))

    def test_rank1_guess(self):
        events = [
            {"generation": -1, "event": "RUN_CONFIG", "details": {"game": "api"}},
            {"generation": 1, "event": "GUESS", "details": {"word": "cement", "rank": 1}},
        ]
        self.assertEqual(recover_target(events), ("cement", "rank1_guess"))

    def test_unrecoverable(self):
        events = [{"generation": -1, "event": "RUN_CONFIG", "details": {"game": "api"}}]
        self.assertEqual(recover_target(events), (None, None))


class LlmOnlyTests(unittest.TestCase):
    TRACE = [
        {"generation": -1, "event": "RUN_CONFIG", "details": {
            "game": "api", "method": "llm_only", "instrumentation_provenance_hash": "h", "trace_schema_version": 3}},
        {"generation": 1, "event": "GUESS", "details": {"word": "bush", "rank": 40, "self_report": _sr(0.5, "top100")}},
        {"generation": 2, "event": "GUESS", "details": {"word": "fern", "rank": 90}},  # no self_report -> ignored
    ]

    def test_guess_selfreport(self):
        inds = extract_individuals(self.TRACE, trace_file="t.json")
        self.assertEqual(len(inds), 1)
        rec = inds[0]
        self.assertEqual(rec.source_event, "GUESS")
        self.assertEqual((rec.proposed_word, rec.realized_rank), ("bush", 40))
        self.assertEqual((rec.child_best_word, rec.child_best_rank), ("bush", 40))
        self.assertEqual(rec.operator, "next_guess")


def _ind(pc, best_rank, bucket=None, *, first_rank=None, method="m", operator="s_mutation",
         parent_rank=None, inherited=False):
    return Individual(
        trace_file="t", game="api", method=method, game_number=1, target="omega",
        target_source="solved", provenance_hash="h", trace_schema_version=3, sigma_mode="adaptive",
        generation=1, source_event="OPERATOR_SAMPLED", operator=operator, hypothesis_name="h",
        parent_id="p", parent_rank=parent_rank, predicted_closeness=pc,
        predicted_closeness_clamped=False, predicted_bucket=bucket, self_report_parse_failed=False,
        injected_rationale_hash=("x" if inherited else None), rationale_inherited=inherited,
        rationale_truncated=False, basis_words_count=2,
        proposed_word="w", realized_rank=(best_rank if first_rank is None else first_rank),
        proposed_word_invalid=False, child_best_word="w", child_best_rank=best_rank,
    )


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.inds = extract_individuals(SELF_ADAPTIVE_TRACE, trace_file="t.json")

    def test_spearman_perfect_negative(self):
        # pc=[0.9,0.2,0.6] vs child_best rank=[8,400,90] -> monotone inverse
        result = M.spearman(self.inds, "child_best")
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["rho"], -1.0)

    def test_bucket_metrics(self):
        b = M.bucket_metrics(self.inds, "child_best")
        self.assertEqual(b["n"], 3)
        self.assertAlmostEqual(b["accuracy"], 2 / 3)  # alpha, gamma correct
        self.assertAlmostEqual(b["mean_signed_bucket_error"], 1 / 3)  # beta beyond vs top500 = +1
        self.assertAlmostEqual(b["off_by_one_rate"], 1 / 3)
        self.assertEqual(b["confusion"]["top10"]["top10"], 1)
        self.assertEqual(b["confusion"]["beyond"]["top500"], 1)
        self.assertEqual(b["confusion"]["top100"]["top100"], 1)

    def test_brier_and_positive_rate(self):
        result = M.brier_score(self.inds, "child_best")
        # y=[1,0,1]; pc=[0.9,0.2,0.6]; MSE=(0.01+0.04+0.16)/3
        self.assertAlmostEqual(result["brier"], 0.21 / 3)
        self.assertAlmostEqual(result["positive_rate"], 2 / 3)

    def test_auroc_perfect(self):
        result = M.auroc(self.inds, "child_best")
        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertEqual((result["n_pos"], result["n_neg"]), (2, 1))

    def test_auroc_tie_aware(self):
        inds = [_ind(0.5, 8), _ind(0.5, 400), _ind(0.8, 8), _ind(0.2, 400)]
        # scores [0.5,0.5,0.8,0.2] labels [1,0,1,0] -> AUC 0.875 (avg-rank ties)
        self.assertAlmostEqual(M.auroc(inds, "child_best")["auroc"], 0.875)

    def test_reliability_ece(self):
        inds = [_ind(0.05, 400), _ind(0.15, 8)]  # labels 0,1
        rel = M.reliability_curve(inds, "child_best", n_bins=10)
        self.assertAlmostEqual(rel["ece"], 0.45)
        self.assertEqual(rel["n"], 2)

    def test_realized_agreement(self):
        agree = M.realized_agreement(self.inds)
        # alpha and gamma have first==best; beta excluded (invalid first)
        self.assertEqual(agree["n"], 2)
        self.assertAlmostEqual(agree["equal_rate"], 1.0)

    def test_splits(self):
        out = M.metrics_with_splits(self.inds, "child_best")
        self.assertIn("operator", out["splits"])
        self.assertIn("inheritance", out["splits"])
        self.assertEqual(out["overall"]["count"], 3)


class MediatorTests(unittest.TestCase):
    def setUp(self):
        self.inds = extract_individuals(SELF_ADAPTIVE_TRACE, trace_file="t.json")

    def test_mediator_metrics_first_proposed(self):
        # glove rank == real rank for known words; "zzz" and one word OOV
        glove = {"ax": 8, "cee": 90}  # bee/zzz/beta-first OOV
        metrics = MED.mediator_metrics(self.inds, lambda w: glove.get(w), "first_proposed")
        # words considered: ax(8), zzz(None real -> skip), cee(90)
        self.assertAlmostEqual(metrics["glove_vs_real"]["rho"], 1.0)
        self.assertEqual(metrics["glove_vs_real"]["n"], 2)
        self.assertEqual(metrics["words_in_glove_vocab"], 2)

    def test_glove_ranker(self):
        ranker = MED.GloveRanker({"cement": 1, "beam": 635})
        self.assertEqual(ranker("Cement"), 1)
        self.assertIsNone(ranker("nope"))


class ReportTests(unittest.TestCase):
    def test_summarize_values(self):
        s = R.summarize_values([1, 2, 3, 4])
        self.assertEqual(s["median"], 2.5)
        self.assertAlmostEqual(s["iqr"], 1.5)
        self.assertEqual(s["n"], 4)

    def test_summarize_drops_none(self):
        s = R.summarize_values([None, 2.0, None, 4.0])
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["median"], 3.0)

    def test_runs_needed(self):
        self.assertEqual(R.runs_needed_per_arm(1.0, 1.0), 16)
        self.assertIsNone(R.runs_needed_per_arm(1.0, 0.0))

    def test_two_arm_mannwhitney(self):
        scalars = {
            "a1": {"ece": 0.1}, "a2": {"ece": 0.2}, "a3": {"ece": 0.15},
            "b1": {"ece": 0.4}, "b2": {"ece": 0.5}, "b3": {"ece": 0.45},
        }
        arm_of = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "b3": "B"}
        out = R.two_arm_comparison(scalars, arm_of, "A", "B", metrics=["ece"])
        self.assertEqual(out["metrics"]["ece"]["test"], "mannwhitneyu")
        self.assertEqual(out["metrics"]["ece"]["n_a"], 3)
        self.assertLess(out["metrics"]["ece"]["median_a"], out["metrics"]["ece"]["median_b"])

    def test_two_arm_wilcoxon_paired(self):
        scalars = {
            "a1": {"brier": 0.30}, "a2": {"brier": 0.20}, "a3": {"brier": 0.25},
            "b1": {"brier": 0.40}, "b2": {"brier": 0.28}, "b3": {"brier": 0.35},
        }
        arm_of = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "b3": "B"}
        paired = {"a1": 1, "b1": 1, "a2": 2, "b2": 2, "a3": 3, "b3": 3}
        out = R.two_arm_comparison(scalars, arm_of, "A", "B", paired_key=paired, metrics=["brier"])
        self.assertEqual(out["metrics"]["brier"]["test"], "wilcoxon")
        self.assertEqual(out["metrics"]["brier"]["n_pairs"], 3)

    def test_variance_table_power_gate(self):
        scalars = {"a": {"ece": 0.1}, "b": {"ece": 0.2}, "c": {"ece": 0.3}}
        table = R.variance_table(scalars, effect=0.1, metrics=["ece"])
        self.assertEqual(table["ece"]["n_runs"], 3)
        self.assertIn("runs_needed_per_arm", table["ece"])


class RecordsTests(unittest.TestCase):
    def test_tidy_row_columns(self):
        inds = extract_individuals(SELF_ADAPTIVE_TRACE, trace_file="t.json")
        row = to_tidy_row(inds[0])
        self.assertEqual(set(row.keys()), set(TIDY_COLUMNS))

    def test_provenance_guard(self):
        a = _ind(0.5, 10)
        b = _ind(0.5, 10)
        object.__setattr__(b, "provenance_hash", "other")
        report = provenance_hashes([a, b])
        self.assertTrue(report.mixed)
        self.assertEqual(len(report.hashes), 2)

    def test_csv_round_trip(self):
        inds = extract_individuals(SELF_ADAPTIVE_TRACE, trace_file="t.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tidy.csv"
            write_tidy_csv(path, inds)
            text = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(text[0], ",".join(TIDY_COLUMNS))
            self.assertEqual(len(text), 1 + len(inds))


class LoadTraceTests(unittest.TestCase):
    def test_summary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps({"runs": [{"trace_path": "x.json"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trace(path)

    def test_read_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps(SELF_ADAPTIVE_TRACE), encoding="utf-8")
            config, inds = read_run(path)
            self.assertEqual(config.method, "ea_llm_self_adaptive")
            self.assertEqual(len(inds), 3)


if __name__ == "__main__":
    unittest.main()
