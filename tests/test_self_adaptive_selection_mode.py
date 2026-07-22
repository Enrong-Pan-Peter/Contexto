"""Construction-level tests for the ea_llm_self_adaptive selection-mode switch.

``tophalf`` (default) must defer to the base top-half rank cull unchanged;
``random`` must pick the same number of survivors uniformly at random with the
run's RNG and log ``selection_mode`` in the SELECT event. No solver runs.
"""

from __future__ import annotations

import unittest

import numpy as np

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.logger import Logger
from contexto_solver.methods.ea_llm_self_adaptive import EALLMSelfAdaptiveConfig, EALLMSelfAdaptiveMethod


class _GameStub:
    def best_so_far(self) -> tuple[str | None, int | None]:
        return ("alpha", 3)

    def total_guesses(self) -> int:
        return 12


def _method(selection_mode: str, n_hypotheses: int = 8, max_active: int = 5) -> EALLMSelfAdaptiveMethod:
    method = EALLMSelfAdaptiveMethod.__new__(EALLMSelfAdaptiveMethod)
    method.config = EALLMSelfAdaptiveConfig(
        max_generations=1,
        candidates_per_hypothesis=1,
        initial_categories=1,
        starter_words_per_category=1,
        mutations_per_generation=1,
        max_active_hypotheses=max_active,
        trace_dir="traces",
        run_label="test",
        selection_mode=selection_mode,
    )
    method.rng = np.random.default_rng(0)
    method.logger = Logger()
    method.game = _GameStub()
    method.generation = 4
    method.hypotheses = []
    for index in range(n_hypotheses):
        hypothesis = Hypothesis(category_name=f"hyp{index}", description="d")
        hypothesis.update(f"word{index}", (index + 1) * 10)  # ranks 10, 20, ..., strictly ordered
        method.hypotheses.append(hypothesis)
    return method


def _select_event(logger: Logger) -> dict:
    return next(entry for entry in logger.trace if entry["event"] == "SELECT")


class TophalfDefaultTests(unittest.TestCase):
    def test_default_mode_keeps_top_half_by_rank(self) -> None:
        method = _method("tophalf")
        method._select()
        event = _select_event(method.logger)
        # 8 hypotheses -> keep_count = min(4, 5) = 4, best ranks first.
        self.assertEqual(event["details"]["kept"], ["hyp0", "hyp1", "hyp2", "hyp3"])
        self.assertEqual(event["details"]["discarded"], ["hyp4", "hyp5", "hyp6", "hyp7"])
        self.assertEqual(event["details"]["elite"], "hyp0")
        # Base event shape unchanged: no selection_mode key in the default path.
        self.assertNotIn("selection_mode", event["details"])
        statuses = {h.category_name: h.status for h in method.hypotheses}
        self.assertEqual([statuses[f"hyp{i}"] for i in range(4)], ["active"] * 4)
        self.assertEqual([statuses[f"hyp{i}"] for i in range(4, 8)], ["dormant"] * 4)


class RandomModeTests(unittest.TestCase):
    def test_random_mode_keeps_same_count_and_logs_mode(self) -> None:
        method = _method("random")
        method._select()
        event = _select_event(method.logger)
        details = event["details"]
        self.assertEqual(details["selection_mode"], "random")
        self.assertEqual(len(details["kept"]), 4)
        self.assertEqual(len(details["kept"]) + len(details["discarded"]), 8)
        self.assertEqual(sum(1 for h in method.hypotheses if h.status == "active"), 4)
        # Elite field reports the best-ranked survivor (informational only).
        kept_ranks = {h.category_name: h.best_rank for h in method.hypotheses if h.status == "active"}
        self.assertEqual(details["elite"], min(kept_ranks, key=kept_ranks.get))

    def test_random_mode_uses_run_rng_deterministically(self) -> None:
        first = _method("random")
        first._select()
        second = _method("random")
        second._select()
        self.assertEqual(
            _select_event(first.logger)["details"]["kept"],
            _select_event(second.logger)["details"]["kept"],
        )

    def test_random_mode_deviates_from_rank_order_sometimes(self) -> None:
        # Over several seeds the random cull must not always equal the top half.
        deviated = False
        for seed in range(10):
            method = _method("random")
            method.rng = np.random.default_rng(seed)
            method._select()
            kept = _select_event(method.logger)["details"]["kept"]
            if set(kept) != {"hyp0", "hyp1", "hyp2", "hyp3"}:
                deviated = True
                break
        self.assertTrue(deviated)

    def test_random_mode_respects_max_active_cap(self) -> None:
        method = _method("random", n_hypotheses=20, max_active=5)
        method._select()
        self.assertEqual(len(_select_event(method.logger)["details"]["kept"]), 5)


if __name__ == "__main__":
    unittest.main()
