"""Offline tests for experiment.py API batch mode (no network/LLM).

Covers the parts of ``--game api`` that do not require a live API or LLM: game
number parsing, input validation, the RUN_CONFIG parity fields, and the resume
key shared with the local batch.
"""

from __future__ import annotations

import argparse
import unittest

from contexto_solver import config
from contexto_solver import experiment as ex


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        game="api", method="ea_llm_self_adaptive", targets=None, target_file=None,
        game_numbers=None, mode="aligned", glove_path=None, game_embedding_path=None,
        solver_embedding_path=None, max_generations=10, runs_per_target=1, random_seed=0,
        seed_count=12, active_count=5, neighbors_per_word=10, llm_workers=4,
        provider="ollama", model=None, ollama_model="qwen3:14b", api_key="x",
        output=None, resume=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class GameNumberParsingTests(unittest.TestCase):
    def test_parses_and_dedupes(self):
        self.assertEqual(ex._load_game_numbers(_args(game_numbers="1387, 1388 1389,1387")),
                         [1387, 1388, 1389])

    def test_empty(self):
        self.assertEqual(ex._load_game_numbers(_args(game_numbers=None)), [])


class ValidationTests(unittest.TestCase):
    def test_missing_game_numbers(self):
        with self.assertRaises(ValueError):
            ex._run_api_batch(_args(game_numbers=None))

    def test_embedding_rejected(self):
        with self.assertRaises(ValueError):
            ex._run_api_batch(_args(game_numbers="1387", method="embedding"))


class RunConfigParityTests(unittest.TestCase):
    def test_api_run_config_fields(self):
        rc = ex._api_run_config(_args(game_numbers="1387"), 1387, 0, "ollama", "qwen3:14b")
        self.assertEqual(rc["game"], "api")
        self.assertIsNone(rc["target"])
        self.assertEqual(rc["game_number"], 1387)
        self.assertEqual(rc["alignment"], "api_unknown")
        self.assertEqual(rc["rank_cache_enabled"], config.RANK_CACHE_ENABLED)
        self.assertEqual(rc["trace_schema_version"], config.TRACE_SCHEMA_VERSION)
        self.assertTrue(rc["instrumentation_provenance_hash"])
        self.assertEqual(rc["self_adaptive_sigma_mode"], config.SELF_ADAPTIVE_SIGMA_MODE)


class CompletedKeyTests(unittest.TestCase):
    def test_local_and_api_keys_disjoint(self):
        local = ex._completed_key({"target": "ivory", "game_number": None, "run_index": 0})
        api = ex._completed_key({"target": None, "game_number": 1387, "run_index": 0})
        self.assertEqual(local, ("ivory", None, 0))
        self.assertEqual(api, (None, 1387, 0))
        self.assertNotEqual(local, api)


if __name__ == "__main__":
    unittest.main()
