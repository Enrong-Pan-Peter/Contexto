"""Tests for the real-game rank lookup cache."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from contexto_solver.game_api import ContextoAPI
from contexto_solver.rank_cache import INVALID_MARKER, RankCache


class RankCacheTests(unittest.TestCase):
    def test_store_and_lookup_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = RankCache(cache_dir, game_number=99, base_url="https://example.test/game")
            cache.store("Pearl", rank=42, invalid=False)
            self.assertEqual(cache.lookup("pearl"), 42)

    def test_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = RankCache(cache_dir, game_number=99, base_url="https://example.test/game")
            cache.store("nope", rank=None, invalid=True)
            self.assertEqual(cache.lookup("nope"), INVALID_MARKER)

    def test_flush_writes_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = RankCache(cache_dir, game_number=7, base_url="https://example.test/game")
            cache.store("word", rank=5, invalid=False)
            payload = json.loads(cache.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["entries"]["word"]["rank"], 5)


class ContextoAPICacheTests(unittest.TestCase):
    def test_cache_hit_skips_http(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            api = ContextoAPI(
                game_number=1314,
                base_url="https://api.contexto.me/machado/en/game",
                rate_limit=0.0,
                rank_cache_enabled=True,
                rank_cache_dir=cache_dir,
            )
            api._rank_cache.store("cachedword", rank=17, invalid=False)

            with mock.patch("contexto_solver.game_api.requests.get") as get_mock:
                rank = api.guess("cachedword")

            self.assertEqual(rank, 17)
            get_mock.assert_not_called()

    def test_api_result_is_cached_for_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            api = ContextoAPI(
                game_number=1314,
                base_url="https://api.contexto.me/machado/en/game",
                rate_limit=0.0,
                rank_cache_enabled=True,
                rank_cache_dir=cache_dir,
            )
            response = mock.Mock(status_code=200)
            response.json.return_value = {"distance": 41}

            with mock.patch("contexto_solver.game_api.requests.get", return_value=response):
                self.assertEqual(api.guess("freshword"), 42)

            self.assertEqual(api._rank_cache.lookup("freshword"), 42)


class ContextoAPICallLoggingTests(unittest.TestCase):
    def _api(self) -> ContextoAPI:
        return ContextoAPI(
            game_number=1314,
            base_url="https://api.contexto.me/machado/en/game",
            rate_limit=0.0,
            rank_cache_enabled=False,
        )

    def test_successful_call_records_status_latency_and_wall_clock(self) -> None:
        api = self._api()
        response = mock.Mock(status_code=200)
        response.json.return_value = {"distance": 4}
        with mock.patch("contexto_solver.game_api.requests.get", return_value=response):
            self.assertEqual(api.guess("hello"), 5)

        self.assertEqual(len(api.call_log), 1)
        record = api.call_log[0]
        self.assertEqual(record["status"], 200)
        self.assertEqual(record["outcome"], "ok")
        self.assertEqual(record["retries"], 0)
        self.assertIsInstance(record["latency_s"], float)
        self.assertIsNotNone(api.network_wall_clock_seconds)

    def test_http_error_and_exception_outcomes_recorded(self) -> None:
        api = self._api()
        error_response = mock.Mock(status_code=404)
        with mock.patch("contexto_solver.game_api.requests.get", return_value=error_response):
            self.assertEqual(api.guess("bad"), -1)
        with mock.patch(
            "contexto_solver.game_api.requests.get", side_effect=__import__("requests").RequestException()
        ):
            self.assertEqual(api.guess("boom"), -1)

        outcomes = [c["outcome"] for c in api.call_log]
        self.assertEqual(outcomes, ["http_error", "exception"])
        self.assertEqual(api.call_log[0]["status"], 404)
        self.assertIsNone(api.call_log[1]["status"])

    def test_cache_hit_records_no_network_call(self) -> None:
        with tempfile.TemporaryDirectory() as cache_dir:
            api = ContextoAPI(
                game_number=1314,
                base_url="https://api.contexto.me/machado/en/game",
                rate_limit=0.0,
                rank_cache_enabled=True,
                rank_cache_dir=cache_dir,
            )
            api._rank_cache.store("cachedword", rank=17, invalid=False)
            with mock.patch("contexto_solver.game_api.requests.get") as get_mock:
                api.guess("cachedword")
            get_mock.assert_not_called()
            self.assertEqual(api.call_log, [])
            self.assertIsNone(api.network_wall_clock_seconds)

    def test_call_metrics_aggregate(self) -> None:
        api = self._api()
        response = mock.Mock(status_code=200)
        response.json.return_value = {"distance": 4}
        with mock.patch("contexto_solver.game_api.requests.get", return_value=response):
            api.guess("one")
            api.guess("two")

        metrics = api.call_metrics()
        self.assertEqual(metrics["network_calls"], 2)
        self.assertEqual(metrics["status_counts"], {"200": 2})
        self.assertEqual(metrics["outcome_counts"], {"ok": 2})
        self.assertIsNotNone(metrics["network_wall_clock_seconds"])


if __name__ == "__main__":
    unittest.main()
