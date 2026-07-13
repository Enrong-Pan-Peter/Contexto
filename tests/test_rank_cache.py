"""Tests for the real-game rank lookup cache."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from contexto_solver import config
from contexto_solver.game_api import ContextoAPI
from contexto_solver.logger import Logger
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

    def test_call_metrics_reports_latency_percentiles(self) -> None:
        api = self._api()
        # Inject a deterministic call log so latency stats are exact.
        api.call_log = [
            {"word": "a", "status": 200, "outcome": "ok", "latency_s": 0.10, "retries": 0},
            {"word": "b", "status": 200, "outcome": "ok", "latency_s": 0.20, "retries": 0},
            {"word": "c", "status": 500, "outcome": "http_error", "latency_s": 0.30, "retries": 0},
            {"word": "d", "status": None, "outcome": "exception", "latency_s": 0.40, "retries": 0},
        ]

        metrics = api.call_metrics()

        self.assertEqual(metrics["network_calls"], 4)
        self.assertEqual(metrics["median_latency_seconds"], 0.25)
        self.assertEqual(metrics["p95_latency_seconds"], 0.40)
        self.assertEqual(metrics["max_latency_seconds"], 0.40)
        self.assertEqual(metrics["status_counts"], {"200": 2, "500": 1, "None": 1})
        self.assertEqual(metrics["outcome_counts"], {"ok": 2, "http_error": 1, "exception": 1})


class NetworkMetricsTraceTests(unittest.TestCase):
    """End-of-run persistence of ContextoAPI telemetry into the trace."""

    def _api_with_calls(self) -> ContextoAPI:
        api = ContextoAPI(
            game_number=1314,
            base_url="https://api.contexto.me/machado/en/game",
            rate_limit=0.0,
            rank_cache_enabled=False,
        )
        response = mock.Mock(status_code=200)
        response.json.return_value = {"distance": 4}
        with mock.patch("contexto_solver.game_api.requests.get", return_value=response):
            api.guess("alpha")
            api.guess("beta")
        return api

    def test_metrics_event_appended_without_call_log_by_default(self) -> None:
        logger = Logger()
        api = self._api_with_calls()

        with mock.patch.object(config, "PERSIST_CALL_LOG", False):
            logger.log_network_metrics(7, api)

        events = [entry for entry in logger.trace if entry["event"] == "NETWORK_METRICS"]
        self.assertEqual(len(events), 1)
        details = events[0]["details"]
        self.assertEqual(events[0]["generation"], 7)
        self.assertEqual(details["network_calls"], 2)
        self.assertIn("median_latency_seconds", details)
        self.assertIn("p95_latency_seconds", details)
        self.assertIn("network_wall_clock_seconds", details)
        self.assertNotIn("call_log", details)

    def test_full_call_log_persisted_behind_env_flag(self) -> None:
        logger = Logger()
        api = self._api_with_calls()

        with mock.patch.object(config, "PERSIST_CALL_LOG", True):
            logger.log_network_metrics(3, api)

        details = next(e["details"] for e in logger.trace if e["event"] == "NETWORK_METRICS")
        self.assertIn("call_log", details)
        self.assertEqual([record["word"] for record in details["call_log"]], ["alpha", "beta"])

    def test_local_game_without_telemetry_is_noop(self) -> None:
        logger = Logger()

        class _LocalStub:
            """A game backend with no network telemetry (mirrors LocalGame)."""

        logger.log_network_metrics(0, _LocalStub())

        self.assertEqual(logger.trace, [])


if __name__ == "__main__":
    unittest.main()
