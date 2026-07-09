"""Wrapper around the public Contexto game API."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import requests

from . import config
from .rank_cache import INVALID_MARKER, RankCache


class ContextoAPI:
    def __init__(
        self,
        game_number: int,
        base_url: str,
        rate_limit: float = 0.5,
        *,
        rank_cache_enabled: bool | None = None,
        rank_cache_dir: str | None = None,
    ) -> None:
        self.game_number = game_number
        self.base_url = base_url.rstrip("/")
        self.rate_limit = rate_limit
        self.guesses: dict[str, int] = {}
        self.invalid_guesses: set[str] = set()
        use_cache = config.RANK_CACHE_ENABLED if rank_cache_enabled is None else rank_cache_enabled
        self._rank_cache = (
            RankCache(rank_cache_dir or config.RANK_CACHE_DIR, game_number, self.base_url)
            if use_cache
            else None
        )
        # Logging-only network telemetry: one record per HTTP call (cache hits
        # never touch the network and are not recorded). Does not affect guess().
        self.call_log: list[dict[str, Any]] = []
        self._run_start_monotonic: float | None = None
        self._run_end_monotonic: float | None = None

    def _record_call(
        self, word: str, *, status: int | None, outcome: str, latency_s: float, start_monotonic: float
    ) -> None:
        if self._run_start_monotonic is None:
            self._run_start_monotonic = start_monotonic
        self._run_end_monotonic = start_monotonic + latency_s
        self.call_log.append(
            {
                "word": word,
                "status": status,
                "outcome": outcome,
                "latency_s": round(latency_s, 4),
                # No retry logic exists on this path; recorded for schema stability.
                "retries": 0,
            }
        )

    @property
    def network_wall_clock_seconds(self) -> float | None:
        """Elapsed wall-clock across all HTTP calls, or ``None`` if none were made."""
        if self._run_start_monotonic is None or self._run_end_monotonic is None:
            return None
        return round(self._run_end_monotonic - self._run_start_monotonic, 4)

    def call_metrics(self) -> dict[str, Any]:
        """Aggregate per-call telemetry (logging-only; safe to call anytime)."""
        latencies = [c["latency_s"] for c in self.call_log]
        status_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        for call in self.call_log:
            status_counts[str(call["status"])] = status_counts.get(str(call["status"]), 0) + 1
            outcome_counts[call["outcome"]] = outcome_counts.get(call["outcome"], 0) + 1
        return {
            "network_calls": len(self.call_log),
            "network_wall_clock_seconds": self.network_wall_clock_seconds,
            "total_latency_seconds": round(sum(latencies), 4) if latencies else 0.0,
            "mean_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
            "max_latency_seconds": max(latencies) if latencies else None,
            "status_counts": status_counts,
            "outcome_counts": outcome_counts,
        }

    def guess(self, word: str) -> int:
        cleaned_word = word.lower().strip()
        if not cleaned_word:
            return -1
        if cleaned_word in self.guesses:
            return self.guesses[cleaned_word]
        if cleaned_word in self.invalid_guesses:
            return -1

        if self._rank_cache is not None:
            cached = self._rank_cache.lookup(cleaned_word)
            if cached == INVALID_MARKER:
                self.invalid_guesses.add(cleaned_word)
                return -1
            if isinstance(cached, int):
                self.guesses[cleaned_word] = cached
                return cached

        time.sleep(self.rate_limit)
        url = f"{self.base_url}/{self.game_number}/{quote(cleaned_word)}"
        start = time.monotonic()
        try:
            response = requests.get(url, timeout=15)
        except requests.RequestException:
            self._record_call(
                cleaned_word, status=None, outcome="exception", latency_s=time.monotonic() - start, start_monotonic=start
            )
            self.invalid_guesses.add(cleaned_word)
            if self._rank_cache is not None:
                self._rank_cache.store(cleaned_word, rank=None, invalid=True)
            return -1

        latency_s = time.monotonic() - start
        if response.status_code >= 400:
            self._record_call(
                cleaned_word, status=response.status_code, outcome="http_error", latency_s=latency_s, start_monotonic=start
            )
            self.invalid_guesses.add(cleaned_word)
            if self._rank_cache is not None:
                self._rank_cache.store(cleaned_word, rank=None, invalid=True)
            return -1

        try:
            rank = int(response.json()["distance"])
        except (KeyError, TypeError, ValueError):
            self._record_call(
                cleaned_word, status=response.status_code, outcome="bad_payload", latency_s=latency_s, start_monotonic=start
            )
            self.invalid_guesses.add(cleaned_word)
            if self._rank_cache is not None:
                self._rank_cache.store(cleaned_word, rank=None, invalid=True)
            return -1

        self._record_call(
            cleaned_word, status=response.status_code, outcome="ok", latency_s=latency_s, start_monotonic=start
        )
        # The public API returns 0 for the answer. The shared interface uses 1.
        normalized_rank = rank + 1
        self.guesses[cleaned_word] = normalized_rank
        if self._rank_cache is not None:
            self._rank_cache.store(cleaned_word, rank=normalized_rank, invalid=False)
        return normalized_rank

    def total_guesses(self) -> int:
        return len(self.guesses)

    def best_so_far(self) -> tuple[str | None, int | None]:
        if not self.guesses:
            return None, None
        best_word = min(self.guesses, key=self.guesses.get)
        return best_word, self.guesses[best_word]

    def is_solved(self) -> bool:
        return any(rank == 1 for rank in self.guesses.values())
