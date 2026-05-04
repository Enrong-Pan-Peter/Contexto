"""Small wrapper around the public Contexto game API."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests


class ContextoAPI:
    def __init__(self, game_number: int, request_delay: float = 0.3) -> None:
        self.game_number = game_number
        self.request_delay = request_delay
        self.guesses: dict[str, int] = {}

    def guess(self, word: str) -> int | None:
        cleaned_word = word.lower().strip()
        if not cleaned_word:
            raise ValueError("Cannot guess an empty word.")

        if cleaned_word in self.guesses:
            return self.guesses[cleaned_word]

        time.sleep(self.request_delay)
        url = f"https://api.contexto.me/machado/en/game/{self.game_number}/{cleaned_word}"
        started_at = time.monotonic()
        try:
            response = requests.get(url, timeout=15)
        except requests.RequestException as exc:
            #region agent log
            _agent_debug_log(
                "contexto_solver/contexto_api.py:guess",
                "contexto request failed",
                {
                    "gameNumber": self.game_number,
                    "word": cleaned_word,
                    "reason": type(exc).__name__,
                    "durationMs": int((time.monotonic() - started_at) * 1000),
                },
                "H13",
            )
            #endregion
            return None

        duration_ms = int((time.monotonic() - started_at) * 1000)
        response_body_prefix = ""
        if response.status_code >= 400:
            response_body_prefix = response.text[:160]
        if response.status_code >= 400:
            #region agent log
            _agent_debug_log(
                "contexto_solver/contexto_api.py:guess",
                "contexto rank unavailable",
                {
                    "gameNumber": self.game_number,
                    "word": cleaned_word,
                    "statusCode": response.status_code,
                    "reason": "http_error",
                    "durationMs": duration_ms,
                    "bodyPrefix": response_body_prefix,
                },
                "H8,H9,H13",
            )
            #endregion
            return None

        try:
            data = response.json()
            rank = int(data["distance"])
        except (ValueError, KeyError, TypeError) as exc:
            #region agent log
            _agent_debug_log(
                "contexto_solver/contexto_api.py:guess",
                "contexto rank unavailable",
                {
                    "gameNumber": self.game_number,
                    "word": cleaned_word,
                    "statusCode": response.status_code,
                    "reason": type(exc).__name__,
                    "bodyPrefix": response.text[:160],
                },
                "H8,H10",
            )
            #endregion
            return None
        self.guesses[cleaned_word] = rank
        return rank


def _agent_debug_log(location: str, message: str, data: dict[str, object], hypothesis_id: str) -> None:
    try:
        payload = {
            "sessionId": "0eedb7",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with Path("debug-0eedb7.log").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass

