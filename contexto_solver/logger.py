"""JSON trace logging for solver runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config


class Logger:
    def __init__(self) -> None:
        self.trace: list[dict[str, Any]] = []

    def log(self, generation: int, event_type: str, details: dict[str, Any]) -> None:
        entry = {
            "generation": generation,
            "event": event_type,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "details": details,
        }
        self.trace.append(entry)

    def log_operator_sampled(
        self,
        generation: int,
        parent_id: str,
        child_id: str,
        sigma_snapshot: list[float] | None,
        sampled_op: str,
        method: str,
    ) -> None:
        self.log(
            generation,
            "OPERATOR_SAMPLED",
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "sigma_snapshot": sigma_snapshot,
                "sampled_op": sampled_op,
                "method": method,
            },
        )

    def log_sigma_trajectory(
        self,
        generation: int,
        mean_sigma: list[float],
        population_size: int,
    ) -> None:
        self.log(
            generation,
            "SIGMA_TRAJECTORY",
            {
                "mean_sigma": mean_sigma,
                "population_size": population_size,
            },
        )

    def log_axis_definition(
        self,
        generation: int,
        grid_resolution: int,
        anchors_concreteness: dict[float, str],
        anchors_specificity: dict[float, str],
    ) -> None:
        self.log(
            generation,
            "AXIS_DEFINITION",
            {
                "grid_resolution": grid_resolution,
                "concreteness": {
                    "label": "0 = most concrete/physical, 1 = most abstract/conceptual",
                    "anchors": {f"{position:.2f}": word for position, word in sorted(anchors_concreteness.items())},
                },
                "specificity": {
                    "label": "0 = most general, 1 = most specific",
                    "anchors": {f"{position:.2f}": word for position, word in sorted(anchors_specificity.items())},
                },
            },
        )

    def log_placement(
        self,
        generation: int,
        word: str,
        coordinates: tuple[float, float],
        cell: tuple[int, int],
        cache_hit: bool,
    ) -> None:
        self.log(
            generation,
            "PLACEMENT",
            {
                "word": word,
                "coordinates": [float(coordinates[0]), float(coordinates[1])],
                "cell": [int(cell[0]), int(cell[1])],
                "cache_hit": bool(cache_hit),
            },
        )

    def log_network_metrics(self, generation: int, game: Any) -> None:
        """Persist real-API network telemetry as a ``NETWORK_METRICS`` event.

        No-op for games without call telemetry (e.g. the local game), so it is
        safe to call unconditionally at end of run. When ``PERSIST_CALL_LOG`` is
        set, the full per-call log is embedded alongside the aggregate metrics.
        """
        metrics_fn = getattr(game, "call_metrics", None)
        if not callable(metrics_fn):
            return
        details: dict[str, Any] = dict(metrics_fn())
        if config.PERSIST_CALL_LOG:
            details["call_log"] = [dict(record) for record in getattr(game, "call_log", [])]
        self.log(generation, "NETWORK_METRICS", details)

    def save(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
        return path

    def print_summary(self) -> None:
        solved = next((entry for entry in reversed(self.trace) if entry["event"] == "SOLVED"), None)
        failed = next((entry for entry in reversed(self.trace) if entry["event"] == "FAILED"), None)
        final = solved or failed
        if final is None:
            print("No solver result logged.")
            return

        details = final["details"]
        print(f"Status: {final['event']}")
        print(f"Best word: {details.get('answer') or details.get('best_word')}")
        print(f"Best rank: {details.get('rank') or details.get('best_rank')}")
        print(f"Total guesses: {details.get('total_guesses')}")

