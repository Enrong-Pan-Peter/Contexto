"""JSON trace logging for solver runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


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

