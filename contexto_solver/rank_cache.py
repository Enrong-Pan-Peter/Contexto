"""Persistent read-through cache for real-game Contexto rank lookups."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


INVALID_MARKER = "invalid"


class RankCache:
    """Crash-safe JSON cache keyed by ``(base_url, game_number, word)``."""

    def __init__(self, cache_dir: str | Path, game_number: int, base_url: str) -> None:
        self.game_number = game_number
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        safe_base = self.base_url.replace("://", "_").replace("/", "_")
        self.path = self.cache_dir / f"game_{game_number}_{safe_base}.json"
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"game_number": self.game_number, "base_url": self.base_url, "entries": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"game_number": self.game_number, "base_url": self.base_url, "entries": {}}
        if not isinstance(payload, dict):
            return {"game_number": self.game_number, "base_url": self.base_url, "entries": {}}
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            payload["entries"] = {}
        payload.setdefault("game_number", self.game_number)
        payload.setdefault("base_url", self.base_url)
        return payload

    def lookup(self, word: str) -> int | None | str:
        """Return a cached rank, ``INVALID_MARKER``, or ``None`` on miss."""
        entry = self._data.get("entries", {}).get(word.lower().strip())
        if not isinstance(entry, dict):
            return None
        if entry.get("invalid"):
            return INVALID_MARKER
        rank = entry.get("rank")
        return int(rank) if isinstance(rank, int) else None

    def store(self, word: str, *, rank: int | None, invalid: bool = False, target_word: str | None = None) -> None:
        cleaned = word.lower().strip()
        if not cleaned:
            return
        entry: dict[str, Any] = {"invalid": bool(invalid)}
        if not invalid and rank is not None:
            entry["rank"] = int(rank)
        if target_word:
            entry["target_word"] = target_word
        self._data.setdefault("entries", {})[cleaned] = entry
        self._flush()

    def _flush(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2, sort_keys=True)
        fd, temp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
