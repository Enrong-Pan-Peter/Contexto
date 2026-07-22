"""List invalid-marked entries in each per-game rank-cache file (read-only audit).

Before the transport-error fix in ``ContextoAPI.guess``, transient network/HTTP
failures were cached as ``invalid`` and could poison the shared per-game cache.
This utility lists every invalid-marked word per cache file so those entries can
be audited (and, if needed, re-verified manually) after a batch. It never writes
anything.

Usage:
    python scripts/list_rank_cache_invalids.py [--cache-dir data/rank_cache] [--game N] [--counts-only]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_invalids(cache_dir: str | Path, game: int | None = None) -> list[dict]:
    """One summary dict per cache file: path, game_number, entry counts, invalid words."""
    summaries: list[dict] = []
    for path in sorted(Path(cache_dir).glob("game_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            summaries.append(
                {"file": path.name, "game_number": None, "entries": 0, "invalid_words": [], "unreadable": True}
            )
            continue
        if not isinstance(payload, dict):
            continue
        game_number = payload.get("game_number")
        if game is not None and game_number != game:
            continue
        entries = payload.get("entries")
        entries = entries if isinstance(entries, dict) else {}
        invalid_words = sorted(
            word for word, entry in entries.items() if isinstance(entry, dict) and entry.get("invalid")
        )
        summaries.append(
            {
                "file": path.name,
                "game_number": game_number,
                "entries": len(entries),
                "invalid_words": invalid_words,
                "unreadable": False,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="List invalid-marked rank-cache entries (read-only).")
    parser.add_argument("--cache-dir", default="data/rank_cache", help="Rank cache directory.")
    parser.add_argument("--game", type=int, default=None, help="Only report this game number.")
    parser.add_argument("--counts-only", action="store_true", help="Suppress the per-word listing.")
    args = parser.parse_args()

    summaries = collect_invalids(args.cache_dir, args.game)
    total_invalid = 0
    for summary in summaries:
        if summary.get("unreadable"):
            print(f"{summary['file']}: UNREADABLE (skipped)")
            continue
        invalid_words = summary["invalid_words"]
        total_invalid += len(invalid_words)
        print(
            f"{summary['file']}: game={summary['game_number']} "
            f"entries={summary['entries']} invalid={len(invalid_words)}"
        )
        if not args.counts_only:
            for word in invalid_words:
                print(f"  {word}")
    print(f"files: {len(summaries)}  total invalid entries: {total_invalid}")


if __name__ == "__main__":
    main()
