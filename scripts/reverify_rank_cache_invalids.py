"""Re-verify invalid-marked rank-cache entries against the live Contexto API.

Invalid entries written before the transport-error fix (commit 1859f2d) may be
cache poison: a transient network/HTTP failure was cached as ``invalid`` as if
the API had definitively rejected the word. This script re-queries every
invalid-marked word through the CURRENT transport path (``ContextoAPI.guess``
with the per-run rank cache disabled, so the stale cache entry cannot satisfy
the lookup): transient failures are retried once and then reported as
unresolved without touching the cache; only a definitive HTTP 404 keeps a word
invalid.

Per-word verdicts:

- ``still-invalid``: the API answered 404 again; the cache entry is correct.
- ``now-valid``: the API returned a rank; the cached ``invalid`` was poison.
- ``unresolved``: transient failure after one retry; no verdict, never rewritten.

Read-only by default. With ``--apply``, ``now-valid`` entries are rewritten in
place (via ``RankCache.store``, which does an atomic temp-file replace) with
their fresh rank; ``still-invalid`` and ``unresolved`` entries are left alone.

This script makes no LLM calls, only Contexto API requests (one or two per
invalid word, rate-limited).

Usage (PowerShell):

    python scripts/reverify_rank_cache_invalids.py --cache-dir traces/batch/data/rank_cache_A1
    python scripts/reverify_rank_cache_invalids.py --cache-dir traces/batch/data/rank_cache_A1 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contexto_solver.game_api import ContextoAPI
from contexto_solver.rank_cache import RankCache


def _load_cache_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _invalid_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {
        word: entry
        for word, entry in sorted(entries.items())
        if isinstance(entry, dict) and entry.get("invalid")
    }


def reverify_file(
    path: Path,
    *,
    rate_limit: float,
    apply: bool,
) -> dict[str, Any] | None:
    """Re-verify one cache file. Returns a summary dict, or ``None`` if skipped."""
    payload = _load_cache_payload(path)
    if payload is None:
        print(f"{path.name}: UNREADABLE (skipped)")
        return None

    game_number = payload.get("game_number")
    base_url = payload.get("base_url")
    if not isinstance(game_number, int) or not isinstance(base_url, str) or not base_url:
        print(f"{path.name}: missing game_number/base_url metadata (skipped)")
        return None

    invalids = _invalid_entries(payload)
    print(f"{path.name}: game={game_number} invalid entries={len(invalids)}")
    if not invalids:
        return {"file": path.name, "now_valid": 0, "still_invalid": 0, "unresolved": 0}

    # rank_cache_enabled=False: force every lookup onto the network through the
    # post-1859f2d path (retry transients once, 404 is the only definitive
    # invalid). The stale on-disk entry can therefore never answer the query.
    api = ContextoAPI(
        game_number,
        base_url,
        rate_limit=rate_limit,
        rank_cache_enabled=False,
    )

    now_valid: dict[str, int] = {}
    still_invalid: list[str] = []
    unresolved: dict[str, str] = {}
    for word, entry in invalids.items():
        try:
            rank = api.guess(word)
        except RuntimeError as exc:
            unresolved[word] = str(exc)
            print(f"  {word}: UNRESOLVED (transient after retry) -- {exc}")
            continue
        if rank == -1:
            still_invalid.append(word)
            print(f"  {word}: still-invalid (definitive 404)")
        else:
            now_valid[word] = rank
            print(f"  {word}: now-valid rank={rank} (cached invalid was poison)")

    if apply and now_valid:
        cache = RankCache(path.parent, game_number, base_url)
        if cache.path != path:
            print(
                f"  WARNING: computed cache path {cache.path.name} != {path.name}; "
                "not rewriting this file"
            )
        else:
            for word, rank in now_valid.items():
                target_word = invalids[word].get("target_word")
                cache.store(word, rank=rank, invalid=False, target_word=target_word)
            print(f"  rewrote {len(now_valid)} entr{'y' if len(now_valid) == 1 else 'ies'} in {path.name}")

    return {
        "file": path.name,
        "now_valid": len(now_valid),
        "still_invalid": len(still_invalid),
        "unresolved": len(unresolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-verify invalid-marked rank-cache entries against the live API."
    )
    parser.add_argument("--cache-dir", default="data/rank_cache", help="Rank cache directory.")
    parser.add_argument("--game", type=int, default=None, help="Only re-verify this game number.")
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds to sleep before each API request (ContextoAPI default: 0.5).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite now-valid entries in place (atomic). Default is read-only.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    paths = sorted(cache_dir.glob("game_*.json"))
    if not paths:
        print(f"No game_*.json cache files found under {cache_dir}")
        return

    summaries: list[dict[str, Any]] = []
    for path in paths:
        if args.game is not None:
            payload = _load_cache_payload(path)
            if payload is None or payload.get("game_number") != args.game:
                continue
        summary = reverify_file(path, rate_limit=args.rate_limit, apply=args.apply)
        if summary is not None:
            summaries.append(summary)

    total_valid = sum(s["now_valid"] for s in summaries)
    total_invalid = sum(s["still_invalid"] for s in summaries)
    total_unresolved = sum(s["unresolved"] for s in summaries)
    mode = "APPLIED" if args.apply else "read-only (use --apply to rewrite)"
    print(
        f"\nfiles checked: {len(summaries)}  now-valid: {total_valid}  "
        f"still-invalid: {total_invalid}  unresolved: {total_unresolved}  [{mode}]"
    )
    if total_unresolved:
        print("Unresolved entries were left untouched; re-run to retry them.")


if __name__ == "__main__":
    main()
