"""Compare the embedding's notion of "close" with the LLM's, per target word.

For each target this lists the top-N closest words two ways and shows them side
by side:
  * Embedding side: the local game's own ranking. ``EmbeddingModel.nearest_neighbors``
    returns the top-N vocabulary words by cosine similarity to the target,
    excluding the target itself, so embedding-rank ``r`` is "the r-th closest word
    the local game would rank".
  * LLM side: the model's notion of meaning-closeness, obtained by asking for an
    ordered list of the N closest words through the existing public JSON path
    (``LLMClient.complete_json_prompt``, the same entrypoint
    ``scripts/calibrate_anchors.py`` uses; not ``place_word``).

It then reports, per target, concise overlap statistics and, most importantly,
the embedding-side BLIND SPOTS: words the game considers close that the LLM never
proposes. Those are the solver's likely stall points (e.g. a rank-5 plateau the
LLM cannot break because it never guesses the close word).

Caveats (verify before citing):
  * The embedding here is the LOCAL game's embedding (MiniLM by default), NOT the
    real Contexto embedding. Findings explain local-game behavior only.
  * LLM words absent from the embedding vocabulary are marked ``n/a`` (the local
    game would reject them); they cannot get an embedding rank.
  * The target is always excluded from its own neighbor list.

This is analysis only: it reads ``EmbeddingModel`` and ``LLMClient`` and changes
no solver, game, or config code, consistent with the analysis-script invariant in
docs/architecture.md.

Usage:
    python scripts/compare_embedding_llm_closeness.py superficial notorious chicken
    python scripts/compare_embedding_llm_closeness.py --targets superficial,notorious
        [--top-n 15] [--provider ollama] [--ollama-model qwen3:14b]
        [--embedding-path data/embeddings/all-MiniLM-L6-v2.npz]
        [--far-rank 100] [--cache PATH] [--report-json PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contexto_solver import config  # noqa: E402
from contexto_solver.embeddings import EmbeddingModel  # noqa: E402
from contexto_solver.llm_client import LLMClient  # noqa: E402
from contexto_solver.local_game import LocalGame  # noqa: E402

DEFAULT_TARGETS = ("superficial", "notorious", "chicken")

# Ordered-list prompt sent through the public complete_json_prompt() path. Mirrors
# calibrate_anchors.py's approach: a task-specific JSON prompt, reusing the shared
# bounded JSON-retry path, with no edits to llm_client.py.
CLOSEST_WORDS_PROMPT = """Return only JSON, no markdown or explanation.
List the {n} single English words whose meaning is most closely related to the
word "{word}", ordered from most closely related first to least.

Rules:
- Each item must be one single lowercase dictionary word: no phrases, no spaces,
  no hyphens, no punctuation, no proper nouns.
- Do not include the word "{word}" itself.
- Give exactly {n} distinct words if you can.
Return JSON only: {{"words": ["word1", "word2", ...]}}"""


# --------------------------------------------------------------------------- #
# LLM side (with per-(model, target, N) cache)
# --------------------------------------------------------------------------- #
def load_cache(path: Path) -> dict[str, list[str]]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {key: [str(word) for word in value]
                        for key, value in data.items() if isinstance(value, list)}
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    return {}


def save_cache(path: Path, cache: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _cache_key(target: str, n: int) -> str:
    return f"{target}::n{n}"


def _extract_word_list(response: Any) -> list[str]:
    """Pull a list of strings out of the JSON response, tolerating a few shapes."""
    if isinstance(response, dict):
        for key in ("words", "closest", "result", "list"):
            value = response.get(key)
            if isinstance(value, list):
                return [str(item) for item in value]
        return []
    if isinstance(response, list):
        return [str(item) for item in response]
    return []


def llm_raw_words(
    target: str,
    n: int,
    client: LLMClient | None,
    cache: dict[str, list[str]],
    cache_path: Path,
) -> tuple[list[str], str]:
    """Return (raw_words, status). status is 'cache', 'llm', or an error string."""
    key = _cache_key(target, n)
    if key in cache:
        return cache[key], "cache"
    if client is None:
        return [], "no-llm"
    prompt = CLOSEST_WORDS_PROMPT.format(word=target, n=n)
    try:
        response = client.complete_json_prompt(prompt)
    except Exception as exc:  # noqa: BLE001 - surface and continue per target
        return [], f"error: {exc}"
    words = _extract_word_list(response)
    if not words:
        return [], f"error: unparseable response {response!r}"
    cache[key] = words
    save_cache(cache_path, cache)
    return words, "llm"


def normalize_llm_words(raw: list[str], target: str, n: int) -> list[str]:
    """Single lowercase dictionary words only (the game's guess constraints):
    drop phrases/hyphens/punctuation, the target itself, and duplicates; cap at N."""
    target_clean = target.lower().strip()
    seen: set[str] = set()
    clean: list[str] = []
    for word in raw:
        candidate = str(word).lower().strip()
        if not candidate or not candidate.isalpha():
            continue  # rejects multi-word, hyphenated, and punctuated entries
        if candidate == target_clean or candidate in seen:
            continue
        seen.add(candidate)
        clean.append(candidate)
        if len(clean) >= n:
            break
    return clean


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _spearman(common: list[str], emb_pos: dict[str, int], llm_pos: dict[str, int]) -> float | None:
    """Spearman rho between embedding-rank and LLM-rank over shared words.

    Positions are already distinct integer ranks, so Pearson over them is exactly
    Spearman. Returns None when fewer than two shared words or zero variance.
    """
    if len(common) < 2:
        return None
    xs = [emb_pos[word] for word in common]
    ys = [llm_pos[word] for word in common]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    return cov / ((var_x ** 0.5) * (var_y ** 0.5))


def analyze_target(
    target: str,
    top_n: int,
    model: EmbeddingModel,
    client: LLMClient | None,
    cache: dict[str, list[str]],
    cache_path: Path,
    far_rank: int,
) -> dict[str, Any] | None:
    target_clean = target.lower().strip()
    if not model.has_word(target_clean):
        print(f"\n[skip] target {target_clean!r} is not in the embedding vocabulary "
              f"({model.path}); cannot rank neighbors.")
        return None

    neighbors = model.nearest_neighbors(target_clean, n=top_n)
    emb_words = [word for word, _ in neighbors]
    emb_cos = {word: cosine for word, cosine in neighbors}

    raw, status = llm_raw_words(target_clean, top_n, client, cache, cache_path)
    llm_words = normalize_llm_words(raw, target_clean, top_n)

    emb_pos = {word: index + 1 for index, word in enumerate(emb_words)}
    llm_pos = {word: index + 1 for index, word in enumerate(llm_words)}
    emb_set, llm_set = set(emb_words), set(llm_words)
    common = [word for word in emb_words if word in llm_set]

    overlap = len(emb_set & llm_set)
    exact_matches = sum(
        1 for index in range(top_n)
        if index < len(emb_words) and index < len(llm_words)
        and emb_words[index] == llm_words[index]
    )
    spearman = _spearman(common, emb_pos, llm_pos)

    # Embedding's full-vocab ranking (target = rank 1) for the LLM-far check.
    game = LocalGame(model, target_clean)
    blind_spots = [
        {"word": word, "embedding_rank": emb_pos[word], "cosine": emb_cos[word]}
        for word in emb_words if word not in llm_set
    ]
    llm_only_far = []
    for word in llm_words:
        if word in emb_set:
            continue
        rank = game.rankings.get(word, -1)
        in_vocab = rank > 0
        if not in_vocab or rank > far_rank:
            llm_only_far.append({
                "word": word,
                "llm_rank": llm_pos[word],
                "embedding_rank": rank if in_vocab else None,
                "in_vocab": in_vocab,
            })

    result = {
        "target": target_clean,
        "top_n": top_n,
        "llm_status": status,
        "llm_word_count": len(llm_words),
        "embedding_word_count": len(emb_words),
        "embedding_words": emb_words,
        "llm_words": llm_words,
        "overlap": overlap,
        "overlap_rate": overlap / top_n if top_n else None,
        "exact_position_matches": exact_matches,
        "exact_match_rate": exact_matches / top_n if top_n else None,
        "spearman": spearman,
        "n_common": len(common),
        "blind_spots": blind_spots,
        "llm_only_far": llm_only_far,
    }
    return result


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_opt_rate(value: float | None) -> str:
    return "  n/a" if value is None else f"{value * 100:5.1f}%"


def _fmt_spearman(value: float | None) -> str:
    return " n/a " if value is None else f"{value:+.2f}"


def print_target_report(result: dict[str, Any]) -> None:
    target = result["target"]
    top_n = result["top_n"]
    emb_words = result["embedding_words"]
    llm_words = result["llm_words"]

    print("\n" + "=" * 64)
    print(f"TARGET: {target}   (top-{top_n})   LLM list source: {result['llm_status']}")
    print("=" * 64)
    if result["llm_word_count"] < top_n:
        print(f"[note] only {result['llm_word_count']} usable LLM word(s) survived "
              f"normalization (wanted {top_n}).")

    print(f"\n{'rank':>4} | {'embedding word':<18} | {'LLM word':<18}")
    print(f"{'-' * 4}-+-{'-' * 18}-+-{'-' * 18}")
    for index in range(top_n):
        emb = emb_words[index] if index < len(emb_words) else "--"
        llm = llm_words[index] if index < len(llm_words) else "--"
        marker = "  <= match" if emb == llm and emb != "--" else ""
        print(f"{index + 1:>4} | {emb:<18} | {llm:<18}{marker}")

    print(f"\nOverlap:          {result['overlap']}/{top_n} words "
          f"({_fmt_opt_rate(result['overlap_rate'])})")
    print(f"Exact position:   {result['exact_position_matches']}/{top_n} "
          f"({_fmt_opt_rate(result['exact_match_rate'])})")
    print(f"Order agreement:  Spearman {_fmt_spearman(result['spearman'])} "
          f"(over {result['n_common']} shared word(s))")

    blind = result["blind_spots"]
    print(f"\nBLIND SPOTS - embedding-close words the LLM never proposed "
          f"({len(blind)}/{top_n}):")
    if blind:
        for item in blind:
            print(f"  emb#{item['embedding_rank']:<2} {item['word']:<18} "
                  f"(cosine {item['cosine']:.3f})")
    else:
        print("  (none — LLM covered the entire embedding top-N)")

    far = result["llm_only_far"]
    if far:
        print(f"\nLLM-only-far - LLM words the embedding ranks far / out of vocab "
              f"({len(far)}):")
        for item in far:
            rank = "n/a (not in vocab)" if not item["in_vocab"] else f"rank {item['embedding_rank']}"
            print(f"  llm#{item['llm_rank']:<2} {item['word']:<18} ({rank})")


def print_aggregate(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 64)
    print("SUMMARY (one row per target)")
    print("=" * 64)
    print(f"{'target':<16} {'overlap':>9} {'pos-match':>10} {'spearman':>9} {'blind':>6} {'llm-far':>8}")
    print("-" * 64)
    for result in results:
        print(f"{result['target']:<16} "
              f"{_fmt_opt_rate(result['overlap_rate']):>9} "
              f"{_fmt_opt_rate(result['exact_match_rate']):>10} "
              f"{_fmt_spearman(result['spearman']):>9} "
              f"{len(result['blind_spots']):>6} "
              f"{len(result['llm_only_far']):>8}")

    overlap_rates = [r["overlap_rate"] for r in results if r["overlap_rate"] is not None]
    match_rates = [r["exact_match_rate"] for r in results if r["exact_match_rate"] is not None]
    spearmans = [r["spearman"] for r in results if r["spearman"] is not None]
    print("-" * 64)
    mean_overlap = sum(overlap_rates) / len(overlap_rates) if overlap_rates else None
    mean_match = sum(match_rates) / len(match_rates) if match_rates else None
    mean_spearman = sum(spearmans) / len(spearmans) if spearmans else None
    print(f"{'MEAN':<16} {_fmt_opt_rate(mean_overlap):>9} "
          f"{_fmt_opt_rate(mean_match):>10} {_fmt_spearman(mean_spearman):>9}")
    print("\nLower overlap / Spearman on a target means the LLM's notion of close")
    print("diverges from the game's there; the blind-spot words are the concrete")
    print("close words the solver is unlikely to ever guess on that target.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare embedding vs LLM closeness per target (analysis only; "
                    "changes no solver code).")
    parser.add_argument("targets", nargs="*", help="Target words (positional).")
    parser.add_argument("--targets", dest="targets_csv",
                        help="Comma-separated target words (merged with positional).")
    parser.add_argument("--top-n", type=int, default=15, help="Words per side (default 15).")
    parser.add_argument("--embedding-path", default=config.GAME_EMBEDDING_PATH,
                        help="Local game embedding file (default: config.GAME_EMBEDDING_PATH).")
    parser.add_argument("--provider", help="LLM provider (default: config.LLM_PROVIDER).")
    parser.add_argument("--model", help="OpenAI/Anthropic model override.")
    parser.add_argument("--ollama-model", help="Ollama model override.")
    parser.add_argument("--cache", help="LLM-list cache path "
                        "(default: <placement_cache_dir>/closeness_<model>.json).")
    parser.add_argument("--far-rank", type=int, default=100,
                        help="LLM words with embedding rank worse than this (or out of "
                             "vocab) are flagged as LLM-only-far (default 100).")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM calls; use cached lists only (embedding side always runs).")
    parser.add_argument("--report-json", help="Optional path to dump the full report as JSON.")
    return parser.parse_args()


def resolve_targets(args: argparse.Namespace) -> list[str]:
    targets: list[str] = list(args.targets or [])
    if args.targets_csv:
        targets.extend(piece for piece in args.targets_csv.split(",") if piece.strip())
    if not targets:
        targets = list(DEFAULT_TARGETS)
    cleaned: list[str] = []
    seen: set[str] = set()
    for target in targets:
        normalized = target.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


def resolve_model(args: argparse.Namespace, provider: str) -> str:
    if provider == "ollama":
        return args.ollama_model or config.OLLAMA_MODEL
    return args.model or config.LLM_MODEL


def main() -> int:
    config.load_dotenv()
    args = parse_args()
    if args.top_n < 1:
        print("--top-n must be >= 1.", file=sys.stderr)
        return 1
    targets = resolve_targets(args)
    print(f"Targets ({len(targets)}): {', '.join(targets)}")

    model = EmbeddingModel(args.embedding_path)

    provider = (args.provider or config.LLM_PROVIDER).lower().strip()
    resolved_model = resolve_model(args, provider)
    safe_model = resolved_model.replace(":", "-").replace("/", "-")
    cache_path = (Path(args.cache) if args.cache
                  else Path(config.MAPELITES_PLACEMENT_CACHE_DIR) / f"closeness_{safe_model}.json")
    cache = load_cache(cache_path)
    print(f"LLM-list cache: {cache_path} ({len(cache)} entr{'y' if len(cache) == 1 else 'ies'} loaded).")

    client: LLMClient | None = None
    if not args.no_llm:
        api_key = "ollama" if provider == "ollama" else config.LLM_API_KEY
        client = LLMClient(provider=provider, api_key=api_key, model=resolved_model)
        print(f"LLM: provider={provider} model={resolved_model}")
    else:
        print("LLM calls disabled (--no-llm); using cached lists only.")

    results: list[dict[str, Any]] = []
    for target in targets:
        result = analyze_target(target, args.top_n, model, client, cache, cache_path, args.far_rank)
        if result is not None:
            results.append(result)
            print_target_report(result)

    if not results:
        print("\nNo targets could be analyzed (all missing from the embedding vocabulary?).",
              file=sys.stderr)
        return 1

    if len(results) > 1:
        print_aggregate(results)

    print("\nCaveat: this is the LOCAL game's embedding "
          f"({model.path}), not the real Contexto embedding; it explains local-game "
          "behavior only.")

    if args.report_json:
        report = {
            "embedding_path": str(model.path),
            "llm_provider": provider,
            "llm_model": resolved_model,
            "top_n": args.top_n,
            "far_rank": args.far_rank,
            "targets": results,
        }
        Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON report to {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
