"""Compare three notions of "close" per target word: real Contexto, embedding, LLM.

For each target this lists the top-N closest words up to three ways and shows
them side by side:
  * Real Contexto side (ground truth, when available): the manually collected
    Contexto backend ranks in ``data/<target>.txt`` (see ``load_contexto_words``).
    This is the anchor: blind spots are defined relative to it.
  * Embedding side: the local game's own ranking. ``EmbeddingModel.nearest_neighbors``
    returns the top-N vocabulary words by cosine similarity to the target,
    excluding the target itself, so embedding-rank ``r`` is "the r-th closest word
    the local game would rank".
  * LLM side: the model's notion of meaning-closeness, obtained by asking for an
    ordered list of the N closest words through the existing public JSON path
    (``LLMClient.complete_json_prompt``, the same entrypoint
    ``scripts/calibrate_anchors.py`` uses; not ``place_word``).

It then reports, per target, pairwise overlap statistics for the three pairs
(Contexto-vs-embedding, Contexto-vs-LLM, embedding-vs-LLM) and, most importantly,
the Contexto-anchored BLIND SPOTS when Contexto data is present: real-Contexto-close
words the LLM never proposes, and real-Contexto-close words the embedding ranks
far. Those are the solver's likely stall points. When no Contexto file exists for
a target, the embedding-vs-LLM comparison runs as usual and the Contexto column is
shown as n/a.

Caveats (verify before citing):
  * Real Contexto data comes from ``data/<target>.txt`` and is capped at the top
    ``CONTEXTO_MAX_RANK`` ranks; positions beyond what the file provides are n/a.
  * The embedding here is the LOCAL game's embedding (MiniLM by default), NOT the
    real Contexto embedding. Embedding-only findings explain local-game behavior.
  * LLM words absent from the embedding vocabulary are marked ``n/a`` (the local
    game would reject them); they cannot get an embedding rank.
  * The target is always excluded from its own neighbor list on every side.

This is analysis only: it reads ``EmbeddingModel``, ``LocalGame``, ``LLMClient``,
and the ``data/<target>.txt`` Contexto files, and changes no solver, game, or
config code, consistent with the analysis-script invariant in docs/architecture.md.

Usage:
    python scripts/compare_embedding_llm_closeness.py superficial notorious chicken
    python scripts/compare_embedding_llm_closeness.py --targets superficial,notorious
        [--top-n 15] [--provider ollama] [--ollama-model qwen3:14b]
        [--embedding-path data/embeddings/all-MiniLM-L6-v2.npz]
        [--contexto-dir data] [--far-rank 100] [--cache PATH] [--report-json PATH]
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

# Real Contexto data files live in this directory as ``<target>.txt`` and are
# capped at this many ranks (the backend only exposes the closest 300 words;
# positions beyond this are reported as n/a).
DEFAULT_CONTEXTO_DIR = "data"
CONTEXTO_MAX_RANK = 300

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
# Real Contexto side (ground truth, from data/<target>.txt)
# --------------------------------------------------------------------------- #
def load_contexto_words(target: str, contexto_dir: str) -> list[str] | None:
    """Ordered closest-first Contexto neighbor words from ``data/<target>.txt``.

    The file has alternating lines: a word then its integer rank, with the target
    itself at rank 1. Returns the closest non-target words in rank order (target
    excluded for parity with the embedding/LLM sides), deduplicated and capped at
    ``CONTEXTO_MAX_RANK``. Returns None when no file exists for the target, so the
    Contexto column/stats fall back to n/a.
    """
    path = Path(contexto_dir) / f"{target}.txt"
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    target_clean = target.lower().strip()
    ranked: list[tuple[int, str]] = []
    for index in range(0, len(lines) - 1, 2):
        word = lines[index].lower().strip()
        try:
            rank = int(lines[index + 1])
        except ValueError:
            continue
        if rank > CONTEXTO_MAX_RANK or word == target_clean:
            continue
        ranked.append((rank, word))
    ranked.sort(key=lambda item: item[0])
    words: list[str] = []
    seen: set[str] = set()
    for _, word in ranked:
        if word and word not in seen:
            seen.add(word)
            words.append(word)
    return words


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


def pairwise_stats(
    words_a: list[str] | None,
    words_b: list[str] | None,
    top_n: int,
) -> dict[str, Any] | None:
    """Overlap, exact-position matches, and Spearman between two ordered lists.

    Returns None when either side is unavailable (e.g. no Contexto file), so the
    caller renders the pair as n/a. Rates divide by ``top_n``; positions present
    in only one list simply do not overlap or align.
    """
    if words_a is None or words_b is None:
        return None
    pos_a = {word: index + 1 for index, word in enumerate(words_a)}
    pos_b = {word: index + 1 for index, word in enumerate(words_b)}
    set_a, set_b = set(words_a), set(words_b)
    common = [word for word in words_a if word in set_b]
    overlap = len(set_a & set_b)
    exact = sum(
        1 for index in range(top_n)
        if index < len(words_a) and index < len(words_b)
        and words_a[index] == words_b[index]
    )
    return {
        "overlap": overlap,
        "overlap_rate": overlap / top_n if top_n else None,
        "exact_position_matches": exact,
        "exact_match_rate": exact / top_n if top_n else None,
        "spearman": _spearman(common, pos_a, pos_b),
        "n_common": len(common),
    }


def analyze_target(
    target: str,
    top_n: int,
    model: EmbeddingModel,
    client: LLMClient | None,
    cache: dict[str, list[str]],
    cache_path: Path,
    far_rank: int,
    contexto_dir: str,
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

    contexto_all = load_contexto_words(target_clean, contexto_dir)
    contexto_words = contexto_all[:top_n] if contexto_all is not None else None

    emb_set, llm_set = set(emb_words), set(llm_words)
    emb_pos = {word: index + 1 for index, word in enumerate(emb_words)}

    # Pairwise comparisons across the three notions of "close".
    pair_contexto_embedding = pairwise_stats(contexto_words, emb_words, top_n)
    pair_contexto_llm = pairwise_stats(contexto_words, llm_words, top_n)
    pair_embedding_llm = pairwise_stats(emb_words, llm_words, top_n)

    # Embedding's full-vocab ranking (target = rank 1), reused for far checks.
    game = LocalGame(model, target_clean)

    # Contexto-anchored blind spots (only when Contexto ground truth is present):
    # ground-truth-close words the LLM never proposes, and ground-truth-close
    # words the embedding ranks far or places out of vocabulary.
    contexto_blind_llm: list[dict[str, Any]] | None = None
    contexto_blind_embedding: list[dict[str, Any]] | None = None
    if contexto_words is not None:
        contexto_pos = {word: index + 1 for index, word in enumerate(contexto_words)}
        contexto_blind_llm = [
            {"word": word, "contexto_rank": contexto_pos[word]}
            for word in contexto_words if word not in llm_set
        ]
        contexto_blind_embedding = []
        for word in contexto_words:
            rank = game.rankings.get(word, -1)
            in_vocab = rank > 0
            if not in_vocab or rank > far_rank:
                contexto_blind_embedding.append({
                    "word": word,
                    "contexto_rank": contexto_pos[word],
                    "embedding_rank": rank if in_vocab else None,
                    "in_vocab": in_vocab,
                })

    # Secondary embedding-anchored diagnostics (retained so targets without a
    # Contexto file still print meaningful blind spots, as before).
    llm_pos = {word: index + 1 for index, word in enumerate(llm_words)}
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
        "contexto_word_count": len(contexto_words) if contexto_words is not None else None,
        "contexto_available": contexto_words is not None,
        "contexto_words": contexto_words,
        "embedding_words": emb_words,
        "llm_words": llm_words,
        "pair_contexto_embedding": pair_contexto_embedding,
        "pair_contexto_llm": pair_contexto_llm,
        "pair_embedding_llm": pair_embedding_llm,
        "contexto_blind_llm": contexto_blind_llm,
        "contexto_blind_embedding": contexto_blind_embedding,
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


def _pair_line(label: str, pair: dict[str, Any] | None, top_n: int) -> str:
    if pair is None:
        return f"  {label:<22} n/a (Contexto data unavailable)"
    return (f"  {label:<22} overlap {pair['overlap']:>3}/{top_n} "
            f"({_fmt_opt_rate(pair['overlap_rate'])}), "
            f"exact {pair['exact_position_matches']:>3} "
            f"({_fmt_opt_rate(pair['exact_match_rate'])}), "
            f"Spearman {_fmt_spearman(pair['spearman'])} "
            f"(n={pair['n_common']})")


def print_target_report(result: dict[str, Any]) -> None:
    target = result["target"]
    top_n = result["top_n"]
    emb_words = result["embedding_words"]
    llm_words = result["llm_words"]
    contexto_words = result["contexto_words"]
    contexto_available = result["contexto_available"]

    print("\n" + "=" * 78)
    print(f"TARGET: {target}   (top-{top_n})   LLM list source: {result['llm_status']}")
    print("=" * 78)
    if not contexto_available:
        print("[note] no Contexto file (data/<target>.txt); Contexto column is n/a.")
    elif result["contexto_word_count"] < top_n:
        print(f"[note] Contexto provides only {result['contexto_word_count']} word(s) "
              f"(cap {CONTEXTO_MAX_RANK}); remaining positions are n/a.")
    if result["llm_word_count"] < top_n:
        print(f"[note] only {result['llm_word_count']} usable LLM word(s) survived "
              f"normalization (wanted {top_n}).")

    contexto_list = contexto_words if contexto_words is not None else []
    print(f"\n{'rank':>4} | {'contexto word':<18} | {'embedding word':<18} | {'LLM word':<18}")
    print(f"{'-' * 4}-+-{'-' * 18}-+-{'-' * 18}-+-{'-' * 18}")
    for index in range(top_n):
        ctx = contexto_list[index] if index < len(contexto_list) else ("n/a" if contexto_available else "--")
        emb = emb_words[index] if index < len(emb_words) else "--"
        llm = llm_words[index] if index < len(llm_words) else "--"
        print(f"{index + 1:>4} | {ctx:<18} | {emb:<18} | {llm:<18}")

    print("\nPairwise agreement:")
    print(_pair_line("contexto vs embedding", result["pair_contexto_embedding"], top_n))
    print(_pair_line("contexto vs LLM", result["pair_contexto_llm"], top_n))
    print(_pair_line("embedding vs LLM", result["pair_embedding_llm"], top_n))

    if contexto_available:
        blind_llm = result["contexto_blind_llm"]
        print(f"\nBLIND SPOTS - Contexto-close words the LLM never proposed "
              f"({len(blind_llm)}/{top_n}):")
        if blind_llm:
            for item in blind_llm:
                print(f"  ctx#{item['contexto_rank']:<3} {item['word']:<18}")
        else:
            print("  (none — LLM covered the entire Contexto top-N)")

        blind_emb = result["contexto_blind_embedding"]
        print(f"\nBLIND SPOTS - Contexto-close words the embedding ranks far / out of vocab "
              f"({len(blind_emb)}/{top_n}):")
        if blind_emb:
            for item in blind_emb:
                rank = "n/a (not in vocab)" if not item["in_vocab"] else f"rank {item['embedding_rank']}"
                print(f"  ctx#{item['contexto_rank']:<3} {item['word']:<18} ({rank})")
        else:
            print("  (none — embedding ranks every Contexto top-N word within far-rank)")
    else:
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


def _pair_overlap_rate(result: dict[str, Any], pair_key: str) -> float | None:
    pair = result.get(pair_key)
    return pair["overlap_rate"] if pair is not None else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def print_aggregate(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY (one row per target; overlap rate per pair)")
    print("=" * 78)
    print(f"{'target':<16} {'ctx-emb':>9} {'ctx-llm':>9} {'emb-llm':>9} "
          f"{'blind-llm':>10} {'blind-emb':>10}")
    print("-" * 78)
    for result in results:
        ctx_blind_llm = result["contexto_blind_llm"]
        ctx_blind_emb = result["contexto_blind_embedding"]
        print(f"{result['target']:<16} "
              f"{_fmt_opt_rate(_pair_overlap_rate(result, 'pair_contexto_embedding')):>9} "
              f"{_fmt_opt_rate(_pair_overlap_rate(result, 'pair_contexto_llm')):>9} "
              f"{_fmt_opt_rate(_pair_overlap_rate(result, 'pair_embedding_llm')):>9} "
              f"{(len(ctx_blind_llm) if ctx_blind_llm is not None else 'n/a'):>10} "
              f"{(len(ctx_blind_emb) if ctx_blind_emb is not None else 'n/a'):>10}")

    print("-" * 78)
    mean_ce = _mean([r for r in (_pair_overlap_rate(x, "pair_contexto_embedding") for x in results) if r is not None])
    mean_cl = _mean([r for r in (_pair_overlap_rate(x, "pair_contexto_llm") for x in results) if r is not None])
    mean_el = _mean([r for r in (_pair_overlap_rate(x, "pair_embedding_llm") for x in results) if r is not None])
    print(f"{'MEAN':<16} {_fmt_opt_rate(mean_ce):>9} {_fmt_opt_rate(mean_cl):>9} "
          f"{_fmt_opt_rate(mean_el):>9}")
    print("\nctx-emb / ctx-llm / emb-llm are top-N set overlap rates between pairs.")
    print("When Contexto data is present it is the ground-truth anchor: blind-llm")
    print("counts Contexto-close words the LLM never proposes, blind-emb counts")
    print("Contexto-close words the embedding ranks far - both are solver stall risks.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare real Contexto vs embedding vs LLM closeness per target "
                    "(analysis only; changes no solver code).")
    parser.add_argument("targets", nargs="*", help="Target words (positional).")
    parser.add_argument("--targets", dest="targets_csv",
                        help="Comma-separated target words (merged with positional).")
    parser.add_argument("--top-n", type=int, default=15, help="Words per side (default 15).")
    parser.add_argument("--contexto-dir", default=DEFAULT_CONTEXTO_DIR,
                        help="Directory of real Contexto rank files <target>.txt "
                             f"(default: {DEFAULT_CONTEXTO_DIR}).")
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
        result = analyze_target(target, args.top_n, model, client, cache, cache_path,
                                args.far_rank, args.contexto_dir)
        if result is not None:
            results.append(result)
            print_target_report(result)

    if not results:
        print("\nNo targets could be analyzed (all missing from the embedding vocabulary?).",
              file=sys.stderr)
        return 1

    if len(results) > 1:
        print_aggregate(results)

    print("\nCaveat: the embedding side is the LOCAL game's embedding "
          f"({model.path}), not the real Contexto embedding. Real Contexto ranks, "
          "when present, come from the manually collected data/<target>.txt files.")

    if args.report_json:
        report = {
            "embedding_path": str(model.path),
            "llm_provider": provider,
            "llm_model": resolved_model,
            "top_n": args.top_n,
            "far_rank": args.far_rank,
            "contexto_dir": args.contexto_dir,
            "contexto_max_rank": CONTEXTO_MAX_RANK,
            "targets": results,
        }
        Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote JSON report to {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
