"""Propose MAP-Elites anchor recalibrations from the empirical distribution of
solver-generated words.

This is a *proposal* tool. It changes no solver or config code. It rates a word
sample on the two MAP-Elites behavior axes using pole-only prompts (no
intermediate anchor examples, so the ratings are not biased toward the current
anchors), then reports the empirical distribution, candidate percentile bin
edges, candidate anchor words, and projected cell coverage under the current
equal-width binning versus a percentile binning.

Usage:
    python scripts/calibrate_anchors.py [--traces-dir traces] [--word-list FILE]
        [--limit N] [--provider ollama] [--ollama-model qwen3:14b]
        [--cache PATH] [--report-json PATH] [--dry-run]

Verified facts this tool relies on (see llm_client.py):
  * LLMClient.place_word() always embeds the formatted anchor scale, so it
    cannot run pole-only. We therefore build a calibration-specific pole-only
    prompt here and submit it through the public complete_json_prompt(), which
    reuses the same bounded JSON-retry path. No edits to llm_client.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contexto_solver import config  # noqa: E402
from contexto_solver.llm_client import LLMClient  # noqa: E402

AXES = ("concreteness", "specificity")
ROUND_QUARTERS = (0.0, 0.25, 0.5, 0.75, 1.0)
ROUND_TENTHS = tuple(round(0.1 * i, 1) for i in range(11))

# Pole-only: poles described in words, NO intermediate anchor examples.
CALIBRATION_PROMPT = """Return only JSON, no markdown or explanation.
Rate the word "{word}" on two independent scales, each a number from 0 to 1.

Concreteness: 0 = most concrete and physical (a tangible object you can touch),
1 = most abstract and conceptual (an idea with no physical form).

Specificity: 0 = most general (a broad umbrella category),
1 = most specific (a single narrow, precise instance).

Use the full range; do not round to convenient values.
Return JSON only: {{"concreteness": <number 0-1>, "specificity": <number 0-1>}}"""


# --------------------------------------------------------------------------- #
# Word sampling
# --------------------------------------------------------------------------- #
def _details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details")
    return details if isinstance(details, dict) else {}


def words_from_traces(traces_dir: Path) -> list[str]:
    """Distinct GUESS words from MAP-Elites traces (those with AXIS_DEFINITION)."""
    seen: dict[str, None] = {}
    for path in sorted(traces_dir.glob("*.json")):
        try:
            events = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(events, list):
            continue
        if not any(e.get("event") == "AXIS_DEFINITION" for e in events if isinstance(e, dict)):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("event") == "GUESS":
                word = _details(event).get("word")
                if isinstance(word, str) and word:
                    seen.setdefault(word, None)
    return list(seen.keys())


def words_from_file(path: Path) -> list[str]:
    seen: dict[str, None] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        word = raw.strip().lower()
        if word and not word.startswith("#"):
            seen.setdefault(word, None)
    return list(seen.keys())


# --------------------------------------------------------------------------- #
# Rating (LLM) with caching
# --------------------------------------------------------------------------- #
def _clamp_unit(value: Any) -> float:
    number = float(value)
    return max(0.0, min(1.0, number))


def load_cache(path: Path) -> dict[str, list[float]]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: [float(v[0]), float(v[1])] for k, v in data.items()}
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError):
            pass
    return {}


def save_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def rate_words(
    words: list[str],
    client: LLMClient,
    cache: dict[str, list[float]],
    cache_path: Path,
) -> dict[str, tuple[float, float]]:
    ratings: dict[str, tuple[float, float]] = {}
    new_count = 0
    for index, word in enumerate(words, start=1):
        if word in cache:
            ratings[word] = (cache[word][0], cache[word][1])
            continue
        prompt = CALIBRATION_PROMPT.format(word=word)
        try:
            response = client.complete_json_prompt(prompt)
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  ! rating failed for {word!r}: {exc}", file=sys.stderr)
            continue
        if not isinstance(response, dict):
            print(f"  ! non-dict response for {word!r}: {response!r}", file=sys.stderr)
            continue
        try:
            concreteness = _clamp_unit(response.get("concreteness"))
            specificity = _clamp_unit(response.get("specificity"))
        except (TypeError, ValueError):
            print(f"  ! non-numeric coords for {word!r}: {response!r}", file=sys.stderr)
            continue
        ratings[word] = (concreteness, specificity)
        cache[word] = [concreteness, specificity]
        new_count += 1
        if new_count % 10 == 0:
            save_cache(cache_path, cache)
            print(f"  ... rated {index}/{len(words)} ({new_count} new, cached)", flush=True)
    save_cache(cache_path, cache)
    print(f"Rated {len(ratings)} words ({new_count} new LLM calls, {len(ratings) - new_count} cache hits).")
    return ratings


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def text_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    counts = [0] * bins
    for value in values:
        index = min(int(value * bins), bins - 1)
        counts[index] += 1
    peak = max(counts) or 1
    lines = []
    for i, count in enumerate(counts):
        lo, hi = i / bins, (i + 1) / bins
        bar = "#" * int(round(width * count / peak))
        lines.append(f"  [{lo:.2f},{hi:.2f}) {count:4d} {bar}")
    return "\n".join(lines)


def percentile_edges(values: list[float]) -> list[float]:
    """20/40/60/80th percentiles (4 interior bin edges -> 5 bins)."""
    return statistics.quantiles(values, n=5)


def nearest_word(target: float, axis_index: int, ratings: dict[str, tuple[float, float]]) -> tuple[str, float]:
    best_word, best_value, best_distance = "", 0.0, float("inf")
    for word, coords in ratings.items():
        value = coords[axis_index]
        distance = abs(value - target)
        if distance < best_distance:
            best_word, best_value, best_distance = word, value, distance
    return best_word, best_value


def round_fraction(values: list[float], targets: tuple[float, ...], tol: float = 1e-6) -> float:
    if not values:
        return 0.0
    hits = sum(1 for v in values if any(abs(v - t) <= tol for t in targets))
    return hits / len(values)


def equal_width_bin(value: float, resolution: int) -> int:
    return min(int(value * resolution), resolution - 1)


def percentile_bin(value: float, edges: list[float]) -> int:
    index = 0
    for edge in edges:
        if value >= edge:
            index += 1
        else:
            break
    return index


def coverage_matrix(
    ratings: dict[str, tuple[float, float]],
    concreteness_binner,
    specificity_binner,
    resolution: int,
) -> list[list[int]]:
    matrix = [[0] * resolution for _ in range(resolution)]
    for coords in ratings.values():
        ci = concreteness_binner(coords[0])
        si = specificity_binner(coords[1])
        matrix[si][ci] += 1
    return matrix


def occupied_cells(matrix: list[list[int]]) -> int:
    return sum(1 for row in matrix for cell in row if cell > 0)


def render_matrix(matrix: list[list[int]]) -> str:
    # Row 0 = lowest specificity at bottom for readability (print top row = highest si).
    lines = []
    for si in range(len(matrix) - 1, -1, -1):
        row = matrix[si]
        lines.append("    spec[{}] ".format(si) + " ".join(f"{c:4d}" for c in row))
    lines.append("            " + " ".join(f"con{c:>1d}".rjust(4) for c in range(len(matrix))))
    return "\n".join(lines)


def axis_report(
    axis_index: int,
    axis_name: str,
    ratings: dict[str, tuple[float, float]],
    resolution: int,
) -> list[float]:
    values = sorted(coords[axis_index] for coords in ratings.values())
    print(f"\n================ AXIS: {axis_name} ================")
    print(f"n={len(values)}  min={values[0]:.3f}  max={values[-1]:.3f}  "
          f"mean={statistics.mean(values):.3f}  median={statistics.median(values):.3f}")
    print("\nHistogram (10 equal-width bins over [0,1]):")
    print(text_histogram(values))

    edges = percentile_edges(values)
    print("\nPercentile bin edges (20/40/60/80) -> candidate equal-count bins:")
    print("  " + "  ".join(f"p{p}={e:.3f}" for p, e in zip((20, 40, 60, 80), edges)))

    print("\nCandidate anchor words:")
    print(f"  pole 0.0 (min)  : {nearest_word(0.0, axis_index, ratings)[0]!r} "
          f"(value {nearest_word(0.0, axis_index, ratings)[1]:.3f})")
    for percentile, target in zip((20, 40, 60, 80), edges):
        word, value = nearest_word(target, axis_index, ratings)
        print(f"  p{percentile} ~ {target:.3f}    : {word!r} (value {value:.3f})")
    print(f"  pole 1.0 (max)  : {nearest_word(1.0, axis_index, ratings)[0]!r} "
          f"(value {nearest_word(1.0, axis_index, ratings)[1]:.3f})")

    print("\nQuantization (fraction of ratings landing exactly on round values):")
    print(f"  on quarters {{0,.25,.5,.75,1}} : {round_fraction(values, ROUND_QUARTERS):.1%}")
    print(f"  on tenths   {{0,.1,...,1}}     : {round_fraction(values, ROUND_TENTHS):.1%}")

    # Marginal coverage current vs percentile.
    current_counts = [0] * resolution
    pct_counts = [0] * resolution
    for value in values:
        current_counts[equal_width_bin(value, resolution)] += 1
        pct_counts[percentile_bin(value, edges)] += 1
    print("\nMarginal bin counts (current equal-width 0.2 bins): " + str(current_counts))
    print("Marginal bin counts (suggested percentile bins)    : " + str(pct_counts))
    return edges


def coverage_report(
    ratings: dict[str, tuple[float, float]],
    concreteness_edges: list[float],
    specificity_edges: list[float],
    resolution: int,
) -> None:
    total_cells = resolution * resolution
    print(f"\n================ PROJECTED CELL COVERAGE ({resolution}x{resolution} = {total_cells} cells) ================")

    current = coverage_matrix(
        ratings,
        lambda v: equal_width_bin(v, resolution),
        lambda v: equal_width_bin(v, resolution),
        resolution,
    )
    print("\nCurrent equal-width binning  -> occupied cells: "
          f"{occupied_cells(current)}/{total_cells}")
    print(render_matrix(current))

    suggested = coverage_matrix(
        ratings,
        lambda v: percentile_bin(v, concreteness_edges),
        lambda v: percentile_bin(v, specificity_edges),
        resolution,
    )
    print("\nSuggested percentile binning -> occupied cells: "
          f"{occupied_cells(suggested)}/{total_cells}")
    print(render_matrix(suggested))
    print("\nNote: per-axis percentile bins are equal-count marginally, but joint cell")
    print("coverage can still be uneven if the two axes are correlated.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose MAP-Elites anchor recalibrations (proposal only; changes no solver/config code).")
    parser.add_argument("--traces-dir", default="traces", help="Directory of trace JSON files (default: traces).")
    parser.add_argument("--word-list", help="Optional file with one word per line; overrides trace scan.")
    parser.add_argument("--limit", type=int, help="Cap the number of sampled words.")
    parser.add_argument("--provider", help="LLM provider (default: config.LLM_PROVIDER).")
    parser.add_argument("--model", help="OpenAI/Anthropic model override.")
    parser.add_argument("--ollama-model", help="Ollama model override.")
    parser.add_argument("--cache", help="Calibration cache file (default: data/placement_cache/calibration_<model>.json).")
    parser.add_argument("--grid", type=int, default=config.MAPELITES_GRID_RESOLUTION, help="Grid resolution (default: config value).")
    parser.add_argument("--report-json", help="Optional path to also dump raw ratings + edges as JSON.")
    parser.add_argument("--dry-run", action="store_true", help="List sampled words and exit; no LLM calls.")
    return parser.parse_args()


def resolve_model(args: argparse.Namespace, provider: str) -> str:
    if provider == "ollama":
        return args.ollama_model or config.OLLAMA_MODEL
    return args.model or config.LLM_MODEL


def main() -> int:
    config.load_dotenv()
    args = parse_args()

    if args.word_list:
        words = words_from_file(Path(args.word_list))
        source = f"word-list {args.word_list}"
    else:
        words = words_from_traces(Path(args.traces_dir))
        source = f"MAP-Elites traces in {args.traces_dir}/"
    if args.limit:
        words = words[: args.limit]
    print(f"Sampled {len(words)} distinct words from {source}.")
    if not words:
        print("No words to rate. Provide --word-list or point --traces-dir at MAP-Elites traces.", file=sys.stderr)
        return 1

    if args.dry_run:
        print("\nDry run — sampled words (no LLM calls):")
        print("  " + ", ".join(words))
        return 0

    provider = (args.provider or config.LLM_PROVIDER).lower().strip()
    model = resolve_model(args, provider)
    api_key = "ollama" if provider == "ollama" else config.LLM_API_KEY
    client = LLMClient(provider=provider, api_key=api_key, model=model)

    safe_model = model.replace(":", "-").replace("/", "-")
    cache_path = Path(args.cache) if args.cache else Path(config.MAPELITES_PLACEMENT_CACHE_DIR) / f"calibration_poleonly_{safe_model}.json"
    cache = load_cache(cache_path)
    print(f"Calibration cache: {cache_path} ({len(cache)} entries loaded).")

    ratings = rate_words(words, client, cache, cache_path)
    if not ratings:
        print("No successful ratings.", file=sys.stderr)
        return 1

    concreteness_edges = axis_report(0, "concreteness", ratings, args.grid)
    specificity_edges = axis_report(1, "specificity", ratings, args.grid)
    coverage_report(ratings, concreteness_edges, specificity_edges, args.grid)

    if args.report_json:
        report = {
            "source": source,
            "n_words": len(ratings),
            "model": model,
            "grid_resolution": args.grid,
            "ratings": {word: list(coords) for word, coords in ratings.items()},
            "concreteness_percentile_edges": concreteness_edges,
            "specificity_percentile_edges": specificity_edges,
        }
        Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote raw report to {args.report_json}")

    print("\nProposal only. Review suggested anchors before changing config.MAPELITES_ANCHORS_*.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
