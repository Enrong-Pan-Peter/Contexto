"""RQ1/RQ3 offline GloVe mediator comparison over solver traces.

Loads a GloVe embedding once and, for each run whose hidden target is recoverable
and in the GloVe vocabulary, ranks the run's proposed words against that target
offline. Reports three Spearman correlations per run and pooled: GloVe rank vs
real game rank, predicted_closeness vs real rank, and predicted_closeness vs
GloVe rank. Runs with an unrecoverable or out-of-vocabulary target are skipped
and listed. Analysis only: no network, no LLM, no trace/cache writes (the GloVe
file is read-only).

Usage (PowerShell):

    python scripts/rq1_mediator.py traces/ea_llm_self_adaptive_api_*.json \
        --glove-path data/glove.6B.300d.txt --output traces/rq1_mediator.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contexto_solver import config
from contexto_solver.embeddings import EmbeddingModel
from contexto_solver.local_game import LocalGame
from contexto_solver.rq1 import mediator as rq1_mediator
from contexto_solver.rq1.reader import Individual, extract_individuals, load_trace, recover_target, run_config


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(sorted(matches))
        elif Path(pattern).exists():
            paths.append(pattern)
    return list(dict.fromkeys(paths))


def _ranker_for_target(model: EmbeddingModel, target: str) -> rq1_mediator.GloveRanker | None:
    """Build a GloVe ranker for a target, or ``None`` if it is out of vocabulary."""
    if not model.has_word(target):
        return None
    game = LocalGame(model, target)
    return rq1_mediator.GloveRanker(game.rankings)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 offline GloVe mediator comparison.")
    parser.add_argument("traces", nargs="+", help="Raw trace JSON path(s) or glob(s).")
    parser.add_argument("--glove-path", default=config.GLOVE_PATH, help="Path to a GloVe text embedding file.")
    parser.add_argument("--which", choices=["first_proposed", "child_best"], default="first_proposed")
    parser.add_argument("--output", help="Path to write the mediator JSON (also printed).")
    args = parser.parse_args()

    paths = _expand_paths(args.traces)
    if not paths:
        raise SystemExit("No trace files matched the given path(s).")

    model = EmbeddingModel(args.glove_path)

    per_run: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    pooled: list[Individual] = []
    # Pooled correlations need a per-word GloVe rank against each word's OWN run
    # target, so accumulate resolved rows rather than re-ranking across targets.
    pooled_rows: list[dict[str, Any]] = []

    for path in paths:
        name = Path(path).name
        events = load_trace(path)
        config_obj = run_config(events)
        target, target_source = recover_target(events, config_obj)
        individuals = extract_individuals(events, trace_file=name)
        if not target:
            skipped.append({"trace": name, "reason": "unrecoverable_target"})
            continue
        ranker = _ranker_for_target(model, target)
        if ranker is None:
            skipped.append({"trace": name, "reason": "target_out_of_vocab", "target": target})
            continue
        per_run[name] = {
            "method": config_obj.method,
            "game_number": config_obj.game_number,
            "target": target,
            "target_source": target_source,
            "provenance_hash": config_obj.provenance_hash,
            "metrics": rq1_mediator.mediator_metrics(individuals, ranker, args.which),
        }
        pooled.extend(individuals)
        pooled_rows.extend(rq1_mediator.mediator_rows(individuals, ranker, args.which))

    pooled_metrics = _pooled_from_rows(pooled_rows)

    report = {
        "traces": [Path(p).name for p in paths],
        "which_realized": args.which,
        "glove_path": args.glove_path,
        "runs_scored": len(per_run),
        "runs_skipped": skipped,
        "per_run": per_run,
        "pooled": pooled_metrics,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"\nWrote mediator JSON: {args.output}", file=sys.stderr)


def _pooled_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the already-resolved mediator rows across runs into the 3 correlations."""
    from scipy import stats

    def corr(pairs: list[tuple[float, float]]) -> dict[str, Any]:
        n = len(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        if n < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
            return {"rho": None, "pvalue": None, "n": n}
        rho, pvalue = stats.spearmanr(xs, ys)
        return {"rho": float(rho), "pvalue": float(pvalue), "n": n}

    real_glove = [(r["glove_rank"], r["real_rank"]) for r in rows if r["glove_rank"] is not None and r["real_rank"] is not None]
    close_real = [(r["predicted_closeness"], r["real_rank"]) for r in rows if r["predicted_closeness"] is not None and r["real_rank"] is not None]
    close_glove = [(r["predicted_closeness"], r["glove_rank"]) for r in rows if r["predicted_closeness"] is not None and r["glove_rank"] is not None]
    in_vocab = sum(1 for r in rows if r["glove_rank"] is not None)
    return {
        "words_total": len(rows),
        "words_in_glove_vocab": in_vocab,
        "glove_vocab_coverage": (in_vocab / len(rows)) if rows else None,
        "glove_vs_real": corr(real_glove),
        "closeness_vs_real": corr(close_real),
        "closeness_vs_glove": corr(close_glove),
    }


if __name__ == "__main__":
    main()
