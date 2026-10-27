"""Re-score self-report calibration against the event the prompt asks about.

The self-report prompt asks for the chance that the operator's BEST proposed
word ranks within the top 100 (and the bucket for that best word). The
submitted analysis graded each report against the FIRST graded word
(``first_proposed`` in contexto_solver/rq1/metrics.py). This script computes
both rules from the logged ranks alone (no model calls):

  first_paper   : the reader's first graded GUESS/SKIP event matched by
                  (hypothesis name, generation)  -- reproduces the paper.
  first_word    : rank of the operator's words[0] (matched by word, so a
                  same-name child in the same generation cannot leak in).
  best_graded   : min rank over the operator's own proposed words that were
                  graded (matched by word).  Records with no graded word are
                  excluded.
  best_strict   : as best_graded, but a report whose proposed words were ALL
                  invalid/ungraded counts as a miss (label 0) for the top-100
                  rate / ECE (bucket accuracy still needs a rank, so those are
                  excluded there).
  best_new_only : as best_graded, but ignoring proposed words that had already
                  been guessed earlier in the run (cache hits).

Standalone cross-check of the reader-based rule (scripts/camera_ready.py is the pipeline;
this script re-derives the same numbers independently of contexto_solver.rq1.metrics).

Usage: python -I camera_ready/rescore_crosscheck.py . camera_ready_outputs_<date>/crosscheck
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

from contexto_solver.rq1.reader import load_trace, run_config, extract_individuals  # noqa: E402
from contexto_solver.llm_client import _first_json_object, _strip_code_fences  # noqa: E402

ARMS = {
    "EA-Qwen (A1)": "traces/batch/traces/rq1_A1/ea_llm_self_adaptive_api_*.json",
    "Direct-Qwen (A2)": "traces/batch/traces/rq1_A2/llm_only_api_*.json",
    "EA-Gemma (A4)": "traces/batch/traces/rq1_A4/ea_llm_self_adaptive_api_*.json",
    "EA-Ministral (A5)": "traces/batch/traces/rq1_A5/ea_llm_self_adaptive_api_*.json",
}
BUCKET_CUTS = ((10, "top10"), (100, "top100"), (500, "top500"))
RULES = ["first_paper", "first_word", "best_graded", "best_strict", "best_new_only"]


def clean_word(word) -> str:
    cleaned = str(word).lower().strip()
    return cleaned if re.fullmatch(r"[a-z]+", cleaned) else ""


def words_from_raw(raw):
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        parsed = json.loads(_strip_code_fences(text))
    except (json.JSONDecodeError, ValueError):
        extracted = _first_json_object(text)
        if extracted is None:
            return None
        try:
            parsed = json.loads(extracted)
        except (json.JSONDecodeError, ValueError):
            return None
    if not isinstance(parsed, dict):
        return None
    if isinstance(parsed.get("words"), list):
        return [str(w) for w in parsed["words"]]
    if "word" in parsed:
        return [str(parsed["word"])]
    return None


def realized_bucket(rank):
    if rank is None or rank <= 0:
        return None
    for upper, label in BUCKET_CUTS:
        if rank <= upper:
            return label
    return "beyond"


def ece_10(scores, labels, n_bins=10):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    if scores.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            ece += (m.sum() / scores.size) * abs(labels[m].mean() - scores[m].mean())
    return float(ece)


def auroc(scores, labels):
    from scipy import stats
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    npos, nneg = int(labels.sum()), int((1 - labels).sum())
    if npos == 0 or nneg == 0:
        return None
    order = stats.rankdata(scores)
    return float((order[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def per_word_outcomes(events, ind, proposed_words):
    """For each proposed word: (cleaned, status, rank, already_tried)."""
    gen = ind.generation
    name = ind.hypothesis_name
    child_guess = {}       # word -> (rank, event index) for this child's GUESS events
    child_skip = set()
    first_seen_guess = {}  # word -> first event index of any GUESS in the run
    skipped_before = {}    # word -> first index of SKIP_INVALID_GUESS anywhere in the run
    for i, e in enumerate(events):
        ev = e.get("event")
        d = e.get("details") or {}
        if ev == "GUESS":
            w = d.get("word")
            first_seen_guess.setdefault(w, i)
            if e.get("generation") == gen and d.get("hypothesis") == name and w not in child_guess:
                child_guess[w] = (d.get("rank"), i)
        elif ev == "SKIP_INVALID_GUESS":
            w = d.get("word")
            skipped_before.setdefault(w, i)
            if e.get("generation") == gen and d.get("hypothesis") == name:
                child_skip.add(w)
    solved_rank1_idx = None
    for w, (r, i) in child_guess.items():
        if r == 1:
            solved_rank1_idx = i
    out = []
    seen = set()
    for raw_w in proposed_words:
        cw = clean_word(raw_w)
        if not cw:
            out.append((raw_w, "uncleanable", None, False))
            continue
        if cw in seen:
            out.append((cw, "duplicate_in_list", None, False))
            continue
        seen.add(cw)
        if cw in child_guess:
            rank, idx = child_guess[cw]
            already = first_seen_guess.get(cw, idx) < idx
            out.append((cw, "graded", rank, already))
        elif cw in child_skip:
            out.append((cw, "invalid", None, False))
        elif cw in skipped_before:
            out.append((cw, "known_invalid_skipped", None, False))
        elif solved_rank1_idx is not None:
            out.append((cw, "after_solve", None, False))
        else:
            out.append((cw, "unknown", None, False))
    return out


def main():
    records = []
    for arm, pattern in ARMS.items():
        for path in sorted(glob.glob(str(REPO / pattern))):
            events = load_trace(path)
            cfg = run_config(events)
            inds = extract_individuals(events, trace_file=Path(path).name)
            for ind in inds:
                raw = None
                for e in events:
                    if e.get("generation") != ind.generation:
                        continue
                    d = e.get("details") or {}
                    if ind.source_event == "OPERATOR_SAMPLED" and e.get("event") == "OPERATOR_SAMPLED" \
                            and d.get("child_hypothesis_name") == ind.hypothesis_name:
                        raw = (d.get("self_report") or {}).get("self_report_raw"); break
                    if ind.source_event == "CROSSOVER" and e.get("event") == "CROSSOVER" \
                            and (d.get("child") or {}).get("category_name") == ind.hypothesis_name:
                        raw = ((d.get("child") or {}).get("self_report") or {}).get("self_report_raw"); break
                    if ind.source_event == "GUESS" and e.get("event") == "GUESS" \
                            and d.get("word") == ind.proposed_word and isinstance(d.get("self_report"), dict):
                        raw = d["self_report"].get("self_report_raw"); break
                proposed = words_from_raw(raw)
                if ind.source_event == "GUESS":
                    outcomes = [(ind.proposed_word, "graded", ind.realized_rank, False)]
                elif proposed is None:
                    outcomes = []
                else:
                    outcomes = per_word_outcomes(events, ind, proposed)
                graded = [(w, r, a) for (w, s, r, a) in outcomes if s == "graded" and isinstance(r, int) and r > 0]
                n_words = len([o for o in outcomes if o[1] not in ("duplicate_in_list",)])
                statuses = collections.Counter(s for (_, s, _, _) in outcomes)
                first_word_rank = None
                if outcomes and outcomes[0][1] == "graded":
                    first_word_rank = outcomes[0][2]
                best_graded = min((r for (_, r, _) in graded), default=None)
                best_new = min((r for (_, r, a) in graded if not a), default=None)
                all_invalid = (proposed is not None and len(outcomes) > 0 and not graded
                               and all(s in ("invalid", "known_invalid_skipped", "uncleanable", "duplicate_in_list") for (_, s, _, _) in outcomes))
                records.append({
                    "arm": arm,
                    "trace_file": ind.trace_file,
                    "game_number": cfg.game_number,
                    "run_index": cfg.raw.get("run_index"),
                    "generation": ind.generation,
                    "source_event": ind.source_event,
                    "operator": ind.operator,
                    "hypothesis_name": ind.hypothesis_name,
                    "parent_rank": ind.parent_rank,
                    "predicted_closeness": ind.predicted_closeness,
                    "predicted_bucket": ind.predicted_bucket,
                    "parse_failed": ind.self_report_parse_failed,
                    "proposed_words": proposed,
                    "n_proposed": (len(proposed) if proposed is not None else None),
                    "word_outcomes": outcomes,
                    "n_graded": len(graded),
                    "n_invalid": statuses["invalid"] + statuses["known_invalid_skipped"],
                    "n_uncleanable": statuses["uncleanable"],
                    "n_after_solve": statuses["after_solve"],
                    "n_unknown": statuses["unknown"],
                    "n_already_tried": sum(1 for (_, _, a) in graded if a),
                    "all_invalid": all_invalid,
                    # rule outcomes (rank used; None = not scoreable)
                    "first_paper": ind.realized_rank,
                    "first_paper_word": ind.proposed_word,
                    "first_paper_invalid": ind.proposed_word_invalid,
                    "first_word": first_word_rank,
                    "best_graded": best_graded,
                    "best_graded_word": min(graded, key=lambda t: t[1])[0] if graded else None,
                    "best_strict": best_graded,  # label handled below
                    "best_new_only": best_new,
                    "reader_child_best": ind.child_best_rank,
                })

    # ---------- per-record CSV ----------
    with open(OUT / "rescored_records.csv", "w", newline="") as f:
        cols = [c for c in records[0].keys() if c != "word_outcomes"] + ["word_outcomes"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            row = dict(r)
            row["proposed_words"] = json.dumps(r["proposed_words"])
            row["word_outcomes"] = json.dumps(r["word_outcomes"])
            w.writerow(row)

    # ---------- coverage counts ----------
    cov = {}
    for arm in ARMS:
        rs = [r for r in records if r["arm"] == arm]
        ea = rs[0]["source_event"] != "GUESS" if rs else True
        c = {
            "reports": len(rs),
            "runs": len(set(r["trace_file"] for r in rs)),
            "with_closeness": sum(r["predicted_closeness"] is not None for r in rs),
            "with_bucket": sum(r["predicted_bucket"] is not None for r in rs),
            "proposed_list_recovered": sum(r["proposed_words"] is not None for r in rs),
            "exactly_3_proposed": sum(r["n_proposed"] == 3 for r in rs),
            "all_proposed_graded": sum(r["n_proposed"] is not None and r["n_graded"] == r["n_proposed"] for r in rs),
            "some_graded": sum(r["n_proposed"] is not None and 0 < r["n_graded"] < r["n_proposed"] for r in rs),
            "none_graded": sum(r["n_graded"] == 0 for r in rs),
            "all_invalid": sum(r["all_invalid"] for r in rs),
            "has_invalid_word": sum(r["n_invalid"] > 0 for r in rs),
            "has_uncleanable_word": sum(r["n_uncleanable"] > 0 for r in rs),
            "has_after_solve_word": sum(r["n_after_solve"] > 0 for r in rs),
            "has_unknown_word": sum(r["n_unknown"] > 0 for r in rs),
            "has_already_tried_word": sum(r["n_already_tried"] > 0 for r in rs),
            "name_collision(first_paper_word not in proposed)": sum(
                r["proposed_words"] is not None and r["first_paper_word"] is not None
                and clean_word(r["first_paper_word"]) not in [clean_word(x) for x in r["proposed_words"]] for r in rs),
            "first_paper_rank_ne_first_word_rank": sum(r["first_paper"] != r["first_word"] for r in rs),
            "best_rank_lt_first_rank": sum(r["first_word"] is not None and r["best_graded"] is not None and r["best_graded"] < r["first_word"] for r in rs),
            "first_word_is_best": sum(r["first_word"] is not None and r["best_graded"] == r["first_word"] for r in rs),
        }
        cov[arm] = c
    with open(OUT / "coverage.json", "w") as f:
        json.dump(cov, f, indent=1)

    # ---------- metrics per arm and rule ----------
    def metrics(rs, rule):
        scores, labels, buck_pred, buck_true = [], [], [], []
        for r in rs:
            if r["predicted_closeness"] is None:
                continue
            rank = r[rule]
            if rule == "best_strict" and rank is None and r["all_invalid"]:
                scores.append(float(r["predicted_closeness"])); labels.append(0)
                continue
            if rank is None:
                continue
            scores.append(float(r["predicted_closeness"]))
            labels.append(1 if rank <= 100 else 0)
        for r in rs:
            rank = r[rule]
            if r["predicted_bucket"] is None or rank is None:
                continue
            buck_pred.append(r["predicted_bucket"]); buck_true.append(realized_bucket(rank))
        n = len(scores)
        stated = float(np.mean(scores)) if n else None
        observed = float(np.mean(labels)) if n else None
        bo = {"top10": 0, "top100": 1, "top500": 2, "beyond": 3}
        signed = [bo[p] - bo[t] for p, t in zip(buck_pred, buck_true)]
        return {
            "n_scored": n,
            "stated_mean_p_top100": stated,
            "observed_top100_rate": observed,
            "ratio_stated_over_observed": (stated / observed) if observed else None,
            "ece_10bins": ece_10(scores, labels),
            "brier": float(np.mean((np.asarray(scores) - np.asarray(labels)) ** 2)) if n else None,
            "auroc": auroc(scores, labels) if n else None,
            "bucket_n": len(buck_pred),
            "bucket_accuracy": (sum(p == t for p, t in zip(buck_pred, buck_true)) / len(buck_pred)) if buck_pred else None,
            "bucket_mean_signed_error(pred-real, neg=overoptimistic)": (float(np.mean(signed)) if signed else None),
            "bucket_off_by_one_rate": (sum(abs(s) == 1 for s in signed) / len(signed)) if signed else None,
        }

    table = {}
    for arm in ARMS:
        rs = [r for r in records if r["arm"] == arm]
        table[arm] = {rule: metrics(rs, rule) for rule in RULES}
    ea_pool = [r for r in records if r["source_event"] != "GUESS"]
    table["EA pooled (A1+A4+A5)"] = {rule: metrics(ea_pool, rule) for rule in RULES}
    table["All four (A1+A2+A4+A5)"] = {rule: metrics(records, rule) for rule in RULES}
    with open(OUT / "rescore_summary.json", "w") as f:
        json.dump(table, f, indent=1)

    # ---------- markdown table ----------
    lines = ["| configuration | rule | n scored | stated mean P(top-100) | observed top-100 rate | stated/observed | ECE (10 bins) | Brier | AUROC | bucket n | bucket accuracy | mean signed bucket error |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, by_rule in table.items():
        for rule, m in by_rule.items():
            def f(x, d=3):
                return "-" if x is None else f"{x:.{d}f}"
            lines.append(f"| {arm} | {rule} | {m['n_scored']} | {f(m['stated_mean_p_top100'])} | {f(m['observed_top100_rate'])} | {f(m['ratio_stated_over_observed'],2)} | {f(m['ece_10bins'])} | {f(m['brier'])} | {f(m['auroc'])} | {m['bucket_n']} | {f(m['bucket_accuracy'])} | {f(m['bucket_mean_signed_error(pred-real, neg=overoptimistic)'],2)} |")
    (OUT / "rescore_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print()
    print(json.dumps(cov, indent=1))


if __name__ == "__main__":
    main()
