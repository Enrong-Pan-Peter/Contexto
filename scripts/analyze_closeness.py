"""Read-only analysis of a closeness-comparison JSON.

This script consumes a comparison JSON produced by
``scripts/compare_embedding_llm_closeness.py`` (either the newer Contexto-anchored
format or the older embedding-vs-LLM-only format) and writes two artifacts:
  * ``<stem>_summary.txt`` - a paste-ready plain-text report, one compact block
    per target plus an aggregate block.
  * ``<stem>_metrics.json`` - a machine-readable mirror of every numeric metric,
    keyed by target plus an ``"aggregate"`` key.

It treats ``contexto_words`` as ground truth and reports, per target and as an
aggregate over usable targets: data health, embedding-vs-real transferability,
LLM-vs-real blind spots, an embedding-vs-LLM secondary overlap, a morphology /
surface-form control, and a decomposition of what "close" means in real Contexto.

Strictly analysis only: it READS one JSON and WRITES into ``--out-dir``. It never
runs the solver, calls an LLM, touches the network, or modifies any existing repo
file, consistent with the analysis-script invariant in docs/architecture.md.
Standard library only (no numpy/scipy dependency; Spearman is implemented here).

Usage:
    python scripts/analyze_closeness.py closeness_contexto_300.json
    python scripts/analyze_closeness.py <input.json> [--out-dir closeness_reports]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUT_DIR = "closeness_reports"

RECALL_KS = (10, 25, 50)
TOP_DECOMP = 50  # real-neighbor window for [B] medians, [C] absences, [F] decomposition
MORPH_PROXY_TOP = 15
TRUNCATE = 25

# Inflectional / derivational suffixes for the lightweight stemmer (longest first).
_SUFFIXES = (
    "ization", "fulness", "iveness", "ousness", "ability", "ibility",
    "ically", "ation", "ities", "ement", "ments", " tions",
    "ness", "ions", "ings", "ment", "tion", "able", "ible", "less", "ful",
    "ies", "ied", "ier", "ous", "ive", "ion", "ize", "ise", "ing", "est",
    "ity", "als", "ic", "ed", "es", "er", "ly", "al", "s", "y",
)


# --------------------------------------------------------------------------- #
# Normalization + helpers
# --------------------------------------------------------------------------- #
def normalize(word: Any) -> str:
    return str(word).lower().strip()


def norm_list(words: Any) -> list[str]:
    """Normalize a list of words, dropping blanks while preserving order."""
    if not isinstance(words, list):
        return []
    out: list[str] = []
    for word in words:
        norm = normalize(word)
        if norm:
            out.append(norm)
    return out


def _stem(word: str) -> str:
    """Tiny suffix-stripping stemmer (no external dependency).

    Strips at most one recognized inflectional/derivational suffix, keeping a
    minimum stem length of 3 so short words are not over-stripped. This is a
    deliberate approximation, not a full Porter implementation.
    """
    word = normalize(word)
    for suffix in _SUFFIXES:
        suffix = suffix.strip()
        if not suffix:
            continue
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def is_morph_variant(word: str, target: str) -> bool:
    """True if ``word`` looks like a surface-form variant of ``target``.

    Two routes: (1) their lightweight stems match; or (2) they share a >=4-char
    common prefix and differ only by a recognized suffix on one or both sides.
    """
    word = normalize(word)
    target = normalize(target)
    if not word or not target:
        return False
    if word == target:
        return True
    if _stem(word) == _stem(target):
        return True
    if _common_prefix_len(word, target) >= 4:
        shorter, longer = sorted((word, target), key=len)
        if longer.startswith(shorter):
            return True
        if _stem(word) == _stem(target):
            return True
    return False


def spearman(words_a: list[str], words_b: list[str]) -> tuple[float | None, int]:
    """Average-rank Spearman over the intersection of two ordered lists.

    Each word's rank is its 1-based position in its own list; ties cannot occur
    within a single ordered list, so this reduces to Pearson over positions.
    Returns ``(rho, n_shared)``; ``rho`` is None when fewer than two shared words
    or when either side has zero variance.
    """
    pos_a = {word: idx + 1 for idx, word in enumerate(words_a)}
    pos_b = {word: idx + 1 for idx, word in enumerate(words_b)}
    shared = [word for word in words_a if word in pos_b]
    n = len(shared)
    if n < 2:
        return None, n
    xs = [pos_a[w] for w in shared]
    ys = [pos_b[w] for w in shared]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None, n
    return cov / ((var_x ** 0.5) * (var_y ** 0.5)), n


# --------------------------------------------------------------------------- #
# Per-target view: defensive extraction of both schema variants
# --------------------------------------------------------------------------- #
class TargetView:
    """Normalized, schema-agnostic view over one target's raw JSON object."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.name = normalize(raw.get("target", "?")) or "?"
        self.embedding_words = norm_list(raw.get("embedding_words"))
        self.llm_words = norm_list(raw.get("llm_words"))
        contexto = raw.get("contexto_words")
        self.has_contexto = contexto is not None
        self.contexto_words = norm_list(contexto)

        self.llm_status = raw.get("llm_status")
        self.llm_word_count = raw.get("llm_word_count")
        self.embedding_word_count = raw.get("embedding_word_count")
        self.contexto_word_count = raw.get("contexto_word_count")
        self.contexto_available = raw.get("contexto_available", self.has_contexto)

        # Embedding global-rank lookup tables (word -> rank) for arbitrary words.
        # New-format files expose ranks beyond the top-N inside these lists.
        self._rank_table: dict[str, int] = {}
        self._oov: set[str] = set()
        for idx, word in enumerate(self.embedding_words):
            self._rank_table.setdefault(word, idx + 1)
        for key in ("contexto_blind_embedding", "llm_only_far"):
            for entry in raw.get(key) or []:
                if not isinstance(entry, dict):
                    continue
                word = normalize(entry.get("word"))
                if not word:
                    continue
                in_vocab = entry.get("in_vocab", True)
                if in_vocab is False:
                    self._oov.add(word)
                rank = entry.get("embedding_rank")
                if isinstance(rank, (int, float)) and word not in self._rank_table:
                    self._rank_table[word] = int(rank)

        # pair_* objects (new format) or flat top-level stats (old format -> emb vs llm).
        self.pair_contexto_embedding = raw.get("pair_contexto_embedding")
        self.pair_contexto_llm = raw.get("pair_contexto_llm")
        pair_el = raw.get("pair_embedding_llm")
        if pair_el is None and "overlap" in raw:
            pair_el = {
                "overlap": raw.get("overlap"),
                "overlap_rate": raw.get("overlap_rate"),
                "exact_position_matches": raw.get("exact_position_matches"),
                "exact_match_rate": raw.get("exact_match_rate"),
                "spearman": raw.get("spearman"),
                "n_common": raw.get("n_common"),
            }
        self.pair_embedding_llm = pair_el

    def embedding_rank(self, word: str) -> tuple[int | None, bool]:
        """Return ``(rank, resolved)``; rank is None when unresolved."""
        word = normalize(word)
        if word in self._rank_table:
            return self._rank_table[word], True
        return None, False

    def is_oov(self, word: str) -> bool:
        return normalize(word) in self._oov

    @property
    def llm_failed(self) -> bool:
        return (not self.llm_words) or (self.llm_status != "llm")


# --------------------------------------------------------------------------- #
# Per-target metric computation
# --------------------------------------------------------------------------- #
def _recall_at_k(real: list[str], pool: set[str], k: int) -> tuple[int, int]:
    """Return (hits, effective_k) for real top-k present in pool."""
    eff = min(k, len(real))
    hits = sum(1 for w in real[:eff] if w in pool)
    return hits, eff


def compute_target(view: TargetView) -> dict[str, Any]:
    real = view.contexto_words
    emb = view.embedding_words
    llm = view.llm_words
    emb_set, llm_set, real_set = set(emb), set(llm), set(real)

    metrics: dict[str, Any] = {"target": view.name}

    # [A] DATA HEALTH ------------------------------------------------------- #
    pair_ce = view.pair_contexto_embedding
    ce_overlap = pair_ce.get("overlap") if isinstance(pair_ce, dict) else None
    emb_top15 = emb[:MORPH_PROXY_TOP]
    resolved_into_real_or_llm = sum(1 for w in emb_top15 if w in real_set or w in llm_set)
    degenerate = False
    if view.has_contexto and ce_overlap == 0:
        degenerate = True
    if resolved_into_real_or_llm < 3:
        degenerate = True
    health = {
        "llm_status": view.llm_status,
        "llm_word_count": view.llm_word_count,
        "embedding_word_count": view.embedding_word_count,
        "contexto_word_count": view.contexto_word_count,
        "contexto_available": view.contexto_available,
        "has_contexto": view.has_contexto,
        "llm_failed": view.llm_failed,
        "degenerate_embedding": degenerate,
        "embedding_top5": emb[:5],
        "embedding_top15_resolved_into_real_or_llm": resolved_into_real_or_llm,
    }
    metrics["health"] = health

    # [B] TRANSFERABILITY (embedding vs real) ------------------------------- #
    if view.has_contexto:
        recalls: dict[str, dict[str, Any]] = {}
        for k in RECALL_KS:
            hits, eff = _recall_at_k(real, emb_set, k)
            cap = min(eff, len(emb))
            hits_capped = sum(1 for w in real[:cap] if w in emb_set)
            recalls[str(k)] = {
                "hits": hits_capped, "k": cap, "requested_k": k,
                "rate": (hits_capped / cap) if cap else None,
                "capped": cap != k,
            }
        rho, n_shared = spearman(emb, real)
        stored_rho = pair_ce.get("spearman") if isinstance(pair_ce, dict) else None
        med15, n_res15, n_unres15 = _median_emb_rank(view, real[:15])
        med50, n_res50, n_unres50 = _median_emb_rank(view, real[:TOP_DECOMP])
        oov = [w for w in real[:TOP_DECOMP] if view.is_oov(w)]
        metrics["transferability"] = {
            "recall": recalls,
            "spearman_intersection": rho,
            "spearman_n_shared": n_shared,
            "spearman_underpowered": n_shared < 5,
            "spearman_stored": stored_rho,
            "spearman_differs": (rho is not None and stored_rho is not None
                                 and abs(rho - stored_rho) > 1e-6),
            "median_emb_rank_real_top15": med15,
            "median_emb_rank_real_top15_n_resolved": n_res15,
            "median_emb_rank_real_top15_n_unresolved": n_unres15,
            "median_emb_rank_real_top50": med50,
            "median_emb_rank_real_top50_n_resolved": n_res50,
            "median_emb_rank_real_top50_n_unresolved": n_unres50,
            "oov_count_real_top50": len(oov),
            "oov_words": oov[:15],
        }
    else:
        metrics["transferability"] = {"reason": "absent (no contexto column)"}

    # [C] BLIND SPOTS (LLM vs real) ----------------------------------------- #
    if not view.has_contexto:
        metrics["blind_spots"] = {"reason": "absent (no contexto column)"}
    elif view.llm_failed:
        metrics["blind_spots"] = {"reason": _llm_fail_reason(view)}
    else:
        recalls = {}
        for k in RECALL_KS:
            cap = min(k, len(real), len(llm))
            hits = sum(1 for w in real[:cap] if w in llm_set)
            recalls[str(k)] = {
                "hits": hits, "k": cap, "requested_k": k,
                "rate": (hits / cap) if cap else None,
                "capped": cap != k,
            }
        absent = [w for w in real[:TOP_DECOMP] if w not in llm_set]
        real_pos = {w: idx + 1 for idx, w in enumerate(real)}
        overreach = []
        for w in llm[:MORPH_PROXY_TOP]:
            overreach.append({
                "word": w,
                "real_rank": real_pos.get(w, None),
            })
        metrics["blind_spots"] = {
            "recall": recalls,
            "real_top50_absent_from_llm_count": len(absent),
            "real_top50_absent_from_llm": absent[:TRUNCATE],
            "real_top50_absent_total": len(absent),
            "llm_overreach": overreach,
        }

    # [D] EMBEDDING vs LLM -------------------------------------------------- #
    if view.llm_failed:
        metrics["embedding_vs_llm"] = {"reason": _llm_fail_reason(view)}
    else:
        cut = min(MORPH_PROXY_TOP, len(emb), len(llm))
        shared = sorted(set(emb[:cut]) & set(llm[:cut]))
        metrics["embedding_vs_llm"] = {
            "cut": cut,
            "overlap": len(shared),
            "shared_words": shared[:TRUNCATE],
        }

    # [E] MORPHOLOGY / SURFACE-FORM CONTROL --------------------------------- #
    morph = {}
    emb_morph = [w for w in emb[:MORPH_PROXY_TOP] if is_morph_variant(w, view.name)]
    morph["embedding_top15_morph_count"] = len(emb_morph)
    morph["embedding_top15_morph_words"] = emb_morph
    if view.llm_failed:
        morph["llm_top15_morph_count"] = None
        morph["llm_top15_morph_words"] = []
        morph["llm_morph_reason"] = _llm_fail_reason(view)
    else:
        llm_morph = [w for w in llm[:MORPH_PROXY_TOP] if is_morph_variant(w, view.name)]
        morph["llm_top15_morph_count"] = len(llm_morph)
        morph["llm_top15_morph_words"] = llm_morph
    if view.has_contexto:
        emb_nm = [w for w in emb if not is_morph_variant(w, view.name)]
        real_nm = [w for w in real if not is_morph_variant(w, view.name)]
        raw_emb_real = len(emb_set & real_set)
        stemmed_emb_real = len(set(emb_nm) & set(real_nm))
        morph["emb_vs_real_overlap_raw"] = raw_emb_real
        morph["emb_vs_real_overlap_stemmed"] = stemmed_emb_real
        if not view.llm_failed:
            llm_nm = [w for w in llm if not is_morph_variant(w, view.name)]
            morph["llm_vs_real_overlap_raw"] = len(llm_set & real_set)
            morph["llm_vs_real_overlap_stemmed"] = len(set(llm_nm) & set(real_nm))
        else:
            morph["llm_vs_real_overlap_raw"] = None
            morph["llm_vs_real_overlap_stemmed"] = None
            morph["llm_vs_real_reason"] = _llm_fail_reason(view)
    else:
        morph["overlap_reason"] = "absent (no contexto column)"
    metrics["morphology"] = morph

    # [F] REAL-NEIGHBOR DECOMPOSITION --------------------------------------- #
    if not view.has_contexto:
        metrics["decomposition"] = {"reason": "absent (no contexto column)"}
    else:
        counts = {"MORPH": 0, "SYNONYM": 0, "DISTRIBUTIONAL": 0, "OTHER": 0}
        synonym_folded = view.llm_failed
        for w in real[:TOP_DECOMP]:
            if is_morph_variant(w, view.name):
                counts["MORPH"] += 1
            elif (not synonym_folded) and w in llm_set:
                counts["SYNONYM"] += 1
            elif w in emb_set:
                counts["DISTRIBUTIONAL"] += 1
            else:
                counts["OTHER"] += 1
        metrics["decomposition"] = {
            "counts": counts,
            "window": min(TOP_DECOMP, len(real)),
            "synonym_folded_into_other": synonym_folded,
        }
        if synonym_folded:
            metrics["decomposition"]["synonym_reason"] = _llm_fail_reason(view)

    return metrics


def _median_emb_rank(view: TargetView, words: list[str]) -> tuple[float | None, int, int]:
    ranks: list[int] = []
    unresolved = 0
    for w in words:
        rank, ok = view.embedding_rank(w)
        if ok and rank is not None:
            ranks.append(rank)
        else:
            unresolved += 1
    if not ranks:
        return None, 0, unresolved
    ranks.sort()
    n = len(ranks)
    mid = n // 2
    median = float(ranks[mid]) if n % 2 else (ranks[mid - 1] + ranks[mid]) / 2
    return median, n, unresolved


def _llm_fail_reason(view: TargetView) -> str:
    status = view.llm_status
    if not view.llm_words:
        return f"LLM_FAILED (empty llm_words; status={status!r})"
    return f"LLM_FAILED (status={status!r})"


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def compute_aggregate(per_target: list[dict[str, Any]]) -> dict[str, Any]:
    usable_emb = [m for m in per_target
                  if not m["health"]["degenerate_embedding"] and m["health"]["has_contexto"]]
    usable_llm = [m for m in per_target if not m["health"]["llm_failed"]]
    usable_llm_ctx = [m for m in usable_llm if m["health"]["has_contexto"]]

    excluded_emb = [(m["target"], "degenerate_embedding" if m["health"]["degenerate_embedding"]
                     else "no contexto column")
                    for m in per_target if m not in usable_emb]
    excluded_llm = [(m["target"], "LLM_FAILED") for m in per_target if m["health"]["llm_failed"]]

    agg: dict[str, Any] = {
        "n_targets": len(per_target),
        "usable_embedding_count": len(usable_emb),
        "usable_llm_count": len(usable_llm),
        "excluded_from_embedding_means": excluded_emb,
        "excluded_from_llm_means": excluded_llm,
    }

    # [B] embedding means over non-degenerate, contexto-bearing targets.
    emb_means: dict[str, Any] = {}
    for k in RECALL_KS:
        emb_means[f"recall@{k}"] = _mean(
            [m["transferability"]["recall"][str(k)]["rate"] for m in usable_emb
             if "recall" in m["transferability"]])
    emb_means["spearman_intersection"] = _mean(
        [m["transferability"]["spearman_intersection"] for m in usable_emb
         if m["transferability"].get("spearman_intersection") is not None])
    emb_means["median_emb_rank_real_top15"] = _mean(
        [m["transferability"]["median_emb_rank_real_top15"] for m in usable_emb
         if m["transferability"].get("median_emb_rank_real_top15") is not None])
    emb_means["median_emb_rank_real_top50"] = _mean(
        [m["transferability"]["median_emb_rank_real_top50"] for m in usable_emb
         if m["transferability"].get("median_emb_rank_real_top50") is not None])
    agg["transferability_means"] = emb_means

    # [C] LLM means over non-failed, contexto-bearing targets.
    llm_means: dict[str, Any] = {}
    for k in RECALL_KS:
        llm_means[f"recall@{k}"] = _mean(
            [m["blind_spots"]["recall"][str(k)]["rate"] for m in usable_llm_ctx
             if "recall" in m["blind_spots"]])
    agg["blind_spot_means"] = llm_means

    # [D] embedding-vs-llm overlap mean over non-failed targets.
    agg["embedding_vs_llm_overlap_mean"] = _mean(
        [m["embedding_vs_llm"]["overlap"] for m in usable_llm
         if "overlap" in m["embedding_vs_llm"]])

    # [F] decomposition means; SYNONYM only over non-failed contexto targets.
    decomp_targets = [m for m in per_target if "counts" in m.get("decomposition", {})]
    decomp_means: dict[str, Any] = {}
    for bucket in ("MORPH", "DISTRIBUTIONAL", "OTHER"):
        decomp_means[bucket] = _mean([m["decomposition"]["counts"][bucket] for m in decomp_targets])
    decomp_means["SYNONYM"] = _mean(
        [m["decomposition"]["counts"]["SYNONYM"] for m in decomp_targets
         if not m["decomposition"].get("synonym_folded_into_other")])
    agg["decomposition_means"] = decomp_means

    return agg


# --------------------------------------------------------------------------- #
# Text rendering
# --------------------------------------------------------------------------- #
def _fmt_rate(num: int | None, den: int | None, rate: float | None) -> str:
    if num is None or den is None:
        return "n/a"
    pct = f" ({rate * 100:.1f}%)" if isinstance(rate, (int, float)) else ""
    return f"{num}/{den}{pct}"


def _fmt_num(value: Any, places: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def _trunc(items: list[Any]) -> str:
    if len(items) <= TRUNCATE:
        return ", ".join(str(i) for i in items) if items else "(none)"
    head = ", ".join(str(i) for i in items[:TRUNCATE])
    return f"first {TRUNCATE} of {len(items)}: {head}"


def render_target(m: dict[str, Any]) -> list[str]:
    h = m["health"]
    lines: list[str] = []
    lines.append(f"##### {m['target']}")
    flags = []
    if h["llm_failed"]:
        flags.append("LLM_FAILED")
    if h["degenerate_embedding"]:
        flags.append("DEGENERATE_EMBEDDING")
    flag_str = ("  [" + ", ".join(flags) + "]") if flags else ""
    lines.append(f"[A] health: status={h['llm_status']!r} llm_N={h['llm_word_count']} "
                 f"emb_N={h['embedding_word_count']} ctx_N={h['contexto_word_count']} "
                 f"contexto_available={h['contexto_available']}{flag_str}")
    if h["degenerate_embedding"]:
        lines.append(f"    emb top5: {_trunc(h['embedding_top5'])} "
                     f"(top15 resolving into real/llm: {h['embedding_top15_resolved_into_real_or_llm']})")

    t = m["transferability"]
    if "reason" in t:
        lines.append(f"[B] transferability: {t['reason']}")
    else:
        rc = t["recall"]
        rec_str = " ".join(
            f"@{k}={_fmt_rate(rc[str(k)]['hits'], rc[str(k)]['k'], rc[str(k)]['rate'])}"
            + ("[capped]" if rc[str(k)]["capped"] else "")
            for k in RECALL_KS)
        lines.append(f"[B] emb-vs-real recall: {rec_str}")
        up = " (underpowered)" if t["spearman_underpowered"] else ""
        diff = " differs-from-stored" if t["spearman_differs"] else ""
        lines.append(f"    spearman(intersection)={_fmt_num(t['spearman_intersection'])} "
                     f"n_shared={t['spearman_n_shared']}{up} "
                     f"stored={_fmt_num(t['spearman_stored'])}{diff}")
        lines.append(f"    median emb-rank real-top15={_fmt_num(t['median_emb_rank_real_top15'], 1)} "
                     f"(res={t['median_emb_rank_real_top15_n_resolved']}, "
                     f"unres={t['median_emb_rank_real_top15_n_unresolved']}); "
                     f"real-top50={_fmt_num(t['median_emb_rank_real_top50'], 1)} "
                     f"(res={t['median_emb_rank_real_top50_n_resolved']}, "
                     f"unres={t['median_emb_rank_real_top50_n_unresolved']})")
        lines.append(f"    OOV in real-top50: {t['oov_count_real_top50']} -> {_trunc(t['oov_words'])}")

    b = m["blind_spots"]
    if "reason" in b:
        lines.append(f"[C] blind spots: {b['reason']}")
    else:
        rc = b["recall"]
        rec_str = " ".join(
            f"@{k}={_fmt_rate(rc[str(k)]['hits'], rc[str(k)]['k'], rc[str(k)]['rate'])}"
            + ("[capped]" if rc[str(k)]["capped"] else "")
            for k in RECALL_KS)
        lines.append(f"[C] llm-vs-real recall: {rec_str}")
        lines.append(f"    real-top50 absent from llm: {b['real_top50_absent_from_llm_count']}/50 -> "
                     f"{_trunc(b['real_top50_absent_from_llm'])}")
        over = ", ".join(f"{o['word']}={o['real_rank'] if o['real_rank'] is not None else 'not-in-real'}"
                         for o in b["llm_overreach"])
        lines.append(f"    llm-top15 real-rank: {over}")

    d = m["embedding_vs_llm"]
    if "reason" in d:
        lines.append(f"[D] emb-vs-llm: {d['reason']}")
    else:
        lines.append(f"[D] emb-vs-llm overlap@{d['cut']}: {d['overlap']}/{d['cut']} -> "
                     f"{_trunc(d['shared_words'])}")

    mo = m["morphology"]
    emb_mc = mo["embedding_top15_morph_count"]
    llm_mc = mo["llm_top15_morph_count"]
    lines.append(f"[E] morph in top15: emb={emb_mc} ({_trunc(mo['embedding_top15_morph_words'])}); "
                 f"llm={llm_mc if llm_mc is not None else 'n/a'} "
                 f"({_trunc(mo['llm_top15_morph_words'])})")
    if "emb_vs_real_overlap_raw" in mo:
        llm_raw = mo.get("llm_vs_real_overlap_raw")
        llm_stem = mo.get("llm_vs_real_overlap_stemmed")
        lines.append(f"    overlap raw->stemmed: emb-vs-real "
                     f"{mo['emb_vs_real_overlap_raw']}->{mo['emb_vs_real_overlap_stemmed']}; "
                     f"llm-vs-real {llm_raw if llm_raw is not None else 'n/a'}->"
                     f"{llm_stem if llm_stem is not None else 'n/a'}")
    elif "overlap_reason" in mo:
        lines.append(f"    overlap raw->stemmed: {mo['overlap_reason']}")

    f = m["decomposition"]
    if "reason" in f:
        lines.append(f"[F] real-top50 decomposition: {f['reason']}")
    else:
        c = f["counts"]
        note = " (SYNONYM folded into OTHER: LLM_FAILED)" if f.get("synonym_folded_into_other") else ""
        lines.append(f"[F] real-top{f['window']} decomp: MORPH={c['MORPH']} SYNONYM={c['SYNONYM']} "
                     f"DISTRIBUTIONAL={c['DISTRIBUTIONAL']} OTHER={c['OTHER']}{note}")
    lines.append("")
    return lines


def render_aggregate(agg: dict[str, Any]) -> list[str]:
    lines = ["=" * 70, "AGGREGATE (means across usable targets)", "=" * 70]
    lines.append(f"targets={agg['n_targets']} usable_embedding={agg['usable_embedding_count']} "
                 f"usable_llm={agg['usable_llm_count']}")
    if agg["excluded_from_embedding_means"]:
        lines.append("  excluded from [B] embedding means: "
                     + ", ".join(f"{t} ({r})" for t, r in agg["excluded_from_embedding_means"]))
    if agg["excluded_from_llm_means"]:
        lines.append("  excluded from [C]/[D]/[F-SYN] llm means: "
                     + ", ".join(f"{t} ({r})" for t, r in agg["excluded_from_llm_means"]))

    tm = agg["transferability_means"]
    lines.append("[B] emb-vs-real recall means: "
                 + " ".join(f"@{k}={_fmt_num(tm.get(f'recall@{k}'))}" for k in RECALL_KS))
    lines.append(f"    spearman={_fmt_num(tm.get('spearman_intersection'))} "
                 f"median-emb-rank real-top15={_fmt_num(tm.get('median_emb_rank_real_top15'), 1)} "
                 f"real-top50={_fmt_num(tm.get('median_emb_rank_real_top50'), 1)}")
    bm = agg["blind_spot_means"]
    lines.append("[C] llm-vs-real recall means: "
                 + " ".join(f"@{k}={_fmt_num(bm.get(f'recall@{k}'))}" for k in RECALL_KS))
    lines.append(f"[D] emb-vs-llm overlap mean: {_fmt_num(agg['embedding_vs_llm_overlap_mean'], 2)}")
    dm = agg["decomposition_means"]
    lines.append(f"[F] real-top50 decomp means: MORPH={_fmt_num(dm.get('MORPH'), 2)} "
                 f"SYNONYM={_fmt_num(dm.get('SYNONYM'), 2)} "
                 f"DISTRIBUTIONAL={_fmt_num(dm.get('DISTRIBUTIONAL'), 2)} "
                 f"OTHER={_fmt_num(dm.get('OTHER'), 2)}")
    return lines


def render_summary(meta: dict[str, Any], per_target: list[dict[str, Any]],
                   agg: dict[str, Any], input_path: Path) -> str:
    lines = ["=" * 70, f"CLOSENESS ANALYSIS: {input_path.name}", "=" * 70]
    meta_bits = []
    for key in ("top_n", "far_rank", "llm_model", "contexto_max_rank", "embedding_path"):
        if key in meta:
            meta_bits.append(f"{key}={meta[key]}")
    if meta_bits:
        lines.append(" ".join(meta_bits))
    lines.append(f"targets in file: {len(per_target)}")
    lines.append("")
    for m in per_target:
        lines.extend(render_target(m))
    lines.extend(render_aggregate(agg))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an existing closeness-comparison JSON (read-only; "
                    "writes a text summary and a metrics JSON).")
    parser.add_argument("input", help="Path to the comparison JSON to analyze.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUT_DIR}).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Could not parse JSON {input_path}: {exc}", file=sys.stderr)
        return 1

    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        print(f"No 'targets' list found in {input_path}.", file=sys.stderr)
        return 1

    meta = {k: data[k] for k in
            ("top_n", "far_rank", "llm_model", "contexto_max_rank", "embedding_path")
            if k in data}

    per_target = [compute_target(TargetView(raw)) for raw in raw_targets
                  if isinstance(raw, dict)]
    if not per_target:
        print(f"No usable target objects in {input_path}.", file=sys.stderr)
        return 1
    agg = compute_aggregate(per_target)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    summary_path = out_dir / f"{stem}_summary.txt"
    metrics_path = out_dir / f"{stem}_metrics.json"

    summary_path.write_text(render_summary(meta, per_target, agg, input_path), encoding="utf-8")
    metrics_payload = {
        "input": str(input_path),
        "meta": meta,
        "targets": {m["target"]: m for m in per_target},
        "aggregate": agg,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Summary written to: {summary_path}")
    print(f"Metrics written to: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
