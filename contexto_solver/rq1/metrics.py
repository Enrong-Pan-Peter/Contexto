"""RQ1 calibration metrics over self-reported individuals.

All metrics operate on the :class:`~contexto_solver.rq1.reader.Individual`
records produced by the reader. Two facts pin the semantics:

- ``predicted_closeness`` is the operator's probability that its best proposed
  word ranks within the **top 100** closest words. The positive label for the
  probabilistic metrics (reliability / ECE / Brier / AUROC) is therefore
  ``realized_rank <= 100``.
- ``predicted_bucket`` is categorical over ``("top10", "top100", "top500",
  "beyond")``; the realized bucket is derived from the realized rank with the
  same cut points.

The "realized rank" used for a record is selected by ``which``:
``"child_best"`` (default) uses the child's realized best rank across all words
it guessed -- the most faithful realization of "best proposed word"; falls back
to the first proposed word's rank when the child logged no valid guess.
``"first_proposed"`` uses only the first evaluated word's rank.

Only numpy + scipy are used (both are project dependencies). No network, LLM, or
writes.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy import stats

from .reader import Individual

BUCKET_ORDER = ("top10", "top100", "top500", "beyond")
_BUCKET_ORDINAL = {bucket: index for index, bucket in enumerate(BUCKET_ORDER)}
# predicted_closeness is P(rank within top 100); the positive class is top-100.
POSITIVE_RANK_THRESHOLD = 100
# Cut points (inclusive upper bounds) for the categorical buckets.
_BUCKET_CUTS = ((10, "top10"), (100, "top100"), (500, "top500"))


def realized_rank(individual: Individual, which: str = "child_best") -> int | None:
    """Select a record's realized rank per the ``which`` policy."""
    if which == "first_proposed":
        return individual.realized_rank
    if which == "child_best":
        if individual.child_best_rank is not None:
            return individual.child_best_rank
        return individual.realized_rank
    raise ValueError(f"unknown realized-rank policy: {which!r}")


def realized_bucket(rank: int | None) -> str | None:
    """Map a realized rank to its categorical bucket, or ``None`` when unknown."""
    if rank is None or rank <= 0:
        return None
    for upper, label in _BUCKET_CUTS:
        if rank <= upper:
            return label
    return "beyond"


def bucket_ordinal(bucket: str | None) -> int | None:
    return _BUCKET_ORDINAL.get(bucket) if bucket is not None else None


def _pairs(individuals: Iterable[Individual], which: str) -> tuple[list[float], list[int]]:
    """(predicted_closeness, realized_rank) pairs where both are present."""
    closeness: list[float] = []
    ranks: list[int] = []
    for individual in individuals:
        rank = realized_rank(individual, which)
        if individual.predicted_closeness is not None and rank is not None:
            closeness.append(float(individual.predicted_closeness))
            ranks.append(int(rank))
    return closeness, ranks


def spearman(individuals: Sequence[Individual], which: str = "child_best") -> dict[str, Any]:
    """Spearman rho between predicted_closeness and realized rank.

    Higher closeness should mean a smaller (better) rank, so a well-calibrated
    signal yields a NEGATIVE rho. Needs >= 2 pairs with variation in each series.
    """
    closeness, ranks = _pairs(individuals, which)
    n = len(closeness)
    if n < 2 or len(set(closeness)) < 2 or len(set(ranks)) < 2:
        return {"rho": None, "pvalue": None, "n": n}
    rho, pvalue = stats.spearmanr(closeness, ranks)
    return {"rho": float(rho), "pvalue": float(pvalue), "n": n}


def bucket_metrics(individuals: Sequence[Individual], which: str = "child_best") -> dict[str, Any]:
    """Accuracy, confusion matrix, mean signed error, and off-by-one rate.

    Uses records with a parsed ``predicted_bucket`` and a known realized rank.
    Ordinals run top10=0 .. beyond=3; signed error is predicted minus realized
    (positive = over-optimistic, i.e. predicted closer than reality).
    """
    confusion = {p: {r: 0 for r in BUCKET_ORDER} for p in BUCKET_ORDER}
    signed_errors: list[int] = []
    correct = 0
    total = 0
    for individual in individuals:
        predicted = individual.predicted_bucket
        actual = realized_bucket(realized_rank(individual, which))
        if predicted is None or actual is None:
            continue
        total += 1
        confusion[predicted][actual] += 1
        if predicted == actual:
            correct += 1
        signed_errors.append(_BUCKET_ORDINAL[predicted] - _BUCKET_ORDINAL[actual])
    off_by_one = sum(1 for error in signed_errors if abs(error) == 1)
    return {
        "n": total,
        "accuracy": (correct / total) if total else None,
        "confusion": confusion,
        "mean_signed_bucket_error": (sum(signed_errors) / len(signed_errors)) if signed_errors else None,
        "off_by_one_rate": (off_by_one / total) if total else None,
    }


def _labels_and_scores(individuals: Iterable[Individual], which: str) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    labels: list[int] = []
    for individual in individuals:
        rank = realized_rank(individual, which)
        if individual.predicted_closeness is not None and rank is not None:
            scores.append(float(individual.predicted_closeness))
            labels.append(1 if rank <= POSITIVE_RANK_THRESHOLD else 0)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)


def reliability_curve(
    individuals: Sequence[Individual], which: str = "child_best", n_bins: int = 10
) -> dict[str, Any]:
    """Binned reliability curve + ECE for P(realized_rank <= 100).

    Equal-width bins over ``[0, 1]``; the top bin includes 1.0. Each bin reports
    its predicted-probability mean, empirical positive rate, and count. ECE is
    the count-weighted mean absolute gap between the two.
    """
    scores, labels = _labels_and_scores(individuals, which)
    total = int(scores.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    if total:
        # Bin index in [0, n_bins-1]; clip so exactly 1.0 lands in the last bin.
        indices = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
        for bin_index in range(n_bins):
            mask = indices == bin_index
            count = int(mask.sum())
            entry = {
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_predicted": float(scores[mask].mean()) if count else None,
                "empirical_rate": float(labels[mask].mean()) if count else None,
            }
            bins.append(entry)
            if count:
                ece += (count / total) * abs(entry["empirical_rate"] - entry["mean_predicted"])
    else:
        for bin_index in range(n_bins):
            bins.append(
                {
                    "bin_lower": float(edges[bin_index]),
                    "bin_upper": float(edges[bin_index + 1]),
                    "count": 0,
                    "mean_predicted": None,
                    "empirical_rate": None,
                }
            )
    return {"n": total, "bins": bins, "ece": ece if total else None}


def brier_score(individuals: Sequence[Individual], which: str = "child_best") -> dict[str, Any]:
    """Mean squared error of predicted_closeness against the top-100 label."""
    scores, labels = _labels_and_scores(individuals, which)
    if scores.size == 0:
        return {"brier": None, "n": 0, "positive_rate": None}
    brier = float(np.mean((scores - labels) ** 2))
    return {"brier": brier, "n": int(scores.size), "positive_rate": float(labels.mean())}


def auroc(individuals: Sequence[Individual], which: str = "child_best") -> dict[str, Any]:
    """Tie-aware AUROC for predicted_closeness predicting realized_rank <= 100.

    Computed from the Mann-Whitney U statistic with average ranks for ties
    (identical to sklearn's ``roc_auc_score`` but dependency-free and easy to
    verify against a hand oracle). ``None`` when only one class is present.
    """
    scores, labels = _labels_and_scores(individuals, which)
    n = int(scores.size)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "n": n, "n_pos": n_pos, "n_neg": n_neg}
    order = stats.rankdata(scores)  # average ranks handle ties correctly
    rank_sum_pos = float(order[labels == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return {"auroc": float(auc), "n": n, "n_pos": n_pos, "n_neg": n_neg}


def realized_agreement(individuals: Sequence[Individual]) -> dict[str, Any]:
    """Agreement between the two realized-rank columns.

    Quantifies how often the first proposed word IS the child's best word (they
    coincide by construction for single-guess modes) and their rank correlation,
    so downstream readers know whether the ``which`` choice materially changes
    the calibration picture.
    """
    both: list[tuple[int, int]] = []
    equal = 0
    for individual in individuals:
        first = individual.realized_rank
        best = individual.child_best_rank
        if first is not None and best is not None:
            both.append((first, best))
            if first == best:
                equal += 1
    n = len(both)
    result: dict[str, Any] = {
        "n": n,
        "equal_count": equal,
        "equal_rate": (equal / n) if n else None,
        "spearman_rho": None,
    }
    if n >= 2:
        firsts = [pair[0] for pair in both]
        bests = [pair[1] for pair in both]
        if len(set(firsts)) >= 2 and len(set(bests)) >= 2:
            rho, _ = stats.spearmanr(firsts, bests)
            result["spearman_rho"] = float(rho)
    return result


def metrics_summary(individuals: Sequence[Individual], which: str = "child_best") -> dict[str, Any]:
    """All RQ1 calibration metrics for one group of individuals."""
    individuals = list(individuals)
    return {
        "count": len(individuals),
        "which_realized": which,
        "spearman": spearman(individuals, which),
        "buckets": bucket_metrics(individuals, which),
        "reliability": reliability_curve(individuals, which),
        "brier": brier_score(individuals, which),
        "auroc": auroc(individuals, which),
    }


# --- splits ------------------------------------------------------------------

def _parent_rank_bin(rank: int | None) -> str:
    if rank is None:
        return "unknown"
    if rank <= 10:
        return "1-10"
    if rank <= 100:
        return "11-100"
    if rank <= 500:
        return "101-500"
    if rank <= 1000:
        return "501-1000"
    return "1000+"


_SPLIT_KEYS: dict[str, Callable[[Individual], Any]] = {
    "operator": lambda i: i.operator,
    "generation": lambda i: i.generation,
    "parent_rank_bin": lambda i: _parent_rank_bin(i.parent_rank),
    "inheritance": lambda i: "inherited" if i.rationale_inherited else "not_inherited",
    "mode": lambda i: i.method,
}


def split_by(individuals: Iterable[Individual], key: Callable[[Individual], Any]) -> dict[Any, list[Individual]]:
    groups: dict[Any, list[Individual]] = {}
    for individual in individuals:
        groups.setdefault(key(individual), []).append(individual)
    return groups


def metrics_with_splits(
    individuals: Sequence[Individual],
    which: str = "child_best",
    *,
    splits: Sequence[str] = tuple(_SPLIT_KEYS),
) -> dict[str, Any]:
    """Overall metrics plus a per-split breakdown for the requested split keys."""
    individuals = list(individuals)
    result: dict[str, Any] = {
        "overall": metrics_summary(individuals, which),
        "realized_agreement": realized_agreement(individuals),
        "splits": {},
    }
    for split_name in splits:
        key = _SPLIT_KEYS.get(split_name)
        if key is None:
            continue
        groups = split_by(individuals, key)
        result["splits"][split_name] = {
            str(label): metrics_summary(members, which)
            for label, members in sorted(groups.items(), key=lambda item: (item[0] is None, str(item[0])))
        }
    return result
