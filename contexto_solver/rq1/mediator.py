"""Offline GloVe mediator comparison for RQ1/RQ3.

For a run whose hidden target is recoverable and present in the GloVe vocabulary,
we can score each proposed word with an offline GloVe cosine rank against that
target and ask three questions:

- ``glove_vs_real``: does the GloVe rank track the real game rank? (mediator
  fidelity; a positive Spearman rho means the offline proxy orders words like the
  real game does).
- ``closeness_vs_real``: does the operator's predicted_closeness track the real
  rank? (calibration; negative rho expected, since higher closeness should mean a
  smaller/closer rank).
- ``closeness_vs_glove``: does predicted_closeness track the GloVe proxy rank?
  (negative rho expected).

The GloVe rank is supplied as a callable ``ranker(word) -> int | None`` so this
module needs neither the embedding matrix nor the network and stays unit-testable
with a dict-backed fake. Runs with an unrecoverable target (unsolved api runs) or
an out-of-vocabulary target are skipped by the caller. No writes.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from scipy import stats

from .metrics import realized_rank
from .reader import Individual

RankFn = Callable[[str], "int | None"]


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    n = len(xs)
    if n < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"rho": None, "pvalue": None, "n": n}
    rho, pvalue = stats.spearmanr(xs, ys)
    return {"rho": float(rho), "pvalue": float(pvalue), "n": n}


def mediator_rows(
    individuals: Iterable[Individual], ranker: RankFn, which: str = "first_proposed"
) -> list[dict[str, Any]]:
    """Per-individual (word, real_rank, glove_rank, closeness) rows.

    ``which`` picks the word/real-rank pair: ``"first_proposed"`` uses the first
    evaluated word and its rank; ``"child_best"`` uses the child's best word and
    rank. ``glove_rank`` is ``None`` when the word is out of the GloVe vocabulary.
    """
    rows: list[dict[str, Any]] = []
    for individual in individuals:
        if which == "child_best":
            word = individual.child_best_word
            real = individual.child_best_rank
        else:
            word = individual.proposed_word
            real = individual.realized_rank
        if not word:
            continue
        glove = ranker(word)
        rows.append(
            {
                "word": word,
                "real_rank": real,
                "glove_rank": glove,
                "predicted_closeness": individual.predicted_closeness,
            }
        )
    return rows


def mediator_metrics(
    individuals: Sequence[Individual], ranker: RankFn, which: str = "first_proposed"
) -> dict[str, Any]:
    """The three Spearman correlations plus GloVe-vocabulary coverage."""
    rows = mediator_rows(individuals, ranker, which)
    total_words = len(rows)
    in_vocab = [row for row in rows if row["glove_rank"] is not None]

    real_glove = [(r["glove_rank"], r["real_rank"]) for r in rows if r["glove_rank"] is not None and r["real_rank"] is not None]
    close_real = [(r["predicted_closeness"], r["real_rank"]) for r in rows if r["predicted_closeness"] is not None and r["real_rank"] is not None]
    close_glove = [(r["predicted_closeness"], r["glove_rank"]) for r in rows if r["predicted_closeness"] is not None and r["glove_rank"] is not None]

    return {
        "which_realized": which,
        "words_total": total_words,
        "words_in_glove_vocab": len(in_vocab),
        "glove_vocab_coverage": (len(in_vocab) / total_words) if total_words else None,
        "glove_vs_real": _spearman([p[0] for p in real_glove], [p[1] for p in real_glove]),
        "closeness_vs_real": _spearman([p[0] for p in close_real], [p[1] for p in close_real]),
        "closeness_vs_glove": _spearman([p[0] for p in close_glove], [p[1] for p in close_glove]),
    }


class GloveRanker:
    """Adapt a ``LocalGame`` ranking dict into a ``ranker(word) -> int | None``.

    Kept out of the metric functions so those stay embedding-free; the CLI builds
    one per run from a shared :class:`~contexto_solver.local_game.LocalGame`.
    """

    def __init__(self, rankings: dict[str, int]) -> None:
        self._rankings = rankings

    def __call__(self, word: str) -> int | None:
        return self._rankings.get(word.lower().strip())
