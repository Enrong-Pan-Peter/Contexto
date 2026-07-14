"""Tidy per-individual table: columns, row projection, and CSV writer.

One row per self-reported individual (see :class:`~contexto_solver.rq1.reader.Individual`).
The column order is stable so downstream tooling (and pgfplots) can rely on it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .reader import Individual

# Stable column order for the tidy table. Grouped run-level -> individual-level
# -> predicted -> realized.
TIDY_COLUMNS = [
    # run-level
    "trace_file",
    "game",
    "method",
    "game_number",
    "target",
    "target_source",
    "provenance_hash",
    "trace_schema_version",
    "sigma_mode",
    # individual-level
    "generation",
    "source_event",
    "operator",
    "hypothesis_name",
    "parent_id",
    "parent_rank",
    # predicted (self-report)
    "predicted_closeness",
    "predicted_closeness_clamped",
    "predicted_bucket",
    "self_report_parse_failed",
    "injected_rationale_hash",
    "rationale_inherited",
    "rationale_truncated",
    "basis_words_count",
    # realized
    "proposed_word",
    "realized_rank",
    "proposed_word_invalid",
    "child_best_word",
    "child_best_rank",
]


def to_tidy_row(individual: Individual) -> dict[str, Any]:
    """Project an :class:`Individual` into a flat dict keyed by ``TIDY_COLUMNS``."""
    return {
        "trace_file": individual.trace_file,
        "game": individual.game,
        "method": individual.method,
        "game_number": individual.game_number,
        "target": individual.target,
        "target_source": individual.target_source,
        "provenance_hash": individual.provenance_hash,
        "trace_schema_version": individual.trace_schema_version,
        "sigma_mode": individual.sigma_mode,
        "generation": individual.generation,
        "source_event": individual.source_event,
        "operator": individual.operator,
        "hypothesis_name": individual.hypothesis_name,
        "parent_id": individual.parent_id,
        "parent_rank": individual.parent_rank,
        "predicted_closeness": individual.predicted_closeness,
        "predicted_closeness_clamped": individual.predicted_closeness_clamped,
        "predicted_bucket": individual.predicted_bucket,
        "self_report_parse_failed": individual.self_report_parse_failed,
        "injected_rationale_hash": individual.injected_rationale_hash,
        "rationale_inherited": individual.rationale_inherited,
        "rationale_truncated": individual.rationale_truncated,
        "basis_words_count": individual.basis_words_count,
        "proposed_word": individual.proposed_word,
        "realized_rank": individual.realized_rank,
        "proposed_word_invalid": individual.proposed_word_invalid,
        "child_best_word": individual.child_best_word,
        "child_best_rank": individual.child_best_rank,
    }


@dataclass(frozen=True)
class ProvenanceReport:
    """Summary of provenance-hash agreement across a set of individuals."""

    hashes: dict[str | None, int]
    mixed: bool

    @property
    def distinct(self) -> list[str | None]:
        return sorted(self.hashes, key=lambda value: (value is None, str(value)))


def provenance_hashes(individuals: Iterable[Individual]) -> ProvenanceReport:
    """Tally ``instrumentation_provenance_hash`` values across individuals.

    ``mixed`` is True when more than one distinct hash is present, which means the
    prompt/instrumentation substrate differed between traces and the records must
    NOT be pooled without accounting for it. Callers decide whether to warn or
    hard-fail on ``mixed``.
    """
    counts: dict[str | None, int] = {}
    for individual in individuals:
        counts[individual.provenance_hash] = counts.get(individual.provenance_hash, 0) + 1
    return ProvenanceReport(hashes=counts, mixed=len(counts) > 1)


def write_tidy_csv(path: str | Path, individuals: Iterable[Individual]) -> Path:
    """Write the tidy per-individual table to ``path`` and return it."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIDY_COLUMNS)
        writer.writeheader()
        for individual in individuals:
            writer.writerow(to_tidy_row(individual))
    return out_path
