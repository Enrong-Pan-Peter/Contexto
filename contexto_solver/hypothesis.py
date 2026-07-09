"""Hypothesis model for the evolutionary Contexto solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np


N_OPERATOR_SIGMA_COMPONENTS = 4


@dataclass
class Hypothesis:
    """A semantic category being explored by the solver."""

    category_name: str
    description: str
    words_tried: dict[str, int] = field(default_factory=dict)
    status: str = "active"
    parent: str | None = None
    origin: str = "init"
    hypothesis_id: str = field(default_factory=lambda: uuid4().hex)
    parent_id: str | None = None
    sigma: np.ndarray = field(default_factory=lambda: np.full(N_OPERATOR_SIGMA_COMPONENTS, 0.25, dtype=np.float64))
    coordinates: tuple[float, float] | None = None
    cell: tuple[int, int] | None = None
    # RQ1 operator self-report instrumentation (logged-only; see self_report.py).
    # All default to None/False so hypotheses in non-instrumented methods and
    # runs serialize byte-identically to before.
    predicted_closeness: float | None = None
    predicted_closeness_clamped: bool = False
    predicted_bucket: str | None = None
    rationale: dict | None = None
    self_report_parse_failed: bool = False
    self_report_raw: str | None = None
    self_report_prompt: str | None = None
    injected_rationale_hash: str | None = None
    rationale_truncated: bool = False

    def __post_init__(self) -> None:
        self.sigma = self._validate_sigma(self.sigma)

    @property
    def best_rank(self) -> int:
        if not self.words_tried:
            return 1_000_000_000
        return min(self.words_tried.values())

    @property
    def best_word(self) -> str | None:
        if not self.words_tried:
            return None
        return min(self.words_tried, key=self.words_tried.get)

    def update(self, word: str, rank: int) -> None:
        self.words_tried[word.lower().strip()] = rank

    def set_sigma(self, sigma: np.ndarray) -> None:
        self.sigma = self._validate_sigma(sigma)

    def to_dict(self) -> dict:
        payload = {
            "hypothesis_id": self.hypothesis_id,
            "category_name": self.category_name,
            "description": self.description,
            "words_tried": dict(sorted(self.words_tried.items(), key=lambda item: item[1])),
            "best_word": self.best_word,
            "best_rank": self.best_rank,
            "status": self.status,
            "parent": self.parent,
            "parent_id": self.parent_id,
            "origin": self.origin,
            "sigma": [float(value) for value in self.sigma],
        }
        # Only emitted by archive-based methods (e.g. MAP-Elites); omitted
        # otherwise so existing methods' traces stay byte-identical.
        if self.coordinates is not None:
            payload["coordinates"] = list(self.coordinates)
        if self.cell is not None:
            payload["cell"] = list(self.cell)
        # Only emitted when self-report instrumentation attached a record, so
        # non-instrumented runs' traces stay byte-identical.
        if self._has_self_report():
            payload["self_report"] = self.self_report_dict()
        return payload

    def _has_self_report(self) -> bool:
        return (
            self.predicted_closeness is not None
            or self.predicted_bucket is not None
            or self.rationale is not None
            or self.self_report_raw is not None
            or self.self_report_prompt is not None
            or self.injected_rationale_hash is not None
            or self.rationale_truncated
            or self.self_report_parse_failed
        )

    def self_report_dict(self) -> dict:
        # Every self-report record carries a uniform key set across all modes:
        # injected_rationale_hash (null when no inheritance injection) and
        # rationale_truncated are always emitted, matching the llm_only writer
        # (resolve_self_report) so schema identity holds without a version bump.
        return {
            "predicted_closeness": self.predicted_closeness,
            "predicted_closeness_clamped": self.predicted_closeness_clamped,
            "predicted_bucket": self.predicted_bucket,
            "rationale": self.rationale,
            "self_report_parse_failed": self.self_report_parse_failed,
            "self_report_raw": self.self_report_raw,
            "self_report_prompt": self.self_report_prompt,
            "injected_rationale_hash": self.injected_rationale_hash,
            "rationale_truncated": self.rationale_truncated,
        }

    @staticmethod
    def _validate_sigma(sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, dtype=np.float64)
        assert sigma.shape == (N_OPERATOR_SIGMA_COMPONENTS,)
        assert np.isclose(sigma.sum(), 1.0, atol=1e-6)
        return sigma

