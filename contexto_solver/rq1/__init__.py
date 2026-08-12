"""RQ1 calibration analysis library (read-only over solver traces).

This package holds the analysis-side building blocks for the RQ1/RQ3 audit
batch: reading solver traces, joining each self-reported individual to its
realized game rank, computing calibration metrics, an offline GloVe mediator
comparison, and cross-run reporting. Nothing here calls the network, runs the
solver, contacts an LLM, or writes to traces/caches; the modules only read the
trace and rank-cache files that already exist on disk.

Thin CLIs live in ``scripts/rq1_*.py`` and import from here so the logic stays
importable and unit-testable against synthetic fixtures.
"""

from __future__ import annotations

from .reader import (
    Individual,
    RunConfig,
    extract_individuals,
    load_trace,
    read_run,
    recover_target,
    run_config,
)
from .records import (
    TIDY_COLUMNS,
    provenance_hashes,
    to_tidy_row,
    write_tidy_csv,
)

__all__ = [
    "Individual",
    "RunConfig",
    "extract_individuals",
    "load_trace",
    "read_run",
    "recover_target",
    "run_config",
    "TIDY_COLUMNS",
    "provenance_hashes",
    "to_tidy_row",
    "write_tidy_csv",
]
