"""EA+LLM Contexto method without stall-pivot operators."""

from __future__ import annotations

from .ea_core import BaseEALLMMethod, EALLMConfig


class EALLMMethod(BaseEALLMMethod):
    """Evolutionary LLM method with selection, mutation, crossover, and local search."""


SolverLLM = EALLMMethod
SolverLLMConfig = EALLMConfig

