"""Self-adaptive mutation operator definitions and sigma utilities."""

from __future__ import annotations

from enum import Enum

import numpy as np

from .llm_client import L_MUTATION_PROMPT, M_MUTATION_PROMPT, ML_MUTATION_PROMPT, S_MUTATION_PROMPT


class Operator(str, Enum):
    S_MUTATION = "s_mutation"
    M_MUTATION = "m_mutation"
    ML_MUTATION = "ml_mutation"
    L_MUTATION = "l_mutation"


OPERATORS = [
    Operator.S_MUTATION,
    Operator.M_MUTATION,
    Operator.ML_MUTATION,
    Operator.L_MUTATION,
]
N_OPERATORS = 4

OPERATOR_PROMPTS: dict[Operator, str] = {
    Operator.S_MUTATION: S_MUTATION_PROMPT,
    Operator.M_MUTATION: M_MUTATION_PROMPT,
    Operator.ML_MUTATION: ML_MUTATION_PROMPT,
    Operator.L_MUTATION: L_MUTATION_PROMPT,
}

OPERATOR_DISTINGUISHING_PHRASES: dict[Operator, str] = {
    Operator.S_MUTATION: "SMALL mutation",
    Operator.M_MUTATION: "MEDIUM mutation",
    Operator.ML_MUTATION: "MEDIUM-LARGE mutation",
    Operator.L_MUTATION: "LARGE mutation",
}


def initial_sigma() -> np.ndarray:
    return np.full(N_OPERATORS, 1.0 / N_OPERATORS, dtype=np.float64)


def validate_sigma(sigma: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64)
    assert sigma.shape == (N_OPERATORS,)
    assert np.isclose(sigma.sum(), 1.0, atol=1e-6)
    return sigma


def sample_operator(sigma: np.ndarray, rng: np.random.Generator) -> Operator:
    sigma = validate_sigma(sigma)
    index = rng.choice(N_OPERATORS, p=sigma)
    return OPERATORS[int(index)]


def perturb_sigma(
    parent_sigma: np.ndarray,
    concentration: float,
    sigma_floor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb a parent's operator probabilities with a floored Dirichlet draw."""
    parent_sigma = validate_sigma(parent_sigma)
    if concentration <= 0:
        raise ValueError("concentration must be positive.")
    if sigma_floor < 0 or sigma_floor * N_OPERATORS >= 1:
        raise ValueError("sigma_floor must be non-negative and leave probability mass to normalize.")

    alpha = np.clip(concentration * parent_sigma, 1e-6, None)
    child = rng.dirichlet(alpha)
    child = np.maximum(child, sigma_floor)
    excess = child - sigma_floor
    if np.isclose(excess.sum(), 0.0):
        child = initial_sigma()
    else:
        child = sigma_floor + excess / excess.sum() * (1.0 - sigma_floor * N_OPERATORS)
    return validate_sigma(child)


def assert_prompt_has_no_sigma_leak(prompt: str, sigma: np.ndarray, operator: Operator) -> None:
    phrase = OPERATOR_DISTINGUISHING_PHRASES[operator]
    if phrase not in prompt:
        raise AssertionError(f"Prompt for {operator.value} is missing distinguishing phrase {phrase!r}.")

    lowered = prompt.lower()
    forbidden_substrings = ("sigma", "σ", "probability")
    for substring in forbidden_substrings:
        if substring in lowered:
            raise AssertionError(f"Prompt for {operator.value} leaked forbidden substring {substring!r}.")

    for value in validate_sigma(sigma):
        token = f"{value:.2f}"
        if token in prompt:
            raise AssertionError(f"Prompt for {operator.value} leaked sigma numeric literal {token!r}.")
