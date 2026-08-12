"""MAP-Elites variant of the self-adaptive EA+LLM Contexto method.

Replaces top-mu selection with an archive over a grid of behavior cells. Each
cell holds at most one elite hypothesis. A hypothesis's behavior coordinates are
derived from an LLM placement call on its single ``best_word`` and are fixed at
creation. The sigma self-adaptation machinery is inherited unchanged from the
self-adaptive parent method.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hypothesis import Hypothesis
from ..operators import (
    N_OPERATORS,
    OPERATOR_PROMPTS,
    assert_prompt_has_no_sigma_leak,
    initial_sigma,
    perturb_sigma,
    sample_operator,
    validate_sigma,
)
from .ea_core import _clean_word, _words_from_category
from .ea_llm_self_adaptive import EALLMSelfAdaptiveConfig, EALLMSelfAdaptiveMethod


@dataclass
class EALLMMapElitesConfig(EALLMSelfAdaptiveConfig):
    grid_resolution: int = 5
    mutations_per_gen: int = 15
    crossovers_per_gen: int = 5
    placement_cache_dir: str = "data/placement_cache"
    anchors_concreteness: dict[float, str] = field(default_factory=dict)
    anchors_specificity: dict[float, str] = field(default_factory=dict)
    sigma_mode: str = "adaptive"
    frozen_sigma: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    ranked_context_k: int = 0


class EALLMMapElitesMethod(EALLMSelfAdaptiveMethod):
    """Archive-based MAP-Elites search layered on the self-adaptive operators."""

    config: EALLMMapElitesConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.archive: dict[tuple[int, int], Hypothesis] = {}
        self._placement_cache: dict[str, list[float]] = {}
        self._placement_lookups = 0
        self._placement_hits = 0
        self._word_ranks: dict[str, int] = {}
        self._anchors_hash = self._compute_anchors_hash()
        self._cache_path = self._placement_cache_path()
        self._load_placement_cache()

    # --- archive-aware overrides -------------------------------------------------

    def _active_hypotheses(self) -> list[Hypothesis]:
        """Treat the archive incumbents as the live population.

        Keeps inherited sigma-trajectory logging meaningful without relying on
        the unused ``status`` field.
        """
        return list(self.archive.values())

    def initialize(self) -> bool:
        self.logger.log_axis_definition(
            self.generation,
            grid_resolution=self.config.grid_resolution,
            anchors_concreteness=self.config.anchors_concreteness,
            anchors_specificity=self.config.anchors_specificity,
        )
        categories = self.llm_client.generate_initial_categories(
            n=self.config.initial_categories,
            starter_words=1,
        )
        for category in categories:
            if not isinstance(category, dict):
                continue
            child = self._hypothesis_from_category(
                category, origin="init", sigma=self._mode_sigma(None)
            )
            word = self._guess_first_valid(_words_from_category(category), child)
            if word is None:
                continue
            self.hypotheses.append(child)
            if self.game.is_solved():
                self._log_solved()
                return True
            self._place_and_compete(child)

        self.logger.log(
            self.generation,
            "INIT",
            {
                "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
                "occupied_cells": len(self.archive),
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        if not self.game.is_solved() and not self.archive:
            raise RuntimeError(
                "MAP-Elites initialization produced an empty archive: none of the "
                f"{len(categories)} initial categories yielded a valid guessed word. "
                "Refusing to run empty generations."
            )
        return self.game.is_solved()

    def run_generation(self) -> bool:
        self.generation += 1

        for _ in range(self.config.mutations_per_gen):
            parents = self._sample_parents(1)
            if not parents:
                break
            child = self._mutation_child(parents[0])
            if child is None:
                continue
            if self.game.is_solved():
                self._log_solved()
                return True
            self._place_and_compete(child)
            if self.game.is_solved():
                self._log_solved()
                return True

        for _ in range(self.config.crossovers_per_gen):
            pair = self._sample_parents(2)
            if len(pair) < 2:
                break
            child = self._crossover_child(pair[0], pair[1])
            if child is None:
                continue
            if self.game.is_solved():
                self._log_solved()
                return True
            self._place_and_compete(child)
            if self.game.is_solved():
                self._log_solved()
                return True

        self._log_archive_snapshot()
        self._log_sigma_trajectory()
        return False

    # --- child production --------------------------------------------------------

    def _mode_sigma(self, base_sigma: Any | None):
        """Return a child sigma per the configured sigma mode.

        ``base_sigma`` is the parent sigma (mutation) or the blended parent sigma
        (crossover), or ``None`` for initial hypotheses. Only ``adaptive`` uses it;
        the frozen and random modes ignore it so the operator-firing distribution
        is held fixed or randomized. Applied at every creation site (init included)
        so frozen/random arms do not start generation 0 from a uniform parent.
        """
        mode = self.config.sigma_mode
        if mode == "frozen_uniform":
            return initial_sigma()
        if mode == "frozen_fixed":
            return validate_sigma(self.config.frozen_sigma)
        if mode == "random":
            return self.rng.dirichlet([1.0] * N_OPERATORS)
        # adaptive (default, current behavior)
        if base_sigma is None:
            return initial_sigma()
        return perturb_sigma(
            base_sigma,
            concentration=self.config.concentration,
            sigma_floor=self.config.sigma_floor,
            rng=self.rng,
        )

    def _mutation_child(self, parent: Hypothesis) -> Hypothesis | None:
        operator = sample_operator(parent.sigma, self.rng)
        prompt_template = OPERATOR_PROMPTS[operator]
        parent_sigma = parent.sigma.copy()
        inheritance_block, inheritance_meta = self._rationale_inheritance_for_parent(parent)
        prompt = self.llm_client.build_operator_mutation_prompt(
            prompt_template,
            parent,
            self._known_words(),
            self.invalid_guesses,
            n=1,
            active_categories=[hypothesis.category_name for hypothesis in self.archive.values()],
            ranked_context=self._render_ranked_context(),
            rationale_inheritance_block=inheritance_block,
            self_report_block=self._self_report_block(),
        )
        assert_prompt_has_no_sigma_leak(prompt, parent_sigma, operator)
        category, raw = self._complete_proposal(prompt)
        if not isinstance(category, dict):
            return None

        child_sigma = self._mode_sigma(parent_sigma)
        child = self._hypothesis_from_category(
            category,
            parent=parent.category_name,
            origin=f"self_adaptive_{operator.value}",
            parent_id=parent.hypothesis_id,
            sigma=child_sigma,
        )
        word = self._guess_first_valid(_words_from_category(category), child)
        if word is None:
            return None
        self.hypotheses.append(child)
        if self.config.self_report:
            self._attach_self_report(
                child,
                category,
                raw,
                prompt,
                word,
                inheritance_meta=inheritance_meta if inheritance_block else None,
            )
        operator_details: dict[str, Any] = {
            "parent_id": parent.hypothesis_id,
            "child_id": child.hypothesis_id,
            "parent_rank": parent.best_rank,
            "sigma_snapshot": [float(value) for value in parent_sigma],
            "child_sigma": [float(value) for value in child.sigma],
            "child_hypothesis_name": child.category_name,
            "sampled_op": operator.value,
            "method": "map_elites",
        }
        if self.config.self_report:
            operator_details["self_report"] = child.self_report_dict()
        self.logger.log(self.generation, "OPERATOR_SAMPLED", operator_details)
        return child

    def _crossover_child(self, parent_a: Hypothesis, parent_b: Hypothesis) -> Hypothesis | None:
        category, raw, rendered_prompt = self._crossover_request(parent_a, parent_b)
        if not isinstance(category, dict):
            return None
        blended_sigma = 0.5 * (parent_a.sigma + parent_b.sigma)
        child_sigma = self._mode_sigma(blended_sigma)
        child = self._hypothesis_from_category(
            category,
            parent=f"{parent_a.category_name}+{parent_b.category_name}",
            origin="crossover",
            sigma=child_sigma,
        )
        word = self._guess_first_valid(_words_from_category(category), child)
        if word is None:
            return None
        self.hypotheses.append(child)
        if self.config.self_report:
            self._attach_self_report(child, category, raw, rendered_prompt, word)
        self.logger.log(
            self.generation,
            "CROSSOVER",
            {
                "parents": [parent_a.category_name, parent_b.category_name],
                "parent_ids": [parent_a.hypothesis_id, parent_b.hypothesis_id],
                "parent_ranks": [parent_a.best_rank, parent_b.best_rank],
                "parent_a_sigma": [float(value) for value in parent_a.sigma],
                "parent_b_sigma": [float(value) for value in parent_b.sigma],
                "child_sigma_pre_perturbation": [float(value) for value in blended_sigma],
                "child": child.to_dict(),
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )
        return child

    def _guess_first_valid(self, words: list[str], hypothesis: Hypothesis) -> str | None:
        known = self._known_words()
        for word in words:
            cleaned_word = _clean_word(word)
            if not cleaned_word or cleaned_word in self.invalid_guesses or cleaned_word in known:
                continue
            rank = self._guess_and_update(cleaned_word, hypothesis)
            if self.game.is_solved():
                self._word_ranks[cleaned_word] = rank
                return cleaned_word
            if rank != -1:
                self._word_ranks[cleaned_word] = rank
                return cleaned_word
        return None

    def _render_ranked_context(self) -> str:
        """Render the global top-K best-ranked guesses for mutation prompts.

        These are game ranks (feedback), not sigma, so the leakage invariant still
        holds. Mirrors the ``{all_guesses}`` slot: the guessed vocabulary is
        injected without reserved-substring filtering. Returns an empty string
        (byte-identical prompt) when the feature is off.
        """
        k = self.config.ranked_context_k
        if k <= 0 or not self._word_ranks:
            return ""
        best = sorted(self._word_ranks.items(), key=lambda item: item[1])[:k]
        rendered = ", ".join(f"{word} ({rank})" for word, rank in best)
        return f"\nClosest words found so far: {rendered}"

    # --- placement and competition ----------------------------------------------

    def _place_and_compete(self, child: Hypothesis) -> None:
        word = child.best_word
        if word is None:
            return
        coordinates, cell = self._place(word)
        child.coordinates = coordinates
        child.cell = cell
        self._compete(child, cell)

    def _place(self, word: str) -> tuple[tuple[float, float], tuple[int, int]]:
        self._placement_lookups += 1
        cached = self._placement_cache.get(word)
        if cached is not None:
            coordinates = (float(cached[0]), float(cached[1]))
            cache_hit = True
            self._placement_hits += 1
        else:
            result = self.llm_client.place_word(
                word,
                self.config.anchors_concreteness or None,
                self.config.anchors_specificity or None,
            )
            coordinates = (float(result["concreteness"]), float(result["specificity"]))
            self._placement_cache[word] = [coordinates[0], coordinates[1]]
            self._persist_placement_cache()
            cache_hit = False
        cell = (self._cell_for(coordinates[0]), self._cell_for(coordinates[1]))
        self.logger.log_placement(self.generation, word, coordinates, cell, cache_hit)
        return coordinates, cell

    def _cell_for(self, coordinate: float) -> int:
        resolution = self.config.grid_resolution
        return int(min(resolution - 1, max(0, math.floor(coordinate * resolution))))

    def _compete(self, child: Hypothesis, cell: tuple[int, int]) -> None:
        incumbent = self.archive.get(cell)
        if incumbent is None:
            self.archive[cell] = child
            self.logger.log(
                self.generation,
                "ARCHIVE_PLACE",
                {
                    "cell": [int(cell[0]), int(cell[1])],
                    "hypothesis_id": child.hypothesis_id,
                    "best_word": child.best_word,
                    "rank": child.best_rank,
                    "sigma": [float(value) for value in child.sigma],
                },
            )
            return

        if child.best_rank < incumbent.best_rank:
            self.archive[cell] = child
            self.logger.log(
                self.generation,
                "ARCHIVE_REPLACE",
                {
                    "cell": [int(cell[0]), int(cell[1])],
                    "old_hypothesis_id": incumbent.hypothesis_id,
                    "new_hypothesis_id": child.hypothesis_id,
                    "old_rank": incumbent.best_rank,
                    "new_rank": child.best_rank,
                    "old_sigma": [float(value) for value in incumbent.sigma],
                    "new_sigma": [float(value) for value in child.sigma],
                },
            )
            return

        self.logger.log(
            self.generation,
            "ARCHIVE_REJECT",
            {
                "cell": [int(cell[0]), int(cell[1])],
                "child_hypothesis_id": child.hypothesis_id,
                "incumbent_hypothesis_id": incumbent.hypothesis_id,
                "child_rank": child.best_rank,
                "incumbent_rank": incumbent.best_rank,
            },
        )

    # --- parent sampling ---------------------------------------------------------

    def _sample_parents(self, count: int) -> list[Hypothesis]:
        occupied = list(self.archive.values())
        if not occupied:
            return []
        indices = self.rng.integers(0, len(occupied), size=count)
        return [occupied[int(index)] for index in indices]

    # --- placement cache ---------------------------------------------------------

    def _compute_anchors_hash(self) -> str:
        items = [
            ["concreteness", round(float(position), 6), word]
            for position, word in self.config.anchors_concreteness.items()
        ]
        items += [
            ["specificity", round(float(position), 6), word]
            for position, word in self.config.anchors_specificity.items()
        ]
        items.sort()
        payload = json.dumps(items, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def _placement_cache_path(self) -> Path:
        safe_model = re.sub(r"[^A-Za-z0-9._-]", "-", self.llm_client.model)
        return Path(self.config.placement_cache_dir) / f"{safe_model}_{self._anchors_hash}.json"

    def _load_placement_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            self._placement_cache = {
                str(word): [float(value[0]), float(value[1])]
                for word, value in data.items()
                if isinstance(value, list) and len(value) == 2
            }

    def _persist_placement_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._placement_cache, indent=2), encoding="utf-8")

    @property
    def placement_cache_hit_rate(self) -> float | None:
        if self._placement_lookups == 0:
            return None
        return self._placement_hits / self._placement_lookups

    # --- logging -----------------------------------------------------------------

    def _log_archive_snapshot(self) -> None:
        cells = []
        for cell, hypothesis in sorted(self.archive.items()):
            cells.append(
                {
                    "cell": [int(cell[0]), int(cell[1])],
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "best_word": hypothesis.best_word,
                    "best_rank": hypothesis.best_rank,
                    "coordinates": list(hypothesis.coordinates) if hypothesis.coordinates is not None else None,
                    "sigma": [float(value) for value in hypothesis.sigma],
                }
            )
        self.logger.log(
            self.generation,
            "ARCHIVE_SNAPSHOT",
            {
                "occupied_cells": len(cells),
                "cells": cells,
                "best_word": self.best_word,
                "best_rank": self.best_rank,
                "total_guesses": self.game.total_guesses(),
            },
        )


SolverLLMMapElites = EALLMMapElitesMethod
SolverLLMMapElitesConfig = EALLMMapElitesConfig
