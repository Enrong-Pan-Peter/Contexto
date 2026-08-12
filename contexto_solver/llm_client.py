"""LLM API wrapper used to generate Contexto search hypotheses."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from . import config
from .hypothesis import Hypothesis


# Minimum viable initial seed set. Every configured caller requests at least 6
# categories (INITIAL_CATEGORIES=6, MAPELITES/SELF_ADAPTIVE_INITIAL_CATEGORIES=15),
# so these are conservative floors: they only reject the degenerate single-object
# collapse (1 category) that Ollama's json_object mode produces, without failing a
# legitimate under-delivering response.
MIN_INITIAL_CATEGORIES = 2
MIN_INITIAL_SEED_WORDS = 2


INITIAL_CATEGORIES_PROMPT = """Return only JSON, no markdown or explanation.
Generate {n} broad semantic categories for exploring a Contexto puzzle.
Each category must include exactly {starter_words} common starter words.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Invalid examples: up-to-date, sour cream, sourcream, wildanimal, dairyproduct.
JSON schema:
{{"categories": [{{"name": "category name", "description": "short description", "words": ["word1", "word2", "word3"]}}]}}"""

PROPOSE_WORDS_PROMPT = """Return only JSON, no markdown or explanation.
Contexto ranks words by semantic closeness. Rank 1 is correct. Lower is better.
Current hypothesis: {name}
Description: {description}
Words already tried in this hypothesis with ranks: {hypothesis_words_tried}
Global best word: {best_word}
Global best rank: {best_rank}
Avoid all already tried words: {all_guesses}
Avoid these invalid or unrecognized words: {invalid_guesses}
Suggest {n} new single-word guesses for this hypothesis.
Every guess must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
Invalid examples: up-to-date, sour cream, sourcream, wildanimal, dairyproduct.
JSON schema:
{{"words": ["word1", "word2", "word3"]}}"""

SPECIALIZE_PROMPT = """Return only JSON, no markdown or explanation.
The category "{name}" had these results: {words_tried}.
The word "{best_word}" scored best with rank {best_rank}.

Suggest {n} new directions to explore. These should NOT all be sub-categories
of "{name}". At least one should be a genuinely different interpretation of
why "{best_word}" might have scored well.

For example, if the category was "food" and "bite" scored well, one direction
could be "small food portions" but another could be "animals that bite" since
"bite" has multiple meanings.

Each direction must include exactly 3 starter words that have not already been tried.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these words: {all_guesses}
Avoid these invalid or unrecognized words: {invalid_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
Invalid examples: up-to-date, sour cream, sourcream, wildanimal, dairyproduct.
JSON schema:
{{"specializations": [{{"name": "direction name", "description": "short description", "words": ["word1", "word2", "word3"]}}]}}"""

S_MUTATION_PROMPT = """Return only JSON, no markdown or explanation.
The current hypothesis is "{name}".
Description: {description}
Results so far in this hypothesis: {words_tried}
Best word so far: "{best_word}" with rank {best_rank}.{ranked_context}

Make a SMALL mutation: produce a child hypothesis that stays in the same
conceptual neighborhood as "{name}" but narrows or refines it. The new starter
words should be close semantic neighbors of "{best_word}" - synonyms, common
descriptors, or words that frequently co-occur with it.

For example, if the parent is "types of plants" and the best word is "shrub",
a small mutation could be "woody plants" with starters like bush, hedge, thicket.

Suggest one refined hypothesis. Include exactly {n} starter words that have not
already been tried.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these words: {all_guesses}
Avoid these invalid or unrecognized words: {invalid_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"name": "direction name", "description": "short description", "words": ["word1", "word2", "word3"]}}"""

M_MUTATION_PROMPT = """Return only JSON, no markdown or explanation.
The current hypothesis is "{name}".
Description: {description}
Results so far in this hypothesis: {words_tried}
Best word so far: "{best_word}" with rank {best_rank}.{ranked_context}

Make a MEDIUM mutation: produce a child hypothesis that reinterprets why
"{best_word}" might have scored well, or shifts to a related sense, lexical
register, or part of speech. The child should still be semantically anchored
to "{best_word}" but approach it from a clearly different angle than "{name}".

For example, if the parent is "food" and the best word is "bite", a medium
mutation could be "animals that bite", "physical sensations from biting", or
"clinical terms for biting and chewing" - each treats "bite" through a
different lens.

Suggest one reframed hypothesis. Include exactly {n} starter words that have
not already been tried.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these words: {all_guesses}
Avoid these invalid or unrecognized words: {invalid_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"name": "direction name", "description": "short description", "words": ["word1", "word2", "word3"]}}"""

ML_MUTATION_PROMPT = """Return only JSON, no markdown or explanation.
The current hypothesis is "{name}".
Description: {description}
Results so far in this hypothesis: {words_tried}
Best word so far: "{best_word}" with rank {best_rank}.{ranked_context}

Make a MEDIUM-LARGE mutation: produce a child hypothesis in an ADJACENT but
distinct category that could still plausibly contain the hidden target given
"{best_word}"'s rank. The child must NOT be a sub-category of "{name}" - it
should sit alongside "{name}" in semantic space, not underneath it.

For example, if the parent is "types of plants" and the best word is "shrub",
an adjacent category could be "plant descriptors", "botanical terminology",
"growth habits", or "ecological roles".

Suggest one adjacent hypothesis. Include exactly {n} starter words that have
not already been tried.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these words: {all_guesses}
Avoid these invalid or unrecognized words: {invalid_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"name": "category name", "description": "short description", "words": ["word1", "word2", "word3"]}}"""

L_MUTATION_PROMPT = """Return only JSON, no markdown or explanation.
The current hypothesis is "{name}".
Description: {description}
Best word so far: "{best_word}" with rank {best_rank}.{ranked_context}
Other active categories already being explored: {active_categories}

Make a LARGE mutation: produce a child hypothesis in a broad, semantically
substantial new direction that is unlike "{name}" and unlike the other active
categories above. The goal is to escape the current semantic region and open
up unexplored territory, while staying plausibly relevant given "{best_word}"'s
rank - lower ranks mean the target is closer to "{best_word}", so the jump
must remain in the same broad semantic area even when reframed.

For example, if the parent is "types of plants" and "shrub" ranks 50, a large
mutation could be "outdoor environments" or "rural landscape features" -
clearly outside the plant taxonomy but still in a plausible neighborhood for
a target close to "shrub".

Suggest one new hypothesis. Include exactly {n} starter words that have not
already been tried.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these words: {all_guesses}
Avoid these invalid or unrecognized words: {invalid_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"name": "direction name", "description": "short description", "words": ["word1", "word2", "word3"]}}"""

CROSSOVER_PROMPT = """Return only JSON, no markdown or explanation.
Category A is {a_name} with results {a_words}.
Category B is {b_name} with results {b_words}.
Suggest one new category that captures the intersection or overlap of A and B.
Include exactly 3 candidate words.
Every word must be one common lowercase dictionary word.
JSON schema:
{{"name": "category name", "description": "short description", "words": ["word1", "word2", "word3"]}}"""

LOCAL_SEARCH_PROMPT = """Return only JSON, no markdown or explanation.
The word {word} has rank {rank} in a word similarity game, meaning it is close to the target.
Suggest {n} words that could be even closer to the hidden target than {word}.
Do not only return synonyms, group members, or subtypes of {word}.
Explore multiple relation types: synonyms, common adjectives/descriptors, collocations,
associated people or groups, causes/effects, and words that commonly appear near {word}.
If {word} is a noun, include plausible adjectives or descriptors associated with it.
Every word must be one common lowercase dictionary word.
Avoid these already tried words: {all_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"words": ["word1", "word2", "word3"]}}"""

PIVOT_MORPHOLOGY_PROMPT = """Return only JSON, no markdown or explanation.
The word {word} has rank {rank}, so it is very close to the hidden target.
Suggest {n} candidate words that pivot across morphology, word forms, parts of speech,
botanical/scientific terms, descriptors, hypernyms, and hyponyms.
For example, from shrub, useful candidates could include shrubby, shrubland,
herbaceous, woody, perennial, bush, foliage.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these already tried words: {all_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"words": ["word1", "word2", "word3"]}}"""

PIVOT_REGISTER_SHIFT_PROMPT = """Return only JSON, no markdown or explanation.
The hidden target is semantically close to {word}, which has rank {rank}, but it may be
in a different lexical register: more technical, more general, more specific, or a
different part of speech.
Suggest {n} candidate single lowercase words across these registers.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these already tried words: {all_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"words": ["word1", "word2", "word3"]}}"""

PIVOT_ADJACENT_CATEGORY_PROMPT = """Return only JSON, no markdown or explanation.
The current best word is {word} with rank {rank}.
Current category: {category_name}
Category description: {category_description}
Words already explored near this category: {words_tried}

Suggest one adjacent but distinct category that might contain the hidden target.
For example, if the current category is "types of plants" and the best word is "shrub",
use an adjacent category such as "plant descriptors", "botanical terminology", or "growth habits".
Include exactly {n} candidate words for that category.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these already tried words: {all_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"name": "category name", "description": "short description", "words": ["word1", "word2", "word3"]}}"""

PIVOT_FRESH_ADJACENT_CATEGORY_PROMPT = """Return only JSON, no markdown or explanation.
The solver is stalled and needs a fresh semantic direction.
Current best word: {word}
Current best rank: {rank}
Existing active categories: {active_categories}

Suggest one broad but relevant adjacent category that is unlike the existing active categories
and could still lead toward the hidden target.
Include exactly {n} candidate words for that category.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Avoid these already tried words: {all_guesses}
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
JSON schema:
{{"name": "category name", "description": "short description", "words": ["word1", "word2", "word3"]}}"""

PLACE_WORD_PROMPT = """Return only JSON, no markdown or explanation.
Rate the word "{word}" on two scales.

Concreteness (0 = most concrete/physical, 1 = most abstract/conceptual):
{concreteness_anchors}

Specificity (0 = most general, 1 = most specific):
{specificity_anchors}

Return JSON only: {{"concreteness": <number 0-1>, "specificity": <number 0-1>}}"""

NEXT_GUESS_PROMPT = """Return only JSON, no markdown or explanation.
You are playing Contexto. Rank 1 is correct. Lower ranks are closer to the hidden target.
Guess history with ranks: {history}
Invalid or unavailable guesses to avoid: {invalid_guesses}

Suggest exactly one new single-word guess that could be closer to the hidden target.
Every guess must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Do not suggest singular or plural forms of already tried words; Contexto treats them as the same guess.
Invalid examples: up-to-date, sour cream, sourcream, wildanimal, dairyproduct.
JSON schema:
{{"word": "guess"}}"""


class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str) -> None:
        self.provider = provider.lower().strip()
        self.api_key = api_key.strip() or ("ollama" if self.provider == "ollama" else "")
        self.model = model
        self.ollama_base_url = config.OLLAMA_BASE_URL
        self.ollama_timeout_seconds = config.OLLAMA_REQUEST_TIMEOUT_SECONDS

        if self.provider not in {"openai", "anthropic", "ollama"}:
            raise ValueError("LLM provider must be 'openai', 'anthropic', or 'ollama'.")
        #region agent log
        _agent_debug_log(
            "contexto_solver/llm_client.py:LLMClient.__init__",
            "llm client configured",
            {
                "provider": self.provider,
                "model": self.model,
                "apiKeyPresent": bool(self.api_key),
                "apiKeyLength": len(self.api_key),
                "apiKeyPlaceholder": self.api_key.startswith("replace-with"),
                "apiKeyHasOpenAIShape": self.api_key.startswith(("sk-", "sk-proj-")),
            },
            "H1,H2,H3,H4",
        )
        #endregion
        if self.provider != "ollama" and (not self.api_key or self.api_key.startswith("replace-with")):
            raise ValueError(f"Missing API key for provider '{self.provider}'. Update your .env file.")

    def generate_initial_categories(self, n: int = 6, starter_words: int = 3) -> list[dict[str, Any]]:
        prompt = INITIAL_CATEGORIES_PROMPT.format(n=n, starter_words=starter_words)
        categories = _normalize_initial_categories(self._json_request_with_retry(prompt))
        if not _initial_categories_sufficient(categories):
            # Ollama's json_object mode frequently collapses the requested
            # category list into a single object. Retry the LLM call once (this is
            # a full extra round-trip, distinct from the JSON-validity retries in
            # ``_json_request_with_retry``) before giving up.
            categories = _normalize_initial_categories(self._json_request_with_retry(prompt))
        if not _initial_categories_sufficient(categories):
            raise ValueError(
                "generate_initial_categories produced a degenerate seed set after "
                f"one retry: {len(categories)} categories / "
                f"{_count_seed_words(categories)} seed words "
                f"(need >= {MIN_INITIAL_CATEGORIES} categories and "
                f">= {MIN_INITIAL_SEED_WORDS} seed words). This usually means the "
                "provider forced a single top-level JSON object (e.g. Ollama "
                "json_object mode) and collapsed the category list."
            )
        return categories

    def _request_json_list(
        self,
        prompt: str,
        expected_key: str,
        element_type: type,
        what: str,
        return_raw: bool = False,
    ) -> Any:
        """Request a JSON list, tolerating json_object's object-only responses.

        Normalizes ``{expected_key: [...]}``, bare arrays, and single-key
        wrappers into a list of ``element_type``. Validates minimally (non-empty);
        on an empty result retries the LLM call once (a full extra round-trip,
        distinct from the JSON-validity retries in ``_json_request_with_retry``)
        before raising a clear error. With ``return_raw`` also returns the raw
        response text of the parsed call and the rendered ``prompt`` (both for
        self-report re-parsing and trace provenance).
        """
        parsed, raw = self._json_request_with_retry_and_raw(prompt)
        items = _normalize_json_list(parsed, expected_key, element_type)
        if not items:
            parsed, raw = self._json_request_with_retry_and_raw(prompt)
            items = _normalize_json_list(parsed, expected_key, element_type)
        if not items:
            raise ValueError(
                f"{what} returned no usable {expected_key} after one retry. This "
                "usually means the provider forced a single top-level JSON object "
                "(e.g. Ollama json_object mode) that did not match the requested "
                f'{{"{expected_key}": [...]}} shape.'
            )
        return (items, raw, prompt) if return_raw else items

    def propose_words(
        self,
        hypothesis: Hypothesis,
        hypothesis_guesses: dict[str, int],
        invalid_guesses: set[str] | None = None,
        n: int = 3,
        global_guesses: dict[str, int] | None = None,
    ) -> list[str]:
        guesses_to_avoid = global_guesses if global_guesses is not None else hypothesis_guesses
        best_word, best_rank = self._global_best(guesses_to_avoid)
        prompt = PROPOSE_WORDS_PROMPT.format(
            name=hypothesis.category_name,
            description=hypothesis.description,
            hypothesis_words_tried=json.dumps(hypothesis_guesses, sort_keys=True),
            best_word=best_word,
            best_rank=best_rank,
            all_guesses=json.dumps(sorted(guesses_to_avoid)),
            invalid_guesses=json.dumps(sorted(invalid_guesses or set())),
            n=n,
        )
        return self._request_json_list(prompt, "words", str, "propose_words")

    def specialize(
        self,
        hypothesis: Hypothesis,
        all_guesses: dict[str, int],
        invalid_guesses: set[str] | None = None,
        n: int = 2,
        rationale_inheritance_block: str = "",
        self_report_block: str = "",
        return_raw: bool = False,
    ) -> Any:
        best_word, best_rank = self._global_best(hypothesis.words_tried)
        prompt = SPECIALIZE_PROMPT.format(
            name=hypothesis.category_name,
            description=hypothesis.description,
            words_tried=json.dumps(hypothesis.words_tried, sort_keys=True),
            best_word=best_word,
            best_rank=best_rank,
            all_guesses=json.dumps(sorted(all_guesses)),
            invalid_guesses=json.dumps(sorted(invalid_guesses or set())),
            n=n,
        ) + rationale_inheritance_block + self_report_block
        return self._request_json_list(
            prompt, "specializations", dict, "specialize", return_raw=return_raw
        )

    def build_operator_mutation_prompt(
        self,
        prompt_template: str,
        hypothesis: Hypothesis,
        all_guesses: set[str],
        invalid_guesses: set[str] | None = None,
        n: int = 3,
        active_categories: list[str] | None = None,
        ranked_context: str = "",
        rationale_inheritance_block: str = "",
        self_report_block: str = "",
    ) -> str:
        best_word, best_rank = self._global_best(hypothesis.words_tried)
        prompt_values = {
            "name": hypothesis.category_name,
            "description": hypothesis.description,
            "words_tried": json.dumps(hypothesis.words_tried, sort_keys=True),
            "best_word": best_word,
            "best_rank": best_rank,
            "all_guesses": json.dumps(sorted(all_guesses)),
            "invalid_guesses": json.dumps(sorted(invalid_guesses or set())),
            "n": n,
        }
        if "{active_categories}" in prompt_template:
            prompt_values["active_categories"] = json.dumps(active_categories or [])
        if "{ranked_context}" in prompt_template:
            prompt_values["ranked_context"] = ranked_context
        # The self-report request (RQ1) is appended at the very end so it renders
        # byte-identically to the pre-instrumentation prompt when the flag is off
        # (self_report_block == ""). Appending, rather than adding a template
        # slot, avoids touching the shared JSON-schema tail used by the
        # out-of-scope pivot prompts.
        return prompt_template.format(**prompt_values) + rationale_inheritance_block + self_report_block

    def complete_json_prompt(self, prompt: str) -> Any:
        return self._json_request_with_retry(prompt)

    def complete_json_prompt_with_raw(self, prompt: str) -> tuple[Any, str]:
        """Return both the parsed JSON and the raw response text.

        Used by the self-report instrumentation so the raw model output can be
        stored for offline re-parsing. Parsing/retry semantics match
        ``complete_json_prompt``.
        """
        return self._json_request_with_retry_and_raw(prompt)

    def place_word(
        self,
        word: str,
        anchors_concreteness: dict[float, str] | None = None,
        anchors_specificity: dict[float, str] | None = None,
    ) -> dict[str, float]:
        """Return MAP-Elites behavior coordinates {"concreteness", "specificity"} in [0, 1]."""
        concreteness_anchors = anchors_concreteness or config.MAPELITES_ANCHORS_CONCRETENESS
        specificity_anchors = anchors_specificity or config.MAPELITES_ANCHORS_SPECIFICITY
        prompt = PLACE_WORD_PROMPT.format(
            word=word,
            concreteness_anchors=_format_anchor_scale(concreteness_anchors),
            specificity_anchors=_format_anchor_scale(specificity_anchors),
        )
        response = self._json_request_with_retry(prompt)
        if not isinstance(response, dict):
            raise ValueError(f"place_word expected a JSON object, got {type(response).__name__}")
        return {
            "concreteness": _clamp_unit(response.get("concreteness")),
            "specificity": _clamp_unit(response.get("specificity")),
        }

    def build_crossover_prompt(
        self,
        hypothesis_a_name: str,
        hypothesis_b_name: str,
        a_words_with_ranks: dict[str, int],
        b_words_with_ranks: dict[str, int],
        self_report_block: str = "",
    ) -> str:
        return (
            CROSSOVER_PROMPT.format(
                a_name=hypothesis_a_name,
                b_name=hypothesis_b_name,
                a_words=json.dumps(a_words_with_ranks, sort_keys=True),
                b_words=json.dumps(b_words_with_ranks, sort_keys=True),
            )
            + self_report_block
        )

    def crossover(
        self,
        hypothesis_a_name: str,
        hypothesis_b_name: str,
        a_words_with_ranks: dict[str, int],
        b_words_with_ranks: dict[str, int],
        self_report_block: str = "",
        return_raw: bool = False,
    ) -> Any:
        prompt = self.build_crossover_prompt(
            hypothesis_a_name,
            hypothesis_b_name,
            a_words_with_ranks,
            b_words_with_ranks,
            self_report_block,
        )
        if return_raw:
            return self._json_request_with_retry_and_raw(prompt)
        return self._json_request_with_retry(prompt)

    def local_search(self, word: str, rank: int, n: int = 5, all_guesses: set[str] | None = None) -> list[str]:
        prompt = LOCAL_SEARCH_PROMPT.format(
            word=word,
            rank=rank,
            n=n,
            all_guesses=json.dumps(sorted(all_guesses or set())),
        )
        return self._request_json_list(prompt, "words", str, "local_search")

    def pivot_morphology(
        self,
        word: str,
        rank: int,
        all_guesses: set[str],
        n: int = 10,
        self_report_block: str = "",
        return_raw: bool = False,
    ) -> Any:
        prompt = PIVOT_MORPHOLOGY_PROMPT.format(
            word=word,
            rank=rank,
            n=n,
            all_guesses=json.dumps(sorted(all_guesses)),
        ) + self_report_block
        return self._request_json_list(
            prompt, "words", str, "pivot_morphology", return_raw=return_raw
        )

    def pivot_register_shift(
        self,
        word: str,
        rank: int,
        all_guesses: set[str],
        n: int = 10,
        self_report_block: str = "",
        return_raw: bool = False,
    ) -> Any:
        prompt = PIVOT_REGISTER_SHIFT_PROMPT.format(
            word=word,
            rank=rank,
            n=n,
            all_guesses=json.dumps(sorted(all_guesses)),
        ) + self_report_block
        return self._request_json_list(
            prompt, "words", str, "pivot_register_shift", return_raw=return_raw
        )

    def pivot_adjacent_category(
        self,
        word: str,
        rank: int,
        category_name: str,
        category_description: str,
        words_tried: dict[str, int],
        all_guesses: set[str],
        n: int = 10,
        self_report_block: str = "",
        return_raw: bool = False,
    ) -> Any:
        prompt = PIVOT_ADJACENT_CATEGORY_PROMPT.format(
            word=word,
            rank=rank,
            category_name=category_name,
            category_description=category_description,
            words_tried=json.dumps(words_tried, sort_keys=True),
            all_guesses=json.dumps(sorted(all_guesses)),
            n=n,
        ) + self_report_block
        if return_raw:
            parsed, raw = self._json_request_with_retry_and_raw(prompt)
            return parsed, raw, prompt
        return self._json_request_with_retry(prompt)

    def pivot_fresh_adjacent_category(
        self,
        word: str,
        rank: int,
        active_categories: list[str],
        all_guesses: set[str],
        n: int = 10,
        self_report_block: str = "",
        return_raw: bool = False,
    ) -> Any:
        prompt = PIVOT_FRESH_ADJACENT_CATEGORY_PROMPT.format(
            word=word,
            rank=rank,
            active_categories=json.dumps(active_categories),
            all_guesses=json.dumps(sorted(all_guesses)),
            n=n,
        ) + self_report_block
        if return_raw:
            parsed, raw = self._json_request_with_retry_and_raw(prompt)
            return parsed, raw, prompt
        return self._json_request_with_retry(prompt)

    def next_guess(
        self,
        history: dict[str, int],
        invalid_guesses: set[str] | None = None,
        self_report_block: str = "",
        return_raw: bool = False,
    ) -> Any:
        prompt = NEXT_GUESS_PROMPT.format(
            history=json.dumps(history, sort_keys=True),
            invalid_guesses=json.dumps(sorted(invalid_guesses or set())),
        ) + self_report_block
        if return_raw:
            response, raw = self._json_request_with_retry_and_raw(prompt)
            word = str(response.get("word", "")) if isinstance(response, dict) else str(response)
            return word, response, raw, prompt
        response = self._json_request_with_retry(prompt)
        if isinstance(response, dict):
            return str(response.get("word", ""))
        return str(response)

    def _json_request_with_retry(self, prompt: str) -> Any:
        parsed, _raw = self._json_request_with_retry_and_raw(prompt)
        return parsed

    def _json_request_with_retry_and_raw(self, prompt: str) -> tuple[Any, str]:
        last_error: Exception | None = None
        max_attempts = 5
        for attempt in range(max_attempts):
            #region agent log
            _agent_debug_log(
                "contexto_solver/llm_client.py:_json_request_with_retry",
                "llm json attempt start",
                {
                    "provider": self.provider,
                    "model": self.model,
                    "attempt": attempt + 1,
                    "maxAttempts": max_attempts,
                    "promptChars": len(prompt),
                    "promptPrefix": prompt[:80],
                },
                "H2,H3,H4",
            )
            #endregion
            try:
                text = self._complete(prompt)
            except requests.RequestException as exc:
                last_error = exc
                #region agent log
                _agent_debug_log(
                    "contexto_solver/llm_client.py:_json_request_with_retry",
                    "llm json attempt request exception",
                    {
                        "provider": self.provider,
                        "model": self.model,
                        "attempt": attempt + 1,
                        "exceptionType": type(exc).__name__,
                        "exception": str(exc)[:240],
                        "retryable": _is_retryable_provider_error(exc),
                    },
                    "H3,H4",
                )
                #endregion
                if not _is_retryable_provider_error(exc) or attempt == max_attempts - 1:
                    raise
                time.sleep(_retry_delay_seconds(exc, attempt))
                continue
            try:
                return json.loads(_strip_code_fences(text)), text
            except json.JSONDecodeError as exc:
                last_error = exc

        raise ValueError(f"LLM did not return valid JSON after retry: {last_error}")

    def _complete(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._complete_openai(prompt)
        if self.provider == "ollama":
            return self._complete_ollama(prompt)
        return self._complete_anthropic(prompt)

    def _complete_openai(self, prompt: str) -> str:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
            },
            timeout=60,
        )
        error_data: Any = None
        if response.status_code >= 400:
            try:
                error_data = response.json().get("error", {})
            except ValueError:
                error_data = {"bodyPrefix": response.text[:160]}
            #region agent log
            _agent_debug_log(
                "contexto_solver/llm_client.py:_complete_openai",
                "openai response failed",
                {
                    "statusCode": response.status_code,
                    "requestId": response.headers.get("x-request-id", ""),
                    "contentType": response.headers.get("content-type", ""),
                    "error": error_data,
                },
                "H3,H4",
            )
            #endregion
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _complete_anthropic(self, prompt: str) -> str:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 1200,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(block.get("text", "") for block in data.get("content", []))

    def _complete_ollama(self, prompt: str) -> str:
        url = f"{self.ollama_base_url.rstrip('/')}/chat/completions"
        started_at = time.perf_counter()
        #region agent log
        _agent_debug_log(
            "contexto_solver/llm_client.py:_complete_ollama",
            "ollama request start",
            {
                "model": self.model,
                "timeoutSeconds": self.ollama_timeout_seconds,
                "promptChars": len(prompt),
                "baseUrl": self.ollama_base_url,
            },
            "H1,H2,H4",
        )
        #endregion
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.ollama_timeout_seconds,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama server not reachable at {self.ollama_base_url}. "
                "Is ollama running? Try ollama list to confirm models are pulled."
            ) from exc
        except requests.RequestException as exc:
            #region agent log
            _agent_debug_log(
                "contexto_solver/llm_client.py:_complete_ollama",
                "ollama request exception",
                {
                    "model": self.model,
                    "timeoutSeconds": self.ollama_timeout_seconds,
                    "elapsedSeconds": round(time.perf_counter() - started_at, 3),
                    "promptChars": len(prompt),
                    "exceptionType": type(exc).__name__,
                    "exception": str(exc)[:240],
                },
                "H1,H2,H4",
            )
            #endregion
            raise

        if response.status_code >= 400:
            response_text = response.text
            if _is_ollama_model_not_found(response.status_code, response_text, self.model):
                raise ValueError(f"Model {self.model} not found. Run ollama pull {self.model}.")
        response.raise_for_status()
        #region agent log
        _agent_debug_log(
            "contexto_solver/llm_client.py:_complete_ollama",
            "ollama request success",
            {
                "model": self.model,
                "statusCode": response.status_code,
                "elapsedSeconds": round(time.perf_counter() - started_at, 3),
                "promptChars": len(prompt),
                "responseChars": len(response.text),
            },
            "H1,H2,H4",
        )
        #endregion
        data = response.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _global_best(all_guesses: dict[str, int]) -> tuple[str | None, int | None]:
        if not all_guesses:
            return None, None
        best_word = min(all_guesses, key=all_guesses.get)
        return best_word, all_guesses[best_word]


def _format_anchor_scale(anchors: dict[float, str]) -> str:
    return " | ".join(f"{position:.2f}: {word}" for position, word in sorted(anchors.items()))


def _clamp_unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"place_word returned a non-numeric coordinate: {value!r}") from exc
    if number != number:  # NaN guard
        raise ValueError("place_word returned NaN coordinate")
    return max(0.0, min(1.0, number))


def _looks_like_category(value: Any) -> bool:
    return isinstance(value, dict) and ("words" in value or "name" in value)


def _normalize_json_list(parsed: Any, expected_key: str, element_type: type) -> list:
    """Coerce a model response into a list of ``element_type`` items.

    Ollama's ``response_format=json_object`` forces a single top-level object, so
    prompts now request ``{expected_key: [...]}``. This tolerates every shape seen
    across providers:
    - ``{expected_key: [ ... ]}`` (the requested shape),
    - a bare top-level array (providers that do not force an object),
    - any other single-key wrapper whose sole list value holds the items,
    - for dict elements only, a bare single element object (json_object collapse).
    Entries not matching ``element_type`` are dropped.
    """
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, element_type)]
    if isinstance(parsed, dict):
        wrapped = parsed.get(expected_key)
        if isinstance(wrapped, list):
            return [item for item in wrapped if isinstance(item, element_type)]
        if element_type is dict and _looks_like_category(parsed):
            return [parsed]
        for value in parsed.values():
            if isinstance(value, list) and any(isinstance(item, element_type) for item in value):
                return [item for item in value if isinstance(item, element_type)]
    return []


def _normalize_initial_categories(parsed: Any) -> list[dict[str, Any]]:
    """Coerce the initial-categories response into a list of category dicts."""
    return _normalize_json_list(parsed, "categories", dict)


def _count_seed_words(categories: list[dict[str, Any]]) -> int:
    total = 0
    for category in categories:
        words = category.get("words")
        if isinstance(words, list):
            total += sum(1 for word in words if isinstance(word, str) and word.strip())
    return total


def _initial_categories_sufficient(categories: list[dict[str, Any]]) -> bool:
    return (
        len(categories) >= MIN_INITIAL_CATEGORIES
        and _count_seed_words(categories) >= MIN_INITIAL_SEED_WORDS
    )


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    # qwen3 reasoning preamble: drop a leading <think>...</think> block if present.
    think_match = re.match(r"^<think>.*?</think>\s*", cleaned, flags=re.DOTALL)
    if think_match:
        cleaned = cleaned[think_match.end():].strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    if not cleaned.startswith("{"):
        extracted = _first_json_object(cleaned)
        if extracted is not None:
            return extracted
    return cleaned


def _is_retryable_provider_error(exc: requests.RequestException) -> bool:
    response = getattr(exc, "response", None)
    if response is None:
        return isinstance(exc, (requests.ConnectionError, requests.Timeout))
    return response.status_code in {429, 500, 502, 503, 504}


def _retry_delay_seconds(exc: requests.RequestException, attempt: int) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except ValueError:
                pass
    return min(5.0 * (2**attempt), 60.0)


def _is_ollama_model_not_found(status_code: int, response_text: str, model: str) -> bool:
    normalized = response_text.lower()
    return status_code == 404 and "not found" in normalized and model.lower() in normalized


def _agent_debug_log(location: str, message: str, data: dict[str, object], hypothesis_id: str) -> None:
    try:
        payload = {
            "sessionId": "f5f8f7",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-f5f8f7.log", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    assert json.loads(_strip_code_fences('<think>reasoning</think>{"words":["a"]}')) == {"words": ["a"]}
    assert json.loads(_strip_code_fences('```json\n{"words":["a"]}\n```')) == {"words": ["a"]}
    assert json.loads(_strip_code_fences('{"words":["a"]}')) == {"words": ["a"]}
    print("ok")

