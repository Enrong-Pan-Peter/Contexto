"""LLM API wrapper used to generate Contexto search hypotheses."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from . import config
from .hypothesis import Hypothesis


INITIAL_CATEGORIES_PROMPT = """Return only JSON, no markdown or explanation.
Generate {n} broad semantic categories for exploring a Contexto puzzle.
Each category must include exactly {starter_words} common starter words.
Every word must be one common lowercase dictionary word.
Do not use spaces, punctuation, hyphens, proper nouns, brands, obscure foreign words, plural-only forms, or phrases joined together.
Invalid examples: up-to-date, sour cream, sourcream, wildanimal, dairyproduct.
JSON schema:
[{{"name": "category name", "description": "short description", "words": ["word1", "word2", "word3"]}}]"""

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
["word1", "word2", "word3"]"""

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
[{{"name": "direction name", "description": "short description", "words": ["word1", "word2", "word3"]}}]"""

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
["word1", "word2", "word3"]"""

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
["word1", "word2", "word3"]"""

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
["word1", "word2", "word3"]"""

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
        return self._json_request_with_retry(prompt)

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
        return self._json_request_with_retry(prompt)

    def specialize(
        self,
        hypothesis: Hypothesis,
        all_guesses: dict[str, int],
        invalid_guesses: set[str] | None = None,
        n: int = 2,
    ) -> list[dict[str, Any]]:
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
        )
        return self._json_request_with_retry(prompt)

    def crossover(
        self,
        hypothesis_a_name: str,
        hypothesis_b_name: str,
        a_words_with_ranks: dict[str, int],
        b_words_with_ranks: dict[str, int],
    ) -> dict[str, Any]:
        prompt = CROSSOVER_PROMPT.format(
            a_name=hypothesis_a_name,
            b_name=hypothesis_b_name,
            a_words=json.dumps(a_words_with_ranks, sort_keys=True),
            b_words=json.dumps(b_words_with_ranks, sort_keys=True),
        )
        return self._json_request_with_retry(prompt)

    def local_search(self, word: str, rank: int, n: int = 5, all_guesses: set[str] | None = None) -> list[str]:
        prompt = LOCAL_SEARCH_PROMPT.format(
            word=word,
            rank=rank,
            n=n,
            all_guesses=json.dumps(sorted(all_guesses or set())),
        )
        return self._json_request_with_retry(prompt)

    def pivot_morphology(self, word: str, rank: int, all_guesses: set[str], n: int = 10) -> list[str]:
        prompt = PIVOT_MORPHOLOGY_PROMPT.format(
            word=word,
            rank=rank,
            n=n,
            all_guesses=json.dumps(sorted(all_guesses)),
        )
        return self._json_request_with_retry(prompt)

    def pivot_register_shift(self, word: str, rank: int, all_guesses: set[str], n: int = 10) -> list[str]:
        prompt = PIVOT_REGISTER_SHIFT_PROMPT.format(
            word=word,
            rank=rank,
            n=n,
            all_guesses=json.dumps(sorted(all_guesses)),
        )
        return self._json_request_with_retry(prompt)

    def pivot_adjacent_category(
        self,
        word: str,
        rank: int,
        category_name: str,
        category_description: str,
        words_tried: dict[str, int],
        all_guesses: set[str],
        n: int = 10,
    ) -> dict[str, Any]:
        prompt = PIVOT_ADJACENT_CATEGORY_PROMPT.format(
            word=word,
            rank=rank,
            category_name=category_name,
            category_description=category_description,
            words_tried=json.dumps(words_tried, sort_keys=True),
            all_guesses=json.dumps(sorted(all_guesses)),
            n=n,
        )
        return self._json_request_with_retry(prompt)

    def pivot_fresh_adjacent_category(
        self,
        word: str,
        rank: int,
        active_categories: list[str],
        all_guesses: set[str],
        n: int = 10,
    ) -> dict[str, Any]:
        prompt = PIVOT_FRESH_ADJACENT_CATEGORY_PROMPT.format(
            word=word,
            rank=rank,
            active_categories=json.dumps(active_categories),
            all_guesses=json.dumps(sorted(all_guesses)),
            n=n,
        )
        return self._json_request_with_retry(prompt)

    def _json_request_with_retry(self, prompt: str) -> Any:
        last_error: Exception | None = None
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                text = self._complete(prompt)
            except requests.RequestException as exc:
                last_error = exc
                if not _is_retryable_provider_error(exc) or attempt == max_attempts - 1:
                    raise
                time.sleep(_retry_delay_seconds(exc, attempt))
                continue
            try:
                return json.loads(_strip_code_fences(text))
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
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                },
                timeout=self.ollama_timeout_seconds,
            )
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama server not reachable at {self.ollama_base_url}. "
                "Is ollama running? Try ollama list to confirm models are pulled."
            ) from exc

        if response.status_code >= 400:
            response_text = response.text
            if _is_ollama_model_not_found(response.status_code, response_text, self.model):
                raise ValueError(f"Model {self.model} not found. Run ollama pull {self.model}.")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    @staticmethod
    def _global_best(all_guesses: dict[str, int]) -> tuple[str | None, int | None]:
        if not all_guesses:
            return None, None
        best_word = min(all_guesses, key=all_guesses.get)
        return best_word, all_guesses[best_word]


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
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
            "sessionId": "0eedb7",
            "runId": "pre-fix",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open("debug-0eedb7.log", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass

