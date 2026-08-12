"""Tests for initial-category parsing under Ollama's json_object mode.

Covers the bug where the prompt requested a top-level JSON array while Ollama's
``response_format=json_object`` forces a single top-level object, collapsing the
category list to one dict and leaving the archive empty.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from contexto_solver.llm_client import (
    MIN_INITIAL_CATEGORIES,
    MIN_INITIAL_SEED_WORDS,
    LLMClient,
    _count_seed_words,
    _initial_categories_sufficient,
    _normalize_initial_categories,
)
from contexto_solver.logger import Logger
from contexto_solver.methods.ea_llm_map_elites import (
    EALLMMapElitesConfig,
    EALLMMapElitesMethod,
)


def _category(name: str, *words: str) -> dict:
    return {"name": name, "description": f"{name} things", "words": list(words)}


class NormalizeInitialCategoriesTests(unittest.TestCase):
    def test_categories_wrapper_object(self) -> None:
        """The shape the prompt now requests: {"categories": [...]}"""
        parsed = {"categories": [_category("colors", "red"), _category("animals", "cat")]}
        result = _normalize_initial_categories(parsed)
        self.assertEqual([c["name"] for c in result], ["colors", "animals"])

    def test_bare_single_category_dict(self) -> None:
        """Ollama json_object collapse: a lone category object is wrapped into a list."""
        parsed = _category("colors", "red", "blue", "green")
        result = _normalize_initial_categories(parsed)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "colors")

    def test_bare_array(self) -> None:
        """Non-Ollama providers may still return a top-level array."""
        parsed = [_category("colors", "red"), _category("animals", "cat")]
        result = _normalize_initial_categories(parsed)
        self.assertEqual(len(result), 2)

    def test_alternate_wrapper_key(self) -> None:
        """A single-key wrapper whose sole list value holds the categories."""
        parsed = {"result": [_category("colors", "red"), _category("animals", "cat")]}
        result = _normalize_initial_categories(parsed)
        self.assertEqual(len(result), 2)

    def test_non_dict_entries_dropped(self) -> None:
        parsed = {"categories": [_category("colors", "red"), "garbage", 7, None]}
        result = _normalize_initial_categories(parsed)
        self.assertEqual(len(result), 1)

    def test_empty_and_garbage_return_empty(self) -> None:
        for parsed in ({}, [], "nonsense", 42, None, {"foo": 1}):
            self.assertEqual(_normalize_initial_categories(parsed), [], msg=repr(parsed))

    def test_count_seed_words_ignores_blanks_and_non_strings(self) -> None:
        cats = [{"words": ["red", "", "blue", 3, None]}, {"words": ["cat"]}]
        self.assertEqual(_count_seed_words(cats), 3)

    def test_sufficiency_thresholds(self) -> None:
        self.assertFalse(_initial_categories_sufficient([]))
        self.assertFalse(_initial_categories_sufficient([_category("colors", "red", "blue")]))
        self.assertTrue(
            _initial_categories_sufficient([_category("colors", "red"), _category("animals", "cat")])
        )
        # Sanity-check the constants are the conservative floors we documented.
        self.assertEqual(MIN_INITIAL_CATEGORIES, 2)
        self.assertEqual(MIN_INITIAL_SEED_WORDS, 2)


class GenerateInitialCategoriesTests(unittest.TestCase):
    def _client(self) -> LLMClient:
        return LLMClient(provider="ollama", api_key="ollama", model="test-model")

    def test_wrapper_response_parses(self) -> None:
        client = self._client()
        client._complete = lambda prompt: (
            '{"categories": ['
            '{"name": "colors", "description": "d", "words": ["red"]},'
            '{"name": "animals", "description": "d", "words": ["cat"]},'
            '{"name": "tools", "description": "d", "words": ["hammer"]}]}'
        )
        categories = client.generate_initial_categories(n=3, starter_words=1)
        self.assertEqual(len(categories), 3)

    def test_below_minimum_retries_then_succeeds(self) -> None:
        client = self._client()
        calls = {"n": 0}

        def fake_complete(prompt: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                # First call: Ollama collapse to a single category (insufficient).
                return '{"name": "colors", "description": "d", "words": ["red", "blue"]}'
            return (
                '{"categories": ['
                '{"name": "colors", "description": "d", "words": ["red"]},'
                '{"name": "animals", "description": "d", "words": ["cat"]}]}'
            )

        client._complete = fake_complete
        categories = client.generate_initial_categories(n=6, starter_words=1)
        self.assertEqual(len(categories), 2)
        self.assertEqual(calls["n"], 2)

    def test_persistent_collapse_retries_once_then_raises(self) -> None:
        client = self._client()
        calls = {"n": 0}

        def fake_complete(prompt: str) -> str:
            calls["n"] += 1
            return '{"name": "colors", "description": "d", "words": ["red", "blue"]}'

        client._complete = fake_complete
        with self.assertRaises(ValueError):
            client.generate_initial_categories(n=6, starter_words=1)
        # Exactly one extra full round-trip beyond the first call.
        self.assertEqual(calls["n"], 2)


class _NoValidWordGame:
    """Every guess is invalid, so no hypothesis can enter the archive."""

    def __init__(self) -> None:
        self.guesses: dict[str, int] = {}

    def guess(self, word: str) -> int:
        self.guesses[word.lower().strip()] = -1
        return -1

    def total_guesses(self) -> int:
        return len(self.guesses)

    def best_so_far(self) -> tuple[str | None, int | None]:
        return None, None

    def is_solved(self) -> bool:
        return False


class _FakeCategoryClient:
    model = "test-model"

    def generate_initial_categories(self, n: int = 6, starter_words: int = 3) -> list[dict]:
        return [_category("colors", "zzqq"), _category("animals", "xxyy")]


class EmptyArchiveFailFastTests(unittest.TestCase):
    def test_map_elites_empty_archive_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = EALLMMapElitesConfig(
                max_generations=10,
                candidates_per_hypothesis=3,
                initial_categories=15,
                starter_words_per_category=1,
                mutations_per_generation=15,
                max_active_hypotheses=5,
                trace_dir=tmp,
                run_label="test",
                placement_cache_dir=tmp,
            )
            method = EALLMMapElitesMethod(
                _NoValidWordGame(), _FakeCategoryClient(), Logger(), config
            )
            with self.assertRaises(RuntimeError):
                method.initialize()
            self.assertEqual(len(method.archive), 0)


@unittest.skipUnless(
    os.getenv("RUN_OLLAMA_SMOKE") == "1",
    "live Ollama smoke; set RUN_OLLAMA_SMOKE=1 to enable",
)
class OllamaLiveSmokeTests(unittest.TestCase):
    def test_live_initial_categories_parse(self) -> None:
        from contexto_solver import config

        client = LLMClient(provider="ollama", api_key="ollama", model=config.OLLAMA_MODEL)
        categories = client.generate_initial_categories(n=15, starter_words=1)
        self.assertGreaterEqual(len(categories), MIN_INITIAL_CATEGORIES)
        self.assertTrue(all(isinstance(c, dict) for c in categories))


if __name__ == "__main__":
    unittest.main()
