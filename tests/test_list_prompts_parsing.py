"""Tests for the array-schema list prompts under Ollama json_object mode.

Covers propose_words, specialize, local_search, pivot_morphology, and
pivot_register_shift, whose prompts now request a single-key object wrapper
instead of a bare top-level array (which Ollama's json_object mode cannot emit).
"""

from __future__ import annotations

import os
import unittest

from contexto_solver.hypothesis import Hypothesis
from contexto_solver.llm_client import LLMClient, _normalize_json_list


class NormalizeJsonListTests(unittest.TestCase):
    def test_wrapped_strings(self) -> None:
        self.assertEqual(_normalize_json_list({"words": ["a", "b"]}, "words", str), ["a", "b"])

    def test_bare_array_strings(self) -> None:
        self.assertEqual(_normalize_json_list(["a", "b"], "words", str), ["a", "b"])

    def test_wrong_key_wrapper_strings(self) -> None:
        self.assertEqual(_normalize_json_list({"result": ["a", "b"]}, "words", str), ["a", "b"])

    def test_wrapped_dicts(self) -> None:
        parsed = {"specializations": [{"name": "x"}, {"name": "y"}]}
        self.assertEqual(len(_normalize_json_list(parsed, "specializations", dict)), 2)

    def test_bare_single_dict_wrapped(self) -> None:
        parsed = {"name": "x", "description": "d", "words": ["a"]}
        self.assertEqual(len(_normalize_json_list(parsed, "specializations", dict)), 1)

    def test_type_filtering(self) -> None:
        self.assertEqual(_normalize_json_list({"words": ["a", 1, None, "b"]}, "words", str), ["a", "b"])
        self.assertEqual(_normalize_json_list({"specializations": [{}, "x", 3]}, "specializations", dict), [{}])

    def test_garbage_returns_empty(self) -> None:
        for parsed in ({}, {"words": "notalist"}, "x", 3, None, []):
            self.assertEqual(_normalize_json_list(parsed, "words", str), [], msg=repr(parsed))


# (method name) -> (wrapper key, inner-array JSON text, expected element type)
_WORDS_INNER = '["red", "blue"]'
_SPEC_INNER = '[{"name": "a", "description": "d", "words": ["x"]}]'
_METHOD_CASES = {
    "propose_words": ("words", _WORDS_INNER, str),
    "specialize": ("specializations", _SPEC_INNER, dict),
    "local_search": ("words", _WORDS_INNER, str),
    "pivot_morphology": ("words", _WORDS_INNER, str),
    "pivot_register_shift": ("words", _WORDS_INNER, str),
}


def _invoke(client: LLMClient, method: str):
    hypothesis = Hypothesis(category_name="colors", description="basic hues")
    if method == "propose_words":
        return client.propose_words(hypothesis, {})
    if method == "specialize":
        return client.specialize(hypothesis, {})
    if method == "local_search":
        return client.local_search("shrub", 50, n=5, all_guesses=set())
    if method == "pivot_morphology":
        return client.pivot_morphology("shrub", 50, set(), n=10)
    if method == "pivot_register_shift":
        return client.pivot_register_shift("shrub", 50, set(), n=10)
    raise AssertionError(method)


class ListPromptMethodTests(unittest.TestCase):
    def _client(self) -> LLMClient:
        return LLMClient(provider="ollama", api_key="ollama", model="test-model")

    def test_each_prompt_shape_variants(self) -> None:
        for method, (key, inner, element_type) in _METHOD_CASES.items():
            shapes = {
                "wrapped": '{"%s": %s}' % (key, inner),
                "bare_array": inner,
                "wrong_key_wrapper": '{"result": %s}' % inner,
            }
            for shape_name, payload in shapes.items():
                with self.subTest(method=method, shape=shape_name):
                    client = self._client()
                    client._complete = lambda prompt, p=payload: p
                    result = _invoke(client, method)
                    self.assertTrue(result)
                    self.assertTrue(all(isinstance(item, element_type) for item in result))

    def test_each_prompt_garbage_retries_then_raises(self) -> None:
        for method in _METHOD_CASES:
            with self.subTest(method=method):
                client = self._client()
                calls = {"n": 0}

                def fake_complete(prompt: str) -> str:
                    calls["n"] += 1
                    return "{}"

                client._complete = fake_complete
                with self.assertRaises(ValueError):
                    _invoke(client, method)
                self.assertEqual(calls["n"], 2)


@unittest.skipUnless(
    os.getenv("RUN_OLLAMA_SMOKE") == "1",
    "live Ollama smoke; set RUN_OLLAMA_SMOKE=1 to enable",
)
class OllamaLiveSmokeTests(unittest.TestCase):
    def test_live_propose_words(self) -> None:
        from contexto_solver import config

        client = LLMClient(provider="ollama", api_key="ollama", model=config.OLLAMA_MODEL)
        hypothesis = Hypothesis(category_name="colors", description="basic hues")
        words = client.propose_words(hypothesis, {}, n=3)
        self.assertTrue(words)
        self.assertTrue(all(isinstance(word, str) for word in words))


if __name__ == "__main__":
    unittest.main()
