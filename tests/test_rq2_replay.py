"""Tests for the RQ2 replay harness and its fixed template library.

Synthetic fixtures only: prompts are built with the real
``rationale_inheritance_block`` + ``SELF_REPORT_BLOCK`` so byte-identity checks
exercise the same string surgery the harness performs on stored traces. Dry-run
tests use the harness's deterministic mocks (no network).
"""

from __future__ import annotations

import json
import random
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from contexto_solver.self_report import SELF_REPORT_BLOCK, rationale_inheritance_block
from scripts.rq2_replay import (
    ARMS,
    MockGrader,
    MockLLMClient,
    REASON_SEPARATOR,
    ReplayEvent,
    build_arm_prompts,
    decompose_block,
    extract_events,
    run_replay,
    sample_stratified,
    write_outputs,
)
from scripts.rq2_templates import (
    FILLER_SENTENCES,
    FORBIDDEN_SUBSTRINGS,
    WRONG_CONTENT_TEMPLATES,
    filler_reason,
    format_words,
    wrong_content_reason,
)

# Contains the s_mutation distinguishing phrase so the isolation guard passes.
BASE_PROMPT = (
    "Return only JSON, no markdown or explanation.\n"
    "Make a SMALL mutation: refine the current idea.\n"
    "JSON schema:\n"
    '{"name": "direction name", "description": "short description", "words": ["word1", "word2", "word3"]}'
)


def _run_config_event() -> dict[str, Any]:
    return {
        "generation": -1,
        "event": "RUN_CONFIG",
        "details": {
            "game": "api",
            "game_number": 42,
            "method": "ea_llm_self_adaptive",
            "self_report": True,
            "rationale_inheritance": True,
            "llm_provider": "ollama",
            "llm_model": "qwen3:14b",
            "instrumentation_provenance_hash": "testhash",
        },
    }


def _mutation_event(
    generation: int,
    child_id: str,
    basis_words: list[str],
    reason: str,
    parent_rank: int = 50,
) -> dict[str, Any]:
    block, meta = rationale_inheritance_block({"basis_words": basis_words, "reason": reason})
    prompt = BASE_PROMPT + block + SELF_REPORT_BLOCK
    return {
        "generation": generation,
        "event": "OPERATOR_SAMPLED",
        "details": {
            "parent_id": f"parent-of-{child_id}",
            "child_id": child_id,
            "parent_rank": parent_rank,
            "sigma_snapshot": [0.25, 0.25, 0.25, 0.25],
            "child_sigma": [0.25, 0.25, 0.25, 0.25],
            "child_hypothesis_name": f"hypothesis {child_id}",
            "sampled_op": "s_mutation",
            "method": "self_adaptive",
            "self_report": {
                "predicted_closeness": 0.4,
                "predicted_closeness_clamped": False,
                "predicted_bucket": "top500",
                "rationale": {"basis_words": basis_words, "reason": reason},
                "self_report_parse_failed": False,
                "self_report_raw": "{}",
                "self_report_prompt": prompt,
                "injected_rationale_hash": meta["hash"],
                "rationale_truncated": False,
            },
        },
    }


def _trace(mutation_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_run_config_event()] + mutation_events


class TemplateLibraryTests(unittest.TestCase):
    def test_wrong_content_deterministic_under_seed(self) -> None:
        basis = ["pearl", "shell", "reef"]
        first = wrong_content_reason(basis, random.Random("seed:trace:child"))
        second = wrong_content_reason(basis, random.Random("seed:trace:child"))
        self.assertEqual(first, second)

    def test_wrong_content_cites_every_basis_word(self) -> None:
        basis = ["pearl", "shell", "reef"]
        for template_index in range(len(WRONG_CONTENT_TEMPLATES) * 3):
            reason = wrong_content_reason(basis, random.Random(template_index))
            for word in basis:
                self.assertIn(word, reason)

    def test_libraries_are_admissible(self) -> None:
        for text in WRONG_CONTENT_TEMPLATES + FILLER_SENTENCES:
            lowered = text.lower()
            for forbidden in FORBIDDEN_SUBSTRINGS:
                self.assertNotIn(forbidden, lowered)
            # No digits at all, so no sigma numeric literal can ever collide.
            self.assertIsNone(re.search(r"\d", text))

    def test_filler_reason_picks_closest_length(self) -> None:
        shortest = min(FILLER_SENTENCES, key=len)
        longest = max(FILLER_SENTENCES, key=len)
        self.assertEqual(filler_reason(1), shortest)
        self.assertEqual(filler_reason(10_000), longest)

    def test_format_words_natural_listing(self) -> None:
        self.assertEqual(format_words(["pearl"]), "pearl")
        self.assertEqual(format_words(["pearl", "shell"]), "pearl and shell")
        self.assertEqual(format_words(["a", "b", "c"]), "a, b and c")


class ExtractEventsTests(unittest.TestCase):
    def test_extract_and_genuine_roundtrip(self) -> None:
        event = _mutation_event(1, "c1", ["pearl", "shell"], "these share a marine theme")
        extracted, skips = extract_events(_trace([event]), "t.json")
        self.assertEqual(skips, {})
        self.assertEqual(len(extracted), 1)
        replay_event = extracted[0]
        stored = event["details"]["self_report"]["self_report_prompt"]
        self.assertEqual(
            replay_event.base_prompt + replay_event.genuine_block + SELF_REPORT_BLOCK, stored
        )
        self.assertEqual(replay_event.basis_words, ["pearl", "shell"])
        self.assertEqual(replay_event.genuine_reason, "these share a marine theme")
        self.assertEqual(replay_event.game_number, 42)
        self.assertEqual(replay_event.parent_rank, 50)

    def test_hash_mismatch_is_skipped(self) -> None:
        event = _mutation_event(1, "c1", ["pearl"], "reason")
        event["details"]["self_report"]["injected_rationale_hash"] = "0" * 16
        extracted, skips = extract_events(_trace([event]), "t.json")
        self.assertEqual(extracted, [])
        self.assertEqual(skips, {"hash_mismatch": 1})

    def test_non_inherited_records_not_eligible(self) -> None:
        event = _mutation_event(1, "c1", ["pearl"], "reason")
        event["details"]["self_report"]["injected_rationale_hash"] = None
        extracted, skips = extract_events(_trace([event]), "t.json")
        self.assertEqual(extracted, [])
        self.assertEqual(skips, {})

    def test_uninstrumented_trace_rejected(self) -> None:
        events = _trace([])
        events[0]["details"]["rationale_inheritance"] = False
        with self.assertRaises(ValueError):
            extract_events(events, "t.json")


class ArmConstructionTests(unittest.TestCase):
    def _event(self) -> ReplayEvent:
        event = _mutation_event(2, "c9", ["pearl", "shell", "reef"], "they orbit the same marine idea")
        extracted, _skips = extract_events(_trace([event]), "t.json")
        return extracted[0]

    def test_genuine_arm_byte_identical_to_stored(self) -> None:
        replay_event = self._event()
        prompts = build_arm_prompts(replay_event, seed=0)
        self.assertEqual(prompts["genuine"], replay_event.stored_prompt)

    def test_absent_arm_cleanly_removes_block(self) -> None:
        replay_event = self._event()
        prompts = build_arm_prompts(replay_event, seed=0)
        self.assertEqual(prompts["absent"], replay_event.base_prompt + SELF_REPORT_BLOCK)
        self.assertNotIn("prior rationale", prompts["absent"])

    def test_wrong_arm_keeps_basis_words_json_byte_identical(self) -> None:
        replay_event = self._event()
        prompts = build_arm_prompts(replay_event, seed=0)
        expected_head = replay_event.base_prompt + replay_event.block_head + REASON_SEPARATOR
        self.assertTrue(prompts["wrong"].startswith(expected_head))
        wrong_block = prompts["wrong"][len(replay_event.base_prompt) : -len(SELF_REPORT_BLOCK)]
        decomposed = decompose_block(wrong_block)
        self.assertIsNotNone(decomposed)
        head, reason, basis_words = decomposed
        self.assertEqual(head, replay_event.block_head)
        self.assertEqual(basis_words, replay_event.basis_words)
        self.assertNotEqual(reason, replay_event.genuine_reason)
        for word in replay_event.basis_words:
            self.assertIn(word, reason)

    def test_filler_arm_has_no_parent_tokens(self) -> None:
        replay_event = self._event()
        prompts = build_arm_prompts(replay_event, seed=0)
        filler_block = prompts["filler"][len(replay_event.base_prompt) : -len(SELF_REPORT_BLOCK)]
        self.assertIn("basis_words=[]", filler_block)
        for word in replay_event.basis_words:
            self.assertNotIn(word, filler_block)
        self.assertNotIn(replay_event.genuine_reason, filler_block)

    def test_arms_deterministic_and_share_frame(self) -> None:
        replay_event = self._event()
        first = build_arm_prompts(replay_event, seed=7)
        second = build_arm_prompts(replay_event, seed=7)
        self.assertEqual(first, second)
        for arm in ARMS:
            self.assertTrue(first[arm].startswith(replay_event.base_prompt))
            self.assertTrue(first[arm].endswith(SELF_REPORT_BLOCK))


class StratifiedSamplingTests(unittest.TestCase):
    def _events(self) -> list[ReplayEvent]:
        events: list[ReplayEvent] = []
        for generation in (1, 2, 3):
            for index in range(4):
                event = _mutation_event(generation, f"g{generation}i{index}", ["pearl"], "reason")
                extracted, _skips = extract_events(_trace([event]), "t.json")
                events.extend(extracted)
        return events

    def test_stratified_across_generations(self) -> None:
        events = self._events()
        sampled = sample_stratified(events, 6, random.Random(0))
        self.assertEqual(len(sampled), 6)
        per_generation: dict[int, int] = {}
        for event in sampled:
            per_generation[event.generation] = per_generation.get(event.generation, 0) + 1
        self.assertEqual(per_generation, {1: 2, 2: 2, 3: 2})

    def test_deterministic_under_seed(self) -> None:
        events = self._events()
        first = [e.child_id for e in sample_stratified(events, 5, random.Random(3))]
        second = [e.child_id for e in sample_stratified(events, 5, random.Random(3))]
        self.assertEqual(first, second)

    def test_returns_all_when_supply_short(self) -> None:
        events = self._events()
        sampled = sample_stratified(events, 100, random.Random(0))
        self.assertEqual(len(sampled), len(events))


class DryRunReplayTests(unittest.TestCase):
    def test_end_to_end_with_mocks_never_touches_the_trace(self) -> None:
        mutation_events = [
            _mutation_event(1, "c1", ["pearl", "shell"], "marine theme"),
            _mutation_event(2, "c2", ["coral"], "reef structures"),
            _mutation_event(2, "c3", ["wave", "tide"], "ocean motion"),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            trace_path.write_text(json.dumps(_trace(mutation_events)), encoding="utf-8")
            original_bytes = trace_path.read_bytes()

            result = run_replay(
                [str(trace_path)],
                events_per_trace=40,
                seed=0,
                client_factory=lambda provider, model: MockLLMClient(),
                grader=MockGrader(),
            )

            self.assertEqual(trace_path.read_bytes(), original_bytes)
            records = result["records"]
            self.assertEqual(len(records), 3 * len(ARMS))
            arms_by_event: dict[str, set[str]] = {}
            for record in records:
                arms_by_event.setdefault(record["child_id"], set()).add(record["arm"])
                self.assertFalse(record["llm_parse_failed"])
                self.assertIsNotNone(record["proposed_word"])
                self.assertIsNotNone(record["realized_rank"])
            self.assertEqual(arms_by_event, {c: set(ARMS) for c in ("c1", "c2", "c3")})

            trace_meta = result["metadata"]["traces"][0]
            self.assertEqual(trace_meta["eligible_events"], 3)
            self.assertEqual(trace_meta["sampled_events"], 3)
            self.assertEqual(trace_meta["shortfall"], 37)

            json_path, csv_path = write_outputs(result, Path(tmp_dir) / "out")
            self.assertTrue(json_path.exists())
            csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(csv_lines) - 1, len(records))

    def test_dry_run_is_deterministic(self) -> None:
        mutation_events = [_mutation_event(1, "c1", ["pearl"], "reason")]
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            trace_path.write_text(json.dumps(_trace(mutation_events)), encoding="utf-8")
            kwargs = dict(
                events_per_trace=5,
                seed=11,
                client_factory=lambda provider, model: MockLLMClient(),
                grader=MockGrader(),
            )
            first = run_replay([str(trace_path)], **kwargs)["records"]
            second = run_replay([str(trace_path)], **kwargs)["records"]
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
