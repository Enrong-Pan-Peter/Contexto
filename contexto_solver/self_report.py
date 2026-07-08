"""RQ1 operator self-report instrumentation (logged-only).

This module owns the prompt text and parsing for the two operator self-report
fields requested from the variation operator:

- ``predicted_closeness``: the operator's forecast in ``[0, 1]`` of how close the
  proposed word is to the hidden target (``1`` = the target itself).
- ``rationale``: ``{"basis_words": [...], "reason": "..."}``.

These fields are measurement only. Nothing here influences selection, fitness,
mutation/crossover mechanics, or sigma self-adaptation. All parsing is defensive
and never raises so a self-report failure cannot abort a run.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .llm_client import _first_json_object, _strip_code_fences


# Appended (only when the ``self_report`` flag is on) to the operator/crossover
# prompt templates via the ``{self_report_block}`` slot. It leads with a newline
# and, when rendered empty (flag off), leaves the prompt byte-identical to the
# pre-instrumentation prompt.
SELF_REPORT_BLOCK = (
    '\nIn the SAME JSON object, also include these three keys: '
    '"predicted_closeness" (a number from 0 to 1 estimating how close your best '
    'proposed word is to the hidden target, where 1 means it is the target), '
    '"basis_words" (a list of words taken from the context above that your choice '
    'relied on), and "reason" (a one or two sentence explanation).'
)

# Targeted follow-up used once when the self-report could not be parsed from the
# operator's response. It asks only for the self-report of the already-proposed
# word, so the accepted word is never affected.
SELF_REPORT_FOLLOWUP_PROMPT = """Return only JSON, no markdown or explanation.
Earlier you proposed the word "{word}" for a Contexto word-guessing puzzle.
Rank 1 is the hidden target; lower ranks are closer.
Context you had available: {context}
Estimate how close "{word}" is to the hidden target and explain briefly.
Return JSON only with exactly these keys:
{{"predicted_closeness": <number 0-1>, "basis_words": ["word", "word"], "reason": "one or two sentences"}}"""


def clamp_predicted_closeness(value: Any) -> tuple[float | None, bool]:
    """Coerce a predicted-closeness value to ``[0, 1]``.

    Returns ``(clamped_value, was_clamped)``. Non-numeric or NaN values return
    ``(None, False)``. Never raises.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, False
    if math.isnan(number) or math.isinf(number):
        return None, False
    if number < 0.0:
        return 0.0, True
    if number > 1.0:
        return 1.0, True
    return number, False


def _sanitize_basis_words(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    words: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                words.append(cleaned)
    return words


def _sanitize_reason(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_to_dict(source: Any) -> dict[str, Any] | None:
    """Return a dict from a parsed object or from raw text; else ``None``."""
    if isinstance(source, dict):
        return source
    if not isinstance(source, str):
        return None
    text = source.strip()
    if not text:
        return None
    try:
        parsed = json.loads(_strip_code_fences(text))
    except (json.JSONDecodeError, ValueError):
        extracted = _first_json_object(text)
        if extracted is None:
            return None
        try:
            parsed = json.loads(extracted)
        except (json.JSONDecodeError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


def empty_self_report() -> dict[str, Any]:
    """A self-report record with all fields absent/failed defaults."""
    return {
        "predicted_closeness": None,
        "predicted_closeness_clamped": False,
        "rationale": None,
        "self_report_parse_failed": True,
    }


def parse_self_report(source: Any) -> dict[str, Any]:
    """Parse a self-report from a parsed dict or raw model text.

    ``self_report_parse_failed`` is ``True`` only when no JSON object could be
    recovered at all. When a JSON object is present, individual missing fields
    become ``None``/empty rather than failing the whole record. Never raises.
    """
    data = _coerce_to_dict(source)
    if data is None:
        return empty_self_report()

    predicted_closeness, clamped = clamp_predicted_closeness(data.get("predicted_closeness"))
    rationale = {
        "basis_words": _sanitize_basis_words(data.get("basis_words")),
        "reason": _sanitize_reason(data.get("reason")),
    }
    return {
        "predicted_closeness": predicted_closeness,
        "predicted_closeness_clamped": clamped,
        "rationale": rationale,
        "self_report_parse_failed": False,
    }


def self_report_block(enabled: bool) -> str:
    """The self-report request block to append to a base prompt, or ``""``.

    Single source of the appended request text: modes pass the result into their
    prompt construction so a flag-off prompt renders byte-identically.
    """
    return SELF_REPORT_BLOCK if enabled else ""


def resolve_self_report(
    llm_client: Any,
    *,
    source: Any,
    raw: str | None,
    context: str,
    proposed_word: str | None,
    rendered_prompt: str | None,
) -> dict[str, Any]:
    """Parse a proposal's self-report, with one targeted follow-up on failure.

    ``source`` is the parsed proposal object (or raw text) that may carry the
    self-report fields; ``raw`` is stored for offline re-parsing. When closeness
    is missing, issues exactly one follow-up asking only about ``proposed_word``
    so the accepted proposal is never affected. Returns a record dict shaped like
    ``Hypothesis.self_report_dict()``. Never raises.
    """
    report = parse_self_report(source)
    final_raw = raw
    if report["predicted_closeness"] is None and proposed_word:
        followup_prompt = SELF_REPORT_FOLLOWUP_PROMPT.format(
            word=proposed_word,
            context=context,
        )
        try:
            parsed, followup_raw = llm_client.complete_json_prompt_with_raw(followup_prompt)
        except Exception:
            parsed, followup_raw = None, None
        if parsed is not None:
            followup = parse_self_report(parsed)
            final_raw = followup_raw if followup_raw is not None else raw
            if followup["predicted_closeness"] is not None:
                report["predicted_closeness"] = followup["predicted_closeness"]
                report["predicted_closeness_clamped"] = followup["predicted_closeness_clamped"]
            if followup["rationale"] and (
                report["rationale"] is None or not report["rationale"]["basis_words"]
            ):
                report["rationale"] = followup["rationale"]

    rationale = report["rationale"]
    parse_failed = report["predicted_closeness"] is None and (
        rationale is None or (not rationale["basis_words"] and not rationale["reason"])
    )
    return {
        "predicted_closeness": report["predicted_closeness"],
        "predicted_closeness_clamped": report["predicted_closeness_clamped"],
        "rationale": rationale,
        "self_report_parse_failed": parse_failed,
        "self_report_raw": final_raw,
        "self_report_prompt": rendered_prompt,
    }


def apply_self_report_to_hypothesis(child: Any, record: dict[str, Any]) -> None:
    """Copy a resolved self-report record onto a hypothesis (logged-only)."""
    child.predicted_closeness = record["predicted_closeness"]
    child.predicted_closeness_clamped = record["predicted_closeness_clamped"]
    child.rationale = record["rationale"]
    child.self_report_parse_failed = record["self_report_parse_failed"]
    child.self_report_raw = record["self_report_raw"]
    child.self_report_prompt = record["self_report_prompt"]


def read_self_report(child_dict: dict[str, Any]) -> dict[str, Any]:
    """Tolerantly read a serialized child's self-report (for old traces).

    Returns a fully-populated record with null defaults when the ``self_report``
    key is absent, so existing trace-reading code opens old traces unchanged.
    """
    defaults: dict[str, Any] = {
        "predicted_closeness": None,
        "predicted_closeness_clamped": False,
        "rationale": None,
        "self_report_parse_failed": False,
        "self_report_raw": None,
        "self_report_prompt": None,
    }
    if not isinstance(child_dict, dict):
        return defaults
    report = child_dict.get("self_report")
    if not isinstance(report, dict):
        return defaults
    merged = dict(defaults)
    for key in defaults:
        if key in report:
            merged[key] = report[key]
    return merged
