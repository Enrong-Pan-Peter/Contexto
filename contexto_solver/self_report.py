"""RQ1 operator self-report instrumentation (logged-only).

This module owns the prompt text and parsing for the operator self-report fields
requested from the variation operator:

- ``predicted_closeness``: a continuous forecast in ``[0, 1]``. Pinned semantics
  (schema v3): the operator's estimated chance that its best proposed word ranks
  within the **top 100 closest words** to the hidden target (``1`` = certain it
  ranks that close). This is the calibratable probability used by RQ1.
- ``predicted_bucket``: a coarse categorical companion to ``predicted_closeness``,
  exactly one of ``PREDICTED_BUCKETS`` (``"top10"``, ``"top100"``, ``"top500"``,
  ``"beyond"``) or ``None``. The two confidence fields are elicited together so
  the categorical and continuous forecasts can be cross-checked in calibration.
- ``rationale``: ``{"basis_words": [...], "reason": "..."}``.

These fields are measurement only. Nothing here influences selection, fitness,
mutation/crossover mechanics, or sigma self-adaptation. All parsing is defensive
and never raises so a self-report failure cannot abort a run.

Field-order note: to keep the flag-off prompts byte-identical, the proposed word
stays in the base prompt template and the self-report keys are appended in a
fixed block. Within ``SELF_REPORT_BLOCK`` the request order is ``basis_words,
reason, predicted_bucket, predicted_closeness``. The rationale is therefore
structurally *post-hoc within a completion*: the model emits the proposed word in
the base object first, then the basis/confidence keys. This ordering is a
documented limitation, not a causal claim that the rationale preceded the choice.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .llm_client import _first_json_object, _strip_code_fences


# The canonical predicted_bucket categories. Coarse closeness buckets over the
# Contexto rank ladder; ``beyond`` means farther than the top-500 closest words.
PREDICTED_BUCKETS = ("top10", "top100", "top500", "beyond")

# Appended (only when the ``self_report`` flag is on) to the operator/crossover
# prompt templates via the ``{self_report_block}`` slot. It leads with a newline
# and, when rendered empty (flag off), leaves the prompt byte-identical to the
# pre-instrumentation prompt. The request order (basis_words, reason,
# predicted_bucket, predicted_closeness) is fixed; see the module docstring.
#
# The continuous field is worded as an "estimated chance" rather than a
# "probability": the operator sigma-leak guard
# (operators.assert_prompt_has_no_sigma_leak) forbids the literal substring
# "probability" in a mutation prompt, and this block is appended to those prompts
# when the flag is on. "estimated chance ... a number from 0 to 1" carries the
# same probabilistic meaning without tripping that guard.
SELF_REPORT_BLOCK = (
    '\nIn the SAME JSON object, also include these four keys, in this order: '
    '"basis_words" (a list of words taken from the context above that your choice '
    'relied on), "reason" (a one or two sentence explanation), "predicted_bucket" '
    '(exactly one of "top10", "top100", "top500", or "beyond", estimating how '
    'close your best proposed word is to the hidden target: "top10" = within the '
    '10 closest words, "top100" = within the 100 closest, "top500" = within the '
    '500 closest, "beyond" = farther than the 500 closest), and '
    '"predicted_closeness" (a number from 0 to 1 giving your estimated chance that '
    'your best proposed word ranks within the top 100 closest words to the hidden '
    'target, where 1 means you are certain it ranks that close).'
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
{{"basis_words": ["word", "word"], "reason": "one or two sentences", "predicted_bucket": "top10|top100|top500|beyond", "predicted_closeness": <number 0-1>}}"""

RATIONALE_INHERITANCE_MAX_REASON_CHARS = 400
RATIONALE_INHERITANCE_MAX_BASIS_WORDS = 8


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


def parse_predicted_bucket(value: Any) -> str | None:
    """Coerce a bucket value to one of ``PREDICTED_BUCKETS`` or ``None``.

    Mirrors ``clamp_predicted_closeness``'s tolerance: whitespace, underscores,
    and hyphens are ignored and matching is case-insensitive, so ``"Top 100"`` and
    ``"top_100"`` both map to ``"top100"``. Any unknown string (or non-string)
    returns ``None``. Never raises.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    return normalized if normalized in PREDICTED_BUCKETS else None


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
        "predicted_bucket": None,
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
    predicted_bucket = parse_predicted_bucket(data.get("predicted_bucket"))
    rationale = {
        "basis_words": _sanitize_basis_words(data.get("basis_words")),
        "reason": _sanitize_reason(data.get("reason")),
    }
    return {
        "predicted_closeness": predicted_closeness,
        "predicted_closeness_clamped": clamped,
        "predicted_bucket": predicted_bucket,
        "rationale": rationale,
        "self_report_parse_failed": False,
    }


def self_report_block(enabled: bool) -> str:
    """The self-report request block to append to a base prompt, or ``""``.

    Single source of the appended request text: modes pass the result into their
    prompt construction so a flag-off prompt renders byte-identically.
    """
    return SELF_REPORT_BLOCK if enabled else ""


def hash_injection_text(text: str) -> str:
    """Stable short hash for an injected prompt suffix (inheritance provenance)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def rationale_inheritance_block(
    parent_rationale: dict[str, Any] | None,
    *,
    replacement: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the parent-rationale injection suffix and provenance metadata.

    Returns ``(block_text, meta)`` where ``meta`` may include ``hash`` and
    ``truncated``. When ``replacement`` is set (tests only), that string is
    returned verbatim. Never raises.
    """
    if replacement is not None:
        return replacement, {"hash": hash_injection_text(replacement), "truncated": False}

    if not isinstance(parent_rationale, dict):
        return "", {}

    reason = _sanitize_reason(parent_rationale.get("reason"))
    basis_words = _sanitize_basis_words(parent_rationale.get("basis_words"))
    if not reason and not basis_words:
        return "", {}

    truncated = False
    if len(reason) > RATIONALE_INHERITANCE_MAX_REASON_CHARS:
        reason = reason[: RATIONALE_INHERITANCE_MAX_REASON_CHARS].rstrip() + "..."
        truncated = True
    if len(basis_words) > RATIONALE_INHERITANCE_MAX_BASIS_WORDS:
        basis_words = basis_words[: RATIONALE_INHERITANCE_MAX_BASIS_WORDS]
        truncated = True

    block = (
        "\nThe parent hypothesis's prior rationale (for context only; do not copy "
        f"blindly): basis_words={json.dumps(basis_words)}, reason={json.dumps(reason)}."
    )
    return block, {"hash": hash_injection_text(block), "truncated": truncated}


def instrumentation_provenance_hash() -> str:
    """Hash prompt templates and instrumentation blocks for ``RUN_CONFIG`` provenance."""
    from . import config as app_config
    from .llm_client import (
        CROSSOVER_PROMPT,
        INITIAL_CATEGORIES_PROMPT,
        L_MUTATION_PROMPT,
        LOCAL_SEARCH_PROMPT,
        M_MUTATION_PROMPT,
        ML_MUTATION_PROMPT,
        NEXT_GUESS_PROMPT,
        PIVOT_ADJACENT_CATEGORY_PROMPT,
        PIVOT_FRESH_ADJACENT_CATEGORY_PROMPT,
        PIVOT_MORPHOLOGY_PROMPT,
        PIVOT_REGISTER_SHIFT_PROMPT,
        PLACE_WORD_PROMPT,
        PROPOSE_WORDS_PROMPT,
        S_MUTATION_PROMPT,
        SPECIALIZE_PROMPT,
    )

    parts = [
        INITIAL_CATEGORIES_PROMPT,
        PROPOSE_WORDS_PROMPT,
        SPECIALIZE_PROMPT,
        S_MUTATION_PROMPT,
        M_MUTATION_PROMPT,
        ML_MUTATION_PROMPT,
        L_MUTATION_PROMPT,
        CROSSOVER_PROMPT,
        LOCAL_SEARCH_PROMPT,
        PIVOT_MORPHOLOGY_PROMPT,
        PIVOT_REGISTER_SHIFT_PROMPT,
        PIVOT_ADJACENT_CATEGORY_PROMPT,
        PIVOT_FRESH_ADJACENT_CATEGORY_PROMPT,
        PLACE_WORD_PROMPT,
        NEXT_GUESS_PROMPT,
        SELF_REPORT_BLOCK,
        SELF_REPORT_FOLLOWUP_PROMPT,
        f"RATIONALE_INHERITANCE_MAX_REASON={RATIONALE_INHERITANCE_MAX_REASON_CHARS}",
        f"RATIONALE_INHERITANCE_MAX_BASIS={RATIONALE_INHERITANCE_MAX_BASIS_WORDS}",
        f"TRACE_SCHEMA_VERSION={app_config.TRACE_SCHEMA_VERSION}",
        f"PREDICTED_BUCKETS={','.join(PREDICTED_BUCKETS)}",
    ]
    digest = hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


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
            if followup["predicted_bucket"] is not None and report["predicted_bucket"] is None:
                report["predicted_bucket"] = followup["predicted_bucket"]
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
        "predicted_bucket": report["predicted_bucket"],
        "rationale": rationale,
        "self_report_parse_failed": parse_failed,
        "self_report_raw": final_raw,
        "self_report_prompt": rendered_prompt,
        "injected_rationale_hash": None,
        "rationale_truncated": False,
    }


def apply_self_report_to_hypothesis(child: Any, record: dict[str, Any]) -> None:
    """Copy a resolved self-report record onto a hypothesis (logged-only)."""
    child.predicted_closeness = record["predicted_closeness"]
    child.predicted_closeness_clamped = record["predicted_closeness_clamped"]
    child.predicted_bucket = record["predicted_bucket"]
    child.rationale = record["rationale"]
    child.self_report_parse_failed = record["self_report_parse_failed"]
    child.self_report_raw = record["self_report_raw"]
    child.self_report_prompt = record["self_report_prompt"]
    child.injected_rationale_hash = record.get("injected_rationale_hash")
    child.rationale_truncated = bool(record.get("rationale_truncated"))


def read_self_report(child_dict: dict[str, Any]) -> dict[str, Any]:
    """Tolerantly read a serialized child's self-report (for old traces).

    Returns a fully-populated record with null defaults when the ``self_report``
    key is absent, so existing trace-reading code opens old traces unchanged.
    """
    defaults: dict[str, Any] = {
        "predicted_closeness": None,
        "predicted_closeness_clamped": False,
        "predicted_bucket": None,
        "rationale": None,
        "self_report_parse_failed": False,
        "self_report_raw": None,
        "self_report_prompt": None,
        "injected_rationale_hash": None,
        "rationale_truncated": False,
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
