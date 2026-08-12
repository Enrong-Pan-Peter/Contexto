"""Fixed, seeded, string-level templates for the RQ2 reason intervention.

Two arm libraries live here:

- Wrong-content reasons: assert a locally plausible but wrong relation. Each
  template MUST cite the real parent basis words (``{words}`` slot), so the
  intervention is content-level (the relation is wrong) while the referents are
  genuine. No LLM generates any of this text.
- Filler reasons: fixed generic sentences with no parent-derived tokens, chosen
  to be comparable in length to the genuine block they replace.

Admissibility: nothing here may contain the operator isolation guard's forbidden
substrings (see ``contexto_solver.operators.assert_prompt_has_no_sigma_leak``):
``"sigma"``, the sigma glyph, or ``"probability"`` (case-insensitive), and no
two-decimal numeric literals that could collide with a sigma component. This is
checked by :func:`assert_admissible` and by tests.
"""

from __future__ import annotations

import random


# Case-insensitive forbidden substrings (mirrors operators.assert_prompt_has_no_sigma_leak).
FORBIDDEN_SUBSTRINGS = ("sigma", "\u03c3", "probability")

# Each template asserts a plausible-sounding but wrong relation and must cite
# the real parent basis words through the {words} slot. Relations are chosen to
# be generically wrong for Contexto (orthography, rhyme, letter counts,
# anagrams, trivia provenance) rather than semantic closeness.
WRONG_CONTENT_TEMPLATES = (
    "The words {words} were kept because they all rhyme with the hidden target, "
    "and rhyme is the strongest signal in this puzzle.",
    "The words {words} share the same number of letters, which means the hidden "
    "target must have exactly that length as well.",
    "The words {words} are all borrowed from maritime vocabulary, so the hidden "
    "target should be a nautical term.",
    "The words {words} appear together in a well-known nursery rhyme, and the "
    "hidden target is the word that completes that rhyme.",
    "The words {words} are each antonyms of the hidden target, so the search "
    "should move in exactly the opposite direction.",
    "The words {words} begin with the same sound as the hidden target, so "
    "alliteration should guide the next step.",
    "The words {words} were picked because each one is an anagram of a close "
    "neighbor of the hidden target.",
    "The words {words} all appeared in the same crossword puzzle as the hidden "
    "target, which fixes the theme for the next guesses.",
)

# Fixed generic sentences, no parent-derived tokens, spread of lengths so the
# harness can pick one comparable in length to the genuine reason it replaces.
FILLER_SENTENCES = (
    "This step keeps the same general approach as before.",
    "The search continues with broadly useful directions that apply to any word puzzle.",
    "This continuation follows standard practice for exploring word guessing games, "
    "without reference to any particular clue.",
    "The next step keeps the overall strategy steady and considers commonly helpful "
    "directions that could apply to any category of words in a puzzle of this kind.",
    "The exploration proceeds in the usual way, weighing general considerations that "
    "are routinely useful for word guessing games of this kind, and it does not lean "
    "on any specific clue or earlier observation about this particular puzzle.",
)


def assert_admissible(text: str) -> None:
    """Raise ``AssertionError`` if ``text`` contains a forbidden substring."""
    lowered = text.lower()
    for substring in FORBIDDEN_SUBSTRINGS:
        if substring in lowered:
            raise AssertionError(f"template text contains forbidden substring {substring!r}: {text!r}")


def format_words(basis_words: list[str]) -> str:
    """Natural-language listing of the parent basis words (verbatim tokens)."""
    words = [str(word) for word in basis_words if str(word)]
    if not words:
        return "from the parent"
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def wrong_content_reason(basis_words: list[str], rng: random.Random) -> str:
    """A seeded wrong-relation reason that cites the real parent basis words."""
    template = rng.choice(WRONG_CONTENT_TEMPLATES)
    reason = template.format(words=format_words(basis_words))
    assert_admissible(reason)
    return reason


def filler_reason(target_chars: int) -> str:
    """The fixed generic sentence whose length is closest to ``target_chars``.

    Deterministic (no randomness): ties resolve to the earlier library entry.
    """
    best = min(FILLER_SENTENCES, key=lambda sentence: (abs(len(sentence) - target_chars)))
    assert_admissible(best)
    return best
