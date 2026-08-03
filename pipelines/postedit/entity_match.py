"""Tiered entity-string matcher for the postedit validator.

When LLMs encounter OSM-style tag class names inside ``{{...}}`` markers
(``{{cafes}}``, ``{{bus_stop}}``, ``{{Pit toilet}}``, ``{{Accesseires}}``)
they naturally try to normalize them — depluralize ``cafes`` → ``cafe``,
fix capitalization ``Pit toilet`` → ``pit toilet``, fix typos
``Accesseires`` → ``Accessories``. Strict byte-identical matching
rejects all these polish edits and the voter gets discarded.

This module decides whether a polish is acceptable. The benchmark tests
spatial reasoning; gold triples join by ``question_id`` (not entity
string), so polished entity strings don't affect evaluation correctness.
The acceptance bands below preserve the safety net for genuinely bad
substitutions (``cafes`` → ``restaurants``) while letting normalizations
through.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

MATCH_EXACT = "exact"
MATCH_CASE = "case"
MATCH_PLURAL = "plural"
MATCH_DIACRITIC = "diacritic"
MATCH_DISTANCE = "edit_distance"
MATCH_NONE = "none"


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of comparing an LLM-emitted entity against the expected one."""
    matches: bool
    tier: str           # one of the MATCH_* constants
    expected: str
    observed: str


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _strip_diacritics(s: str) -> str:
    """NFKD-decompose then drop all combining marks. Folds ``Café`` → ``Cafe``."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _english_plural_match(a: str, b: str) -> bool:
    """Return True if ``a`` and ``b`` are English singular/plural variants.

    Handles the common -s, -es, and -y→-ies suffix patterns. Symmetric: the
    caller doesn't need to know which side is singular vs plural.

    Guarded against too-short stems (length < 3 after stem) so we don't
    spuriously match short words like ``a`` ↔ ``as``.
    """
    if a == b:
        return False  # exact match isn't a plural — that's MATCH_EXACT
    # Normalise case + diacritics so this works alongside the case + diacritic tiers.
    al = _strip_diacritics(_nfc(a)).casefold()
    bl = _strip_diacritics(_nfc(b)).casefold()
    short, long_ = (al, bl) if len(al) <= len(bl) else (bl, al)
    if len(short) < 3:
        return False
    # -s suffix
    if long_ == short + "s":
        return True
    # -es suffix
    if long_ == short + "es":
        return True
    # -y → -ies (short = "berry", long = "berries")
    if short.endswith("y") and long_ == short[:-1] + "ies":
        return True
    return False


def _levenshtein(a: str, b: str) -> int:
    """Wagner-Fischer edit distance. O(len(a) * len(b)) time + len(a) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Row-by-row DP with a single previous row.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,        # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
        prev = curr
    return prev[-1]


def _edit_distance_match(a: str, b: str) -> bool:
    """Return True if ``a`` and ``b`` differ by a bounded edit distance
    AND have a reasonable length ratio.

    Bounds:
      * Both strings have length >= 4 (typo fixes only make sense on
        non-trivial words; below this every Tier-1-passing pair is
        already a plural/case/diacritic match, and beyond Tier 1 a
        2-char Levenshtein distance can swap ``X`` ↔ ``Y``).
      * Levenshtein <= max(2, len(longer)//4)
      * length ratio in [0.5, 2.0]

    The bounds together reject long substitutions
    (``cafes`` ↔ ``restaurants`` fails the ratio).
    """
    if not a or not b:
        return False
    short_len = min(len(a), len(b))
    long_len = max(len(a), len(b))
    if short_len < 4:
        return False
    if short_len / long_len < 0.5:
        return False
    ceiling = max(2, long_len // 4)
    return _levenshtein(a, b) <= ceiling


def match_entity(observed: str, expected: str) -> MatchResult:
    """Tiered acceptance test for an LLM-emitted entity polish.

    Order matters: try Tier 1 (exact, case, plural, diacritic) first,
    then Tier 2 (bounded edit distance) only if no Tier 1 hit.
    """
    if observed == expected:
        return MatchResult(True, MATCH_EXACT, expected, observed)

    # Tier 1: case-fold compare (after NFC normalisation).
    if _nfc(observed).casefold() == _nfc(expected).casefold():
        return MatchResult(True, MATCH_CASE, expected, observed)

    # Tier 1: plural/singular variants.
    if _english_plural_match(observed, expected):
        return MatchResult(True, MATCH_PLURAL, expected, observed)

    # Tier 1: diacritic-folded compare.
    if _strip_diacritics(_nfc(observed)) == _strip_diacritics(_nfc(expected)):
        return MatchResult(True, MATCH_DIACRITIC, expected, observed)

    # Tier 2: bounded edit distance (typo fixes).
    if _edit_distance_match(observed, expected):
        return MatchResult(True, MATCH_DISTANCE, expected, observed)

    return MatchResult(False, MATCH_NONE, expected, observed)
