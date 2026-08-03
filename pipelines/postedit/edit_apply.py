"""Validate the LLM's corrected_question and strip the entity braces.

The validator accepts minor "polish" edits to entity strings inside
``{{...}}`` markers via :mod:`entity_match` (case-fold, pluralisation,
diacritic-fold, bounded edit distance). Genuinely wrong substitutions
still fail, voter abstains, ensemble falls back. See the rationale in
``docs/analyses/stratified_sampling.md`` §6.7 and the implementation
plan ``async-shimmying-truffle.md``.
"""

import re

from pipelines.postedit.entity_match import MATCH_EXACT, MATCH_NONE, match_entity

_MARKER_RE = re.compile(r"\{\{([^{}]+)\}\}")
_NESTED_BRACE_RE = re.compile(r"\{\{[^{}]*\{\{|\}\}[^{}]*\}\}")


def strip_entity_braces(text: str, entity_values: list[str]) -> str:
    """Replace each ``{{<entity>}}`` with ``<entity>`` for known entity values.

    Tolerant of whitespace inside the braces (``{{ <entity> }}``) so that the
    output matches what ``_extract_marker_values`` already accepts.
    """
    known = set(entity_values)

    def repl(match: "re.Match[str]") -> str:
        inner = match.group(1).strip()
        if inner in known:
            return inner
        return match.group(0)

    return _MARKER_RE.sub(repl, text)


def _extract_marker_values(text: str) -> list[str]:
    # Strip whitespace inside the braces so `{{ X }}` (a common LLM artefact,
    # especially from Qwen) is treated the same as `{{X}}`. Outer-brace
    # matching is unchanged.
    return [match.group(1).strip() for match in _MARKER_RE.finditer(text)]


def _iter_outside_marker_segments(text: str) -> list[str]:
    segments: list[str] = []
    cursor = 0
    for match in _MARKER_RE.finditer(text):
        segments.append(text[cursor:match.start()])
        cursor = match.end()
    segments.append(text[cursor:])
    return segments


def _has_bare_entity_occurrence(text: str, value: str) -> bool:
    if not value:
        return False
    if f'"{value}"' in text:
        return True
    # ASCII-only word boundaries: under default Unicode \w, CJK
    # characters count as word chars, so an entity followed by kana
    # or ideographs (no whitespace) would slip past this check.
    # ``re.ASCII`` restricts \w to ASCII letters/digits/underscore so
    # adjacent non-Latin script is treated as a boundary, matching the
    # convention in ``pipelines/postedit/normalize.py``.
    return bool(
        re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.ASCII)
    )


def validate_and_strip(
    corrected_question: str,
    instantiations: dict[str, str],
    marked_question: str,
) -> tuple[str, bool, str, list[dict]]:
    """Verify each entity is preserved (modulo allowed polish) and strip braces.

    Returns ``(final_text, ok, error_message, entity_edits)``.
    * ``final_text`` — the brace-stripped question (whether ok or not, for
      inspection).
    * ``ok`` — True if all positional + structural checks passed.
    * ``error_message`` — empty on success; reason on failure.
    * ``entity_edits`` — list of polish edits accepted by the matcher, each
      ``{"expected": "cafes", "observed": "cafe", "tier": "plural"}``. Empty
      list when every marker was exact-match (the common case).
    """
    if not corrected_question:
        return "", False, "empty corrected_question", []

    if _NESTED_BRACE_RE.search(corrected_question):
        return corrected_question, False, "nested braces are not allowed", []

    expected_markers = _extract_marker_values(marked_question)
    observed_markers = _extract_marker_values(corrected_question)
    if len(observed_markers) != len(expected_markers):
        return (
            corrected_question,
            False,
            f"marker sequence mismatch: expected {expected_markers}, got {observed_markers}",
            [],
        )

    entity_edits: list[dict] = []
    for obs, exp in zip(observed_markers, expected_markers):
        result = match_entity(obs, exp)
        if not result.matches:
            return (
                corrected_question,
                False,
                f"marker sequence mismatch: expected {expected_markers}, got {observed_markers}",
                [],
            )
        if result.tier != MATCH_EXACT:
            entity_edits.append({
                "expected": exp, "observed": obs, "tier": result.tier,
            })

    # Verify each observed marker is some allowed polish of a known
    # instantiation value (not just any string).
    known_values = list(instantiations.values())
    unknown: list[str] = []
    for obs in observed_markers:
        if not any(
            match_entity(obs, kv).tier != MATCH_NONE for kv in known_values
        ):
            unknown.append(obs)
    if unknown:
        return corrected_question, False, f"unknown marker(s): {unknown}", []

    outside_segments = _iter_outside_marker_segments(corrected_question)
    if any("{{" in segment or "}}" in segment for segment in outside_segments):
        return (
            corrected_question, False,
            "malformed brace sequence outside markers", [],
        )

    # Bare-entity check: reject if EITHER an original entity value OR any
    # polished form (the observed markers) appears bare outside the braces.
    bare_check_values = set(known_values) | set(observed_markers)
    for value in bare_check_values:
        if any(
            _has_bare_entity_occurrence(segment, value)
            for segment in outside_segments
        ):
            return (
                corrected_question,
                False,
                f"entity {value!r} appears outside {{{{...}}}}",
                [],
            )

    # Strip braces around every observed marker — known_values for the
    # ORIGINAL entities AND the polished observed forms, so both shapes
    # survive the stripper.
    final_text = strip_entity_braces(
        corrected_question, list(bare_check_values),
    )
    return final_text, True, "", entity_edits
