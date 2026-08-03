"""Shared helpers for template-variant recovery.

Two consumers:

* ``scripts.fix_data_defects`` — D8 (worldkg trailing-quote-drop fix):
  uses :func:`build_pattern` (quote-tolerant) to identify the variant,
  :func:`render_template` to reconstruct the corrected ``question``.

* ``pipelines.translation.recover_template_variant`` — production matcher:
  identifies the FTIDv variant for every English benchmark row and
  writes ``variant_ftidv`` (+ recovered slot values) back. Also uses
  :func:`build_pattern`, :func:`render_template`.

Single source of truth for placeholder mapping, regex construction,
and template rendering — avoids drift between the data fix and the
production matcher.
"""

from __future__ import annotations

import re
from typing import Any


PLACEHOLDER_RE = re.compile(r"\{([^}]+)\}")

# Entity-style placeholders backed by row fields. The value is
# substituted literally into the question text by the renderer.
ENTITY_PLACEHOLDER_TO_FIELD: dict[str, str] = {
    "entity_A": "entity_a_name",
    "entity_B": "entity_b_name",
    "entity_C": "entity_c_name",
    "entity_D": "entity_d_name",
    "event_A": "event_a_name",
    "event_B": "event_b_name",
    # Source-data typo at FTIDv 44.1.1: ``{entity A}`` (with space).
    # Preserved in the data; resolved against the same field.
    "entity A": "entity_a_name",
}

# Non-entity placeholders. Their values are NOT stored in any row
# field — they're rendered into the question text and lost from the
# structured row. The matcher captures them via regex wildcards;
# downstream code can write them back into the row to close the
# data-loophole.
NON_ENTITY_PLACEHOLDERS: frozenset[str] = frozenset({
    "value",
    "value_1",
    "value_2",
    "cardinal_direction",
    "time_period",
    "attribute_A",
    "grid_cell_ID",
    "projection_1",
    "projection_2",
})


def get_entity_value(row: dict[str, Any], placeholder: str) -> str | None:
    """Return the row's field value for ``placeholder`` (an entity
    placeholder like ``"entity_A"``). Returns ``None`` when the
    placeholder is unknown OR the row doesn't have a populated value
    for the corresponding field — the matcher uses ``None`` to skip
    candidates that the row can't satisfy.
    """
    field = ENTITY_PLACEHOLDER_TO_FIELD.get(placeholder)
    if field is None:
        return None
    val = row.get(field)
    if val is None or val == "":
        return None
    return str(val)


def get_image_slots(row: dict[str, Any]) -> set[str]:
    """Return the set of slot letters (``"A"``, ``"B"``, ...) that
    have an image attached on this row. Empty set for non-multimodal
    rows.

    The benchmark generator's renderer keeps ``[IMAGE_<slot>]``
    literal in the question text for slots that have an image
    attached, while substituting ``entity_<slot>_name`` for slots
    that don't (per the D1 fix landed in 2026-05-07). Pass the
    returned set to :func:`build_pattern` to enable matching against
    multimodal questions.
    """
    images = row.get("images") or []
    if isinstance(images, str):
        try:
            import json as _json
            images = _json.loads(images)
        except Exception:
            return set()
    out: set[str] = set()
    for img in images:
        if isinstance(img, dict):
            slot = img.get("slot")
            if isinstance(slot, str) and slot:
                out.add(slot)
    return out


def build_pattern(
    template: str,
    row: dict[str, Any],
    *,
    quote_tolerant: bool = False,
    image_slots: set[str] | None = None,
) -> tuple[str, list[str]] | None:
    """Build a regex pattern + the list of non-entity placeholder
    names (in order of appearance) that became wildcard groups.

    Args:
        template: An English template string with ``{...}`` placeholders.
        row: A benchmark row dict.
        quote_tolerant: When True, every ``"`` character in entity
            literal substitutions is rewritten as ``"?`` (each quote
            optional). This tolerates the worldkg renderer's
            trailing-quote-drop bug. Use ``False`` for round-trip
            verification (the corrected question must match strictly
            against the same regex without tolerance).
        image_slots: Set of slot letters (e.g. ``{"A"}``) for which
            the row has an image attached. For each entity placeholder
            ``{entity_<slot>}`` whose ``<slot>`` is in this set, the
            literal string ``[IMAGE_<slot>]`` is substituted in the
            regex (mirroring the benchmark generator's multimodal
            rendering). Pass ``None`` (default) for regular,
            non-multimodal rows.

    Returns:
        ``(pattern_str, non_entity_names_in_order)`` on success, or
        ``None`` when an entity placeholder's row field is missing
        AND its slot doesn't have an image either.
    """
    parts: list[str] = []
    last_end = 0
    non_entity_order: list[str] = []
    image_slots = image_slots or set()
    for m in PLACEHOLDER_RE.finditer(template):
        parts.append(re.escape(template[last_end:m.start()]))
        ph = m.group(1)
        if ph in ENTITY_PLACEHOLDER_TO_FIELD:
            slot_letter = ph[-1] if ph.startswith(("entity_", "event_")) else None
            if slot_letter and slot_letter in image_slots:
                # Multimodal: this slot has an image; the renderer
                # substitutes ``[IMAGE_<slot>]`` literal text rather
                # than the entity name.
                parts.append(re.escape(f"[IMAGE_{slot_letter}]"))
            else:
                v = get_entity_value(row, ph)
                if v is None:
                    return None
                esc = re.escape(v)
                if quote_tolerant:
                    # ``re.escape`` does not escape ``"``, so direct
                    # replacement is safe.
                    esc = esc.replace('"', '"?')
                parts.append(esc)
        else:
            # Any non-entity placeholder (whether in
            # NON_ENTITY_PLACEHOLDERS or unknown) becomes a wildcard.
            parts.append(r"(.+?)")
            non_entity_order.append(ph)
        last_end = m.end()
    parts.append(re.escape(template[last_end:]))
    return "".join(parts), non_entity_order


def render_template(
    template: str,
    row: dict[str, Any],
    non_entity_values: dict[str, str] | None = None,
    *,
    image_slots: set[str] | None = None,
) -> str | None:
    """Render ``template`` with literal entity-name substitution and
    explicit values for non-entity placeholders.

    ``non_entity_values`` maps placeholder names (without braces) to
    the values to substitute (typically captured from a regex match).
    When a non-entity placeholder is required by the template but
    missing from the dict, returns ``None``.

    ``image_slots`` (multimodal): for each entity placeholder
    ``{entity_<slot>}`` whose ``<slot>`` is in this set, render the
    literal string ``[IMAGE_<slot>]`` instead of the entity name.

    Used both for question reconstruction (D8 fix) and for verifying
    that the recovered variant round-trips against the row.
    """
    if non_entity_values is None:
        non_entity_values = {}
    image_slots = image_slots or set()
    out_parts: list[str] = []
    last_end = 0
    for m in PLACEHOLDER_RE.finditer(template):
        out_parts.append(template[last_end:m.start()])
        ph = m.group(1)
        if ph in ENTITY_PLACEHOLDER_TO_FIELD:
            slot_letter = ph[-1] if ph.startswith(("entity_", "event_")) else None
            if slot_letter and slot_letter in image_slots:
                out_parts.append(f"[IMAGE_{slot_letter}]")
            else:
                v = get_entity_value(row, ph)
                if v is None:
                    return None
                out_parts.append(v)
        else:
            v = non_entity_values.get(ph)
            if v is None:
                return None
            out_parts.append(v)
        last_end = m.end()
    out_parts.append(template[last_end:])
    return "".join(out_parts)


def find_variant(
    candidates: list[tuple[str, str]],
    row: dict[str, Any],
    *,
    quote_tolerant: bool = False,
    image_slots: set[str] | None = None,
) -> list[tuple[str, str, dict[str, str]]]:
    """For every (ftidv, template) candidate, try to match against
    ``row['question']`` and return the matches.

    Each match is ``(ftidv, template, captured_non_entity_values)``.
    Empty list = no candidate matched. Length > 1 = ambiguous.

    ``quote_tolerant=True`` makes entity literals accept missing
    ``"`` characters — used to recover variants from rows where the
    worldkg renderer dropped trailing quotes.

    ``image_slots`` enables multimodal matching: entity placeholders
    for those slot letters are substituted as ``[IMAGE_<slot>]``
    literals (mirroring the benchmark renderer). Pass ``None``
    (default) for non-multimodal rows.
    """
    question = row.get("question", "")
    out: list[tuple[str, str, dict[str, str]]] = []
    for ftidv, template in candidates:
        result = build_pattern(
            template, row,
            quote_tolerant=quote_tolerant,
            image_slots=image_slots,
        )
        if result is None:
            continue
        pattern, non_entity_names = result
        m = re.fullmatch(pattern, question)
        if m is None:
            continue
        groups = m.groups()
        captures = dict(zip(non_entity_names, groups, strict=True))
        out.append((ftidv, template, captures))
    return out


__all__ = [
    "PLACEHOLDER_RE",
    "ENTITY_PLACEHOLDER_TO_FIELD",
    "NON_ENTITY_PLACEHOLDERS",
    "get_entity_value",
    "get_image_slots",
    "build_pattern",
    "render_template",
    "find_variant",
]
