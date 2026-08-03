"""Normalize input JSONL records into the canonical InputQuestion shape."""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from pipelines.postedit.schema import InputQuestion

# Match {{placeholder_name}} in templates
PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")
PROTECTED_PLACEHOLDER_RE = re.compile(
    r"^(?:entity|event)_[A-Z](?:_(?:category|subcategory))?$"
)


def strip_outer_quotes(value: str) -> str:
    """Strip surrounding double quotes from an entity name (benchmark records wrap them)."""
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value.replace('\\"', '"')


def extract_placeholder_occurrences(base_template: str) -> list[str]:
    """Return placeholder names in the order they appear in the template."""
    return PLACEHOLDER_RE.findall(base_template)


def extract_placeholders(base_template: str) -> list[str]:
    """Return placeholder names in first-occurrence order, deduped."""
    seen: list[str] = []
    for name in extract_placeholder_occurrences(base_template):
        if name not in seen:
            seen.append(name)
    return seen


def is_protected_placeholder(placeholder: str) -> bool:
    """Return whether a placeholder names a protected proper-noun slot."""
    return bool(PROTECTED_PLACEHOLDER_RE.fullmatch(placeholder))


def _load_parameter_map(record: dict) -> dict[str, str]:
    """Return lower-cased parameter keys from the record's JSON parameters."""
    raw = record.get("parameters")
    if not raw:
        return {}
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {}
    return {
        str(key).lower(): str(value)
        for key, value in data.items()
        if value is not None
    }


def build_instantiations(record: dict, placeholders: list[str]) -> dict[str, str]:
    """Map placeholder names → entity values from a benchmark record.

    Convention: a placeholder named ``entity_A`` is matched to the field
    ``entity_a_name`` (case-insensitive). Event placeholders reuse the same
    slot naming in benchmark rows (for example ``event_A`` may still be stored
    in ``entity_a_name``). Protected category/subcategory placeholders can come
    from dedicated fields such as ``entity_a_subcategory`` or from parameter
    entries such as ``entity_A_node_type``. Other placeholders (e.g. ``value``,
    ``time_period``) are looked up by ``<name>`` if a matching field exists.
    """
    instantiations: dict[str, str] = {}
    record_lookup = {
        str(key).lower(): str(value)
        for key, value in record.items()
        if value is not None
    }
    parameter_lookup = _load_parameter_map(record)
    for placeholder in placeholders:
        lower = placeholder.lower()
        parts = placeholder.split("_")
        slot = parts[1].lower() if len(parts) >= 2 else ""
        qualifier = parts[2].lower() if len(parts) >= 3 else ""
        candidates = [f"{lower}_name", lower]

        if placeholder.startswith("event_") and slot:
            entity_slot = f"entity_{slot}"
            candidates.extend([f"{entity_slot}_name", entity_slot])

        # entity_A_category and entity_A_subcategory already match
        # f"{lower}_name" / lower above; only the parameter-side alias
        # needs an extra candidate.
        if qualifier == "category" and slot:
            candidates.append(f"entity_{slot}_node_type")

        value: Optional[str] = None
        for key in candidates:
            if key in record_lookup:
                value = record_lookup[key]
                break
            if key in parameter_lookup:
                value = parameter_lookup[key]
                break
        if value is None:
            continue
        instantiations[placeholder] = strip_outer_quotes(value)
    return instantiations


def _ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _overlaps_any(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(_ranges_overlap(span, other) for other in occupied)


def _find_exact_spans(text: str, literal: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(literal, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(literal)))
        start = idx + len(literal)
    return spans


def _find_value_spans(
    question: str,
    value: str,
    occupied: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Find all non-overlapping spans for one value, preferring quoted matches."""
    if not value:
        return []

    spans: list[tuple[int, int]] = []
    quoted_literal = f'"{value}"'
    for start, end in _find_exact_spans(question, quoted_literal):
        inner_span = (start + 1, end - 1)
        if not _overlaps_any(inner_span, occupied):
            spans.append(inner_span)

    blocked = occupied + spans
    # ASCII-only word boundaries: under default Unicode \w, CJK characters
    # count as "word" characters, so an entity literal followed by Japanese
    # kana / Chinese ideographs (no space between) failed the boundary check
    # and the marker skipped the question. re.ASCII restricts \w to ASCII
    # so adjacent CJK acts as a boundary, while still blocking substring
    # matches inside Latin words (e.g. "Café" inside "Caféteria").
    pattern = re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.ASCII)
    for match in pattern.finditer(question):
        span = match.span()
        if _overlaps_any(span, blocked):
            continue
        spans.append(span)
    return spans


def _expected_value_counts(
    placeholder_occurrences: list[str],
    instantiations: dict[str, str],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for placeholder in placeholder_occurrences:
        value = instantiations.get(placeholder)
        if value:
            counts[value] += 1
    return counts


def _apply_marker_spans(question: str, spans: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        parts.append(question[cursor:start])
        parts.append("{{")
        parts.append(question[start:end])
        parts.append("}}")
        cursor = end
    parts.append(question[cursor:])
    return "".join(parts)


def _mark_question_with_reason(
    question: str,
    instantiations: dict[str, str],
    placeholder_occurrences: Optional[list[str]] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return a safely marked question or an error explaining why it was skipped."""
    # NFC-normalise both the question and the entity values so that decomposed
    # diacritics (e.g. "a" + U+0301) match the composed form ("á") that
    # tends to come out of the upstream MT pipelines. Without this, span
    # lookup fails on Vietnamese / Korean / accented Greek even when the
    # entity is visually present.
    question = unicodedata.normalize("NFC", question)
    instantiations = {
        ph: unicodedata.normalize("NFC", v) for ph, v in instantiations.items()
    }

    if placeholder_occurrences is None:
        expected_counts = Counter(value for value in instantiations.values() if value)
    else:
        expected_counts = _expected_value_counts(placeholder_occurrences, instantiations)

    if not expected_counts:
        return None, "no resolved placeholder values to mark"

    occupied: list[tuple[int, int]] = []
    selected_spans: list[tuple[int, int]] = []
    for value, expected_count in sorted(
        expected_counts.items(),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        spans = _find_value_spans(question, value, occupied)
        if len(spans) != expected_count:
            return (
                None,
                f"could not safely mark {value!r}: expected {expected_count} occurrence(s), found {len(spans)}",
            )
        occupied.extend(spans)
        selected_spans.extend(spans)

    return _apply_marker_spans(question, selected_spans), None


def mark_question(
    question: str,
    instantiations: dict[str, str],
    placeholder_occurrences: Optional[list[str]] = None,
) -> Optional[str]:
    """Wrap entity spans in ``question`` with double curly braces.

    Matches are found on the original instantiated question text, not rebuilt
    from the template. Quoted exact matches are preferred, then unquoted
    boundary-safe matches. If the expected multiplicity cannot be satisfied
    without ambiguity, ``None`` is returned so the caller can skip the record.
    """
    marked, _ = _mark_question_with_reason(
        question,
        instantiations,
        placeholder_occurrences=placeholder_occurrences,
    )
    return marked


def normalize_record(record: dict, language: str = "en") -> Optional[InputQuestion]:
    """Convert one benchmark record to an InputQuestion.

    Returns ``None`` if the record is missing required fields or no
    instantiations could be matched.
    """
    required = ("question_id", "question", "base_template")
    if not all(k in record and record[k] for k in required):
        return None

    base_template = record["base_template"]
    placeholder_occurrences = extract_placeholder_occurrences(base_template)
    if not placeholder_occurrences:
        return None

    protected_occurrences = [
        placeholder
        for placeholder in placeholder_occurrences
        if is_protected_placeholder(placeholder)
    ]
    if not protected_occurrences:
        return None

    protected_placeholders = []
    for placeholder in protected_occurrences:
        if placeholder not in protected_placeholders:
            protected_placeholders.append(placeholder)

    instantiations = build_instantiations(record, protected_placeholders)

    # Multimodal: image slots replace entity values with the literal
    # placeholder text the renderer substituted into the question
    # (e.g. ``[IMAGE_A]``). The marker hunt then finds the bracketed
    # text and wraps it as ``{{[IMAGE_A]}}`` — a protected span the
    # LLM treats like any other entity, so it postedits surrounding
    # grammar without ever modifying the bracketed placeholder.
    if record.get("is_multimodal") in (True, "true"):
        for img in record.get("images") or []:
            slot = img.get("slot")
            if not slot:
                continue
            placeholder = f"entity_{slot}"
            if placeholder in protected_placeholders:
                instantiations[placeholder] = f"[IMAGE_{slot}]"

    missing_protected = [
        placeholder
        for placeholder in protected_placeholders
        if placeholder not in instantiations
    ]
    if missing_protected:
        logger.warning(
            f"Skipping question {record['question_id']}: missing protected placeholder(s) "
            f"{missing_protected}"
        )
        return None
    if not instantiations:
        return None

    marked, error = _mark_question_with_reason(
        record["question"],
        instantiations,
        placeholder_occurrences=protected_occurrences,
    )
    if marked is None:
        logger.warning(
            f"Skipping question {record['question_id']}: could not build marked_question ({error})"
        )
        return None

    source_metadata: dict = {}
    # ``vid`` is required by the per-language template lookup
    # (``templates_lookup.TemplateRegistry``) — without it the registry
    # silently falls back to the English ``base_template`` for every
    # row, defeating the purpose of the multilingual prompt.
    for key in ("vid", "tid", "sfid", "kg_source"):
        if key in record and record[key]:
            source_metadata[key] = record[key]

    return InputQuestion(
        question_id=record["question_id"],
        question=record["question"],
        base_template=base_template,
        marked_question=marked,
        instantiations=instantiations,
        language=language,
        source_metadata=source_metadata,
    )


def load_jsonl(path: Path) -> Iterator[dict]:
    """Yield each JSON object from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_input(path: Path, language: str = "en") -> Iterator[InputQuestion]:
    """Read a JSONL file and yield InputQuestion records, skipping invalid rows."""
    for record in load_jsonl(path):
        normalized = normalize_record(record, language=language)
        if normalized is not None:
            yield normalized
