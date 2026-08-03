"""Pydantic models and helpers for input/output records."""

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from config import TAXONOMY


class InputQuestion(BaseModel):
    """A normalized input question ready for LLM post-editing."""

    question_id: str
    question: str
    base_template: str
    marked_question: str
    instantiations: dict[str, str]
    language: str = "en"
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class Edit(BaseModel):
    """A single edit proposed by an LLM."""

    op: str
    orig: str
    fix: str

    @field_validator("op")
    @classmethod
    def validate_op(cls, v: str) -> str:
        if v not in TAXONOMY:
            # Unknown ops are coerced to OTHER rather than rejected outright,
            # so we don't lose the edit if a model returns a slightly off op name.
            return "OTHER"
        return v


class ModelRawOutput(BaseModel):
    """Raw structured output from one LLM call."""

    needs_edit: bool
    edits: list[Edit] = Field(default_factory=list)
    corrected_question: Optional[str] = None


class ModelResult(BaseModel):
    """Per-model result after parsing, validation, and brace stripping.

    ``entity_edits`` records any non-exact polish the LLM applied to
    text inside ``{{...}}`` markers (pluralisation, capitalisation,
    typo fixes — see ``entity_match`` for the tiered acceptance rules).
    Empty for rows where every marker was byte-identical to the input,
    which is the common case.
    """

    raw: Optional[dict[str, Any]] = None
    ops: list[str] = Field(default_factory=list)
    final_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ok: bool = False
    error: Optional[str] = None
    entity_edits: list[dict[str, str]] = Field(default_factory=list)


class EnsembleRecord(BaseModel):
    """Final output record per question.

    The ``decision`` label (unanimous / majority / fallback_gemini) is
    the only signal about voting confidence. There's intentionally no
    boolean "needs human review" — at 1.5M questions per run no one is
    going to review per-row. Filter on ``decision == 'fallback_gemini'``
    if you want to sample low-confidence cases.
    """

    question_id: str
    language: str
    original_question: str
    base_template: str
    marked_question: str
    instantiations: dict[str, str]
    source_metadata: dict[str, Any]
    model_outputs: dict[str, ModelResult]
    decision: str  # unanimous | majority | fallback_gemini
    final_question: str
    selected_model: Optional[str] = None  # name of the model whose output became final


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_first_json_object(text: str) -> Optional[str]:
    """Find and return the first complete top-level JSON object in ``text``.

    Performs a brace-balancing scan that respects strings (so ``{`` and ``}``
    inside string literals don't count) and escaped characters. Returns the
    matching substring including its outer braces, or ``None`` if no balanced
    object is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_model_json(raw: str) -> ModelRawOutput:
    """Parse a raw LLM response string into a ModelRawOutput.

    Tolerates accidental markdown fences and extracts the first complete
    JSON object if the model added surrounding prose.
    """
    if raw is None:
        raise ValueError("empty response")
    text = _JSON_FENCE_RE.sub("", raw).strip()

    # Try direct parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: brace-balanced scan for the first complete {...} object.
        # Handles nested braces and ignores braces inside string literals.
        candidate = _extract_first_json_object(text)
        if candidate is None:
            raise ValueError(f"no JSON object found in response: {raw[:120]}...")
        data = json.loads(candidate)

    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object, got {type(data).__name__}")

    return ModelRawOutput(**data)
