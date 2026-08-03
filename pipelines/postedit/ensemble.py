"""Ensemble voting: decide the final corrected question from per-model results."""

import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from pipelines.postedit.config import MODEL_NAMES
from pipelines.postedit.schema import ModelResult


@dataclass
class Decision:
    final_text: str
    decision: str  # "unanimous" | "majority" | "fallback_gemini"
    selected_model: Optional[str]  # name of the model whose text became final_text


def normalize_for_compare(text: str) -> str:
    """Normalize text for equality comparison: NFC + collapse whitespace + strip."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()


def _pick_model_with_text(
    target_norm: str, ok_results: dict[str, ModelResult]
) -> tuple[str, str]:
    """Return ``(model_name, original_final_text)`` for the first ok model
    whose normalized final_text matches ``target_norm``."""
    for name in MODEL_NAMES:
        r = ok_results.get(name)
        if r and normalize_for_compare(r.final_text) == target_norm:
            return name, r.final_text

    for name, r in ok_results.items():
        if normalize_for_compare(r.final_text) == target_norm:
            return name, r.final_text
    # Should not happen — caller guarantees target_norm came from ok_results
    name, r = next(iter(ok_results.items()))
    return name, r.final_text


def decide(results: dict[str, ModelResult], original_question: str) -> Decision:
    """Vote on the final text across model results.

    Decision labels:
    * ``unanimous`` — every successful model agreed.
    * ``majority`` — more than half of the successful models agreed
      (e.g. 2/3); there's at least one outlier.
    * ``fallback_gemini`` — no clear majority *or* fewer than two
      models succeeded; the result falls back to Gemini if available,
      else the first ok model in MODEL_NAMES order, else the original
      question.

    The label is the only signal the pipeline emits about voting
    confidence — there's no separate "needs human review" boolean
    because at 1.5M questions per run no one is reviewing per-row.
    Downstream filters can key on ``decision == 'fallback_gemini'`` if
    they want to sample the low-confidence cases.
    """
    ok = {n: r for n, r in results.items() if r.ok}

    # Insufficient successful models
    if len(ok) == 0:
        return Decision(original_question, "fallback_gemini", None)
    if len(ok) == 1:
        only_name, only = next(iter(ok.items()))
        return Decision(only.final_text, "fallback_gemini", only_name)

    # Compare normalized final texts
    norm_texts = {n: normalize_for_compare(r.final_text) for n, r in ok.items()}
    counts = Counter(norm_texts.values())
    top_norm, top_count = counts.most_common(1)[0]

    if top_count == len(ok) and len(ok) >= 2:
        # All ok models agree — unanimous, regardless of how many
        # models we ran with. (2/2 agreed has no outlier, unlike 2/3
        # which is "majority".)
        selected, final = _pick_model_with_text(top_norm, ok)
        return Decision(final, "unanimous", selected)

    if top_count >= 2:
        # Majority (e.g. 2/3) — at least one outlier in the ensemble.
        selected, final = _pick_model_with_text(top_norm, ok)
        return Decision(final, "majority", selected)

    # No majority — fall back to Gemini if available, else the first ok
    # model in the canonical MODEL_NAMES order. Iterating ok.items() directly
    # would be timing-dependent because results is populated in completion
    # order by the threaded pipeline.
    if "gemini" in ok:
        return Decision(ok["gemini"].final_text, "fallback_gemini", "gemini")
    for name in MODEL_NAMES:
        r = ok.get(name)
        if r is not None:
            return Decision(r.final_text, "fallback_gemini", name)
    # Defensive: should not be reached because ok is non-empty here.
    first_name, first = next(iter(ok.items()))
    return Decision(first.final_text, "fallback_gemini", first_name)
