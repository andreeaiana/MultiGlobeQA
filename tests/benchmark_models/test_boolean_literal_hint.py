"""Tests for ``boolean_literal_hint`` on the tier-1 prompts.

The T1 / T1-oracle prompts ask booleans for "yes or no" while the strict
schema only allows true/false/null; "no" shares its first token with null.
The hint inserts the tier-2 true/false sentence on boolean questions only.
"""
from __future__ import annotations

import pytest

from benchmark_models.base import Example
from benchmark_models.oracle import OracleContext, Triple
from benchmark_models.prompts import PromptRenderer
from benchmark_models.schemas import TemplateInfo

T2_SENTENCE = (
    'For yes/no questions, the "answer" value must be the JSON literal true '
    'or false — not the strings "yes" or "no".'
)


@pytest.fixture
def example() -> Example:
    return Example(question_id="q1", tid="1.1", sfid="A", question="Is A east of B?",
                   base_template="Is {{entity_A}} east of {{entity_B}}?",
                   kg_source="WorldKG", language="en")


def _ti(answer_type: str, unit: str | None = None) -> TemplateInfo:
    return TemplateInfo(tid="1.1", sfid="A", ssfid="I", answer_type=answer_type, answer_unit=unit)


BOOL = "Boolean (yes/no, true/false)"
NUM = "Numeric (float)"


def _ctx() -> OracleContext:
    t: Triple = {"subject": "A", "predicate": "http://ex/isEastOf", "object": "B"}
    return OracleContext(mode="verbalized", gold_triples_map={"q1": [t]})


def test_t2_sentence_matches_tier2_template_verbatim() -> None:
    from importlib.resources import files
    txt = files("benchmark_models.prompts").joinpath("tier2_thinking.txt").read_text(encoding="utf-8")
    assert T2_SENTENCE in txt


@pytest.mark.parametrize("oracle", [False, True])
def test_boolean_prompt_gets_hint_once(example: Example, oracle: bool) -> None:
    kw = {"oracle_context": _ctx()} if oracle else {}
    base = PromptRenderer(tier="tier1", **kw).render(example, _ti(BOOL))
    hint = PromptRenderer(tier="tier1", boolean_literal_hint=True, **kw).render(example, _ti(BOOL))
    assert T2_SENTENCE not in base[0]
    assert hint[0].count(T2_SENTENCE) == 1
    assert hint[0].replace(T2_SENTENCE + " ", "", 1) == base[0]
    assert hint[1] == base[1] + "+boolhint"
    assert hint[2] == base[2]


@pytest.mark.parametrize("oracle", [False, True])
def test_non_boolean_prompt_byte_identical(example: Example, oracle: bool) -> None:
    kw = {"oracle_context": _ctx()} if oracle else {}
    base = PromptRenderer(tier="tier1", **kw).render(example, _ti(NUM, "kilometres"))
    hint = PromptRenderer(tier="tier1", boolean_literal_hint=True, **kw).render(example, _ti(NUM, "kilometres"))
    assert hint == base


def test_default_off_is_unchanged(example: Example) -> None:
    r = PromptRenderer(tier="tier1")
    assert T2_SENTENCE not in r.render(example, _ti(BOOL))[0]


def test_tier2_prompted_cot_gets_hint(example: Example) -> None:
    base = PromptRenderer(tier="tier2").render(example, _ti(BOOL))
    hint = PromptRenderer(tier="tier2", boolean_literal_hint=True).render(example, _ti(BOOL))
    assert T2_SENTENCE not in base[0] and hint[0].count(T2_SENTENCE) == 1
    assert hint[0].replace(T2_SENTENCE + " ", "", 1) == base[0]


def test_tier2_thinking_is_idempotent(example: Example) -> None:
    kw = {"tier": "tier2", "prompt_variant": "tier2_thinking"}
    base = PromptRenderer(**kw).render(example, _ti(BOOL))
    hint = PromptRenderer(boolean_literal_hint=True, **kw).render(example, _ti(BOOL))
    assert hint == base and base[0].count(T2_SENTENCE) == 1


HINT_VARIANTS = [
    f"{m}-plain" for m in ("qwen35-27b", "qwen35-a3b", "gemma-3-27b-it", "qwen38-27b",
                           "gemma4-31b", "glm47-flash", "gptoss-120b", "gptoss-20b")
] + [f"{m}-image" for m in ("qwen35-27b", "qwen35-a3b", "gemma-3-27b-it", "qwen38-27b", "gemma4-31b")] + [
    "qwen35-27b-tier2", "gemma-3-27b-it-tier2", "qwen35-a3b-thinking-tier2",
    "qwen38-27b-thinking-tier2", "gemma4-31b-thinking-tier2", "glm47-flash-thinking-tier2",
    "gptoss-120b-thinking-tier2", "gptoss-20b-thinking-tier2",
    "qwen35-27b-tier2-image", "gemma-3-27b-it-tier2-image", "qwen35-a3b-thinking-tier2-image",
    "qwen38-27b-thinking-tier2-image", "gemma4-31b-thinking-tier2-image",
]


def test_hint_variants_set_it_and_default_stays_off() -> None:
    from benchmark_models.config import VariantConfigBase, load_models_catalogue
    assert VariantConfigBase.model_fields["boolean_literal_hint"].default is False
    cat = load_models_catalogue()
    for name in HINT_VARIANTS:
        assert cat.get(name).boolean_literal_hint is True, name
    for name in ("gemini-3-flash-plain", "gemini-3-flash-image", "gemini-3-flash-thinking-tier2",
                 "gemini-3-flash-thinking-tier2-image"):   # Gemini variants do not set the hint
        assert cat.get(name).boolean_literal_hint is False, name
    for name in cat.list_variants():                  # nothing outside tiers 1/2 carries it
        if cat.get(name).boolean_literal_hint:
            assert cat.get(name).tier in ("plain", "reasoning"), name
