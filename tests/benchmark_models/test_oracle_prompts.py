"""Tests for oracle template rendering via :func:`render_prompt` and
:class:`PromptRenderer`.

Covers:

* Each oracle template substitutes ``$triple_context`` correctly.
* Missing ``triple_context`` for an oracle template raises
  :class:`PromptRenderingError`.
* Supplying ``triple_context`` for a non-oracle template raises too.
* ``PromptRenderer.render`` routes to oracle templates when an
  :class:`OracleContext` is attached, and returns the matching
  ``prompt_template_id``.
* ``PromptRenderer`` construction rejects invalid combinations
  (tier 2 + oracle, T3 oracle mode, non-OracleContext arg).
"""

from __future__ import annotations

import pytest

from benchmark_models.base import Example
from benchmark_models.oracle import OracleContext, Triple
from benchmark_models.prompts import (
    PromptRenderer,
    PromptRenderingError,
    render_prompt,
)
from benchmark_models.schemas import TemplateInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def example() -> Example:
    return Example(
        question_id="q1",
        tid="1.1",
        sfid="A",
        question="What is the distance from A to B?",
        base_template="Distance from {{entity_A}} to {{entity_B}}?",
        kg_source="WorldKG",
        language="en",
    )


@pytest.fixture
def template_info() -> TemplateInfo:
    return TemplateInfo(
        tid="1.1",
        sfid="A",
        ssfid="I",
        answer_type="Numeric (float)",
        answer_unit="kilometres",
    )


@pytest.fixture
def tiny_triples() -> list[Triple]:
    return [
        {"subject": "s1", "predicate": "http://ex/pred", "object": "o1"},
    ]


# ---------------------------------------------------------------------------
# render_prompt with oracle variants
# ---------------------------------------------------------------------------


class TestRenderPromptOracleTemplates:
    """Direct ``render_prompt`` calls for each oracle variant."""

    def test_tier1_oracle_structured_substitutes(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        out = render_prompt(
            "tier1",
            example,
            template_info,
            prompt_variant="tier1_oracle_structured",
            triple_context="FAKE_CONTEXT",
        )
        assert "FAKE_CONTEXT" in out
        assert "What is the distance from A to B?" in out
        # The structured wrapper mentions "structured triples".
        assert "structured" in out.lower()

    def test_tier1_oracle_raw_substitutes(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        out = render_prompt(
            "tier1",
            example,
            template_info,
            prompt_variant="tier1_oracle_raw",
            triple_context="FAKE_TRIPLES",
        )
        assert "FAKE_TRIPLES" in out
        assert "RDF triples" in out

    def test_tier1_oracle_verbalized_substitutes(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        out = render_prompt(
            "tier1",
            example,
            template_info,
            prompt_variant="tier1_oracle_verbalized",
            triple_context="FAKE_PROSE",
        )
        assert "FAKE_PROSE" in out
        assert "Relevant facts" in out

    def test_tier3_oracle_substitutes(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        out = render_prompt(
            "tier3",
            example,
            template_info,
            prompt_variant="tier3_oracle",
            triple_context="FAKE_T3_CONTEXT",
            max_tool_calls=10,
        )
        assert "FAKE_T3_CONTEXT" in out
        # T3-oracle must clarify the no-retrieval-tools policy.
        assert "NOT have access" in out
        # Should mention the Python sandbox.
        assert "shapely" in out

    def test_oracle_template_without_context_raises(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        with pytest.raises(PromptRenderingError):
            render_prompt(
                "tier1",
                example,
                template_info,
                prompt_variant="tier1_oracle_raw",
                # Missing: triple_context=...
            )

    def test_non_oracle_template_with_context_raises(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        """Passing triple_context to a vanilla template is a misuse and raises."""
        with pytest.raises(PromptRenderingError):
            render_prompt(
                "tier1",
                example,
                template_info,
                triple_context="should_not_be_here",
            )


# ---------------------------------------------------------------------------
# PromptRenderer adapter
# ---------------------------------------------------------------------------


class TestPromptRendererConstruction:
    def test_plain_tier1_constructs(self) -> None:
        r = PromptRenderer(tier="tier1")
        assert r.oracle_mode is None

    def test_plain_tier2_constructs(self) -> None:
        r = PromptRenderer(tier="tier2")
        assert r.oracle_mode is None

    def test_tier3_rejected(self) -> None:
        """Tier 3 must use SmolagentsModel/SmolagentsOracleModel directly."""
        with pytest.raises(ValueError):
            PromptRenderer(tier="tier3")  # type: ignore[arg-type]

    def test_oracle_with_tier2_rejected(self) -> None:
        """Oracle mode is not supported for tier 2 in this adapter."""
        ctx = OracleContext(mode="structured", gold_triples_map={})
        with pytest.raises(ValueError):
            PromptRenderer(tier="tier2", oracle_context=ctx)

    def test_oracle_t3_mode_rejected_on_adapter(self) -> None:
        """The adapter refuses T3 oracle modes — those use SmolagentsOracleModel."""
        ctx = OracleContext(mode="t3", gold_triples_map={})
        with pytest.raises(ValueError):
            PromptRenderer(tier="tier1", oracle_context=ctx)

    def test_bad_oracle_context_type_rejected(self) -> None:
        with pytest.raises(TypeError):
            PromptRenderer(tier="tier1", oracle_context="not a context")  # type: ignore[arg-type]


class TestPromptRendererRender:
    def test_non_oracle_returns_canonical_id(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        r = PromptRenderer(tier="tier1")
        prompt, ptid, _tctx = r.render(example, template_info)
        assert ptid == "tier1.canonical"
        assert "FAKE" not in prompt  # no injection
        assert example.question in prompt

    def test_oracle_structured_renders_and_tags(
        self,
        example: Example,
        template_info: TemplateInfo,
        tiny_triples: list[Triple],
    ) -> None:
        ctx = OracleContext(
            mode="structured",
            gold_triples_map={example.question_id: tiny_triples},
        )
        r = PromptRenderer(tier="tier1", oracle_context=ctx)
        prompt, ptid, _tctx = r.render(example, template_info)
        assert ptid == "tier1.oracle.structured"
        assert "s1" in prompt
        assert "PREFIXES" in prompt

    def test_oracle_raw_renders_and_tags(
        self,
        example: Example,
        template_info: TemplateInfo,
        tiny_triples: list[Triple],
    ) -> None:
        ctx = OracleContext(
            mode="raw",
            gold_triples_map={example.question_id: tiny_triples},
        )
        r = PromptRenderer(tier="tier1", oracle_context=ctx)
        prompt, ptid, _tctx = r.render(example, template_info)
        assert ptid == "tier1.oracle.raw"
        # Raw form quotes the object as a literal.
        assert '"o1"' in prompt or "<o1>" in prompt

    def test_oracle_verbalized_renders_and_tags(
        self,
        example: Example,
        template_info: TemplateInfo,
        tiny_triples: list[Triple],
    ) -> None:
        ctx = OracleContext(
            mode="verbalized",
            gold_triples_map={example.question_id: tiny_triples},
        )
        r = PromptRenderer(tier="tier1", oracle_context=ctx)
        prompt, ptid, _tctx = r.render(example, template_info)
        assert ptid == "tier1.oracle.verbalized"
        # Verbalized form produces "Entity s1 has ..." for unknown predicates.
        assert "Entity s1 has" in prompt

    def test_oracle_missing_qid_raises(
        self,
        example: Example,
        template_info: TemplateInfo,
    ) -> None:
        """Missing coverage should not happen (caught at run start) — but
        if it does, render raises LookupError, not a silent fallback."""
        ctx = OracleContext(mode="raw", gold_triples_map={})  # empty!
        r = PromptRenderer(tier="tier1", oracle_context=ctx)
        with pytest.raises(LookupError):
            r.render(example, template_info)
