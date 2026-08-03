"""Tests for ``benchmark_models.prompts``.

Verifies the master prompt templates render correctly for every tier,
that all slots get substituted, that unit handling is correct for both
the per-template override and the type-spec default, that tier-3
requires a tools_section and tiers 1/2 reject one, and that
dynamic-type templates raise PromptRenderingError.
"""

from __future__ import annotations

import pytest

from benchmark_models.base import Example
from benchmark_models.prompts import (
    DEFAULT_MAX_TOOL_CALLS,
    PromptRenderingError,
    _build_distance_clause,
    _build_unit_clause,
    _load_template,
    _resolve_unit,
    reload_templates,
    render_prompt,
)
from benchmark_models.schemas import TYPE_REGISTRY, TemplateInfo

# ============================================================================
# Helpers
# ============================================================================


def _example(
    *,
    question_id: str = "qid",
    tid: str = "1.1",
    sfid: str = "A",
    question: str = "How far apart are X and Y?",
    language: str = "en",
) -> Example:
    return Example.from_benchmark_row(
        {
            "question_id": question_id,
            "tid": tid,
            "sfid": sfid,
            "question": question,
            "base_template": "What is the distance between {{entity_A}} and {{entity_B}}?",
            "kg_source": "osmh3kg",
            "entity_a_id": "1",
            "entity_a_name": "X",
            "entity_b_id": "2",
            "entity_b_name": "Y",
            "language": language,
        }
    )


def _template_info(
    *,
    tid: str = "1.1",
    sfid: str = "A",
    ssfid: str = "I",
    answer_type: str = "Numeric (float)",
    answer_unit: str | None = "kilometres",
) -> TemplateInfo:
    return TemplateInfo.model_validate(
        {
            "tid": tid,
            "sfid": sfid,
            "ssfid": ssfid,
            "answer_type": answer_type,
            "answer_unit": answer_unit,
        }
    )


# ============================================================================
# _build_unit_clause
# ============================================================================


# ============================================================================
# _build_distance_clause
# ============================================================================


class TestBuildDistanceClause:
    def test_distance_unit_produces_clause(self) -> None:
        clause = _build_distance_clause("kilometres")
        assert "geodesic distances" in clause
        assert "closest points" in clause

    def test_distance_squared_no_clause(self) -> None:
        # Area questions should NOT get the distance clause.
        assert _build_distance_clause("kilometres squared") == ""

    def test_none_unit_no_clause(self) -> None:
        assert _build_distance_clause(None) == ""

    def test_empty_unit_no_clause(self) -> None:
        assert _build_distance_clause("") == ""

    def test_case_insensitive(self) -> None:
        clause = _build_distance_clause("Kilometres")
        assert "geodesic" in clause

    # B1: clause must fire for every distance unit, not just km. The
    # geodesic-distance instruction is a learning signal — withholding
    # it from m/mi/nmi questions while giving it to km questions
    # creates instruction-bias across unit groups.
    def test_metres_produce_clause(self) -> None:
        assert "geodesic" in _build_distance_clause("metres")
        assert "geodesic" in _build_distance_clause("m")

    def test_miles_produce_clause(self) -> None:
        assert "geodesic" in _build_distance_clause("miles")
        assert "geodesic" in _build_distance_clause("mi")

    def test_feet_produce_clause(self) -> None:
        assert "geodesic" in _build_distance_clause("feet")
        assert "geodesic" in _build_distance_clause("ft")

    def test_nautical_miles_produce_clause(self) -> None:
        assert "geodesic" in _build_distance_clause("nautical miles")
        assert "geodesic" in _build_distance_clause("nmi")

    def test_area_units_negative_cases(self) -> None:
        # All area units must NOT get the distance clause.
        assert _build_distance_clause("km^2") == ""
        assert _build_distance_clause("km²") == ""
        assert _build_distance_clause("m²") == ""
        assert _build_distance_clause("square miles") == ""

    def test_non_unit_strings(self) -> None:
        # Non-unit strings (counts, etc.) get no clause.
        assert _build_distance_clause("count") == ""
        assert _build_distance_clause("number") == ""


# ============================================================================
# _build_unit_clause
# ============================================================================


class TestBuildUnitClause:
    def test_with_unit(self) -> None:
        assert _build_unit_clause("kilometres") == ", in kilometres"

    def test_with_km_squared(self) -> None:
        assert _build_unit_clause("kilometres squared") == ", in kilometres squared"

    def test_none(self) -> None:
        assert _build_unit_clause(None) == ""

    def test_empty_string(self) -> None:
        assert _build_unit_clause("") == ""


# ============================================================================
# _resolve_unit
# ============================================================================


class TestResolveUnit:
    def test_template_override_takes_precedence(self) -> None:
        info = _template_info(answer_type="Numeric (float)", answer_unit="kilometres squared")
        spec = TYPE_REGISTRY["Numeric (float)"]  # default_unit = "km"
        assert _resolve_unit(info, spec) == "kilometres squared"

    def test_falls_back_to_type_spec_default(self) -> None:
        info = _template_info(answer_type="Numeric (float)", answer_unit=None)
        spec = TYPE_REGISTRY["Numeric (float)"]
        assert _resolve_unit(info, spec) == spec.default_unit

    def test_non_numeric_returns_none(self) -> None:
        info = _template_info(answer_type="Boolean (yes/no, true/false)", answer_unit=None)
        spec = TYPE_REGISTRY["Boolean (yes/no, true/false)"]
        assert _resolve_unit(info, spec) is None

    def test_entity_name_returns_none(self) -> None:
        info = _template_info(answer_type="Entity name (single)", answer_unit=None)
        spec = TYPE_REGISTRY["Entity name (single)"]
        assert _resolve_unit(info, spec) is None


# ============================================================================
# Template loading
# ============================================================================


class TestTemplateLoading:
    def test_tier1_template_loads(self) -> None:
        reload_templates()
        t = _load_template("tier1")
        assert t is not None

    def test_tier2_template_loads(self) -> None:
        t = _load_template("tier2")
        assert t is not None

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_tier3_subtier_templates_load(self, variant: str) -> None:
        t = _load_template(variant)
        assert t is not None

    def test_template_caching_returns_same_object(self) -> None:
        a = _load_template("tier1")
        b = _load_template("tier1")
        assert a is b


# ============================================================================
# render_prompt — tier 1
# ============================================================================


class TestRenderTier1:
    def test_includes_question(self) -> None:
        ex = _example(question="How far apart are Berlin and Paris?")
        info = _template_info()
        out = render_prompt("tier1", ex, info)
        assert "How far apart are Berlin and Paris?" in out

    def test_numeric_includes_unit_clause(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Numeric (float)", answer_unit="kilometres")
        out = render_prompt("tier1", ex, info)
        assert "in kilometres" in out

    def test_numeric_unit_falls_back_to_type_spec_default(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Numeric (float)", answer_unit=None)
        out = render_prompt("tier1", ex, info)
        # Default for Numeric (float) is "km".
        assert "in km" in out

    def test_boolean_no_unit_clause(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Boolean (yes/no, true/false)", answer_unit=None)
        out = render_prompt("tier1", ex, info)
        assert "in km" not in out
        assert "in kilometres" not in out

    def test_includes_natural_language_description(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Numeric (float)")
        out = render_prompt("tier1", ex, info)
        # The type registry says "a numeric value" for Numeric (float).
        assert "a numeric value" in out

    def test_boolean_uses_yes_or_no_description(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Boolean (yes/no, true/false)", answer_unit=None)
        out = render_prompt("tier1", ex, info)
        assert "yes or no" in out

    def test_includes_abstention_instruction(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier1", ex, info)
        assert '"answer": null' in out

    def test_does_not_include_reasoning_field(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier1", ex, info)
        assert "reasoning" not in out.lower()

    def test_does_not_include_tools(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier1", ex, info)
        assert "tool" not in out.lower()

    def test_distance_clause_present_for_distance_questions(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Numeric (float)", answer_unit="kilometres")
        out = render_prompt("tier1", ex, info)
        assert "geodesic distances" in out
        assert "closest points" in out

    def test_distance_clause_absent_for_boolean_questions(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Boolean (yes/no, true/false)", answer_unit=None)
        out = render_prompt("tier1", ex, info)
        assert "geodesic" not in out

    def test_distance_clause_absent_for_area_questions(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Numeric (float)", answer_unit="kilometres squared")
        out = render_prompt("tier1", ex, info)
        assert "geodesic" not in out

    def test_no_unsubstituted_slots(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier1", ex, info)
        assert "$question" not in out
        assert "$answer_type_description" not in out
        assert "$unit_clause" not in out
        assert "$distance_clause" not in out


# ============================================================================
# render_prompt — tier 2
# ============================================================================


class TestRenderTier2:
    def test_includes_reasoning_instruction(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier2", ex, info)
        assert "reasoning" in out.lower()
        assert "step by step" in out.lower()

    def test_includes_question(self) -> None:
        ex = _example(question="A pretty question")
        info = _template_info()
        out = render_prompt("tier2", ex, info)
        assert "A pretty question" in out

    def test_unit_clause_works_for_tier2(self) -> None:
        ex = _example()
        info = _template_info(answer_type="Numeric (float)", answer_unit="kilometres")
        out = render_prompt("tier2", ex, info)
        assert "in kilometres" in out

    def test_abstention_instruction_includes_reasoning_field(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier2", ex, info)
        # Tier 2 abstention has both fields populated.
        assert '"reasoning"' in out
        assert '"answer": null' in out

    def test_does_not_include_tools(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier2", ex, info)
        assert "tool" not in out.lower()

    def test_no_unsubstituted_slots(self) -> None:
        ex = _example()
        info = _template_info()
        out = render_prompt("tier2", ex, info)
        assert "$question" not in out
        assert "$answer_type_description" not in out
        assert "$unit_clause" not in out


# ============================================================================
# render_prompt — tier 3
# ============================================================================


class TestRenderTier3:
    """Tests for all three tier-3 sub-tier prompts (3a, 3b, 3c)."""

    # Mock tools_section content that mirrors what
    # ``benchmark_models.tier3.render_tools_section`` would produce
    # for each sub-tier in production. The tests only exercise the
    # template substitution layer; the realistic-content rendering
    # itself is covered by tests that actually instantiate the
    # smolagents Tool subclasses (under tests requiring smolagents).
    _TOOLS_SECTION_BY_VARIANT: dict[str, str] = {
        "tier3a": (
            "Available tools (call as Python functions):\n"
            "- find_entity(name: str) -> str — search the knowledge graph\n"
            "- find_entities_by_type(category: str, region: str | None) -> str — search by category\n"
            "- get_entity(entity_id: str) -> str — fetch full properties\n"
            "- list_categories() -> str — list categories\n"
            "- final_answer(answer) — submit your final answer and stop\n"
            "\n"
            "Pre-imported libraries you can use in your code:\n"
            "- shapely (shapely.wkt.loads for parsing WKT geometry strings)\n"
            "- h3 (H3 spatial indexing)"
        ),
        "tier3b": (
            "Available tools (call as Python functions):\n"
            "- web_search(query: str) -> str — search the web\n"
            "- visit_webpage(url: str) -> str — fetch a web page's text content\n"
            "- final_answer(answer) — submit your final answer and stop\n"
            "\n"
            "Pre-imported libraries you can use in your code:\n"
            "- shapely\n"
            "- h3"
        ),
        "tier3c": (
            "Available tools (call as Python functions):\n"
            "- find_entity(name: str) -> str — search the knowledge graph\n"
            "- find_entities_by_type(category: str, region: str | None) -> str — search by category\n"
            "- get_entity(entity_id: str) -> str — fetch full properties\n"
            "- list_categories() -> str — list categories\n"
            "- web_search(query: str) -> str — search the web\n"
            "- visit_webpage(url: str) -> str — fetch a web page's text content\n"
            "- final_answer(answer) — submit your final answer and stop\n"
            "\n"
            "Pre-imported libraries you can use in your code:\n"
            "- shapely.wkt.loads\n"
            "- h3"
        ),
    }

    def _render_tier3(self, variant: str, **kwargs) -> str:
        """Helper: render a tier-3 prompt with the given sub-tier variant."""
        ex = kwargs.pop("example", _example())
        info = kwargs.pop("info", _template_info())
        tools_section = kwargs.pop(
            "tools_section",
            self._TOOLS_SECTION_BY_VARIANT[variant],
        )
        return render_prompt(
            "tier3", ex, info,
            tools_section=tools_section,
            prompt_variant=variant,
            **kwargs,
        )

    # -- 3a (KG + Python) --

    def test_3a_includes_kg_tools(self) -> None:
        out = self._render_tier3("tier3a")
        assert "find_entity" in out
        assert "get_entity" in out
        assert "find_entities_by_type" in out
        assert "list_categories" in out

    def test_3a_excludes_web_tools(self) -> None:
        out = self._render_tier3("tier3a")
        assert "web_search" not in out
        assert "visit_webpage" not in out

    def test_3a_mentions_wkt_parsing(self) -> None:
        out = self._render_tier3("tier3a")
        assert "shapely.wkt.loads" in out

    # -- 3b (Web + Python) --

    def test_3b_includes_web_tools(self) -> None:
        out = self._render_tier3("tier3b")
        assert "web_search" in out
        assert "visit_webpage" in out

    def test_3b_excludes_kg_tools(self) -> None:
        out = self._render_tier3("tier3b")
        assert "find_entity" not in out
        assert "get_entity" not in out

    # -- 3c (KG + Web + Python) --

    def test_3c_includes_all_tools(self) -> None:
        out = self._render_tier3("tier3c")
        assert "find_entity" in out
        assert "get_entity" in out
        assert "web_search" in out
        assert "visit_webpage" in out

    # -- Shared behavior across all sub-tiers --

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_includes_question(self, variant: str) -> None:
        out = self._render_tier3(variant, example=_example(question="My question"))
        assert "My question" in out

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_unit_clause_works(self, variant: str) -> None:
        info = _template_info(answer_type="Numeric (float)", answer_unit="kilometres")
        out = self._render_tier3(variant, info=info)
        assert "in kilometres" in out

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_max_tool_calls_substituted(self, variant: str) -> None:
        out = self._render_tier3(variant)
        assert f"up to {DEFAULT_MAX_TOOL_CALLS}" in out
        assert "$max_tool_calls" not in out

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_custom_tool_call_cap(self, variant: str) -> None:
        out = self._render_tier3(variant, max_tool_calls=20)
        assert "up to 20" in out

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_includes_abstention_instruction(self, variant: str) -> None:
        out = self._render_tier3(variant)
        assert '"answer": None' in out or '"answer": null' in out

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_includes_code_format_instruction(self, variant: str) -> None:
        out = self._render_tier3(variant)
        assert "Write ONLY valid Python code" in out

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_includes_final_answer_format(self, variant: str) -> None:
        out = self._render_tier3(variant)
        assert "final_answer" in out

    @pytest.mark.parametrize("variant", ["tier3a", "tier3b", "tier3c"])
    def test_no_unsubstituted_slots(self, variant: str) -> None:
        out = self._render_tier3(variant)
        for slot in ("$question", "$answer_type_description", "$unit_clause",
                      "$distance_clause", "$max_tool_calls"):
            assert slot not in out

    def test_missing_tools_section_raises(self) -> None:
        """``tools_section`` is required for tier-3 non-oracle prompts.

        The templates reference ``$tools_section`` and the model
        section's tier-3 driver passes a freshly-rendered section
        per call (built from the actual loaded tool list). Refusing
        ``None`` here surfaces a misuse — silently substituting an
        empty section would leave the model without a tool catalogue.
        """
        ex = _example()
        info = _template_info()
        with pytest.raises(PromptRenderingError, match="tools_section"):
            render_prompt("tier3", ex, info, prompt_variant="tier3a")


# ============================================================================
# Tier mismatches
# ============================================================================


class TestTierMismatches:
    def test_tier1_with_tools_section_raises(self) -> None:
        ex = _example()
        info = _template_info()
        with pytest.raises(PromptRenderingError) as excinfo:
            render_prompt("tier1", ex, info, tools_section="oops")
        assert "tools_section" in str(excinfo.value)

    def test_tier2_with_tools_section_raises(self) -> None:
        ex = _example()
        info = _template_info()
        with pytest.raises(PromptRenderingError) as excinfo:
            render_prompt("tier2", ex, info, tools_section="oops")
        assert "tools_section" in str(excinfo.value)

    def test_tier1_with_max_tool_calls_raises(self) -> None:
        ex = _example()
        info = _template_info()
        with pytest.raises(PromptRenderingError) as excinfo:
            render_prompt("tier1", ex, info, max_tool_calls=20)
        assert "max_tool_calls" in str(excinfo.value)

    def test_tier2_with_max_tool_calls_raises(self) -> None:
        ex = _example()
        info = _template_info()
        with pytest.raises(PromptRenderingError) as excinfo:
            render_prompt("tier2", ex, info, max_tool_calls=20)
        assert "max_tool_calls" in str(excinfo.value)


# ============================================================================
# Multilingual question text
# ============================================================================


class TestMultilingualQuestion:
    def test_question_substituted_verbatim_for_non_english(self) -> None:
        # The wrapper is English; the question is in the target language
        # and gets substituted as-is.
        german = "Wie weit voneinander entfernt sind Berlin und Paris?"
        ex = _example(question=german, language="de")
        info = _template_info()
        out = render_prompt("tier1", ex, info)
        assert german in out
        # The wrapper is still English.
        assert "spatial question" in out

    def test_chinese_question(self) -> None:
        chinese = "柏林和巴黎相距多远？"
        ex = _example(question=chinese, language="zh-CN")
        info = _template_info()
        out = render_prompt("tier1", ex, info)
        assert chinese in out


# ============================================================================
# All canonical answer types render
# ============================================================================


class TestAllAnswerTypesRender:
    @pytest.mark.parametrize("answer_type", sorted(TYPE_REGISTRY.keys()))
    def test_every_canonical_type_renders_in_tier1(
        self, answer_type: str
    ) -> None:
        ex = _example()
        info = _template_info(answer_type=answer_type, answer_unit=None)
        out = render_prompt("tier1", ex, info)
        spec = TYPE_REGISTRY[answer_type]
        # The natural-language description appears in the output.
        assert spec.nl_description in out
