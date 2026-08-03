"""Tests for the Level 1 prompt tracking work.

Two halves:

* :class:`Prediction` carries ``triple_context``, populated only for
  oracle runs (tier-1 oracle modes via :class:`PromptRenderer`, tier-3
  oracle via :class:`SmolagentsOracleModel`).
* :func:`benchmark_models.mlflow_logging._resolve_template_paths`
  maps the prompt-template-ids on a run's predictions back to the
  template files that should be logged as MLflow artifacts.

The MLflow side of the integration (the ``log_artifact`` call) is
exercised only at the resolver-pure-function layer; running the real
``log_run`` would require a working MLflow tracking store, which the
existing module-level tests already cover.
"""

from __future__ import annotations

import pytest

from benchmark_models.base import Example, Prediction, Usage
from benchmark_models.mlflow_logging import _resolve_template_paths
from benchmark_models.oracle import OracleContext, Triple
from benchmark_models.prompts import PromptRenderer
from benchmark_models.schemas import TemplateInfo

# ---------------------------------------------------------------------------
# Prediction.triple_context default + oracle-population
# ---------------------------------------------------------------------------


class TestPredictionTripleContextDefault:
    def test_default_is_none(self) -> None:
        """A bare-bones Prediction has triple_context=None — non-oracle
        runs must not populate it."""
        p = Prediction(
            question_id="q1",
            tid="1.1",
            sfid="A",
            answer_type="Numeric (float)",
            model_id="m",
            tier="tier1",
            language="en",
            prompt_template_id="tier1.canonical",
            seed=42,
            snapshot_id="x",
            answer=1.0,
            usage=Usage(),
        )
        assert p.triple_context is None

    def test_round_trip_through_jsonl(self) -> None:
        """``triple_context`` survives a model_dump_json → validate cycle.

        Any drift in the field name or type breaks downstream
        consumers (the eval pipeline reads JSONL line-by-line).
        """
        p = Prediction(
            question_id="q1",
            tid="1.1",
            sfid="A",
            answer_type="Numeric (float)",
            model_id="m",
            tier="tier1",
            language="en",
            prompt_template_id="tier1.oracle.raw",
            seed=42,
            snapshot_id="x",
            answer=12.5,
            triple_context="<s> <p> <o> .",
            usage=Usage(),
        )
        roundtrip = Prediction.model_validate_json(p.model_dump_json())
        assert roundtrip.triple_context == "<s> <p> <o> ."


# ---------------------------------------------------------------------------
# PromptRenderer threads triple_context out to the caller
# ---------------------------------------------------------------------------


@pytest.fixture
def example() -> Example:
    return Example(
        question_id="q1",
        tid="1.1",
        sfid="A",
        question="How far apart are X and Y?",
        base_template="Distance {{entity_A}} to {{entity_B}}?",
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
        {"subject": "s", "predicate": "http://ex/pred", "object": "o"},
    ]


class TestPromptRendererReturnsTripleContext:
    def test_non_oracle_returns_none(
        self, example: Example, template_info: TemplateInfo
    ) -> None:
        r = PromptRenderer(tier="tier1")
        prompt, ptid, tctx = r.render(example, template_info)
        assert ptid == "tier1.canonical"
        assert tctx is None

    def test_oracle_returns_formatted_triples(
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
        prompt, ptid, tctx = r.render(example, template_info)
        assert ptid == "tier1.oracle.raw"
        assert tctx is not None
        # The returned triple_context is the same string injected into
        # the prompt body.
        assert tctx in prompt
        # Distinctive marker from the raw formatter.
        assert "<http://ex/pred>" in tctx


# ---------------------------------------------------------------------------
# MLflow template artifact resolver
# ---------------------------------------------------------------------------


class TestResolveTemplatePaths:
    def test_canonical_tier1_resolves(self) -> None:
        paths = _resolve_template_paths({"tier1.canonical"})
        assert len(paths) == 1
        assert paths[0].name == "tier1.txt"
        assert paths[0].is_file()

    def test_oracle_modes_resolve(self) -> None:
        paths = _resolve_template_paths({
            "tier1.oracle.structured",
            "tier1.oracle.raw",
            "tier1.oracle.verbalized",
            "tier3.oracle",
        })
        names = {p.name for p in paths}
        assert names == {
            "tier1_oracle_structured.txt",
            "tier1_oracle_raw.txt",
            "tier1_oracle_verbalized.txt",
            "tier3_oracle.txt",
        }

    def test_tier2_canonical_and_sc_dedupe_to_one_file(self) -> None:
        """Both ids point at ``tier2.txt`` — only log it once."""
        paths = _resolve_template_paths({"tier2.canonical", "tier2.sc"})
        assert len(paths) == 1
        assert paths[0].name == "tier2.txt"

    def test_tier3_subtiers_resolve(self) -> None:
        paths = _resolve_template_paths({
            "tier3.3a",
            "tier3.3b",
            "tier3.3c",
        })
        names = {p.name for p in paths}
        assert names == {"tier3a.txt", "tier3b.txt", "tier3c.txt"}

    def test_unknown_id_skipped(self) -> None:
        """``"error"`` and any other unrecognized id yield no paths,
        not an exception."""
        paths = _resolve_template_paths({"error", "completely.bogus"})
        assert paths == []

    def test_mixed_known_and_unknown(self) -> None:
        """Unknown ids are silently filtered; known ids still resolve."""
        paths = _resolve_template_paths({
            "tier1.canonical",
            "error",
            "tier1.oracle.raw",
        })
        assert len(paths) == 2
        assert {p.name for p in paths} == {"tier1.txt", "tier1_oracle_raw.txt"}

    def test_empty_input(self) -> None:
        assert _resolve_template_paths(set()) == []

    def test_returned_paths_are_real_files(self) -> None:
        """Sanity: every returned path actually exists on disk so
        ``mlflow.log_artifact`` won't error."""
        all_known = {
            "tier1.canonical",
            "tier2.canonical",
            "tier1.oracle.structured",
            "tier1.oracle.raw",
            "tier1.oracle.verbalized",
            "tier3.3a",
            "tier3.3b",
            "tier3.3c",
            "tier3.oracle",
        }
        for path in _resolve_template_paths(all_known):
            assert path.is_file(), f"{path} does not exist"
            assert path.stat().st_size > 0, f"{path} is empty"
