"""End-to-end oracle tests using a mocked backend.

Exercises the full path from :func:`create_model` through the
backend's ``predict`` call, observing the actual prompt string sent
to the mock SDK. Split into two halves:

1. **Oracle path behaves as expected**: with an
   :class:`OracleContext` attached to ``create_model``, the prompt
   sent to the SDK contains the formatted triple context, and the
   resulting :class:`Prediction` records ``prompt_template_id ==
   "tier1.oracle.<mode>"``.
2. **Leak regression**: with the SAME gold triples file present in
   the fixture but NO :class:`OracleContext` set, neither the
   rendered prompt nor the prediction record contains any triple
   content. This is the critical guarantee that makes the no-field-
   on-``Example`` design trustworthy.

The mock is the Google SDK because its response shape is the simplest
to fake; the adapter layer is backend-agnostic so the guarantees hold
for OpenAI / Anthropic / vLLM as well.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmark_models.base import Example
from benchmark_models.config import GoogleVariantConfig
from benchmark_models.oracle import OracleContext, Triple
from benchmark_models.registry import create_model


@pytest.fixture
def example() -> Example:
    return Example.from_benchmark_row(
        {
            "question_id": "qid-1.1",
            "tid": "1.1",
            "sfid": "A",
            "question": "How far apart are X and Y?",
            "base_template": "Distance {{entity_A}} to {{entity_B}}?",
            "kg_source": "osmh3kg",
            "entity_a_id": "1",
            "entity_a_name": "X",
            "entity_b_id": "2",
            "entity_b_name": "Y",
        }
    )


@pytest.fixture
def triples_by_qid() -> dict[str, list[Triple]]:
    """A small gold-triples map covering the example's question_id.

    The distinctive literal ``SECRET_WKT_STRING`` serves as a canary —
    it should appear in oracle prompts and *never* in non-oracle
    prompts or prediction records.
    """
    # Mirror the shape of the real gold triples file (indirect geometry):
    # entity → hasGeometry → geomNode → asWKT → WKT literal.
    return {
        "qid-1.1": [
            {
                "subject": "1",
                "predicate": "http://www.opengis.net/ont/geosparql#hasGeometry",
                "object": "geo1",
            },
            {
                "subject": "geo1",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POINT SECRET_WKT_STRING (1 2)",
            },
            {
                "subject": "2",
                "predicate": "http://www.opengis.net/ont/geosparql#hasGeometry",
                "object": "geo2",
            },
            {
                "subject": "geo2",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POINT SECRET_WKT_STRING (3 4)",
            },
        ]
    }


def _config() -> GoogleVariantConfig:
    return GoogleVariantConfig(
        backend="google",
        model_id="gemini-3-flash-preview",
        tier="plain",
        thinking_level="minimal",
        temperature=0.0,
    )


def _mock_response(text: str = '{"answer": 1.0, "unit": "km"}') -> MagicMock:
    """Build a minimal mock genai response."""
    answer_part = MagicMock()
    answer_part.text = text
    answer_part.thought = False
    candidate = MagicMock()
    candidate.content.parts = [answer_part]
    meta = MagicMock()
    meta.prompt_token_count = 100
    meta.candidates_token_count = 20
    meta.thoughts_token_count = 0
    resp = MagicMock()
    resp.candidates = [candidate]
    resp.usage_metadata = meta
    return resp


class TestOracleEndToEnd:
    """With oracle_context set, the prompt carries the formatted triples."""

    @pytest.mark.parametrize(
        "mode,marker",
        [
            ("structured", '"subject": "1"'),
            ("raw", '"POINT SECRET_WKT_STRING (1 2)"'),
            ("verbalized", "Entity 1 has geometry POINT"),
        ],
    )
    def test_prompt_contains_triple_context_per_mode(
        self,
        registry: Any,
        example: Example,
        triples_by_qid: dict[str, list[Triple]],
        mode: str,
        marker: str,
    ) -> None:
        """Each mode produces its distinctive marker in the sent prompt."""
        ctx = OracleContext(mode=mode, gold_triples_map=triples_by_qid)  # type: ignore[arg-type]
        with patch("google.genai.Client") as mock_client_cls:
            client = MagicMock()
            client.models.generate_content.return_value = _mock_response()
            mock_client_cls.return_value = client

            model = create_model(
                "test-variant",
                _config(),
                registry=registry,
                api_key="fake",
                oracle_context=ctx,
            )
            pred = model.predict(example)

            # Inspect the prompt actually passed to the SDK.
            call = client.models.generate_content.call_args
            sent_prompt: str = call.kwargs["contents"]
            assert marker in sent_prompt, (
                f"Mode {mode!r}: expected marker {marker!r} in the sent "
                f"prompt but it was missing."
            )

            # Prediction records which oracle mode produced it.
            assert pred.prompt_template_id == f"tier1.oracle.{mode}"

    def test_leak_regression_no_triples_in_non_oracle_prompt(
        self,
        registry: Any,
        example: Example,
        triples_by_qid: dict[str, list[Triple]],
    ) -> None:
        """The critical non-leak regression test.

        We construct a non-oracle model the same way the CLI does
        (without passing ``oracle_context``), even though the gold
        triples map exists in the surrounding test scope. The
        contract: nothing from the gold triples map should appear in
        the rendered prompt or in the resulting prediction record.
        """
        canary = "SECRET_WKT_STRING"  # distinctive literal from triples_by_qid
        assert any(
            canary in t["object"]
            for t in triples_by_qid[example.question_id]
        ), "Canary setup sanity check failed"

        with patch("google.genai.Client") as mock_client_cls:
            client = MagicMock()
            client.models.generate_content.return_value = _mock_response()
            mock_client_cls.return_value = client

            # Critical: oracle_context NOT passed.
            model = create_model(
                "test-variant",
                _config(),
                registry=registry,
                api_key="fake",
            )
            pred = model.predict(example)

            sent_prompt: str = client.models.generate_content.call_args.kwargs[
                "contents"
            ]

            # The canary must not appear in anything the model saw,
            # nor in the prediction record that is written to disk.
            assert canary not in sent_prompt, (
                "LEAK: gold-triples canary leaked into a non-oracle prompt. "
                "Gold triples must never flow into non-oracle code paths."
            )
            assert canary not in pred.raw_completion
            assert canary not in pred.model_dump_json()

            # And prompt_template_id should mark this as canonical (not oracle).
            assert pred.prompt_template_id == "tier1.canonical"
