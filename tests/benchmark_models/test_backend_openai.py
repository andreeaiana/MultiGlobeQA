"""Tests for ``benchmark_models.backends.openai.OpenAIModel``.

All tests mock ``openai.OpenAI`` — no real API calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from benchmark_models.backends.openai import OpenAIModel
from benchmark_models.base import Example, Prediction
from benchmark_models.config import OpenAIVariantConfig


def _config(
    *,
    tier: str = "plain",
    reasoning_effort: str = "none",
    self_consistency_n: int = 1,
    temperature: float = 0.0,
    **kwargs: Any,
) -> OpenAIVariantConfig:
    return OpenAIVariantConfig(
        backend="openai",
        model_id="gpt-5.4",
        tier=tier,
        reasoning_effort=reasoning_effort,
        self_consistency_n=self_consistency_n,
        temperature=temperature,
        **kwargs,
    )


def _example(tid: str = "1.1") -> Example:
    return Example.from_benchmark_row({
        "question_id": f"qid-{tid}",
        "tid": tid,
        "sfid": "A",
        "question": "How far apart are X and Y?",
        "base_template": "What is the distance between {{entity_A}} and {{entity_B}}?",
        "kg_source": "osmh3kg",
        "entity_a_id": "1",
        "entity_a_name": "X",
        "entity_b_id": "2",
        "entity_b_name": "Y",
    })


def _mock_response(
    text: str = '{"answer": 12.5, "unit": "km"}',
    *,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning_tokens: int = 0,
) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = text

    choice = MagicMock()
    choice.message = message

    details = MagicMock()
    details.reasoning_tokens = reasoning_tokens

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.completion_tokens_details = details

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ============================================================================
# Tier 1
# ============================================================================


class TestOpenAIModelTier1:
    @patch("benchmark_models.backends.openai.OpenAI")
    def test_successful_prediction(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"answer": 12.5, "unit": "km"}',
            prompt_tokens=150, completion_tokens=20,
        )
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain", _config(), registry=registry
        )
        pred = model.predict(_example())

        assert isinstance(pred, Prediction)
        assert pred.answer == 12.5
        assert pred.unit == "km"
        assert pred.refusal is False
        assert pred.tier == "tier1"
        assert pred.model_id == "gpt-5.4-plain"
        assert pred.sfid == "A"
        assert pred.usage.input_tokens == 150
        assert pred.usage.output_tokens == 20
        assert pred.latency_ms > 0

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_null_answer_is_refusal(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"answer": null, "unit": null}'
        )
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain", _config(), registry=registry
        )
        pred = model.predict(_example())
        assert pred.answer is None
        assert pred.refusal is True

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_api_exception_produces_error_prediction(
        self, MockClient, registry
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("rate limit")
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain", _config(), registry=registry
        )
        pred = model.predict(_example())
        assert pred.final_state == "error"
        assert "rate limit" in pred.raw_completion

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_malformed_json_produces_refusal(
        self, MockClient, registry
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            "not json at all"
        )
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain", _config(), registry=registry
        )
        pred = model.predict(_example())
        assert pred.answer is None
        assert pred.refusal is True

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_schema_has_additional_properties_false(
        self, MockClient, registry
    ) -> None:
        """OpenAI strict mode requires additionalProperties: false."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response()
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain", _config(), registry=registry
        )
        model.predict(_example())

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        schema = call_kwargs["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False


# ============================================================================
# Tier 2 — reasoning
# ============================================================================


class TestOpenAIModelTier2:
    @patch("benchmark_models.backends.openai.OpenAI")
    def test_reasoning_model_allocates_large_token_budget(
        self, MockClient, registry
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"reasoning": "I think...", "answer": 12.5, "unit": "km"}',
            reasoning_tokens=400,
        )
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-thinking-tier2",
            _config(tier="reasoning", reasoning_effort="high"),
            registry=registry,
        )
        pred = model.predict(_example())

        assert pred.answer == 12.5
        assert pred.usage.reasoning_tokens == 400

        # Verify max_completion_tokens was bumped for reasoning.
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] >= 4096
        # Verify reasoning_effort was passed.
        assert call_kwargs["reasoning_effort"] == "high"

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_prompted_cot_bumps_max_tokens_via_schema_introspection(
        self, MockClient, registry
    ) -> None:
        """Tier-2 variants without a native reasoning channel (vLLM
        Qwen3.5-27B, prompted-CoT only) emit the chain-of-thought
        *inside* the JSON ``reasoning`` field. The base 512-token
        budget truncates that string mid-prose → JSON parse failure
        → silent refusal. Schema-introspection bumps the budget to
        2048 for any call whose schema declares a ``reasoning``
        property *without* a native reasoning_effort channel.

        Regression: qwen35-27b-tier2 smoke produced 1/5 parse-fail
        refusals at 512 max_tokens; bumping to 2048 fits a several-
        sentence CoT paragraph comfortably.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"reasoning": "I computed it.", "answer": 12.5, "unit": "km"}',
        )
        MockClient.return_value = mock_client

        # Reasoning tier + reasoning_effort="none" mirrors the
        # vLLM-tier-2 shape (VLLMHttpVariantConfig.reasoning_effort
        # is a property pinned to "none").
        model = OpenAIModel(
            "gpt-5.4-tier2-prompted",
            _config(
                tier="reasoning",
                reasoning_effort="none",  # prompted-CoT, not native
                temperature=0.0,
            ),
            registry=registry,
        )
        pred = model.predict(_example())

        assert pred.answer == 12.5
        # max_completion_tokens bumped to 2048 (or more) for the
        # prompted-CoT branch, *not* the 25k native-reasoning floor.
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert 2048 <= call_kwargs["max_completion_tokens"] < 25000
        # No reasoning_effort flag — the API would reject it for
        # non-reasoning models, and the bump came from schema
        # introspection, not from reasoning_effort.
        assert "reasoning_effort" not in call_kwargs

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_tier1_keeps_base_max_tokens(
        self, MockClient, registry
    ) -> None:
        """Plain tier-1 (no reasoning in schema, no reasoning_effort)
        keeps ``config.max_tokens`` untouched — the prompted-CoT bump
        must not leak into tier-1 calls."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"answer": 12.5, "unit": "km"}',
        )
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain",
            _config(tier="plain", max_tokens=512),
            registry=registry,
        )
        model.predict(_example())

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["max_completion_tokens"] == 512

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_self_consistency_majority_vote(
        self, MockClient, registry
    ) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _mock_response('{"reasoning": "A", "answer": 12.5, "unit": "km"}'),
            _mock_response('{"reasoning": "B", "answer": 13.0, "unit": "km"}'),
            _mock_response('{"reasoning": "C", "answer": 12.5, "unit": "km"}'),
        ]
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-thinking-tier2",
            _config(
                tier="reasoning", reasoning_effort="high",
                self_consistency_n=3, temperature=0.7,
            ),
            registry=registry,
        )
        pred = model.predict(_example())
        assert pred.answer == 12.5
        assert pred.sc_samples is not None
        assert len(pred.sc_samples) == 3
        assert pred.usage.api_call_count == 3

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_self_consistency_carries_winner_reasoning(
        self, MockClient, registry
    ) -> None:
        """Regression: _predict_self_consistency used to store only
        per-sample answers and discard the reasoning trace at
        aggregation, so every SC-aggregated Prediction came back
        with ``reasoning=None`` even though the per-sample schemas
        required the field and the per-sample Predictions had it
        populated. Observed empty reasoning on every
        qwen35-a3b-thinking-tier2 row. The winning sample's reasoning
        must now ride through to the final Prediction so downstream
        eval can compare CoT depth / quality across tier-2 variants.
        """
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _mock_response('{"reasoning": "compute via haversine first", "answer": 12.5, "unit": "km"}'),
            _mock_response('{"reasoning": "tried Vincenty formula", "answer": 13.0, "unit": "km"}'),
            _mock_response('{"reasoning": "verified via geodesic", "answer": 12.5, "unit": "km"}'),
        ]
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-thinking-tier2",
            _config(
                tier="reasoning", reasoning_effort="high",
                self_consistency_n=3, temperature=0.7,
            ),
            registry=registry,
        )
        pred = model.predict(_example())
        # Winning answer is 12.5 (2 of 3 votes); reasoning should be
        # the FIRST sample whose answer matches the winning key.
        assert pred.answer == 12.5
        assert pred.reasoning == "compute via haversine first"

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_self_consistency_no_consensus_keeps_reasoning_none(
        self, MockClient, registry
    ) -> None:
        """When every sample returned ``answer=None`` (model refused
        across all SC draws), the aggregated answer is None and the
        aggregated reasoning is also None — we don't pick a refused
        sample's reasoning since the run has no winning answer to
        anchor it to. Matches the existing behaviour for the
        ``answer`` field."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _mock_response('{"reasoning": "I dont know", "answer": null, "unit": "km"}'),
            _mock_response('{"reasoning": "I dont know", "answer": null, "unit": "km"}'),
            _mock_response('{"reasoning": "I dont know", "answer": null, "unit": "km"}'),
        ]
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-thinking-tier2",
            _config(
                tier="reasoning", reasoning_effort="high",
                self_consistency_n=3, temperature=0.7,
            ),
            registry=registry,
        )
        pred = model.predict(_example())
        assert pred.answer is None
        assert pred.refusal is True
        assert pred.reasoning is None


# ============================================================================
# Edge cases
# ============================================================================


class TestOpenAIModelEdgeCases:
    @patch("benchmark_models.backends.openai.OpenAI")
    def test_unknown_tid_emits_error(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain", _config(), registry=registry
        )
        pred = model.predict(_example(tid="999.999"))
        assert pred.final_state == "error"
        mock_client.chat.completions.create.assert_not_called()


# ============================================================================
# Batch + concurrency
# ============================================================================


class TestOpenAIModelBatch:
    @patch("benchmark_models.backends.openai.OpenAI")
    def test_batch_returns_in_order(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_response(
            '{"answer": 42.0, "unit": "km"}'
        )
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain", _config(), registry=registry
        )
        preds = model.predict_batch([_example(), _example(), _example()])
        assert len(preds) == 3
        assert all(p.answer == 42.0 for p in preds)

    @patch("benchmark_models.backends.openai.OpenAI")
    def test_batch_runs_concurrently(self, MockClient, registry) -> None:
        import time

        def _slow_generate(*args: Any, **kwargs: Any) -> MagicMock:
            time.sleep(0.1)
            return _mock_response('{"answer": 1.0, "unit": "km"}')

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _slow_generate
        MockClient.return_value = mock_client

        model = OpenAIModel(
            "gpt-5.4-plain",
            _config(),
            registry=registry,
            max_concurrent=5,
        )
        examples = [_example() for _ in range(5)]

        t0 = time.perf_counter()
        preds = model.predict_batch(examples)
        elapsed = time.perf_counter() - t0

        assert len(preds) == 5
        assert elapsed < 0.3, f"Batch took {elapsed:.2f}s — not concurrent?"
