"""Tests for ``benchmark_models.backends.google.GoogleModel``.

All tests mock the ``google.genai.Client`` so no real API calls are
made. The mock returns canned responses that exercise:
- Successful tier-1 plain prediction with JSON answer.
- Tier-2 reasoning with thinking parts.
- Self-consistency N=3 majority voting.
- Dynamic-type template strict-error handling.
- Unknown tid error handling.
- API exception handling (network errors, rate limits, etc.).
- JSON parse failure handling (malformed response text).
- Usage metadata extraction (prompt tokens, output tokens, thinking tokens).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from benchmark_models.backends.google import GoogleModel
from benchmark_models.base import Example, Prediction
from benchmark_models.config import GoogleVariantConfig


def _config(
    *,
    tier: str = "plain",
    thinking_level: str = "minimal",
    self_consistency_n: int = 1,
    temperature: float = 0.0,
    **kwargs: Any,
) -> GoogleVariantConfig:
    return GoogleVariantConfig(
        backend="google",
        model_id="gemini-3-flash-preview",
        tier=tier,
        thinking_level=thinking_level,
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
    thought_text: str | None = None,
    prompt_tokens: int = 100,
    output_tokens: int = 50,
    thought_tokens: int = 0,
) -> MagicMock:
    """Build a mock genai response object."""
    parts = []
    if thought_text is not None:
        thought_part = MagicMock()
        thought_part.text = thought_text
        thought_part.thought = True
        parts.append(thought_part)

    answer_part = MagicMock()
    answer_part.text = text
    answer_part.thought = False
    parts.append(answer_part)

    candidate = MagicMock()
    candidate.content.parts = parts

    meta = MagicMock()
    meta.prompt_token_count = prompt_tokens
    meta.candidates_token_count = output_tokens
    meta.thoughts_token_count = thought_tokens

    response = MagicMock()
    response.candidates = [candidate]
    response.usage_metadata = meta
    return response


# ============================================================================
# Tier 1 — plain
# ============================================================================


class TestGoogleModelTier1:
    @patch("benchmark_models.backends.google.genai.Client")
    def test_successful_prediction(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            '{"answer": 12.5, "unit": "km"}',
            prompt_tokens=150, output_tokens=20,
        )
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        pred = model.predict(_example())

        assert isinstance(pred, Prediction)
        assert pred.answer == 12.5
        assert pred.unit == "km"
        assert pred.refusal is False
        assert pred.tier == "tier1"
        assert pred.model_id == "gemini-3-flash-plain"
        assert pred.sfid == "A"
        assert pred.answer_type == "Numeric (float)"
        assert pred.usage.input_tokens == 150
        assert pred.usage.output_tokens == 20
        assert pred.latency_ms > 0

    @patch("benchmark_models.backends.google.genai.Client")
    def test_null_answer_is_refusal(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            '{"answer": null}'
        )
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        pred = model.predict(_example())
        assert pred.answer is None
        assert pred.refusal is True

    @patch("benchmark_models.backends.google.genai.Client")
    def test_api_exception_produces_error_prediction(
        self, MockClient, registry
    ) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("rate limit")
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        pred = model.predict(_example())
        assert pred.final_state == "error"
        assert "rate limit" in pred.raw_completion

    @patch("benchmark_models.backends.google.genai.Client")
    def test_malformed_json_produces_refusal(
        self, MockClient, registry
    ) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            "not json at all"
        )
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        pred = model.predict(_example())
        assert pred.answer is None
        assert pred.refusal is True

    @patch("benchmark_models.backends.google.genai.Client")
    def test_boolean_answer_type(self, MockClient, registry) -> None:
        # TID that requires a boolean answer (need to find one in the
        # registry). Check a few.
        bool_tid = None
        for tid, info in registry.templates.items():
            if info.answer_type == "Boolean (yes/no, true/false)":
                bool_tid = tid
                break
        if bool_tid is None:
            pytest.skip("no boolean template in registry")

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            '{"answer": true}'
        )
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        pred = model.predict(_example(tid=bool_tid))
        assert pred.answer is True
        assert pred.refusal is False


# ============================================================================
# Tier 2 — reasoning
# ============================================================================


class TestGoogleModelTier2:
    @patch("benchmark_models.backends.google.genai.Client")
    def test_reasoning_with_thinking(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            '{"reasoning": "Step 1: lookup. Step 2: compute.", "answer": 12.5, "unit": "km"}',
            thought_text="Let me think about this...",
            thought_tokens=80,
        )
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-thinking-tier2",
            _config(tier="reasoning", thinking_level="high"),
            registry=registry,
        )
        pred = model.predict(_example())
        assert pred.answer == 12.5
        assert pred.reasoning is not None
        assert "think about this" in pred.reasoning
        assert pred.usage.reasoning_tokens == 80

    @patch("benchmark_models.backends.google.genai.Client")
    def test_self_consistency_majority_vote(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        # Three calls: two return 12.5, one returns 13.0 → majority is 12.5
        mock_client.models.generate_content.side_effect = [
            _mock_response('{"reasoning": "A", "answer": 12.5, "unit": "km"}'),
            _mock_response('{"reasoning": "B", "answer": 13.0, "unit": "km"}'),
            _mock_response('{"reasoning": "C", "answer": 12.5, "unit": "km"}'),
        ]
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-thinking-tier2",
            _config(tier="reasoning", thinking_level="high",
                    self_consistency_n=3, temperature=0.7),
            registry=registry,
        )
        pred = model.predict(_example())
        assert pred.answer == 12.5
        assert pred.sc_samples is not None
        assert len(pred.sc_samples) == 3
        assert pred.usage.api_call_count == 3


# ============================================================================
# Dynamic-type and unknown-tid
# ============================================================================


class TestGoogleModelEdgeCases:
    @patch("benchmark_models.backends.google.genai.Client")
    def test_unknown_tid_emits_error(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        pred = model.predict(_example(tid="999.999"))
        assert pred.final_state == "error"
        assert "unknown tid" in pred.raw_completion
        mock_client.models.generate_content.assert_not_called()


# ============================================================================
# predict_batch
# ============================================================================


class TestGoogleModelBatch:
    @patch("benchmark_models.backends.google.genai.Client")
    def test_batch_returns_in_order(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            '{"answer": 42.0, "unit": "km"}'
        )
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        preds = model.predict_batch([_example(), _example(), _example()])
        assert len(preds) == 3
        assert all(p.answer == 42.0 for p in preds)

    @patch("benchmark_models.backends.google.genai.Client")
    def test_batch_runs_concurrently(self, MockClient, registry) -> None:
        """Verify threaded batch is faster than sequential.

        Each mock call sleeps 0.1s. Sequential for 5 calls = 0.5s.
        With 5 concurrent threads, should be ~0.1s. We assert < 0.3s
        to leave margin for thread overhead.
        """
        import time

        def _slow_generate(*args: Any, **kwargs: Any) -> MagicMock:
            time.sleep(0.1)
            return _mock_response('{"answer": 1.0, "unit": "km"}')

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = _slow_generate
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain",
            _config(),
            registry=registry,
            max_concurrent=5,
        )
        examples = [_example() for _ in range(5)]

        t0 = time.perf_counter()
        preds = model.predict_batch(examples)
        elapsed = time.perf_counter() - t0

        assert len(preds) == 5
        assert all(p.answer == 1.0 for p in preds)
        # Sequential would be >= 0.5s. Concurrent should be ~0.1s.
        assert elapsed < 0.3, f"Batch took {elapsed:.2f}s — not concurrent?"


# ============================================================================
# Usage metadata
# ============================================================================


class TestUsageMetadata:
    @patch("benchmark_models.backends.google.genai.Client")
    def test_tokens_extracted(self, MockClient, registry) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = _mock_response(
            '{"answer": 1.0, "unit": "km"}',
            prompt_tokens=200, output_tokens=30, thought_tokens=50,
        )
        MockClient.return_value = mock_client

        model = GoogleModel(
            "gemini-3-flash-plain", _config(), registry=registry
        )
        pred = model.predict(_example())
        assert pred.usage.input_tokens == 200
        assert pred.usage.output_tokens == 30
        assert pred.usage.reasoning_tokens == 50
        assert pred.usage.api_call_count == 1
