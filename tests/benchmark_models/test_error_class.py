"""Tests for :mod:`benchmark_models.backends._error_class`.

Covers the classifier (``classify_error``) across its four primary
categories (``infra``, ``code_gen``, ``schema_violation``, ``unknown``)
and the exception-chain walking that the classifier relies on.

These tests mirror the style of :mod:`test_retry` — fabricated
exception classes with controlled ``__module__`` / name, no real SDK
imports required.
"""

from __future__ import annotations

import json

import pytest

from benchmark_models.backends._error_class import (
    classify_error,
    classify_error_message,
)


def _make_exc(
    *, module: str, name: str, status: int | None = None, message: str = ""
) -> Exception:
    cls = type(name, (Exception,), {"__module__": module})
    exc = cls(message or f"fake {name}")
    if status is not None:
        exc.status_code = status  # type: ignore[attr-defined]
    return exc


# ---------------------------------------------------------------------------
# code_gen
# ---------------------------------------------------------------------------


class TestCodeGen:
    """Python-exec and smolagents failures map to ``code_gen``."""

    @pytest.mark.parametrize(
        "exc",
        [
            UnboundLocalError("cannot access local variable 'final_answer'"),
            NameError("name 'foo' is not defined"),
            SyntaxError("invalid syntax"),
            IndentationError("unexpected indent"),
        ],
    )
    def test_python_exec_errors(self, exc: Exception) -> None:
        assert classify_error(exc) == "code_gen"

    @pytest.mark.parametrize(
        "name",
        [
            "AgentGenerationError",
            "AgentParsingError",
            "AgentExecutionError",
            "AgentMaxStepsError",
        ],
    )
    def test_smolagents_wrapper_errors(self, name: str) -> None:
        exc = _make_exc(module="smolagents", name=name)
        assert classify_error(exc) == "code_gen"

    def test_substring_match_on_wrapped_message(self) -> None:
        """A generic Exception whose message contains the final_answer
        substring is still ``code_gen``. Handles the case where smolagents
        re-raises a plain Exception with the original message."""
        exc = RuntimeError(
            "cannot access local variable 'final_answer' where it is "
            "not associated with a value"
        )
        assert classify_error(exc) == "code_gen"

    def test_code_gen_in_cause_chain(self) -> None:
        """A wrapper whose ``__cause__`` is a code-gen error classifies
        as ``code_gen``. Smolagents sometimes wraps execution errors in
        its own exception type."""
        inner = UnboundLocalError("cannot access local variable 'final_answer'")
        outer = RuntimeError("agent run failed")
        outer.__cause__ = inner
        assert classify_error(outer) == "code_gen"

    def test_code_gen_wins_over_infra(self) -> None:
        """If a chain contains both a retryable and a code_gen link,
        ``code_gen`` wins (the more specific category)."""
        retryable_inner = _make_exc(module="openai", name="RateLimitError")
        codegen_outer = UnboundLocalError(
            "cannot access local variable 'final_answer'"
        )
        codegen_outer.__cause__ = retryable_inner
        assert classify_error(codegen_outer) == "code_gen"

    def test_context_overflow_beats_smolagents_wrapper_name(self) -> None:
        """An over-context provider 400 wrapped by smolagents as
        ``AgentGenerationError``/``AgentMaxStepsError`` (both code-gen
        names) must classify as ``context_overflow``, not ``code_gen`` —
        the specific message wins over the generic wrapper name. This is
        the tier-3 mislabel the classifier fix corrects."""
        for wrapper in ("AgentGenerationError", "AgentMaxStepsError"):
            exc = _make_exc(
                module="smolagents",
                name=wrapper,
                message="Error in generating model output: "
                "litellm.ContextWindowExceededError: input is too long "
                "for the model's maximum context length",
            )
            assert classify_error(exc) == "context_overflow", wrapper

    def test_bare_code_gen_name_unaffected_by_hoist(self) -> None:
        """A genuine SyntaxError with no context-overflow markers stays
        ``code_gen`` — the hoisted context-overflow check must not
        swallow real capability failures."""
        assert classify_error(SyntaxError("invalid syntax")) == "code_gen"


# ---------------------------------------------------------------------------
# infra
# ---------------------------------------------------------------------------


class TestInfra:
    """Retryable transport errors that reached the classifier map to
    ``infra`` — i.e., they weren't retried or the retries were exhausted."""

    @pytest.mark.parametrize(
        "exc",
        [
            _make_exc(module="openai", name="RateLimitError"),
            _make_exc(module="openai", name="InternalServerError"),
            _make_exc(module="google.api_core.exceptions", name="ServiceUnavailable"),
            _make_exc(module="httpx", name="ReadTimeout"),
            TimeoutError("boom"),
        ],
    )
    def test_retryable_exceptions(self, exc: Exception) -> None:
        assert classify_error(exc) == "infra"

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_by_status_code(self, status: int) -> None:
        exc = _make_exc(module="random.sdk", name="HttpError", status=status)
        assert classify_error(exc) == "infra"

    def test_infra_via_cause_chain(self) -> None:
        inner = _make_exc(module="openai", name="RateLimitError")
        outer = RuntimeError("litellm wrapped this")
        outer.__cause__ = inner
        assert classify_error(outer) == "infra"


# ---------------------------------------------------------------------------
# schema_violation
# ---------------------------------------------------------------------------


class TestSchemaViolation:
    def test_json_decode_error(self) -> None:
        try:
            json.loads("{not valid")
        except json.JSONDecodeError as exc:
            assert classify_error(exc) == "schema_violation"
        else:
            pytest.fail("JSONDecodeError was not raised")

    def test_pydantic_validation_error_by_name(self) -> None:
        """Pydantic raises a class named ``ValidationError`` from the
        ``pydantic`` module. We match by module prefix + name substring."""
        exc = _make_exc(module="pydantic", name="ValidationError")
        assert classify_error(exc) == "schema_violation"


# ---------------------------------------------------------------------------
# unknown
# ---------------------------------------------------------------------------


class TestUnknown:
    def test_plain_value_error(self) -> None:
        """A generic ValueError with no diagnostic content is ``unknown``."""
        assert classify_error(ValueError("something went wrong")) == "unknown"

    def test_key_error(self) -> None:
        assert classify_error(KeyError("missing")) == "unknown"

    def test_never_raises_on_weird_exception(self) -> None:
        """The classifier must not raise even if the exception does
        something exotic in ``__str__`` or ``__cause__``. Protects the
        error path from being broken by a broken exception."""

        class WeirdError(Exception):
            def __str__(self) -> str:
                raise RuntimeError("boom from __str__")

        assert classify_error(WeirdError()) in {"infra", "code_gen", "unknown"}


# ---------------------------------------------------------------------------
# Integration — Prediction accepts and round-trips error_class
# ---------------------------------------------------------------------------


class TestPredictionRoundTrip:
    def test_prediction_accepts_error_class(self) -> None:
        from benchmark_models.base import Prediction, Usage

        pred = Prediction(
            question_id="q1",
            tid="1.1",
            sfid="A",
            answer_type="Boolean",
            model_id="m",
            tier="tier1",
            language="en",
            prompt_template_id="error",
            seed=0,
            snapshot_id="s",
            answer=None,
            refusal=True,
            final_state="error",
            error_class="code_gen",
            usage=Usage(),
            latency_ms=0.0,
        )
        dumped = pred.model_dump_json()
        restored = Prediction.model_validate_json(dumped)
        assert restored.error_class == "code_gen"

    def test_prediction_defaults_error_class_to_none(self) -> None:
        """Old rows (without the field) deserialize with ``None``.
        Protects the mixed-schema-file case when landing mid-run."""
        from benchmark_models.base import Prediction

        raw = {
            "question_id": "q1",
            "tid": "1.1",
            "sfid": "A",
            "answer_type": "Boolean",
            "model_id": "m",
            "tier": "tier1",
            "language": "eng",
            "prompt_template_id": "tier1.canonical",
            "seed": 0,
            "snapshot_id": "s",
            "answer": True,
            "refusal": False,
        }
        pred = Prediction.model_validate(raw)
        assert pred.error_class is None

    def test_invalid_error_class_rejected(self) -> None:
        """Literal type enforcement: an unknown value fails validation."""
        from pydantic import ValidationError

        from benchmark_models.base import Prediction, Usage

        with pytest.raises(ValidationError):
            Prediction(
                question_id="q1",
                tid="1.1",
                sfid="A",
                answer_type="Boolean",
                model_id="m",
                tier="tier1",
                language="en",
                prompt_template_id="error",
                seed=0,
                snapshot_id="s",
                answer=None,
                refusal=True,
                final_state="error",
                error_class="bogus",  # type: ignore[arg-type]
                usage=Usage(),
                latency_ms=0.0,
            )


# ---------------------------------------------------------------------------
# Message-based classifier (context_overflow / infra) + reclassification
# ---------------------------------------------------------------------------


class TestClassifyErrorMessage:
    def test_context_overflow_from_400(self) -> None:
        msg = ("GoogleModel error: API call failed: 400 The input token "
               "count exceeds the maximum context length")
        assert classify_error_message(msg) == "context_overflow"

    def test_transport_drop_is_infra(self) -> None:
        msg = ("GoogleModel error: API call failed: Server disconnected "
               "without sending a response.")
        assert classify_error_message(msg) == "infra"

    def test_benign_message_is_none(self) -> None:
        assert classify_error_message('{"answer": null, "unit": "km"}') is None
        assert classify_error_message("") is None

    def test_classify_error_context_overflow_via_message(self) -> None:
        """A non-retryable request-too-large error (no distinctive type)
        is caught by the message-based branch, not left as ``unknown``."""
        exc = RuntimeError(
            "API call failed: 400 request exceeds the maximum context length"
        )
        assert classify_error(exc) == "context_overflow"


class TestLoadTimeReclassification:
    def test_load_predictions_upgrades_unknown_context_overflow(
        self, tmp_path
    ) -> None:
        """An already-run error row stored as ``error_class='unknown'``
        whose raw_completion is a context-overflow message is upgraded to
        ``context_overflow`` at load time — no re-inference needed."""
        from benchmark_eval.loaders import load_predictions

        row = {
            "question_id": "q1", "tid": "1.1", "sfid": "A",
            "answer_type": "Boolean", "model_id": "m", "tier": "tier1",
            "language": "eng", "prompt_template_id": "t", "seed": 0,
            "snapshot_id": "s", "answer": None, "refusal": True,
            "final_state": "error", "error_class": "unknown",
            "raw_completion": (
                "GoogleModel error: API call failed: 400 exceeds the "
                "maximum context length"
            ),
        }
        p = tmp_path / "predictions.jsonl"
        p.write_text(json.dumps(row) + "\n")
        preds = load_predictions(p)
        assert len(preds) == 1
        assert preds[0].error_class == "context_overflow"
