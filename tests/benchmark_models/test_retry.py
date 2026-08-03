"""Tests for :mod:`benchmark_models.backends._retry`.

Covers the classifier (``is_retryable``) and the driver
(``retry_with_backoff``):

* Retryable exceptions trigger the backoff loop up to ``max_attempts``.
* Non-retryable exceptions bubble immediately (no sleep, no loop).
* Success on the first attempt returns the value without sleeping.
* Success after one retry returns the value and sleeps exactly once.

The driver uses ``time.sleep`` for backoff — tests monkeypatch it to
a no-op so the suite runs instantly.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmark_models.backends import _retry
from benchmark_models.backends._retry import (
    is_retry_enabled,
    is_retryable,
    retry_with_backoff,
    set_retry_enabled,
)

# ---------------------------------------------------------------------------
# Fake provider exceptions
# ---------------------------------------------------------------------------


def _make_exc(
    *, module: str, name: str, status: int | None = None
) -> Exception:
    """Fabricate an exception with the right ``__module__`` / class name
    and an optional ``status_code``. Used to simulate provider SDK
    exceptions without importing the real SDK types everywhere."""
    cls = type(name, (Exception,), {"__module__": module})
    exc = cls(f"fake {name}")
    if status is not None:
        exc.status_code = status  # type: ignore[attr-defined]
    return exc


# ---------------------------------------------------------------------------
# is_retryable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        _make_exc(module="openai", name="RateLimitError"),
        _make_exc(module="openai", name="APITimeoutError"),
        _make_exc(module="openai", name="APIConnectionError"),
        _make_exc(module="openai", name="InternalServerError"),
        _make_exc(module="google.api_core.exceptions", name="ServiceUnavailable"),
        _make_exc(module="google.api_core.exceptions", name="InternalServerError"),
        _make_exc(module="google.api_core.exceptions", name="GatewayTimeout"),
        _make_exc(module="google.api_core.exceptions", name="ResourceExhausted"),
        _make_exc(module="google.genai.errors", name="APIError", status=503),
        _make_exc(module="httpx", name="ConnectError"),
        _make_exc(module="httpx", name="ReadTimeout"),
        TimeoutError("boom"),
    ],
)
def test_retryable_exceptions(exc: Exception) -> None:
    """These exception types should all be classified as retryable."""
    assert is_retryable(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        _make_exc(module="openai", name="AuthenticationError"),
        _make_exc(module="openai", name="BadRequestError"),
        _make_exc(module="openai", name="NotFoundError"),
        _make_exc(module="openai", name="PermissionDeniedError"),
        _make_exc(module="google.genai.errors", name="ClientError", status=400),
        _make_exc(module="builtins", name="ValueError"),
        ValueError("invalid"),
        KeyError("missing"),
    ],
)
def test_non_retryable_exceptions(exc: Exception) -> None:
    """These exception types should never trigger retry."""
    assert is_retryable(exc) is False


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_by_status_code(status: int) -> None:
    """Any exception carrying a retryable HTTP status code retries."""
    exc = _make_exc(module="random.sdk", name="HttpError", status=status)
    assert is_retryable(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_by_status_code(status: int) -> None:
    """Client-error 4xx (except 408 / 425 / 429) should not retry."""
    exc = _make_exc(module="random.sdk", name="HttpError", status=status)
    assert is_retryable(exc) is False


def test_retryable_via_cause_chain() -> None:
    """A wrapper exception whose ``__cause__`` is retryable retries too.

    LiteLLM in particular wraps provider errors; the classifier walks
    the cause chain so those still work.
    """
    inner = _make_exc(module="openai", name="RateLimitError")
    outer = RuntimeError("litellm wrapped this")
    outer.__cause__ = inner
    assert is_retryable(outer) is True


# ---------------------------------------------------------------------------
# retry_with_backoff driver
# ---------------------------------------------------------------------------


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``time.sleep`` with a recorder so tests run instantly."""
    slept: list[float] = []
    monkeypatch.setattr(_retry.time, "sleep", lambda s: slept.append(s))
    return slept


def test_success_first_try(no_sleep: list[float]) -> None:
    """A successful call on the first try returns the value and never sleeps."""
    result = retry_with_backoff(lambda: 42, provider="test")
    assert result == 42
    assert no_sleep == []


def test_retry_then_succeed(no_sleep: list[float]) -> None:
    """A retryable failure followed by success returns the value.

    Sleeps exactly once (after the first failure, before the second try).
    """
    counter = {"n": 0}

    def fn() -> str:
        counter["n"] += 1
        if counter["n"] == 1:
            raise _make_exc(module="openai", name="RateLimitError")
        return "ok"

    result = retry_with_backoff(fn, provider="test")
    assert result == "ok"
    assert counter["n"] == 2
    assert len(no_sleep) == 1  # one backoff between the two attempts


def test_non_retryable_bubbles_immediately(no_sleep: list[float]) -> None:
    """Non-retryable errors raise on the first attempt with no sleep."""
    def fn() -> Any:
        raise _make_exc(module="openai", name="AuthenticationError")

    with pytest.raises(Exception) as excinfo:
        retry_with_backoff(fn, provider="test")
    assert excinfo.value.__class__.__name__ == "AuthenticationError"
    assert no_sleep == []


def test_exhausted_retries_bubble_last_exception(no_sleep: list[float]) -> None:
    """After ``max_attempts`` failures, the last exception is re-raised.

    The driver sleeps between attempts but not after the last one, so
    ``len(no_sleep) == max_attempts - 1``.
    """
    def fn() -> Any:
        raise _make_exc(module="openai", name="RateLimitError")

    with pytest.raises(Exception) as excinfo:
        retry_with_backoff(fn, provider="test", max_attempts=3)
    assert excinfo.value.__class__.__name__ == "RateLimitError"
    assert len(no_sleep) == 2  # between-attempt backoffs for 3 tries


# ---------------------------------------------------------------------------
# set_retry_enabled / is_retry_enabled
# ---------------------------------------------------------------------------


class TestRetryToggle:
    """The ``--no-retry`` CLI flag flips a module-level switch."""

    def test_default_is_enabled(self) -> None:
        assert is_retry_enabled() is True

    def test_disable_makes_retryable_errors_bubble(
        self, no_sleep: list[float]
    ) -> None:
        """When disabled, the first exception is re-raised on the first
        attempt — no retry loop, no sleep."""
        set_retry_enabled(False)
        try:
            calls = {"n": 0}

            def fn() -> Any:
                calls["n"] += 1
                raise _make_exc(module="openai", name="RateLimitError")

            with pytest.raises(Exception) as excinfo:
                retry_with_backoff(fn, provider="test")
            assert excinfo.value.__class__.__name__ == "RateLimitError"
            assert calls["n"] == 1  # called exactly once
            assert no_sleep == []   # no backoff
        finally:
            set_retry_enabled(True)  # restore default for other tests

    def test_disable_does_not_affect_success_path(
        self, no_sleep: list[float]
    ) -> None:
        """A successful call still returns its value when retry is off."""
        set_retry_enabled(False)
        try:
            assert retry_with_backoff(lambda: 7, provider="test") == 7
            assert no_sleep == []
        finally:
            set_retry_enabled(True)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_max_attempts_is_3(self) -> None:
        """Default reduced from 5 → 3 (4th/5th retries rarely helped)."""
        assert _retry.MAX_ATTEMPTS == 3

    def test_max_delay_is_15s(self) -> None:
        """Default reduced from 60s → 15s for tighter wall-clock."""
        assert _retry.MAX_DELAY == 15.0


def test_question_id_logged(
    no_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """The ``question_id`` argument surfaces in retry log lines.

    Useful for tracing a specific flaky question in the run log.
    """
    import logging

    # loguru routes through its own sink by default; bridge to stdlib
    # so caplog can observe the message.
    from loguru import logger as loguru_logger

    handler_id = loguru_logger.add(
        lambda msg: logging.getLogger("loguru").handle(
            logging.LogRecord(
                name="loguru",
                level=logging.WARNING,
                pathname="",
                lineno=0,
                msg=str(msg),
                args=(),
                exc_info=None,
            )
        ),
        level="WARNING",
    )

    counter = {"n": 0}

    def fn() -> str:
        counter["n"] += 1
        if counter["n"] == 1:
            raise _make_exc(module="openai", name="RateLimitError")
        return "ok"

    with caplog.at_level(logging.WARNING, logger="loguru"):
        retry_with_backoff(fn, provider="test", question_id="q_abc_123")
    loguru_logger.remove(handler_id)

    assert any("q_abc_123" in rec.message for rec in caplog.records)
