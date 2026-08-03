"""Retry with exponential backoff for transient API errors.

Used by every backend (``GoogleModel``, ``OpenAIModel``,
``AnthropicModel``, ``VLLMModel``) and by ``SmolagentsModel`` (tier 3)
to wrap the provider SDK call and retry on transient failures:

* Rate limits (HTTP 429)
* Server errors (HTTP 500, 502, 503, 504)
* Connection / timeout / SSL errors
* Provider-specific overload / unavailable exceptions

The classifier :func:`is_retryable` dispatches on exception type and,
where available, the HTTP status code. Non-retryable errors (auth
failures, 400 bad requests, content filters, schema validation) bubble
up immediately so the caller can emit a proper ``final_state="error"``
prediction and stop wasting time.

The backoff is exponential with jitter: ``min(60, 2**attempt) + U(0,2)``
seconds. At most :data:`MAX_ATTEMPTS` = 5 tries (1 initial + 4 retries).
Worst-case wait ~62s × 4 ≈ 4 minutes for a single question in a
sustained outage, after which the error bubbles and the runner records
``final_state="error"``.

The helper is synchronous (matches today's backends) and thread-safe
(the ThreadPoolExecutor in ``predict_batch`` calls it from multiple
workers). Retries are per-worker with independent jitter — a sustained
provider outage is not handled here; see the plan's Option B for global
rate limiting, which is out of scope.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from loguru import logger

T = TypeVar("T")

MAX_ATTEMPTS: int = 3
"""Total attempts (1 initial + 2 retries). Reduced from the original 5 —
empirically the 4th and 5th retries rarely succeeded for the transient
errors we see (Gemini 500s, brief overload), so they mostly added wait
time rather than recovered failures."""

BASE_DELAY: float = 1.0
"""Base for ``2**attempt`` seconds."""

MAX_DELAY: float = 15.0
"""Per-attempt delay cap. Reduced from 60s — Gemini 500s typically
clear within a few seconds; sustained outages need a different
recovery (``--retry-errors`` after the run)."""

JITTER: float = 2.0


# Module-level toggle controlled by the CLI's ``--no-retry`` flag. When
# False, :func:`retry_with_backoff` calls ``fn()`` exactly once and
# bubbles the first exception. Useful for tier 1 / tier 2 runs where
# the natural recovery path is to fail fast and ``--retry-errors``
# the failed questions afterward (cheap to re-run a single LLM call).
# Tier 3 should leave this on: a transient failure mid-agent-loop
# wastes the steps already taken.
_RETRY_ENABLED: bool = True


def set_retry_enabled(enabled: bool) -> None:
    """Globally enable or disable retry. Called once at process start
    by the CLI; not thread-safe but the toggle is set before any
    workers spawn."""
    global _RETRY_ENABLED
    _RETRY_ENABLED = enabled


def is_retry_enabled() -> bool:
    """Return the current retry state. Used by tests."""
    return _RETRY_ENABLED


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def _status_code(exc: BaseException) -> int | None:
    """Extract an HTTP status code from a provider exception.

    Different SDKs store it on different attributes. Falls through to
    ``None`` when no status is available (e.g., pre-request connection
    errors).
    """
    for attr in ("status_code", "code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value
    return None


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if ``exc`` is worth retrying.

    Retryable: HTTP 429, 500, 502, 503, 504; connection / timeout
    errors; provider-specific overload / unavailable exceptions.
    Non-retryable: 400 / 401 / 403 / 404, content filters, schema
    validation, and anything else (defensive default).
    """
    # Provider-agnostic check by HTTP status code first.
    status = _status_code(exc)
    if status is not None and status in (408, 425, 429, 500, 502, 503, 504):
        return True
    if status is not None and 400 <= status < 500 and status not in (408, 425, 429):
        return False

    # Provider-specific exception types — imported lazily so this
    # module is importable without every SDK installed.
    exc_type = type(exc)
    exc_module = getattr(exc_type, "__module__", "") or ""
    exc_name = exc_type.__name__

    # OpenAI
    if exc_module.startswith("openai"):
        if exc_name in (
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
        ):
            return True
        if exc_name in (
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
            "BadRequestError",
            "UnprocessableEntityError",
            "ContentFilterFinishReasonError",
            "LengthFinishReasonError",
        ):
            return False

        if exc_name in (
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
            "BadRequestError",
            "UnprocessableEntityError",
        ):
            return False

    # Google (google-genai and google.api_core)
    if exc_module.startswith("google"):
        if exc_name in (
            "ServiceUnavailable",
            "InternalServerError",
            "GatewayTimeout",
            "ServerError",
            "DeadlineExceeded",
            "TooManyRequests",
            "ResourceExhausted",
        ):
            return True
        if exc_name in ("ClientError",):
            # google-genai's ClientError covers all 4xx — not retryable.
            return False

    # Smolagents / LiteLLM — check for upstream HTTP status in the
    # message or wrapped cause. LiteLLM often wraps provider errors.
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return is_retryable(cause)

    # Generic network / connection layer — httpx, requests, socket.
    if exc_module.startswith(("httpx", "requests", "urllib3")) and exc_name in (
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ReadError",
        "ConnectionError",
        "Timeout",
    ):
        return True

    # TimeoutError / OSError from the standard library. ECONNRESET,
    # EPIPE, ECONNREFUSED etc. at the socket layer are retryable.
    # Default otherwise: not retryable. A false-negative wastes one
    # question; a false-positive wastes up to 4 minutes per question.
    return isinstance(exc, (TimeoutError, OSError))


# ---------------------------------------------------------------------------
# Retry driver
# ---------------------------------------------------------------------------


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    provider: str,
    question_id: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> T:
    """Call ``fn`` with exponential-backoff retry on transient errors.

    When :func:`set_retry_enabled` has been called with ``False``
    (CLI flag ``--no-retry``), this function calls ``fn`` exactly
    once and re-raises any exception immediately. Intended workflow
    in that mode: let transient failures land as
    ``final_state="error"`` predictions, then run ``--retry-errors``
    on a clean pass afterward.

    Args:
        fn: Zero-argument callable wrapping the SDK call. Return value
            bubbles up unchanged.
        provider: Short provider name (``"google"``, ``"openai"``,
            ``"vllm"``, ``"smolagents"``) used in retry log lines.
        question_id: Optional question id for the log context.
        max_attempts: Override the default cap (rarely needed; tests
            use this to exercise the cap in one step).

    Returns:
        Whatever ``fn()`` returns on success.

    Raises:
        The last exception from ``fn()`` if all retries fail, or the
        first non-retryable exception encountered.
    """
    qid = question_id or "?"

    # Global toggle short-circuit: --no-retry mode skips the loop and
    # bubbles the first exception. Saves wall clock when the user
    # prefers the "fail fast, retry-errors later" recovery path.
    if not _RETRY_ENABLED:
        return fn()

    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            last_exc = exc
            if not is_retryable(exc):
                # Non-retryable: re-raise immediately.
                raise
            if attempt >= max_attempts:
                logger.warning(
                    "{}: retry exhausted after {} attempts for qid={}: "
                    "{}: {}",
                    provider,
                    attempt,
                    qid,
                    type(exc).__name__,
                    exc,
                )
                raise
            delay = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, JITTER)
            logger.warning(
                "{}: retryable error on attempt {}/{} for qid={} "
                "({}: {}); sleeping {:.1f}s",
                provider,
                attempt,
                max_attempts,
                qid,
                type(exc).__name__,
                exc,
                delay,
            )
            time.sleep(delay)

    # Unreachable: either we returned or re-raised above. Guard against
    # a future refactor that swallows the raise.
    assert last_exc is not None
    raise last_exc


__all__ = [
    "MAX_ATTEMPTS",
    "is_retryable",
    "retry_with_backoff",
]
