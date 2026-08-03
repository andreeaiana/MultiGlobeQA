"""Error classifier for :class:`~benchmark_models.base.Prediction`.

Populates the ``error_class`` field on failure predictions so the paper's
tool-use analysis can separate *infrastructure* failures (API 5xx,
network) from *capability* failures (bad code generation, schema
violations). Without this separation the headline "errors = wrong"
policy obscures the qualitative difference between "retryable noise"
and "the model did something wrong".

Categories (see :data:`~benchmark_models.base.ErrorClass` for the
canonical enum):

* ``infra`` — retryable transport error that exhausted
  :func:`~benchmark_models.backends._retry.retry_with_backoff`. Includes
  HTTP 429/5xx, connection/timeout errors, provider-specific overload
  exceptions. In a retry policy these would be retried; the
  ``--no-retry`` + ``--retry-errors`` workflow lets us re-run them.
* ``context_overflow`` — the provider rejected the request as too large
  / over the context window (a non-retryable 400/413), so the model
  never produced an answer. Like ``infra`` this is a harness failure,
  not a capability one, but retrying does NOT help. Common in tier-3
  agents that accumulate tool output and in large oracle contexts.
* ``code_gen`` — the tier-3 CodeAgent produced Python that failed at
  execution time (``UnboundLocalError``, ``NameError``, ``SyntaxError``,
  ``IndentationError``), or smolagents' generation/parsing layer
  rejected the model's output (``AgentGenerationError``,
  ``AgentParsingError``). Stochastic; a retry usually draws a different
  code path.
* ``schema_violation`` — the model's structured output failed JSON or
  pydantic validation. Provider-specific parse errors land here.
* ``config`` — the runner couldn't even attempt the question:
  unknown ``tid``, coverage gap in the oracle map, wrong tier for
  the backend. Not the model's fault.
* ``unknown`` — fallback. Indicates a classification gap, not a
  successful categorisation. Fixing these as they surface keeps the
  headline breakdown honest.

The classifier walks the exception's ``__cause__`` / ``__context__``
chain the same way :func:`~benchmark_models.backends._retry.is_retryable`
does, so wrapped exceptions (LiteLLM wrapping provider SDK errors,
smolagents wrapping execution errors) are still classified correctly.
"""

from __future__ import annotations

from benchmark_models.backends._retry import is_retryable

_CODE_GEN_NAMES: frozenset[str] = frozenset(
    {
        # Python execution errors from the model's generated code.
        "SyntaxError",
        "IndentationError",
        "NameError",
        "UnboundLocalError",
        # Smolagents wrappers — present when the agent's own layer
        # detected the failure before it bubbled as a plain Python
        # exception. Matched by name so we don't import smolagents.
        "AgentGenerationError",
        "AgentParsingError",
        "AgentExecutionError",
        "AgentMaxStepsError",
    }
)
"""Exception class names that indicate model-generated code errors.

Matched by ``type(exc).__name__`` so we don't need to import the real
classes (smolagents is an optional dependency, and the bare Python
ones could be wrapped by anything)."""


_CODE_GEN_SUBSTRINGS: tuple[str, ...] = (
    # The specific UnboundLocalError we see in tier 3 when the model
    # writes ``final_answer = X`` in a branch: Python then treats the
    # name as local in the whole function, so the earlier
    # ``final_answer({...})`` call raises.
    "cannot access local variable 'final_answer'",
    "local variable 'final_answer'",
)
"""Message substrings that indicate code-gen failure even when the
class name alone isn't diagnostic (wrapped exceptions, generic
``Exception`` from user-written tools)."""


_CONTEXT_OVERFLOW_SUBSTRINGS: tuple[str, ...] = (
    "context length",
    "maximum context",
    "context_length_exceeded",
    "context window",
    "maximum number of tokens",
    "too many tokens",
    "token limit",
    "reduce the length",
    "input is too long",
    "request entity too large",
    "request too large",
    "payload too large",
)
"""Message markers for a provider rejecting the request as too large /
over the context window (non-retryable ``context_overflow`` harness)."""


_TRANSPORT_SUBSTRINGS: tuple[str, ...] = (
    "server disconnected",
    "connection reset",
    "connection aborted",
    "connection error",
    "remote end closed",
    "broken pipe",
    "eof occurred",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
)
"""Message markers for a transport drop / provider-overload wrap that
``is_retryable`` may miss once flattened into a string (``infra``)."""


def classify_error_message(message: str) -> str | None:
    """Best-effort classification of a *stored* error string.

    Unlike :func:`classify_error` (which needs a live exception object and
    its cause chain), this works from the ``raw_completion`` text a failed
    prediction preserves — e.g. ``"GoogleModel error: API call failed:
    Server disconnected without sending a response."`` — so an already-run
    predictions JSONL can be reclassified at load time without re-running
    inference.

    Returns ``"context_overflow"`` for request-too-large / over-context
    rejections (a non-retryable harness failure — the model never got to
    answer), ``"infra"`` for transport drops / timeouts / overload wraps,
    or ``None`` when no marker matches (leave the existing class as-is).
    """
    if not message:
        return None
    m = message.lower()
    for s in _CONTEXT_OVERFLOW_SUBSTRINGS:
        if s in m:
            return "context_overflow"
    for s in _TRANSPORT_SUBSTRINGS:
        if s in m:
            return "infra"
    return None


def _walk(exc: BaseException) -> list[BaseException]:
    """Return ``exc`` followed by its ``__cause__`` / ``__context__``
    chain. Bounded to avoid pathological cycles."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen and len(seen) < 8:
        seen.append(current)
        nxt = getattr(current, "__cause__", None)
        if nxt is None or nxt is current:
            nxt = getattr(current, "__context__", None)
        if nxt is current:
            break
        current = nxt
    return seen


def classify_error(exc: BaseException) -> str:
    """Map an exception to an :data:`ErrorClass` value.

    Precedence (most-specific first):

    1. Context-overflow: a provider "request too large / over context
       window" rejection anywhere in the chain. Checked *before*
       code-gen because smolagents wraps the provider's 400 in a
       generic ``AgentGenerationError`` / ``AgentMaxStepsError`` — both
       in :data:`_CODE_GEN_NAMES` — so the wrapper name would otherwise
       win over the specific message and mislabel a harness failure as
       a capability one. The bare-Python code-gen names (``SyntaxError``
       …) are not affected: their messages don't carry these markers.
    2. Code-gen: class name or message substring matches the known
       code-gen set, anywhere in the cause chain. Checked before
       ``infra`` because smolagents can wrap a code error in an
       exception that might look like transport noise.
    3. Infra: :func:`is_retryable` returns ``True`` for any link in
       the chain. These reached the classifier only because retry
       was disabled or attempts exhausted.
    4. Schema violation: pydantic / JSON-decode errors.
    5. Message markers: provider request-failure text in the flattened
       message — remaining ``infra`` (transport drop / overload wrap that
       ``is_retryable`` missed). Context-overflow already handled at (1).
    6. Unknown: fallback.

    Never raises — classification must not be able to break an already
    failing prediction.
    """
    try:
        chain = _walk(exc)

        # 1. Context-overflow beats the code-gen wrapper names: a
        #    provider over-context 400 wrapped by smolagents as a generic
        #    AgentGenerationError/AgentMaxStepsError is a harness failure,
        #    not capability. The message is specific; the wrapper is not.
        for link in chain:
            m = str(link).lower()
            if any(s in m for s in _CONTEXT_OVERFLOW_SUBSTRINGS):
                return "context_overflow"

        for link in chain:
            name = type(link).__name__
            if name in _CODE_GEN_NAMES:
                return "code_gen"
            msg = str(link)
            for substr in _CODE_GEN_SUBSTRINGS:
                if substr in msg:
                    return "code_gen"

        for link in chain:
            if is_retryable(link):
                return "infra"

        for link in chain:
            module = getattr(type(link), "__module__", "") or ""
            name = type(link).__name__
            if module.startswith("pydantic") and "Validation" in name:
                return "schema_violation"
            if name in ("JSONDecodeError", "ValidationError"):
                return "schema_violation"

        # Message-based fallback for provider request failures that are
        # not distinguishable by exception type: a non-retryable 400 /
        # context-overflow, or a transport drop flattened into a string
        # wrapper (``is_retryable`` can't see the original type then).
        for link in chain:
            hit = classify_error_message(str(link))
            if hit is not None:
                return hit

        return "unknown"
    except Exception:
        # Never let the classifier itself error out a prediction.
        return "unknown"


__all__ = ["classify_error", "classify_error_message"]
