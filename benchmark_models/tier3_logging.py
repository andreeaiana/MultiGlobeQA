"""Smolagents → loguru bridge for tier-3 agent traces.

Smolagents' ``CodeAgent`` writes step banners, code blocks, and tool
results to a ``rich.Console`` (stdout by default). Those don't flow
through loguru, so the per-run ``run.log`` file the runner sets up
in ``benchmark_models.runner.run`` ends up empty of tier-3 content —
the only way to see what an agent did mid-run was to ``tail`` stderr,
which is not captured by SLURM into the prediction directory.

This module supplies :class:`LoguruAgentLogger`, a drop-in replacement
for :class:`smolagents.monitoring.AgentLogger` that tees every log
event through loguru in addition to the rich console. The result:

* Live stdout output is preserved (pretty banners, code highlighting,
  tool tables — useful when running interactively).
* A plain-text mirror lands in every loguru sink — including the
  per-run ``run.log`` — so post-mortem and remote ``tail -f`` work
  off a single file regardless of tier.

The bridge is wired in :func:`benchmark_models.tier3.SmolagentsModel._build_code_agent`
and the oracle-t3 variant's constructor; nothing else needs to change.
"""

from __future__ import annotations

from typing import Any

from loguru import logger as _loguru_logger

from smolagents.monitoring import AgentLogger, LogLevel  # type: ignore[import-not-found]


def _smolagents_level_to_loguru(level: LogLevel | int) -> str:
    """Map a Smolagents :class:`LogLevel` to a loguru severity name."""
    try:
        lvl = LogLevel(level) if not isinstance(level, LogLevel) else level
    except ValueError:
        return "DEBUG"
    if lvl == LogLevel.ERROR:
        return "ERROR"
    if lvl == LogLevel.DEBUG:
        return "DEBUG"
    # OFF and INFO both map to INFO; OFF won't fire log() anyway because
    # the parent's level gate suppresses it.
    return "INFO"


def _plain_text(*args: Any) -> str:
    """Best-effort plain-text rendering of rich-printable arguments.

    Smolagents' ``AgentLogger.log`` accepts rich types (``Text``,
    ``Group``, ``Rule``, ``Syntax``). Calling ``str(...)`` on them
    produces something readable enough for loguru sinks. We strip
    surrounding whitespace and skip empty fragments.
    """
    parts = []
    for a in args:
        if a is None:
            continue
        s = str(a).strip()
        if s:
            parts.append(s)
    return " ".join(parts)


class LoguruAgentLogger(AgentLogger):
    """``AgentLogger`` that also forwards events to loguru.

    Behavioural contract:

    * Calls ``super().<method>(...)`` first so the rich console output
      is preserved (live stdout for interactive use).
    * After the parent call, emits a plain-text version through
      ``loguru.logger`` at a severity mapped from the Smolagents
      :class:`LogLevel`.

    The ``log_messages`` override is intentionally a no-op on the
    loguru side: chat-message dumps are verbose and already captured
    in ``Prediction.tool_calls``, so mirroring them in ``run.log``
    would blow it up to MB-scale without adding diagnostic value.
    """

    _LOG_TAG = "smolagents"

    def log(self, *args: Any, level: int | str | LogLevel = LogLevel.INFO, **kwargs: Any) -> None:
        super().log(*args, level=level, **kwargs)
        # Respect Smolagents' own level gate: if the parent would
        # have suppressed this line, don't surface it through loguru
        # either.
        if isinstance(level, str):
            level_resolved: LogLevel = LogLevel[level.upper()]
        else:
            try:
                level_resolved = LogLevel(level)
            except ValueError:
                level_resolved = LogLevel.DEBUG
        if level_resolved > self.level:
            return
        text = _plain_text(*args)
        if not text:
            return
        _loguru_logger.log(
            _smolagents_level_to_loguru(level_resolved),
            "{}: {}", self._LOG_TAG, text,
        )

    def log_error(self, error_message: str) -> None:
        super().log_error(error_message)
        if error_message:
            _loguru_logger.error("{}: {}", self._LOG_TAG, error_message)

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        super().log_rule(title, level=level)
        if title and LogLevel(level) <= self.level:
            _loguru_logger.log(
                _smolagents_level_to_loguru(level),
                "{}: ── {} ──", self._LOG_TAG, title,
            )

    def log_task(
        self,
        content: str,
        subtitle: str,
        title: str | None = None,
        level: LogLevel = LogLevel.INFO,
    ) -> None:
        super().log_task(content, subtitle, title=title, level=level)
        if LogLevel(level) <= self.level:
            label = title or "task"
            _loguru_logger.log(
                _smolagents_level_to_loguru(level),
                "{}: {} — {}\n{}",
                self._LOG_TAG, label, subtitle, content,
            )

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        super().log_code(title, content, level=level)
        if LogLevel(level) <= self.level:
            _loguru_logger.log(
                _smolagents_level_to_loguru(level),
                "{}: {}:\n{}", self._LOG_TAG, title, content,
            )

    def log_markdown(
        self,
        content: str,
        title: str | None = None,
        level: LogLevel = LogLevel.INFO,
        style: str = "",
    ) -> None:
        super().log_markdown(content, title=title, level=level, style=style)
        if LogLevel(level) <= self.level:
            label = title or "markdown"
            _loguru_logger.log(
                _smolagents_level_to_loguru(level),
                "{}: {}:\n{}", self._LOG_TAG, label, content,
            )

    # ``log_messages`` deliberately not bridged — message dumps are
    # large and already preserved in ``Prediction.tool_calls`` per
    # the module docstring.


__all__ = ["LoguruAgentLogger"]
