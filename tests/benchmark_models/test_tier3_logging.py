"""Tests for the Smolagents → loguru bridge in tier-3 agent runs.

The bridge tees Smolagents' rich-console output through loguru so
per-run ``run.log`` files capture tier-3 agent traces in the same
file as tier-1/2 warnings. These tests confirm the bridge fires for
each public log method and respects the parent class's level gate.
"""

from __future__ import annotations

from contextlib import contextmanager

from smolagents.monitoring import LogLevel  # type: ignore[import-not-found]

from benchmark_models.tier3_logging import LoguruAgentLogger


@contextmanager
def _loguru_capture(level: str = "DEBUG"):
    """Capture loguru log records into a list. Mirrors the helper used
    by ``tests/benchmark_eval/test_duplicate_qid_warnings.py``."""
    from loguru import logger

    captured: list[str] = []

    def _sink(message):  # type: ignore[no-untyped-def]
        captured.append(str(message))

    sink_id = logger.add(_sink, level=level)
    try:
        yield captured
    finally:
        logger.remove(sink_id)


class TestLoguruAgentLoggerForwarding:
    def test_log_forwards_to_loguru(self) -> None:
        agent_logger = LoguruAgentLogger(level=LogLevel.INFO)
        with _loguru_capture() as captured:
            agent_logger.log("hello world", level=LogLevel.INFO)
        assert any("hello world" in m for m in captured), captured
        assert any("smolagents" in m for m in captured), captured

    def test_log_error_emits_error_level(self) -> None:
        agent_logger = LoguruAgentLogger(level=LogLevel.INFO)
        with _loguru_capture() as captured:
            agent_logger.log_error("boom")
        assert any("boom" in m for m in captured)
        # The captured message includes the loguru level token.
        assert any("ERROR" in m for m in captured), captured

    def test_log_rule_emits_step_banner(self) -> None:
        agent_logger = LoguruAgentLogger(level=LogLevel.INFO)
        with _loguru_capture() as captured:
            agent_logger.log_rule("Step 1")
        assert any("Step 1" in m for m in captured)

    def test_log_code_includes_title_and_content(self) -> None:
        agent_logger = LoguruAgentLogger(level=LogLevel.INFO)
        with _loguru_capture() as captured:
            agent_logger.log_code("Code:", "print(42)")
        joined = "\n".join(captured)
        assert "Code:" in joined
        assert "print(42)" in joined

    def test_log_task_includes_subtitle_and_content(self) -> None:
        agent_logger = LoguruAgentLogger(level=LogLevel.INFO)
        with _loguru_capture() as captured:
            agent_logger.log_task("Compute distance", "tier3a", title="Task")
        joined = "\n".join(captured)
        assert "Compute distance" in joined
        assert "tier3a" in joined

    def test_level_gate_suppresses_debug_when_logger_is_info(self) -> None:
        """When the agent logger is at INFO, DEBUG-level events must
        not reach loguru — matches the parent's rich-console gate."""
        agent_logger = LoguruAgentLogger(level=LogLevel.INFO)
        with _loguru_capture(level="DEBUG") as captured:
            agent_logger.log("noisy debug", level=LogLevel.DEBUG)
        assert not any("noisy debug" in m for m in captured), captured

    def test_level_gate_permits_debug_when_logger_is_debug(self) -> None:
        agent_logger = LoguruAgentLogger(level=LogLevel.DEBUG)
        with _loguru_capture(level="DEBUG") as captured:
            agent_logger.log("noisy debug", level=LogLevel.DEBUG)
        assert any("noisy debug" in m for m in captured), captured

    def test_empty_arguments_do_not_emit(self) -> None:
        """``log()`` with only whitespace / None must not produce a
        loguru line."""
        agent_logger = LoguruAgentLogger(level=LogLevel.INFO)
        with _loguru_capture() as captured:
            agent_logger.log("", level=LogLevel.INFO)
            agent_logger.log(None, level=LogLevel.INFO)
        assert not any("smolagents" in m for m in captured), captured

    def test_log_messages_not_overridden(self) -> None:
        """``log_messages`` is intentionally not bridged — chat dumps
        are large and already preserved in ``Prediction.tool_calls``.
        Verify by inspecting the MRO: our subclass must not override
        the parent's implementation."""
        from smolagents.monitoring import AgentLogger as _ParentLogger

        assert (
            LoguruAgentLogger.log_messages
            is _ParentLogger.log_messages
        ), (
            "LoguruAgentLogger should NOT override log_messages — see the "
            "module docstring for rationale."
        )
