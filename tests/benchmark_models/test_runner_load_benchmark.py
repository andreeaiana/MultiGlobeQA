"""Tests for :func:`benchmark_models.runner.load_benchmark`.

Multilingual layout: rows live under ``benchmark/<lang>/<kg>/<sfid>/``
shard directories using ISO 639-3 codes. The ``--language`` flag is
authoritative; a row whose own ``language`` field disagrees produces
a one-time WARNING per loaded directory, helping catch the
"pointed at the wrong language" failure mode.
"""

from __future__ import annotations

import json
import logging
from io import StringIO
from pathlib import Path

import pytest

from benchmark_models.runner import load_benchmark


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(qid: str, language: str | None = None) -> dict:
    out = {
        "question_id": qid,
        "tid": "1.1",
        "sfid": "A",
        "question": "How far is X from Y?",
        "base_template": "What is the distance between {{entity_A}} and {{entity_B}}?",
        "kg_source": "osmh3kg",
        "is_valid": True,
    }
    if language is not None:
        out["language"] = language
    return out


class TestLoadBenchmarkLanguageOverride:
    def test_language_override_wins_over_row(
        self, tmp_path: Path
    ) -> None:
        """The ``--language`` override applies even when the row
        carries a different ``language`` field."""
        path = tmp_path / "bench.jsonl"
        _write_jsonl(path, [_row("q1", language="eng"), _row("q2", language="eng")])
        out = load_benchmark(path, language="deu")
        assert all(ex.language == "deu" for ex in out)

    def test_language_mismatch_warns_once_per_run(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When the override disagrees with row-level languages, the
        loader emits a single WARNING per distinct disagreeing code."""
        path = tmp_path / "bench.jsonl"
        # Two rows tagged 'eng', one untagged. With --language=deu, only
        # the 'eng' tag should produce a warning, and only once.
        _write_jsonl(
            path,
            [
                _row("q1", language="eng"),
                _row("q2", language="eng"),
                _row("q3"),
            ],
        )
        # Capture loguru warnings via a stream-bound handler.
        from loguru import logger as loguru_logger

        sink = StringIO()
        sink_id = loguru_logger.add(sink, level="WARNING")
        try:
            out = load_benchmark(path, language="deu")
        finally:
            loguru_logger.remove(sink_id)

        assert all(ex.language == "deu" for ex in out)
        warnings_text = sink.getvalue()
        # One warning line total: 'eng' was the only mismatched
        # language, and the loader dedupes per code.
        warning_lines = [
            line for line in warnings_text.splitlines()
            if "Row in" in line and "carries language=" in line
        ]
        assert len(warning_lines) == 1
        assert "'eng'" in warning_lines[0] and "'deu'" in warning_lines[0]

    def test_no_language_override_keeps_row_language(
        self, tmp_path: Path
    ) -> None:
        """Without an override, row-level ``language`` flows through."""
        path = tmp_path / "bench.jsonl"
        _write_jsonl(
            path,
            [_row("q1", language="zho_CN"), _row("q2", language="ita")],
        )
        out = load_benchmark(path)
        by_qid = {ex.question_id: ex.language for ex in out}
        assert by_qid["q1"] == "zho_CN"
        assert by_qid["q2"] == "ita"

    def test_no_warning_when_override_matches_row(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "bench.jsonl"
        _write_jsonl(path, [_row("q1", language="deu")])
        from loguru import logger as loguru_logger

        sink = StringIO()
        sink_id = loguru_logger.add(sink, level="WARNING")
        try:
            load_benchmark(path, language="deu")
        finally:
            loguru_logger.remove(sink_id)
        assert "language=" not in sink.getvalue()


class TestLoadBenchmarkSchemaDriftWarnOnce:
    """Unrecognized top-level fields trigger a single WARNING per
    field name, not per row. Otherwise a 33K-row run with one
    drifted column would produce 33K identical warnings."""

    def test_schema_drift_warns_once_per_field(
        self, tmp_path: Path
    ) -> None:
        from benchmark_models.base import reset_schema_drift_warnings
        from loguru import logger as loguru_logger

        reset_schema_drift_warnings()

        path = tmp_path / "bench.jsonl"
        _write_jsonl(
            path,
            [
                {**_row(f"q{i}"), "novel_field": f"value-{i}"}
                for i in range(5)
            ],
        )
        sink = StringIO()
        sink_id = loguru_logger.add(sink, level="WARNING")
        try:
            load_benchmark(path)
        finally:
            loguru_logger.remove(sink_id)

        warning_lines = [
            line for line in sink.getvalue().splitlines()
            if "unrecognized field" in line
        ]
        # 5 rows with the same novel field → 1 warning, not 5.
        assert len(warning_lines) == 1
        assert "'novel_field'" in warning_lines[0]
        # The cleanup helper resets state so subsequent tests in
        # the same process see a fresh warning if they need to.
        reset_schema_drift_warnings()


@pytest.fixture(autouse=True)
def _isolate_loguru():
    """``load_benchmark`` doesn't add handlers, but defensively sweep
    so test-level loguru sinks don't accumulate across tests."""
    from loguru import logger

    yield
    # No-op; sinks added inside tests are removed inside the test.
    _ = logging  # silence unused-import warnings (kept for symmetry)
    _ = logger
