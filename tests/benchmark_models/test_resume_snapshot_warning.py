"""Resume warns loudly on snapshot_id mismatch.

``run(resume=True)`` filters out completed ``question_id``s but a
regenerated benchmark with identical qids would silently mix stale
predictions with fresh ones if the snapshot changed between runs.
The runner emits a RESUME SNAPSHOT MISMATCH warning in that case;
the user keeps agency to continue if the change was intentional.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

from benchmark_models.runner import load_existing_snapshot_id, run


def _write_jsonl_with_snapshot(
    path: Path, rows: list[dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@contextmanager
def _loguru_capture():
    """Capture loguru WARNING+ messages into a list. ``caplog`` doesn't
    intercept loguru output, so the runner's warnings (which use
    loguru) need a direct sink. Mirrors the helper in
    ``tests/benchmark_eval/test_duplicate_qid_warnings.py``.
    """
    from loguru import logger

    captured: list[str] = []

    def _sink(message):  # type: ignore[no-untyped-def]
        captured.append(str(message))

    sink_id = logger.add(_sink, level="WARNING")
    try:
        yield captured
    finally:
        logger.remove(sink_id)


def test_load_existing_snapshot_id_returns_none_for_missing_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "nope.jsonl"
    assert load_existing_snapshot_id(missing) is None


def test_load_existing_snapshot_id_returns_none_for_empty_file(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert load_existing_snapshot_id(empty) is None


def test_load_existing_snapshot_id_returns_last_line(
    tmp_path: Path,
) -> None:
    """The most recent line wins — runs in progress may have older
    snapshot_ids early in the file from a prior partial run."""
    p = tmp_path / "predictions.jsonl"
    _write_jsonl_with_snapshot(p, [
        {"question_id": "q1", "snapshot_id": "abc123abc123"},
        {"question_id": "q2", "snapshot_id": "abc123abc123"},
        {"question_id": "q3", "snapshot_id": "def456def456"},  # newer
    ])
    assert load_existing_snapshot_id(p) == "def456def456"


def test_load_existing_snapshot_id_skips_malformed_lines(
    tmp_path: Path,
) -> None:
    p = tmp_path / "predictions.jsonl"
    p.write_text(
        json.dumps({"question_id": "q1", "snapshot_id": "good"}) + "\n"
        "not-json\n"
    )
    assert load_existing_snapshot_id(p) == "good"


def test_resume_warns_on_snapshot_mismatch(tmp_path: Path) -> None:
    """``run(resume=True)`` against a predictions file with a different
    snapshot_id emits a RESUME SNAPSHOT MISMATCH warning. Empty
    ``examples`` short-circuits the work loop after the check fires.
    """
    output_path = tmp_path / "predictions.jsonl"
    _write_jsonl_with_snapshot(output_path, [
        {"question_id": "q1", "snapshot_id": "old_snapshot_111"},
    ])

    model = MagicMock()
    model.snapshot_id = "new_snapshot_222"
    model.model_id = "test-model"
    model.tier = "tier1"

    with _loguru_capture() as captured:
        run(model, examples=[], output_path=output_path, resume=True)

    mismatch_msgs = [m for m in captured if "RESUME SNAPSHOT MISMATCH" in m]
    assert mismatch_msgs, (
        f"expected RESUME SNAPSHOT MISMATCH warning, got: {captured!r}"
    )
    assert any("old_snapshot_111" in m for m in mismatch_msgs)
    assert any("new_snapshot_222" in m for m in mismatch_msgs)


def test_resume_skips_warning_when_snapshot_matches(tmp_path: Path) -> None:
    """No mismatch warning when existing and current snapshot_ids agree."""
    output_path = tmp_path / "predictions.jsonl"
    _write_jsonl_with_snapshot(output_path, [
        {"question_id": "q1", "snapshot_id": "same_snapshot"},
    ])

    model = MagicMock()
    model.snapshot_id = "same_snapshot"
    model.model_id = "test-model"
    model.tier = "tier1"

    with _loguru_capture() as captured:
        run(model, examples=[], output_path=output_path, resume=True)

    assert not any("RESUME SNAPSHOT MISMATCH" in m for m in captured), (
        f"unexpected mismatch warning: {captured!r}"
    )


def test_resume_skips_warning_when_no_existing_snapshot(tmp_path: Path) -> None:
    """First run on a fresh path — no warning, since there's no
    existing snapshot to compare against."""
    output_path = tmp_path / "predictions.jsonl"

    model = MagicMock()
    model.snapshot_id = "any_snapshot"
    model.model_id = "test-model"
    model.tier = "tier1"

    with _loguru_capture() as captured:
        run(model, examples=[], output_path=output_path, resume=True)

    assert not any("RESUME SNAPSHOT MISMATCH" in m for m in captured)
    assert load_existing_snapshot_id(output_path) is None
