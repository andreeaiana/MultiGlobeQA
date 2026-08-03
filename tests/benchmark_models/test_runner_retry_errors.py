"""Tests for the ``--retry-errors`` path in :mod:`benchmark_models.runner`.

Verifies that :func:`rewrite_dropping_errored`:

* Removes rows whose ``final_state == "error"``.
* Preserves every other row untouched (answered / abstained / cap_hit).
* Creates a timestamped ``.bak`` file before rewriting.
* Is a no-op when the file doesn't exist.
* Leaves the file unchanged when there are no errored rows to drop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_models.runner import rewrite_dropping_errored


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to ``path`` as JSONL."""
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file back into a list of dicts."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def mixed_predictions(tmp_path: Path) -> Path:
    """A predictions.jsonl with a mix of error and non-error rows."""
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        [
            {"question_id": "q1", "final_state": "answered", "answer": 12.5},
            {"question_id": "q2", "final_state": "error", "answer": None},
            {"question_id": "q3", "final_state": "abstained", "answer": None},
            {"question_id": "q4", "final_state": "error", "answer": None},
            {"question_id": "q5", "final_state": "cap_hit", "answer": None},
            {"question_id": "q6", "final_state": "answered", "answer": "Berlin"},
        ],
    )
    return path


def test_drops_errored_rows(mixed_predictions: Path) -> None:
    """Only rows with ``final_state == "error"`` are dropped."""
    kept, dropped, backup = rewrite_dropping_errored(mixed_predictions)
    assert kept == 4
    assert dropped == 2

    surviving = _read_jsonl(mixed_predictions)
    surviving_ids = [r["question_id"] for r in surviving]
    assert surviving_ids == ["q1", "q3", "q5", "q6"]
    assert all(r["final_state"] != "error" for r in surviving)


def test_creates_timestamped_backup(mixed_predictions: Path) -> None:
    """A ``.bak`` file with a timestamp in the name is created."""
    original = _read_jsonl(mixed_predictions)

    _, _, backup = rewrite_dropping_errored(mixed_predictions)

    assert backup is not None
    assert backup.exists()
    assert backup.name.endswith(".bak")
    # Timestamp is YYYY-MM-DDTHH-MM-SS → contains two 'T'/':'/'-' shapes.
    # The actual format uses "-" to be filesystem-safe.
    assert ".jsonl." in backup.name

    # Backup contents match the pre-rewrite state exactly.
    backup_rows = _read_jsonl(backup)
    assert backup_rows == original


def test_no_file_is_noop(tmp_path: Path) -> None:
    """If the file doesn't exist, the function returns (0, 0, None)."""
    path = tmp_path / "does_not_exist.jsonl"
    kept, dropped, backup = rewrite_dropping_errored(path)
    assert (kept, dropped, backup) == (0, 0, None)
    assert not path.exists()


def test_all_clean_file(tmp_path: Path) -> None:
    """A file with no errored rows: nothing dropped, backup still made."""
    path = tmp_path / "predictions.jsonl"
    _write_jsonl(
        path,
        [
            {"question_id": "q1", "final_state": "answered"},
            {"question_id": "q2", "final_state": "abstained"},
        ],
    )
    before = _read_jsonl(path)
    kept, dropped, backup = rewrite_dropping_errored(path)
    assert kept == 2
    assert dropped == 0
    assert backup is not None
    assert backup.exists()
    assert _read_jsonl(path) == before


def test_malformed_line_preserved(tmp_path: Path) -> None:
    """Malformed JSON lines are preserved untouched rather than silently lost.

    The resumer already tolerates these (it just skips them); dropping
    them here would throw away partial user data.
    """
    path = tmp_path / "predictions.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"question_id": "q1", "final_state": "answered"}) + "\n")
        f.write("{not valid json\n")
        f.write(json.dumps({"question_id": "q2", "final_state": "error"}) + "\n")

    kept, dropped, _ = rewrite_dropping_errored(path)
    # 1 answered + 1 malformed line preserved; 1 errored dropped.
    assert kept == 2
    assert dropped == 1
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert "{not valid json" in lines
