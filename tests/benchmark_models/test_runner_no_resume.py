"""Tests for the ``resume=False`` (CLI: ``--no-resume``) behaviour of
:func:`benchmark_models.runner.run`.

Regression context: prior to commit landing alongside these tests,
``--no-resume`` only disabled the resume-skip logic but left the
existing predictions file untouched. The runner opens the file in
``"a"`` mode, so a re-run with ``--no-resume`` *appended* its records
to whatever was already there. A 28-question re-run on top of a prior
28-question file produced a 56-record file with each ``question_id``
appearing twice, requiring ``tail -N`` to disentangle. The fix is a
file unlink before the run starts when ``resume=False``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark_models.base import Example, Prediction, Usage
from benchmark_models.runner import run


class _FakeModel:
    """Minimal :class:`Model`-shaped object for runner tests.

    Returns a deterministic prediction per example. We can't import a
    real backend without a live API; this stub mirrors only the
    surface ``run`` actually touches (``predict`` / ``predict_batch``).
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict(self, example: Example) -> Prediction:
        self.calls.append(example.question_id)
        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=example.sfid,
            answer_type="Boolean (yes/no, true/false)",
            model_id="fake",
            tier="tier1",
            language="en",
            prompt_template_id="fake.t1",
            seed=42,
            snapshot_id="x",
            answer=True,
            usage=Usage(),
        )

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        return [self.predict(ex) for ex in examples]


def _example(qid: str) -> Example:
    return Example(
        question_id=qid,
        tid="1.1",
        sfid="A",
        question="dummy",
        base_template="dummy template",
        kg_source="worldkg",
        language="en",
        raw={},
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRunNoResumeTruncates:
    """When ``resume=False``, the existing predictions file is wiped
    before the run starts. Without the fix, predictions from prior
    runs end up mixed with the new run's output."""

    def test_existing_file_cleared(self, tmp_path: Path) -> None:
        out = tmp_path / "predictions.jsonl"
        # Pre-populate with a stale prior-run record.
        out.write_text(
            json.dumps(
                {"question_id": "stale", "answer": "old", "final_state": "answered"}
            )
            + "\n"
        )
        assert out.read_text(), "fixture: file must be non-empty before the run"

        run(_FakeModel(), [_example("q1"), _example("q2")], out, resume=False)

        records = _read_jsonl(out)
        # The stale record must be gone — only the new run's predictions
        # survive.
        assert [r["question_id"] for r in records] == ["q1", "q2"]
        assert all(r.get("answer") is True for r in records)

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        """Removing a non-existent file must not raise."""
        out = tmp_path / "predictions.jsonl"
        assert not out.exists()
        run(_FakeModel(), [_example("q1")], out, resume=False)
        records = _read_jsonl(out)
        assert [r["question_id"] for r in records] == ["q1"]


class TestRunResumeAppends:
    """When ``resume=True`` (the default), an existing file is preserved
    and the runner skips already-completed question_ids."""

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        out = tmp_path / "predictions.jsonl"
        # Prior run: q1 already done (matching the schema enough that
        # load_completed_ids picks it up).
        out.write_text(
            json.dumps(
                {"question_id": "q1", "answer": "old", "final_state": "answered"}
            )
            + "\n"
        )

        model = _FakeModel()
        run(model, [_example("q1"), _example("q2")], out, resume=True)

        # The model only saw q2 — q1 was skipped.
        assert model.calls == ["q2"]
        # The original q1 record is still in the file (unchanged).
        records = _read_jsonl(out)
        assert {r["question_id"] for r in records} == {"q1", "q2"}


@pytest.fixture(autouse=True)
def _isolate_logger():
    """``runner.run`` adds a loguru file handler. Prevent test-level
    handler accumulation by removing the handler set after each test."""
    from loguru import logger

    yield
    # The runner creates and removes its own handler in a finally block;
    # this fixture is a defensive sweep in case a test fails mid-run and
    # leaves a handler registered.
    logger.remove()
