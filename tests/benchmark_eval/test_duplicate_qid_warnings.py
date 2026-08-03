"""R4-3: duplicate ``question_id`` rows on the core eval path emit
warnings (last-wins semantics preserved).

Pre-R4-3:
- ``load_gold_rows``: ``out[qid] = GoldRow(...)`` silently last-
  wins. Overlapping shards corrupt the gold set with no signal.
- ``Scorer.score_batch``: NOW dedups on ``question_id`` (last-wins,
  mirroring ``load_gold_rows``) and warns. Previously it scored every
  row with no dedup, so partial re-runs concatenated onto an old
  predictions JSONL silently double-counted the duplicated qids.

``load_gold_rows`` warns and keeps last-wins; ``score_batch`` warns
and dedups last-wins. Both surface the duplicate loudly.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_eval.aliases import AliasStore
from benchmark_eval.loaders import load_gold_rows
from benchmark_eval.scorer import Scorer
from benchmark_models.base import Prediction, Usage


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _gold_row(qid: str, ground_truth_type: str = "boolean", gt=True) -> dict:
    """Minimal benchmark-row dict accepted by ``load_gold_rows``."""
    return {
        "question_id": qid,
        "tid": "1.1",
        "sfid": "A",
        "ground_truth_type": ground_truth_type,
        "ground_truth": gt,
        "language": "eng",
        "kg_source": "kwg",
        "is_true_premise": True,
    }


def _loguru_capture():
    """Context manager that swaps in a list-capturing sink on loguru
    and yields the messages list. ``caplog`` doesn't intercept
    loguru output, so eval-side warnings (which use loguru) need
    direct capture."""
    from contextlib import contextmanager

    from loguru import logger

    @contextmanager
    def _cm():
        captured: list[str] = []

        def _sink(message):
            captured.append(str(message))

        sink_id = logger.add(_sink, level="WARNING")
        try:
            yield captured
        finally:
            logger.remove(sink_id)

    return _cm()


class TestLoadGoldRowsDuplicateWarning:
    def test_duplicate_qids_emit_warning(self, tmp_path: Path) -> None:
        """Two gold rows with the same qid → one warning naming the
        path and the duplicate count."""
        p = tmp_path / "benchmark.jsonl"
        _write_jsonl(p, [
            _gold_row("q1", gt=True),
            _gold_row("q1", gt=False),  # duplicate — last-wins
            _gold_row("q2", gt=True),
        ])
        with _loguru_capture() as captured:
            golds = load_gold_rows(p)
        # Last-wins kept the False value.
        assert golds["q1"].ground_truth is False
        # Warning fired with the right count.
        assert any(
            "1 duplicate question_id rows" in m for m in captured
        ), f"expected duplicate-qid warning in {captured!r}"

    def test_no_duplicates_no_warning(self, tmp_path: Path) -> None:
        p = tmp_path / "benchmark.jsonl"
        _write_jsonl(p, [
            _gold_row("q1"), _gold_row("q2"), _gold_row("q3"),
        ])
        with _loguru_capture() as captured:
            load_gold_rows(p)
        assert not any(
            "duplicate question_id rows" in m for m in captured
        )


def _pred(qid: str, *, answer=True) -> Prediction:
    return Prediction(
        question_id=qid,
        tid="1.1",
        sfid="A",
        answer_type="Boolean (yes/no, true/false)",
        model_id="m",
        tier="tier1",
        language="eng",
        prompt_template_id="t",
        seed=42,
        snapshot_id="x",
        answer=answer,
        unit=None,
        refusal=False,
        usage=Usage(),
    )


class TestScoreBatchDuplicateWarning:
    def test_duplicate_predictions_deduped_last_wins(self) -> None:
        """Two prediction rows with the same qid → one warning, and only
        the LAST occurrence is scored (last-wins dedup, mirroring
        load_gold_rows), so the aggregate never double-counts."""
        from benchmark_eval.base import GoldRow

        scorer = Scorer(alias_store=AliasStore.from_entries([]))
        golds = {
            "q1": GoldRow(
                question_id="q1", tid="1.1", sfid="A",
                answer_type="Boolean (yes/no, true/false)",
                answer_unit=None,
                ground_truth=True, language="eng",
                kg_source="kwg", is_true_premise=True,
            ),
        }
        preds = [
            _pred("q1", answer=True),   # earlier — dropped
            _pred("q1", answer=False),  # last — wins
        ]
        with _loguru_capture() as captured:
            records = scorer.score_batch(preds, golds)
        # Deduped to a single record (no double-count).
        assert len(records) == 1
        # Last-wins: the surviving record is the second pred
        # (answer=False vs gold True -> em=0).
        assert records[0].em == 0
        # Warning fired.
        assert any(
            "dropped 1 duplicate question_id" in m
            for m in captured
        ), f"expected duplicate-prediction dedup warning in {captured!r}"

    def test_no_duplicates_no_warning(self) -> None:
        from benchmark_eval.base import GoldRow

        scorer = Scorer(alias_store=AliasStore.from_entries([]))
        golds = {
            qid: GoldRow(
                question_id=qid, tid="1.1", sfid="A",
                answer_type="Boolean (yes/no, true/false)",
                answer_unit=None,
                ground_truth=True, language="eng",
                kg_source="kwg", is_true_premise=True,
            )
            for qid in ("q1", "q2", "q3")
        }
        preds = [_pred("q1"), _pred("q2"), _pred("q3")]
        with _loguru_capture() as captured:
            records = scorer.score_batch(preds, golds)
        assert len(records) == 3
        assert not any(
            "duplicate question_id prediction rows" in m
            for m in captured
        )
