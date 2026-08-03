"""End-to-end tests for the CLI and loaders.

Scores the committed sample oracle subset (39 distance questions)
against handcrafted predictions and asserts the report shape is
sensible. Uses the real benchmark + real alias store on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from benchmark_eval.cli import cli
from benchmark_eval.loaders import load_gold_rows
from benchmark_models.base import Prediction, Usage

FIXTURES = Path(__file__).parents[1] / "fixtures"
SAMPLE_BENCHMARK = FIXTURES / "sample_benchmark.jsonl"
ALIASES = FIXTURES / "entity_aliases.json"


@pytest.fixture
def predictions_jsonl(tmp_path: Path) -> Path:
    """Write a predictions.jsonl with 3 handcrafted predictions against
    real question_ids from the sample benchmark.

    The sample is heterogeneous (17 answer types across 28 rows), so
    the fixture keeps the assertions type-agnostic: it picks any three
    ``boolean`` rows (the sample reliably holds 6) and submits two
    correct answers plus one inverted answer, yielding EM=2/3
    independent of tolerance hyperparameters.
    """
    assert SAMPLE_BENCHMARK.exists(), (
        f"{SAMPLE_BENCHMARK} is expected to be committed to the repo; "
        "run scripts/build_oracle_subset.py to regenerate it."
    )

    # Select by the loaded gold's ``answer_type`` (which drives scorer
    # dispatch), NOT the raw ``ground_truth_type`` field. Some rows carry
    # ``ground_truth_type == "boolean"`` yet a non-boolean ``answer_type``
    # (e.g. tid 24.1 is a "Set enumeration" whose ground_truth happens to
    # be a bool) — those dispatch to a different scorer, so a boolean
    # answer against them is a guaranteed miss. Filtering on answer_type
    # keeps this test measuring pure boolean equality.
    gold_by_qid: list[tuple[str, bool]] = []
    for qid, gold in load_gold_rows(SAMPLE_BENCHMARK).items():
        if not str(getattr(gold, "answer_type", "") or "").startswith("Boolean"):
            continue
        gt = gold.ground_truth
        # Restrict to unambiguous "true"/"false" string golds so the
        # fixture's bool parse agrees with the scorer's. Some Boolean-typed
        # sample rows encode the truth value as "0"/"1"/"0.0" or a list; a
        # naive ``== "true"`` parse would read "1" as False, and the
        # "inverted" prediction (not False = True) could then accidentally
        # match the scorer's parse of "1" as True — silently turning the
        # 2/3 invariant into 3/3. Which rows land first here depends on
        # answer-type resolution, so skipping the ambiguous ones keeps this
        # a pure boolean-equality test regardless of TID renumbering.
        if not isinstance(gt, str) or gt.strip().lower() not in ("true", "false"):
            continue
        gt_bool = gt.strip().lower() == "true"
        gold_by_qid.append((qid, gt_bool))
        if len(gold_by_qid) == 3:
            break
    assert len(gold_by_qid) == 3, (
        "sample benchmark must contain at least 3 Boolean answer_type rows"
    )

    # Two correct predictions + one inverted → EM = 2/3 regardless of
    # tolerance flags (boolean scoring is pure equality).
    invert = [False, False, True]
    preds = []
    for (qid, gold), flip in zip(gold_by_qid, invert, strict=True):
        answer = (not gold) if flip else gold
        preds.append(
            Prediction(
                question_id=qid,
                tid="24.1",
                sfid="C",
                answer_type="Boolean",
                model_id="test-model",
                tier="tier1",
                language="en",
                prompt_template_id="tier1.canonical",
                seed=42,
                snapshot_id="x",
                answer=answer,
                usage=Usage(),
            )
        )

    path = tmp_path / "predictions.jsonl"
    with path.open("w") as f:
        for p in preds:
            f.write(p.model_dump_json() + "\n")
    return path


class TestLoaders:
    def test_load_gold_rows_from_sample_benchmark(self) -> None:
        assert SAMPLE_BENCHMARK.exists(), (
            f"{SAMPLE_BENCHMARK} is expected to be committed to the repo."
        )
        golds = load_gold_rows(SAMPLE_BENCHMARK)
        assert len(golds) > 0
        # Every loaded row should have a canonicalised answer_type
        # (the loader skips rows with unknown ground_truth_type).
        for sample in golds.values():
            assert sample.answer_type is not None
            assert sample.tid


class TestCliEndToEnd:
    def test_score_runs_and_writes_outputs(
        self, tmp_path: Path, predictions_jsonl: Path
    ) -> None:
        if not ALIASES.exists():
            pytest.skip(f"{ALIASES} not present")
        out_dir = tmp_path / "scores"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "score",
                "--predictions", str(predictions_jsonl),
                "--benchmark", str(SAMPLE_BENCHMARK),
                "--aliases", str(ALIASES),
                "--output", str(out_dir),
                "--no-mlflow",
            ],
        )
        assert result.exit_code == 0, result.output
        records_path = out_dir / "score_records.jsonl"
        report_path = out_dir / "aggregate_report.json"
        assert records_path.exists()
        assert report_path.exists()

        # Record count matches prediction count (3).
        with records_path.open() as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 3

        report = json.loads(report_path.read_text())
        for key in (
            "em_micro",
            "em_macro",
            "em_conditional",
            "ne_micro",
            "ne_macro",
            "coverage_rate",
        ):
            assert key in report

        # Two correct boolean predictions + one inverted → EM = 2/3.
        assert report["em_micro"] == pytest.approx(2 / 3)

    def test_score_alpha_override_runs(
        self, tmp_path: Path, predictions_jsonl: Path
    ) -> None:
        """Smoke test that ``--alpha`` is accepted and the run completes.

        Boolean scoring is equality-only, so alpha does not change the
        outcome; the test just exercises the CLI override plumbing.
        """
        if not ALIASES.exists():
            pytest.skip(f"{ALIASES} not present")
        out_dir = tmp_path / "scores_loose"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "score",
                "--predictions", str(predictions_jsonl),
                "--benchmark", str(SAMPLE_BENCHMARK),
                "--aliases", str(ALIASES),
                "--output", str(out_dir),
                "--alpha", "50.0",
                "--no-mlflow",
            ],
        )
        assert result.exit_code == 0, result.output
        report = json.loads((out_dir / "aggregate_report.json").read_text())
        assert report["em_micro"] == pytest.approx(2 / 3)
