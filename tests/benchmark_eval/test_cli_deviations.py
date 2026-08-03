"""P1-5: ``--include-invalid`` recorded as a deviation.

Pre-fix: ``scoring_spec_deviations`` only carried fields from
``ScorerConfig.deviations_from_spec()``. ``--include-invalid``
materially changes the denominator (admits ``is_valid=False`` rows
into the score) but went untracked. A re-score with the flag set
produced numbers untraceable to a single canonical spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from click.testing import CliRunner

from benchmark_eval.cli import score


@pytest.fixture
def make_minimal_inputs(tmp_path: Path):
    """Build a minimal valid input set: one predictions row + one
    gold row sharing the same question_id; an alias store; an
    output directory."""

    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({}))

    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({
        "question_id": "q1",
        "tid": "1.1",
        "sfid": "A",
        "answer_type": "Boolean (yes/no, true/false)",
        "model_id": "m",
        "tier": "tier1",
        "language": "eng",
        "prompt_template_id": "tier1",
        "seed": 42,
        "snapshot_id": "x",
        "answer": True,
        "usage": {},
    }) + "\n")

    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps({
        "question_id": "q1",
        "tid": "1.1",
        "sfid": "A",
        "ground_truth_type": "boolean",
        "ground_truth": True,
        "language": "eng",
        "kg_source": "kwg",
        "is_true_premise": True,
    }) + "\n")

    output_dir = tmp_path / "out"
    return predictions, benchmark, aliases, output_dir


def test_include_invalid_appears_in_deviations(make_minimal_inputs) -> None:
    """End-to-end: passing ``--include-invalid`` produces an
    AggregateReport whose ``scoring_spec_deviations`` carries
    ``include_invalid: True``."""
    predictions, benchmark, aliases, output_dir = make_minimal_inputs
    runner = CliRunner()
    result = runner.invoke(score, [
        "--predictions", str(predictions),
        "--benchmark", str(benchmark),
        "--aliases", str(aliases),
        "--output", str(output_dir),
        "--include-invalid",
        "--no-mlflow",
    ])
    assert result.exit_code == 0, result.output

    report_path = output_dir / "aggregate_report.json"
    report = json.loads(report_path.read_text())
    assert report["scoring_spec_deviations"].get("include_invalid") is True


def test_include_invalid_absent_when_flag_not_set(make_minimal_inputs) -> None:
    """Without the flag, ``scoring_spec_deviations`` does not carry
    ``include_invalid`` (default policy)."""
    predictions, benchmark, aliases, output_dir = make_minimal_inputs
    runner = CliRunner()
    result = runner.invoke(score, [
        "--predictions", str(predictions),
        "--benchmark", str(benchmark),
        "--aliases", str(aliases),
        "--output", str(output_dir),
        "--no-mlflow",
    ])
    assert result.exit_code == 0, result.output

    report_path = output_dir / "aggregate_report.json"
    report = json.loads(report_path.read_text())
    assert "include_invalid" not in report["scoring_spec_deviations"]
