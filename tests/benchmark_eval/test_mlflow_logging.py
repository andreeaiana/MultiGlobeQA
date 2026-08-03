"""Tests for :mod:`benchmark_eval.mlflow_logging`.

Covers:

* ``_metric_safe`` scrubs answer-type names into MLflow-safe keys.
* ``_build_run_name`` includes alpha when present.
* ``log_eval_run`` makes the expected ``log_param`` / ``log_metric``
  / ``log_artifact`` calls for both standalone and nested runs.
* The CLI wiring respects ``--no-mlflow``.

The MLflow SDK is mocked via :func:`unittest.mock.patch` — no real
tracking store is touched. The tests only verify the calls we
make, not MLflow's internals.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark_eval.base import AggregateReport, GroupStats
from benchmark_eval.mlflow_logging import (
    _build_run_name,
    _metric_safe,
    log_eval_run,
)

# ---------------------------------------------------------------------------
# _metric_safe
# ---------------------------------------------------------------------------


class TestMetricSafe:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Numeric", "Numeric"),
            ("Numeric (float)", "Numeric_float"),
            ("Boolean (yes/no, true/false)", "Boolean_yes_no_true_false"),
            ("Cardinal direction (predefined)", "Cardinal_direction_predefined"),
            ("Set enumeration (list of entity names)", "Set_enumeration_list_of_entity_names"),
            ("  leading/trailing  ", "leading_trailing"),
            ("a---b", "a---b"),
            ("a__b__c", "a_b_c"),  # collapsed
        ],
    )
    def test_transforms(self, raw: str, expected: str) -> None:
        assert _metric_safe(raw) == expected


# ---------------------------------------------------------------------------
# _build_run_name
# ---------------------------------------------------------------------------


def _report(**kwargs) -> AggregateReport:
    """Build a minimal AggregateReport with sensible defaults."""
    defaults = {
        "model_id": "gemini-3-flash-plain",
        "tier": "tier1",
        "n_total": 10,
        "n_covered": 10,
        "coverage_rate": 1.0,
        "n_format_errors": 0,
        "format_error_rate": 0.0,
        "em_micro": 0.9,
        "em_macro": 0.85,
        "em_conditional": 0.9,
        "ne_micro": 0.1,
        "ne_macro": 0.15,
    }
    defaults.update(kwargs)
    return AggregateReport(**defaults)


class TestBuildRunName:
    def test_with_alpha(self) -> None:
        r = _report()
        name = _build_run_name(r, {"alpha": 0.05})
        assert name == "eval:gemini-3-flash-plain:alpha=0.05"

    def test_without_alpha(self) -> None:
        r = _report()
        name = _build_run_name(r, {})
        assert name == "eval:gemini-3-flash-plain"


# ---------------------------------------------------------------------------
# log_eval_run
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """An output dir with stub artifact files — used to verify that
    log_eval_run invokes log_artifact on them."""
    (tmp_path / "score_records.jsonl").write_text("{}\n")
    (tmp_path / "aggregate_report.json").write_text("{}")
    return tmp_path


class TestLogEvalRun:
    def test_standalone_run(self, tmp_output_dir: Path) -> None:
        """Without parent_run_id: single (non-nested) start_run call."""
        fake_mlflow = MagicMock()
        fake_run = MagicMock()
        fake_run.info.run_id = "fake-eval-run-id"
        fake_mlflow.start_run.return_value.__enter__.return_value = fake_run

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            run_id = log_eval_run(
                _report(),
                output_dir=tmp_output_dir,
                config_params={"alpha": 0.05},
            )
        assert run_id == "fake-eval-run-id"
        # One start_run — no nesting.
        assert fake_mlflow.start_run.call_count == 1
        # Called with run_name containing the alpha marker.
        call_kwargs = fake_mlflow.start_run.call_args.kwargs
        assert "alpha=0.05" in call_kwargs["run_name"]

    def test_nested_under_parent(self, tmp_output_dir: Path) -> None:
        """With parent_run_id: outer start_run(run_id=parent) opens
        the parent, inner start_run(nested=True) creates the child."""
        fake_mlflow = MagicMock()
        fake_child = MagicMock()
        fake_child.info.run_id = "fake-eval-child-id"
        # Two separate start_run calls — outer and inner — both used as
        # context managers.
        fake_mlflow.start_run.return_value.__enter__.side_effect = [
            MagicMock(),  # outer (parent) context
            fake_child,   # inner (child) context
        ]

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            run_id = log_eval_run(
                _report(),
                output_dir=tmp_output_dir,
                config_params={"alpha": 0.05},
                parent_run_id="parent-123",
            )
        assert run_id == "fake-eval-child-id"
        assert fake_mlflow.start_run.call_count == 2
        outer_call = fake_mlflow.start_run.call_args_list[0]
        inner_call = fake_mlflow.start_run.call_args_list[1]
        assert outer_call.kwargs == {"run_id": "parent-123"}
        assert inner_call.kwargs.get("nested") is True

    def test_logs_all_headline_metrics(self, tmp_output_dir: Path) -> None:
        """Every headline field on AggregateReport is logged at
        least once via ``log_metric``."""
        fake_mlflow = MagicMock()
        fake_run = MagicMock()
        fake_run.info.run_id = "r"
        fake_mlflow.start_run.return_value.__enter__.return_value = fake_run

        report = _report(
            per_answer_type={
                "Numeric": GroupStats(n=5, n_covered=5, em=1.0, ne=0.0, em_conditional=1.0),
                "Boolean (yes/no, true/false)": GroupStats(
                    n=5, n_covered=5, em=0.8, ne=0.2, em_conditional=0.8
                ),
            },
            per_sfid={"A": GroupStats(n=10, n_covered=10, em=0.9, ne=0.1, em_conditional=0.9)},
            per_language={"en": GroupStats(n=10, n_covered=10, em=0.9, ne=0.1, em_conditional=0.9)},
            resolve_failure_rate=0.1,
            unit_match_rate=0.95,
            numeric_failure_breakdown={"both_wrong": 2},
        )

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            log_eval_run(
                report,
                output_dir=tmp_output_dir,
                config_params={"alpha": 0.05},
            )

        logged_metrics = {
            call.args[0]
            for call in fake_mlflow.log_metric.call_args_list
        }

        # Headline metrics all present.
        for m in (
            "em_micro", "em_macro", "em_conditional",
            "ne_micro", "ne_macro",
            "coverage_rate", "format_error_rate",
            "n_total", "n_covered", "n_format_errors",
            "resolve_failure_rate", "unit_match_rate",
        ):
            assert m in logged_metrics, f"Missing metric: {m}"

        # Per-answer-type metrics (with transformed names).
        assert "em_type_Numeric" in logged_metrics
        assert "em_type_Boolean_yes_no_true_false" in logged_metrics
        assert "ne_type_Numeric" in logged_metrics

        # Per-SFID / per-language.
        assert "em_sfid_A" in logged_metrics
        assert "em_lang_en" in logged_metrics

        # Numeric failure breakdown.
        assert "numeric_failure_both_wrong" in logged_metrics

    def test_logs_params(self, tmp_output_dir: Path) -> None:
        fake_mlflow = MagicMock()
        fake_run = MagicMock()
        fake_run.info.run_id = "r"
        fake_mlflow.start_run.return_value.__enter__.return_value = fake_run

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            log_eval_run(
                _report(),
                output_dir=tmp_output_dir,
                config_params={"alpha": 0.05, "polygon_iou_threshold": 0.7},
                predictions_path=Path("/tmp/preds.jsonl"),
                benchmark_path=Path("/tmp/bench.jsonl"),
                alias_store_path=Path("/tmp/aliases.json"),
            )

        logged_params = {
            call.args[0]: call.args[1]
            for call in fake_mlflow.log_param.call_args_list
        }
        assert logged_params["alpha"] == 0.05
        assert logged_params["polygon_iou_threshold"] == 0.7
        assert logged_params["predictions_path"] == "/tmp/preds.jsonl"
        assert logged_params["benchmark_path"] == "/tmp/bench.jsonl"
        assert logged_params["alias_store_path"] == "/tmp/aliases.json"
        assert logged_params["eval_model_id"] == "gemini-3-flash-plain"
        assert logged_params["eval_tier"] == "tier1"

    def test_logs_artifacts(self, tmp_output_dir: Path) -> None:
        fake_mlflow = MagicMock()
        fake_run = MagicMock()
        fake_run.info.run_id = "r"
        fake_mlflow.start_run.return_value.__enter__.return_value = fake_run

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            log_eval_run(
                _report(),
                output_dir=tmp_output_dir,
                config_params={"alpha": 0.05},
            )

        logged_artifacts = {
            Path(call.args[0]).name
            for call in fake_mlflow.log_artifact.call_args_list
        }
        assert "score_records.jsonl" in logged_artifacts
        assert "aggregate_report.json" in logged_artifacts

    def test_failure_returns_none(self, tmp_output_dir: Path) -> None:
        """An MLflow exception bubbles up as None (logged, not raised)."""
        fake_mlflow = MagicMock()
        fake_mlflow.start_run.side_effect = RuntimeError("tracking dead")

        with patch.dict("sys.modules", {"mlflow": fake_mlflow}):
            run_id = log_eval_run(
                _report(),
                output_dir=tmp_output_dir,
                config_params={"alpha": 0.05},
            )
        assert run_id is None
