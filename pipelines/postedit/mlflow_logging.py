"""MLflow integration for the post-edit pipeline.

Mirrors :mod:`benchmark_models.mlflow_logging`: per-(language, kg)
SLURM job is one MLflow parent run; each cell is a nested child run.
The pipeline calls :func:`init_experiment` once at startup and
:func:`log_cell_run` per cell after writing its output JSONL.

Tracking URI defaults to the same SQLite backend the rest of the
project uses (``sqlite:///logs/mlflow.db``); override with
``MLFLOW_TRACKING_URI``. Experiment defaults to ``"Postedit"``;
override with ``MLFLOW_EXPERIMENT_NAME``.

The dependency is optional — if ``mlflow`` isn't importable or the
user passed ``--no-mlflow``, every function in this module is a
no-op so the pipeline keeps running.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_EXPERIMENT_NAME = "Postedit"
DEFAULT_TRACKING_URI = "sqlite:///logs/mlflow.db"


_state: dict[str, Any] = {
    "enabled": False,
    "parent_run_id": None,
}


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        # OSError covers FileNotFoundError (git not on PATH) and
        # PermissionError (git present but unexecutable from this node —
        # seen on some HPC compute nodes that restrict /usr/bin).
        return None


def init_experiment(
    *,
    enabled: bool = True,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Initialise MLflow tracking. Idempotent.

    If ``enabled=False`` (e.g. via ``--no-mlflow`` or import failure),
    every subsequent call in this module is a silent no-op.
    """
    if not enabled:
        _state["enabled"] = False
        return
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("mlflow not installed — postedit MLflow logging disabled")
        _state["enabled"] = False
        return

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    name = experiment_name or os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME,
    )

    # Ensure logs/ exists for the SQLite backend.
    if uri.startswith("sqlite:///"):
        db_path = Path(uri.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    _state["enabled"] = True
    logger.info("MLflow experiment '{}' at {}", name, uri)


def start_parent_run(
    *,
    language: str,
    kg: str | None,
    sfid: str | None,
    models: list[str],
    question_concurrency: int,
    per_question_timeout_s: float,
    resume_mode: str,
    benchmark_root: Path,
) -> None:
    """Open the SLURM-job-level parent run. Stores the run id in module
    state so each per-cell call can nest under it. No-op if disabled.
    """
    if not _state["enabled"]:
        return
    import mlflow  # type: ignore[import-not-found]

    run_name = f"postedit_{language}_{kg or 'allkg'}"
    if sfid:
        run_name += f"_{sfid}"
    run = mlflow.start_run(run_name=run_name)
    _state["parent_run_id"] = run.info.run_id

    mlflow.log_param("language", language)
    mlflow.log_param("kg_filter", kg or "all")
    mlflow.log_param("sfid_filter", sfid or "all")
    mlflow.log_param("models", ",".join(models))
    mlflow.log_param("question_concurrency", question_concurrency)
    mlflow.log_param("per_question_timeout_s", per_question_timeout_s)
    mlflow.log_param("resume_mode", resume_mode)
    mlflow.log_param("benchmark_root", str(benchmark_root))
    mlflow.log_param("slurm_job_id", os.environ.get("SLURM_JOB_ID", ""))
    commit = _get_git_commit()
    if commit:
        mlflow.log_param("git_commit", commit)


def log_cell_run(
    *,
    language: str,
    kg: str,
    sfid_dir: str,
    n_done: int,
    n_errors: int,
    decisions: dict[str, int],
    out_path: Path,
) -> None:
    """Log a per-cell child run nested under the parent. No-op if
    MLflow is disabled.

    Metrics include ensemble-decision distribution + fallback_gemini
    rate, computed by re-reading the output JSONL — cheap (≤15k rows
    per cell, ~50 ms).
    """
    if not _state["enabled"]:
        return
    import mlflow  # type: ignore[import-not-found]

    parent_id = _state["parent_run_id"]
    with mlflow.start_run(
        run_name=f"{kg}_{sfid_dir}",
        nested=parent_id is not None,
    ):
        mlflow.log_param("language", language)
        mlflow.log_param("kg", kg)
        mlflow.log_param("sfid_dir", sfid_dir)
        mlflow.log_param("output_path", str(out_path))

        mlflow.log_metric("rows_written", n_done)
        mlflow.log_metric("rows_errored", n_errors)

        # Per-decision counts.
        for decision, count in decisions.items():
            mlflow.log_metric(f"decision_{decision}", count)

        # Re-read the output to compute downstream-relevant rates.
        if out_path.is_file():
            stats = _summarise_output(out_path)
            for k, v in stats.items():
                mlflow.log_metric(k, v)

            # Log the output JSONL as an artifact (skip very large files
            # to avoid bloating the MLflow store; reference path is
            # always logged via the param above).
            try:
                if out_path.stat().st_size <= 100 * 1024 * 1024:  # 100 MB
                    mlflow.log_artifact(str(out_path), artifact_path="output")
            except OSError:
                pass


def end_parent_run(*, status: str = "FINISHED") -> None:
    """Close the parent run. ``status`` should be ``"FINISHED"`` on
    clean exit or ``"FAILED"`` on uncaught error."""
    if not _state["enabled"] or _state["parent_run_id"] is None:
        return
    import mlflow  # type: ignore[import-not-found]
    mlflow.end_run(status=status)
    _state["parent_run_id"] = None


def _summarise_output(out_path: Path) -> dict[str, float]:
    """Compute decision-distribution + token-usage metrics from the
    cell's output JSONL.
    """
    n = 0
    n_fallback = 0
    n_edit_made = 0
    decision_counts: dict[str, int] = {}
    op_counts: dict[str, int] = {}
    total_in = 0
    total_out = 0
    entity_polish_total = 0
    entity_polish_by_tier: dict[str, int] = {}
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            d = row.get("decision", "unknown")
            decision_counts[d] = decision_counts.get(d, 0) + 1
            if d == "fallback_gemini":
                n_fallback += 1
            if row.get("final_question") and row.get("original_question") and \
                    row["final_question"] != row["original_question"]:
                n_edit_made += 1
            for mr in (row.get("model_outputs") or {}).values():
                total_in += int(mr.get("prompt_tokens", 0) or 0)
                total_out += int(mr.get("completion_tokens", 0) or 0)
                for op in mr.get("ops") or []:
                    op_counts[op] = op_counts.get(op, 0) + 1
                # Entity-polish audit: per-tier counts across all voters.
                for edit in (mr.get("entity_edits") or []):
                    entity_polish_total += 1
                    tier = edit.get("tier", "unknown")
                    entity_polish_by_tier[tier] = (
                        entity_polish_by_tier.get(tier, 0) + 1
                    )

    metrics: dict[str, float] = {
        "rows_total": float(n),
        "rows_fallback_gemini": float(n_fallback),
        "rows_edit_made": float(n_edit_made),
        "fallback_rate": (n_fallback / n) if n else 0.0,
        "edit_rate": (n_edit_made / n) if n else 0.0,
        "total_prompt_tokens": float(total_in),
        "total_completion_tokens": float(total_out),
        "entity_polish_total": float(entity_polish_total),
    }
    for op, count in op_counts.items():
        metrics[f"op_{op}"] = float(count)
    for tier, count in entity_polish_by_tier.items():
        metrics[f"entity_polish_{tier}"] = float(count)
    return metrics
