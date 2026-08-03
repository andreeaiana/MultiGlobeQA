"""MLflow integration for benchmark_models.

Provides a thin wrapper around the ``mlflow`` SDK to log experiment
runs. Each (variant × seed) becomes a separate MLflow child run under
a parent run for the experiment. The runner calls :func:`log_run`
after a batch of predictions is complete.

Tracking configuration comes from the ``MLFLOW_TRACKING_URI`` and
``MLFLOW_EXPERIMENT_NAME`` environment variables. The model section
uses a separate experiment name (``Benchmarked_Models``) by default.

Usage::

    from benchmark_models.mlflow_logging import init_experiment, log_run

    init_experiment()  # creates the MLflow experiment if needed
    log_run(
        variant_name="gemini-3-flash-plain",
        config=config,
        predictions=predictions,
        output_path=Path("results/smoke/..."),
        snapshot_id="abc123",
    )
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from statistics import mean, quantiles
from typing import Any

from loguru import logger

from benchmark_models.base import Prediction

# Default experiment name; override via MLFLOW_EXPERIMENT_NAME env var.
DEFAULT_EXPERIMENT_NAME = "Benchmarked_Models"

# Default tracking URI; override via MLFLOW_TRACKING_URI env var.
# Uses a SQLite backend (recommended by MLflow since Feb 2026) stored
# in the project's logs/ directory. The filesystem backend
# (file://./mlruns) is deprecated.
DEFAULT_TRACKING_URI = "sqlite:///logs/mlflow.db"


def init_experiment(
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Initialize the MLflow experiment (create if it doesn't exist).

    Call once at runner startup before any :func:`log_run` calls.
    Safe to call multiple times — MLflow is idempotent.

    Args:
        experiment_name: Override the experiment name. Defaults to
            ``MLFLOW_EXPERIMENT_NAME`` env var or ``"Benchmarked_Models"``.
        tracking_uri: Override the tracking URI. Defaults to
            ``MLFLOW_TRACKING_URI`` env var or ``"sqlite:///logs/mlflow.db"``.
    """
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    logger.info("MLflow experiment '{}' at {}", name, uri)


def _get_git_commit() -> str | None:
    """Best-effort git commit hash for the current repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


# Mapping from the dotted ``prompt_template_id`` (as stored on every
# :class:`Prediction`) to the underscore-keyed tier file key that
# :data:`benchmark_models.prompts._TIER_TO_TEMPLATE_FILE` uses. Anything
# outside this map (e.g. ``"error"``) has no associated template file
# and is silently skipped when logging template artifacts.
_PROMPT_TEMPLATE_ID_TO_TIER_KEY: dict[str, str] = {
    "tier1.canonical": "tier1",
    "tier2.canonical": "tier2",
    "tier2.sc": "tier2",
    "tier1.oracle.structured": "tier1_oracle_structured",
    "tier1.oracle.raw": "tier1_oracle_raw",
    "tier1.oracle.verbalized": "tier1_oracle_verbalized",
    "tier3.3a": "tier3a",
    "tier3.3b": "tier3b",
    "tier3.3c": "tier3c",
    "tier3.oracle": "tier3_oracle",
}


def _resolve_template_paths(prompt_template_ids: set[str]) -> list[Path]:
    """Return absolute paths to the template files actually used in a run.

    Takes the set of unique ``prompt_template_id`` values observed on
    the run's predictions, maps each to its source ``.txt`` file, and
    returns the deduplicated list of files that exist. Skips
    ``"error"`` and any other ids that do not correspond to a real
    template (so the caller gets back only loggable artifacts).

    Resolution uses :func:`importlib.resources.as_file` so the right
    file is materialised on disk regardless of how the package is
    installed (editable, wheel, zipped). For zipped wheels ``as_file``
    extracts to a temp path; for editable installs it returns the
    source file directly. Caller is responsible for using the path
    *before* the underlying ``as_file`` context exits — we materialise
    + copy to a long-lived temp dir so MLflow logging can read the
    file even after this function returns.
    """
    import shutil
    import tempfile
    from importlib.resources import as_file, files

    prompts_pkg = files("benchmark_models.prompts")
    seen: set[str] = set()
    out: list[Path] = []
    for ptid in prompt_template_ids:
        tier_key = _PROMPT_TEMPLATE_ID_TO_TIER_KEY.get(ptid)
        if tier_key is None:
            continue
        filename = f"{tier_key}.txt"
        if filename in seen:
            continue
        seen.add(filename)
        resource = prompts_pkg.joinpath(filename)
        try:
            with as_file(resource) as src_path:
                if not src_path.is_file():
                    continue
                # Copy to a temp dir so the path remains valid after
                # the as_file context exits (zipped-wheel installs
                # otherwise lose the extracted file).
                tmp_dir = Path(tempfile.mkdtemp(prefix="bm_prompts_"))
                dst = tmp_dir / filename
                shutil.copy2(src_path, dst)
                out.append(dst)
        except Exception:
            continue
    return out


def log_run(
    variant_name: str,
    config: Any,
    predictions: list[Prediction],
    output_path: Path,
    *,
    snapshot_id: str = "unset",
    parent_run_id: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str | None:
    """Log a completed run to MLflow.

    Creates an MLflow run, logs params/metrics/artifacts, and returns
    the run ID. If ``parent_run_id`` is given, the run is nested under
    the parent (used for multi-variant experiment YAML invocations).

    Args:
        variant_name: The variant id from models.yaml.
        config: The validated :data:`VariantConfig` for this variant.
        predictions: The list of Prediction records produced.
        output_path: Path to the predictions JSONL file (logged as artifact).
        snapshot_id: KG snapshot identifier.
        parent_run_id: Optional parent MLflow run ID for nesting.
        extra_params: Additional params to log (e.g., experiment-level
            metadata).

    Returns:
        The MLflow run ID, or ``None`` if logging failed.
    """
    if not predictions:
        logger.warning("No predictions to log for variant '{}'", variant_name)
        return None

    # When ``parent_run_id`` is given, reopen the parent run so the
    # subsequent ``nested=True`` start_run actually attaches to it.
    # Without reopening, calling this function with ``parent_run_id``
    # outside an active run wouldn't nest at all.
    from contextlib import ExitStack

    import mlflow

    try:
        with ExitStack() as stack:
            if parent_run_id is not None:
                stack.enter_context(mlflow.start_run(run_id=parent_run_id))
            run = stack.enter_context(
                mlflow.start_run(
                    run_name=variant_name,
                    nested=parent_run_id is not None,
                )
            )
            _ = run  # keep reference for the body below
            # --- Params ---
            mlflow.log_param("variant", variant_name)
            mlflow.log_param("backend", config.backend)
            mlflow.log_param("tier", config.tier)
            mlflow.log_param("temperature", config.temperature)
            mlflow.log_param("max_tokens", config.max_tokens)
            mlflow.log_param("seed", config.seed)
            mlflow.log_param("snapshot_id", snapshot_id)

            # Backend-specific params.
            if hasattr(config, "model_id"):
                mlflow.log_param("model_id", config.model_id)
            if hasattr(config, "model_path"):
                mlflow.log_param("model_path", config.model_path)
            if hasattr(config, "reasoning_effort"):
                mlflow.log_param("reasoning_effort", config.reasoning_effort)
            if hasattr(config, "thinking_level"):
                mlflow.log_param("thinking_level", config.thinking_level)
            if hasattr(config, "extended_thinking"):
                mlflow.log_param("extended_thinking", config.extended_thinking)
            if hasattr(config, "self_consistency_n"):
                mlflow.log_param("self_consistency_n", config.self_consistency_n)
            if hasattr(config, "dtype"):
                mlflow.log_param("dtype", config.dtype)
            if hasattr(config, "quantization") and config.quantization:
                mlflow.log_param("quantization", config.quantization)

            # Git commit.
            commit = _get_git_commit()
            if commit:
                mlflow.log_param("git_commit", commit)

            # Extra params.
            if extra_params:
                for k, v in extra_params.items():
                    mlflow.log_param(k, v)

            # --- Metrics ---
            total = len(predictions)
            answered = sum(1 for p in predictions if not p.refusal)
            refused = sum(1 for p in predictions if p.refusal)
            errors = sum(1 for p in predictions if p.final_state == "error")

            mlflow.log_metric("total_questions", total)
            mlflow.log_metric("answered", answered)
            mlflow.log_metric("refused", refused)
            mlflow.log_metric("errors", errors)
            mlflow.log_metric("refusal_rate", refused / total if total else 0)

            # Per-class error breakdown. Logged with ``errors_`` prefix so
            # the MLflow UI groups them next to the flat ``errors`` metric.
            # Only emitted for error rows (error_class is None elsewhere).
            error_classes: dict[str, int] = {}
            for p in predictions:
                if p.final_state == "error":
                    key = p.error_class or "unknown"
                    error_classes[key] = error_classes.get(key, 0) + 1
            for ec, count in error_classes.items():
                mlflow.log_metric(f"errors_{ec}", count)

            # Token usage.
            input_tokens = [p.usage.input_tokens for p in predictions]
            output_tokens = [p.usage.output_tokens for p in predictions]
            reasoning_tokens = [p.usage.reasoning_tokens for p in predictions]
            mlflow.log_metric("total_input_tokens", sum(input_tokens))
            mlflow.log_metric("total_output_tokens", sum(output_tokens))
            mlflow.log_metric("total_reasoning_tokens", sum(reasoning_tokens))
            if input_tokens:
                mlflow.log_metric("mean_input_tokens", mean(input_tokens))
                mlflow.log_metric("mean_output_tokens", mean(output_tokens))

            # Latency.
            latencies = [p.latency_ms for p in predictions]
            total_latency = sum(latencies)
            mlflow.log_metric("total_latency_ms", total_latency)
            if latencies:
                mlflow.log_metric("mean_latency_ms", mean(latencies))
                if len(latencies) >= 4:
                    q = quantiles(latencies, n=100)
                    mlflow.log_metric("p50_latency_ms", q[49])
                    mlflow.log_metric("p95_latency_ms", q[94])
                    mlflow.log_metric("p99_latency_ms", q[98])

            # Tier 3 tool metrics. Filter on ``tool_calls is not None``
            # for both the numerator and the denominator so the rate
            # is over the same population. Counting ``cap_hit`` over
            # all predictions against the tool-call-only denominator
            # would produce a rate inconsistent with the eval-side
            # ``cap_hit_rate`` over the same data.
            tier3_predictions = [
                p for p in predictions if p.tool_calls is not None
            ]
            tool_call_counts = [len(p.tool_calls or []) for p in tier3_predictions]
            cap_hits = sum(1 for p in tier3_predictions if p.cap_hit)
            if tier3_predictions:
                mlflow.log_metric("mean_tool_calls", mean(tool_call_counts))
                mlflow.log_metric("cap_hit_count", cap_hits)
                mlflow.log_metric(
                    "cap_hit_rate",
                    cap_hits / len(tier3_predictions),
                )

            # --- Tables ---
            # Import pandas once so both tables can use it; importing
            # inside the predictions_table branch would leave ``pd``
            # unbound when only the traces_table is populated.
            import pandas as pd

            # Predictions summary table (viewable in MLflow UI).
            # The full ``answer`` value is logged, not a truncation:
            # notebooks that pull the table can otherwise not recover
            # the full answer for long sets / geometry without
            # separately reading predictions.jsonl. The full JSONL is
            # also logged as an artefact below; the table stays
            # self-contained.
            predictions_table = []
            for p in predictions:
                row = {
                    "question_id": p.question_id,
                    "tid": p.tid,
                    "sfid": p.sfid,
                    "answer_type": p.answer_type,
                    "answer": (
                        json.dumps(p.answer, ensure_ascii=False)
                        if p.answer is not None
                        else None
                    ),
                    "unit": p.unit,
                    "refusal": p.refusal,
                    "latency_ms": round(p.latency_ms, 1),
                    "input_tokens": p.usage.input_tokens,
                    "output_tokens": p.usage.output_tokens,
                    "num_steps": len(p.tool_calls) if p.tool_calls else 0,
                    "final_state": p.final_state,
                }
                predictions_table.append(row)
            if predictions_table:
                mlflow.log_table(
                    data=pd.DataFrame(predictions_table),
                    artifact_file="predictions_summary.json",
                )

            # Tier-3 execution traces table (viewable in MLflow UI).
            # One row per agent step per question — lets reviewers
            # browse code, output, and errors directly in the UI.
            traces_table = []
            for p in predictions:
                if not p.tool_calls:
                    continue
                for step_idx, tc in enumerate(p.tool_calls):
                    traces_table.append({
                        "question_id": p.question_id,
                        "step": step_idx,
                        "step_type": tc.name,
                        "code": tc.args.get("code", "")[:1000],
                        "output": (tc.result or "")[:1000],
                        "error": tc.error,
                    })
            if traces_table:
                mlflow.log_table(
                    data=pd.DataFrame(traces_table),
                    artifact_file="execution_traces.json",
                )

            # --- Artifacts ---
            output_path = Path(output_path)
            if output_path.exists():
                mlflow.log_artifact(str(output_path))

            # Also log the run log if it exists alongside.
            log_path = output_path.parent / "run.log"
            if log_path.exists():
                mlflow.log_artifact(str(log_path))

            # Log the prompt template files actually used in this run
            # under ``prompts/``. A tier-2 SC run with an oracle-tier-1
            # sibling would log both templates; a vanilla tier-1 run
            # logs just ``tier1.txt``. The artifact makes the exact
            # prompt scaffolding auditable from the MLflow UI without
            # cross-referencing against the git commit hash.
            unique_ptids = {
                p.prompt_template_id
                for p in predictions
                if p.prompt_template_id
            }
            for template_path in _resolve_template_paths(unique_ptids):
                mlflow.log_artifact(str(template_path), artifact_path="prompts")

            logger.info(
                "Logged MLflow run '{}' (id={}): {} predictions, "
                "{} answered, {} refused, {} errors",
                variant_name,
                run.info.run_id,
                total,
                answered,
                refused,
                errors,
            )
            return run.info.run_id

    except Exception as exc:
        logger.warning(
            "MLflow logging failed for variant '{}': {}",
            variant_name,
            exc,
        )
        return None
