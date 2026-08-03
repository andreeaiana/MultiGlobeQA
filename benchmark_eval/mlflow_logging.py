"""MLflow integration for the eval pipeline.

Mirrors :mod:`benchmark_models.mlflow_logging`: every scored run
becomes an MLflow run (or a child of the model-side run when a
``parent_run_id`` is given). Reuses the same
``Benchmarked_Models`` experiment and ``sqlite:///logs/mlflow.db``
tracking URI by default, so eval runs show up next to — or nested
under — the model runs that produced the predictions.

One ``benchmark_eval score`` invocation → one MLflow run. Running
the same predictions under a different ``α`` creates a separate
child run; the two won't overwrite each other.

Logged surface:

* **Params**: the scorer config (``alpha``, the three ``delta``
  values, ``use_scale_from_magnitude``, ``polygon_iou_threshold``),
  plus the file paths used (predictions, benchmark, alias store).
* **Metrics**: headline EM / NE micro / macro / conditional,
  coverage / format-error / resolve-failure / unit-match rates,
  per-answer-type / per-SFID / per-language EM and NE, and sample
  counts per group.
* **Artifacts**: ``score_records.jsonl`` and ``aggregate_report.json``
  from the output directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from benchmark_eval.base import AggregateReport

DEFAULT_EXPERIMENT_NAME = "Benchmarked_Models"
"""Reuse the model section's experiment so eval runs group naturally
with the model runs they score."""

DEFAULT_TRACKING_URI = "sqlite:///logs/mlflow.db"


def init_experiment(
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> None:
    """Idempotently configure the MLflow experiment.

    Safe to call multiple times; respects ``MLFLOW_TRACKING_URI`` and
    ``MLFLOW_EXPERIMENT_NAME`` env vars for override.
    """
    import mlflow

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    name = experiment_name or os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)
    logger.info("MLflow eval experiment '{}' at {}", name, uri)


def _metric_safe(name: str) -> str:
    """Return an MLflow-safe metric-name suffix.

    MLflow metric names accept alphanumerics, underscores, dashes,
    periods, slashes, and spaces — but to keep names easy to grep
    we replace anything else with an underscore and collapse runs.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    out = "".join(c if c in allowed else "_" for c in name)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _build_run_name(report: AggregateReport, config_params: dict[str, Any]) -> str:
    """Construct a human-readable eval run name.

    Format: ``eval:<model_id>:alpha=<α>`` — visually distinguishable
    from the model-side run (named ``<model_id>``) and telegraphs
    the hyperparameter knob that matters most for sensitivity
    sweeps.
    """
    alpha = config_params.get("alpha")
    if alpha is not None:
        return f"eval:{report.model_id}:alpha={alpha}"
    return f"eval:{report.model_id}"


def log_eval_run(
    report: AggregateReport,
    *,
    output_dir: Path,
    config_params: dict[str, Any],
    parent_run_id: str | None = None,
    predictions_path: Path | None = None,
    benchmark_path: Path | None = None,
    alias_store_path: Path | None = None,
) -> str | None:
    """Log a completed eval run to MLflow.

    Args:
        report: The :class:`AggregateReport` returned by
            :func:`benchmark_eval.aggregate.build_report`.
        output_dir: Directory containing ``score_records.jsonl`` and
            ``aggregate_report.json`` to log as artifacts.
        config_params: The scorer config as a flat dict; every item
            is logged verbatim via ``mlflow.log_param``.
        parent_run_id: Optional MLflow run id to nest the eval run
            under (typically the id of the ``benchmark_models`` run
            that produced the predictions). When given, the eval run
            is created as a child of this run so the two are linked
            in the UI.
        predictions_path / benchmark_path / alias_store_path: Logged
            as params for provenance.

    Returns:
        The MLflow run id, or ``None`` if logging failed.
    """
    import mlflow

    run_name = _build_run_name(report, config_params)

    try:
        if parent_run_id is not None:
            # Reopen the parent run so the subsequent nested
            # ``start_run`` attaches to it.
            with (
                mlflow.start_run(run_id=parent_run_id),
                mlflow.start_run(run_name=run_name, nested=True) as run,
            ):
                _log_payload(
                    run,
                    report,
                    output_dir=output_dir,
                    config_params=config_params,
                    predictions_path=predictions_path,
                    benchmark_path=benchmark_path,
                    alias_store_path=alias_store_path,
                )
                run_id = run.info.run_id
            return run_id
        else:
            with mlflow.start_run(run_name=run_name) as run:
                _log_payload(
                    run,
                    report,
                    output_dir=output_dir,
                    config_params=config_params,
                    predictions_path=predictions_path,
                    benchmark_path=benchmark_path,
                    alias_store_path=alias_store_path,
                )
                return run.info.run_id
    except Exception as exc:
        logger.warning("MLflow eval logging failed: {}", exc)
        return None


def _log_payload(
    run: Any,
    report: AggregateReport,
    *,
    output_dir: Path,
    config_params: dict[str, Any],
    predictions_path: Path | None,
    benchmark_path: Path | None,
    alias_store_path: Path | None,
) -> None:
    """Write all params / metrics / artifacts for an open run.

    Extracted so the nested and non-nested paths in
    :func:`log_eval_run` share identical payload logic.
    """
    import mlflow

    # Params — scorer config + provenance.
    for k, v in config_params.items():
        mlflow.log_param(k, v)
    mlflow.log_param("eval_model_id", report.model_id)
    mlflow.log_param("eval_tier", report.tier)
    if report.scoring_spec_deviations:
        mlflow.log_param(
            "scoring_spec_deviations", dict(report.scoring_spec_deviations)
        )
    if predictions_path is not None:
        mlflow.log_param("predictions_path", str(predictions_path))
    if benchmark_path is not None:
        mlflow.log_param("benchmark_path", str(benchmark_path))
    if alias_store_path is not None:
        mlflow.log_param("alias_store_path", str(alias_store_path))

    # Headline metrics.
    mlflow.log_metric("em_micro", report.em_micro)
    mlflow.log_metric("em_macro", report.em_macro)
    mlflow.log_metric("em_conditional", report.em_conditional)
    mlflow.log_metric("ne_micro", report.ne_micro)
    mlflow.log_metric("ne_macro", report.ne_macro)
    # No-reasoning floor (majority-class baseline). Model-independent;
    # logged beside em_micro/em_macro so a reader immediately sees how
    # much of the headline is reachable without spatial reasoning.
    if report.majority_baseline_em is not None:
        mlflow.log_metric("majority_baseline_em", report.majority_baseline_em)
    if report.majority_baseline_em_macro is not None:
        mlflow.log_metric(
            "majority_baseline_em_macro", report.majority_baseline_em_macro
        )
    mlflow.log_metric("coverage_rate", report.coverage_rate)
    mlflow.log_metric("format_error_rate", report.format_error_rate)
    mlflow.log_metric("n_total", report.n_total)
    mlflow.log_metric("n_covered", report.n_covered)
    mlflow.log_metric("n_format_errors", report.n_format_errors)
    # v2 coverage breakdown — refusals + gold-missing split out from
    # format errors. answer_rate is an alias for coverage_rate; both
    # logged so the MLflow UI's natural-name search hits.
    mlflow.log_metric("answer_rate", report.answer_rate)
    mlflow.log_metric("refusal_rate", report.refusal_rate)
    mlflow.log_metric("gold_missing_rate", report.gold_missing_rate)
    mlflow.log_metric("n_refusals", report.n_refusals)
    mlflow.log_metric("n_gold_missing", report.n_gold_missing)
    # Harness-error split: ``em_adjusted`` is the capability-only
    # number (headline with harness errors excluded from the
    # denominator); per-class counts surface as individual metrics
    # so the MLflow UI can chart them next to the flat em_micro.
    mlflow.log_metric("em_adjusted", report.em_adjusted)
    mlflow.log_metric("n_harness_errors", report.n_harness_errors)
    for cls, count in report.error_class_breakdown.items():
        mlflow.log_metric(f"errors_{cls}", count)
    if report.resolve_failure_rate is not None:
        mlflow.log_metric("resolve_failure_rate", report.resolve_failure_rate)
    if report.unit_match_rate is not None:
        mlflow.log_metric("unit_match_rate", report.unit_match_rate)

    # Numeric-only EM/NE summary (strict vs unit-normalized). Logged
    # independently so the gap between them (spatial reasoning vs
    # unit-instruction compliance) is visible in the MLflow UI.
    if report.numeric_em_strict is not None:
        mlflow.log_metric("numeric_em_strict", report.numeric_em_strict)
    if report.numeric_em_unit_normalized is not None:
        mlflow.log_metric(
            "numeric_em_unit_normalized", report.numeric_em_unit_normalized
        )
    if report.numeric_ne_strict is not None:
        mlflow.log_metric("numeric_ne_strict", report.numeric_ne_strict)
    if report.numeric_ne_unit_normalized is not None:
        mlflow.log_metric(
            "numeric_ne_unit_normalized", report.numeric_ne_unit_normalized
        )
    # Denominator visibility (#6): the conditional (scored-subset) EM and
    # the retained fraction. When retained < 1.0 the conditional number is
    # a model-chosen denominator and not comparable across models.
    if report.numeric_em_conditional is not None:
        mlflow.log_metric(
            "numeric_em_conditional", report.numeric_em_conditional
        )
    if report.numeric_retained_fraction is not None:
        mlflow.log_metric(
            "numeric_retained_fraction", report.numeric_retained_fraction
        )

    # Numeric failure breakdown — one metric per category so the UI
    # can chart them independently.
    if report.numeric_failure_breakdown:
        for category, count in report.numeric_failure_breakdown.items():
            mlflow.log_metric(f"numeric_failure_{_metric_safe(category)}", count)

    # Grid-cell spatial companions (strict exact-ID vs geodesic
    # proximity). Logged independently so the gap — spatial competence
    # hidden by a non-canonical grid-encoding convention — is visible
    # in the MLflow UI. Populated only for runs with decodable grid-cell
    # rows (MGRS-like grid cell + the H3 half of Composite string).
    if report.n_grid_spatial > 0:
        mlflow.log_metric("n_grid_spatial", report.n_grid_spatial)
    if report.grid_em_strict is not None:
        mlflow.log_metric("grid_em_strict", report.grid_em_strict)
    if report.grid_em_spatial is not None:
        mlflow.log_metric("grid_em_spatial", report.grid_em_spatial)
    if report.grid_ne_strict is not None:
        mlflow.log_metric("grid_ne_strict", report.grid_ne_strict)
    if report.grid_ne_spatial is not None:
        mlflow.log_metric("grid_ne_spatial", report.grid_ne_spatial)

    # False-premise stratification.
    if report.n_true_premise > 0:
        mlflow.log_metric("n_true_premise", report.n_true_premise)
        if report.em_true_premise is not None:
            mlflow.log_metric("em_true_premise", report.em_true_premise)
    if report.n_false_premise > 0:
        mlflow.log_metric("n_false_premise", report.n_false_premise)
        if report.negative_rejection_rate is not None:
            mlflow.log_metric(
                "negative_rejection_rate", report.negative_rejection_rate
            )

    # Set-enumeration aggregate. Gate on n_set_typed (all set-typed rows)
    # so the metrics still log when every set row was refused.
    if report.n_set_typed > 0:
        mlflow.log_metric("n_set_records", report.n_set_records)
        mlflow.log_metric("n_set_typed", report.n_set_typed)
        if report.mean_set_precision is not None:
            mlflow.log_metric("mean_set_precision", report.mean_set_precision)
        if report.mean_set_recall is not None:
            mlflow.log_metric("mean_set_recall", report.mean_set_recall)
        if report.mean_set_f1 is not None:
            mlflow.log_metric("mean_set_f1", report.mean_set_f1)
        if report.mean_set_f1_conditional is not None:
            mlflow.log_metric(
                "mean_set_f1_conditional", report.mean_set_f1_conditional
            )
        if report.set_retained_fraction is not None:
            mlflow.log_metric(
                "set_retained_fraction", report.set_retained_fraction
            )

    # Tier-3 operational metrics.
    if report.n_tier3_records > 0:
        mlflow.log_metric("n_tier3_records", report.n_tier3_records)
        if report.cap_hit_rate is not None:
            mlflow.log_metric("cap_hit_rate", report.cap_hit_rate)
        if report.tool_error_rate is not None:
            mlflow.log_metric("tool_error_rate", report.tool_error_rate)
        if report.no_final_answer_rate is not None:
            mlflow.log_metric("no_final_answer_rate", report.no_final_answer_rate)
        if report.mean_tool_calls is not None:
            mlflow.log_metric("mean_tool_calls", report.mean_tool_calls)
        if report.mean_tool_errors is not None:
            mlflow.log_metric("mean_tool_errors", report.mean_tool_errors)

    # Self-consistency agreement.
    if report.n_sc_records > 0:
        mlflow.log_metric("n_sc_records", report.n_sc_records)
        if report.sc_agreement_rate is not None:
            mlflow.log_metric("sc_agreement_rate", report.sc_agreement_rate)

    # Per-group breakdowns.
    for group_name, group_dict in (
        ("type", report.per_answer_type),
        ("sfid", report.per_sfid),
        ("lang", report.per_language),
        ("strata", report.per_strata_cell),
        ("income", report.per_income_tier),
    ):
        for key, stats in group_dict.items():
            safe = _metric_safe(key)
            mlflow.log_metric(f"em_{group_name}_{safe}", stats.em)
            mlflow.log_metric(f"ne_{group_name}_{safe}", stats.ne)
            mlflow.log_metric(f"n_{group_name}_{safe}", stats.n)

    # Artifacts.
    for name in ("score_records.jsonl", "aggregate_report.json"):
        path = output_dir / name
        if path.exists():
            mlflow.log_artifact(str(path))

    logger.info(
        "Logged MLflow eval run '{}' (id={}): em_micro={:.3f} em_macro={:.3f}",
        run.info.run_name if hasattr(run.info, "run_name") else "?",
        run.info.run_id,
        report.em_micro,
        report.em_macro,
    )


__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_TRACKING_URI",
    "init_experiment",
    "log_eval_run",
]
