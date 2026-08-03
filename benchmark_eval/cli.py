"""Click-based CLI for :mod:`benchmark_eval`.

Entry point: ``python -m benchmark_eval <subcommand> ...``.

Subcommands:

* ``score`` — score a predictions JSONL against a benchmark JSONL,
  writing ``score_records.jsonl`` and ``aggregate_report.json`` to
  an output directory.

Flags on ``score`` cover the tolerance hyperparameters and the
scale-from-magnitude toggle so the user can run sensitivity sweeps
from the command line without editing code.
"""

from __future__ import annotations

from pathlib import Path

import click

from benchmark_eval._provenance import (
    compute_alias_store_sha256,
    compute_benchmark_snapshot_sha256,
)
from benchmark_eval.aggregate import build_report
from benchmark_eval.aliases import AliasStore
from benchmark_eval.loaders import load_gold_rows, load_predictions
from benchmark_eval.mlflow_logging import init_experiment, log_eval_run
from benchmark_eval.scorer import _SPEC as _SCORING_SPEC
from benchmark_eval.scorer import Scorer, ScorerConfig


@click.group()
def cli() -> None:
    """benchmark_eval — primary-metric scoring for MultiGlobeQA."""


@cli.command()
@click.option(
    "--predictions",
    "predictions_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the predictions JSONL produced by benchmark_models.",
)
@click.option(
    "--benchmark",
    "benchmark_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the benchmark — either a single JSONL file, or a "
    "directory of Spark ``part-*.json`` shards. Carries ground_truth "
    "per question_id.",
)
@click.option(
    "--aliases",
    "aliases_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the entity alias JSON. Used by the resolve() "
    "function for entity / cardinal / predefined types. "
    "Multilingual runs typically pass a per-language file from "
    "``output/aliases/<iso639-3>.json``.",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory to write score_records.jsonl and aggregate_report.json.",
)
# CLI defaults pull from the canonical scoring spec rather than
# duplicating literals. Edit ``scoring_spec.json`` and the no-flags
# CLI invocation tracks it automatically; the drift-guard test in
# tests/benchmark_eval/test_spec_drift.py asserts the two stay in
# sync.
@click.option(
    "--alpha",
    type=float,
    default=_SCORING_SPEC["alpha"],
    show_default=True,
    help="Relative tolerance for numeric / coordinate types.",
)
@click.option(
    "--delta-distance",
    "default_delta_distance_km",
    type=float,
    default=_SCORING_SPEC["default_delta_distance_km"],
    show_default=True,
    help="Default absolute tolerance (km) for distance answers.",
)
@click.option(
    "--delta-area",
    "default_delta_area_km2",
    type=float,
    default=_SCORING_SPEC["default_delta_area_km2"],
    show_default=True,
    help="Default absolute tolerance (km^2) for area answers.",
)
@click.option(
    "--no-scale-from-magnitude",
    is_flag=True,
    help="Disable scale-from-magnitude δ selection for distance / "
    "area numerics. The default behaviour picks δ from the gold "
    "magnitude (urban: |y*| < 50 → δ=0.5; regional: |y*| < 500 → "
    "δ=5; national: otherwise → δ=10). Pass this flag to use a "
    "flat global δ (set via --delta-distance / --delta-area). "
    "Useful for sensitivity analysis.",
)
@click.option(
    "--polygon-iou-threshold",
    type=float,
    default=_SCORING_SPEC["polygon_iou_threshold"],
    show_default=True,
    help="Polygon EM threshold: EM=1 iff IoU >= this value.",
)
@click.option(
    "--polar-angle-tol",
    "polar_angle_tol_degrees",
    type=float,
    default=_SCORING_SPEC["polar_angle_tol_degrees"],
    show_default=True,
    help="Bearing tolerance (degrees) for Polar coordinates (TID 70.1).",
)
@click.option(
    "--parent-run-id",
    default=None,
    help="Optional MLflow run id to nest the eval run under "
    "(typically the benchmark_models run that produced the "
    "predictions). When given, the eval run is linked to the parent "
    "in the MLflow UI; otherwise it runs standalone.",
)
@click.option(
    "--no-mlflow",
    is_flag=True,
    help="Skip MLflow logging for this eval run. Useful for quick "
    "local scoring without touching the tracking store.",
)
@click.option(
    "--include-invalid",
    is_flag=True,
    help="Include rows flagged ``is_valid=False`` by the benchmark "
    "validator. The default drops them from the denominator — they "
    "represent questions the upstream generator already marked as "
    "broken. Turn on for diagnostic re-scoring.",
)
@click.option(
    "--language",
    default=None,
    help="Restrict gold loading to this language's shards "
    "(<kg>/<sfid>/<language>/). The eval scores one language per cell, so "
    "passing it avoids loading all 17 languages and collapsing them "
    "last-wins (~17x less I/O; also silences the duplicate-qid warning). "
    "Omit to load every language.",
)
def score(
    predictions_path: Path,
    benchmark_path: Path,
    aliases_path: Path,
    output_dir: Path,
    alpha: float,
    default_delta_distance_km: float,
    default_delta_area_km2: float,
    no_scale_from_magnitude: bool,
    polygon_iou_threshold: float,
    polar_angle_tol_degrees: float,
    parent_run_id: str | None,
    no_mlflow: bool,
    include_invalid: bool,
    language: str | None,
) -> None:
    """Score a predictions JSONL and write score_records + aggregate report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Loading aliases from {aliases_path}")
    alias_store = AliasStore.load(aliases_path)
    click.echo(f"  {len(alias_store)} entries")
    alias_store_sha256 = compute_alias_store_sha256(aliases_path)
    click.echo(f"  alias_store_sha256: {alias_store_sha256[:16]}…")

    click.echo(f"Loading gold rows from {benchmark_path}")
    golds = load_gold_rows(
        benchmark_path, include_invalid=include_invalid, language=language
    )
    click.echo(f"  {len(golds)} rows")
    # Snapshot hash only when ``benchmark_path`` is the layout root
    # (a directory containing per-shard ``part-*.json`` files).
    # Single-file inputs (test fixtures, ad-hoc JSONLs) get ``None``.
    benchmark_snapshot_sha256: str | None = None
    if benchmark_path.is_dir():
        benchmark_snapshot_sha256 = compute_benchmark_snapshot_sha256(
            benchmark_path
        )
        click.echo(
            f"  benchmark_snapshot_sha256: {benchmark_snapshot_sha256[:16]}…"
        )

    click.echo(f"Loading predictions from {predictions_path}")
    predictions = load_predictions(predictions_path)
    click.echo(f"  {len(predictions)} predictions")

    config = ScorerConfig(
        alpha=alpha,
        default_delta_distance_km=default_delta_distance_km,
        default_delta_area_km2=default_delta_area_km2,
        use_scale_from_magnitude=not no_scale_from_magnitude,
        polygon_iou_threshold=polygon_iou_threshold,
        polar_angle_tol_degrees=polar_angle_tol_degrees,
    )
    scorer = Scorer(alias_store=alias_store, config=config)

    records = scorer.score_batch(predictions, golds)

    # Write score_records.jsonl.
    records_path = output_dir / "score_records.jsonl"
    with records_path.open("w") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    click.echo(f"Wrote {len(records)} score records to {records_path}")

    if not records:
        click.echo("No records scored; skipping aggregate report.")
        return

    deviations = config.deviations_from_spec()
    # ``--include-invalid`` materially changes the denominator (the
    # scoring spec's default policy is to drop ``is_valid=False``
    # rows). Record it as a deviation so the resulting AggregateReport
    # is traceable to the policy that produced it.
    if include_invalid:
        deviations["include_invalid"] = True
    report = build_report(
        records,
        scoring_spec_deviations=deviations,
        alias_store_sha256=alias_store_sha256,
        benchmark_snapshot_sha256=benchmark_snapshot_sha256,
        n_gold=len(golds),
    )
    report_path = output_dir / "aggregate_report.json"
    report_path.write_text(report.model_dump_json(indent=2))
    click.echo(f"Wrote aggregate report to {report_path}")
    if deviations:
        click.echo(f"  Scoring spec deviations: {deviations}")

    # Print a short headline summary.
    click.echo("")
    click.echo(f"  Coverage:       {report.coverage_rate:.1%} "
               f"({report.n_covered}/{report.n_total})")
    if report.n_missing_predictions:
        click.echo(
            f"  Missing preds:  {report.missing_prediction_rate:.1%} "
            f"({report.n_missing_predictions} gold qids with no prediction "
            f"— incomplete run)"
        )
    click.echo(f"  EM micro:       {report.em_micro:.3f}")
    click.echo(f"  EM macro:       {report.em_macro:.3f}")
    click.echo(f"  EM conditional: {report.em_conditional:.3f}")
    click.echo(f"  NE micro:       {report.ne_micro:.3f}")
    click.echo(f"  NE macro:       {report.ne_macro:.3f}")
    if report.format_error_rate > 0:
        click.echo(
            f"  Format errors:  {report.format_error_rate:.1%} "
            f"({report.n_format_errors}/{report.n_total})"
        )
    if report.refusal_rate > 0:
        click.echo(
            f"  Refusals:       {report.refusal_rate:.1%} "
            f"({report.n_refusals}/{report.n_total})"
        )
    if report.gold_missing_rate > 0:
        click.echo(
            f"  Gold missing:   {report.gold_missing_rate:.1%} "
            f"({report.n_gold_missing}/{report.n_total})"
        )
    if report.n_harness_errors > 0:
        click.echo(
            f"  Harness errors: {report.n_harness_errors}/{report.n_total} "
            f"(EM adjusted: {report.em_adjusted:.3f})"
        )
        for cls in sorted(report.error_class_breakdown):
            click.echo(
                f"    {cls}: {report.error_class_breakdown[cls]}"
            )
    if report.resolve_failure_rate is not None:
        click.echo(f"  Resolve failure rate: {report.resolve_failure_rate:.1%}")
    if report.unit_match_rate is not None:
        click.echo(f"  Unit match rate:      {report.unit_match_rate:.1%}")
    if report.numeric_em_strict is not None:
        click.echo(
            f"  Numeric EM strict:      {report.numeric_em_strict:.3f}"
        )
        assert report.numeric_em_unit_normalized is not None
        click.echo(
            f"  Numeric EM unit-norm.:  {report.numeric_em_unit_normalized:.3f}"
        )
    if report.n_false_premise > 0 or report.n_true_premise > 0:
        click.echo(
            f"  True-premise rows:      {report.n_true_premise} "
            f"(EM: {report.em_true_premise:.3f})"
            if report.em_true_premise is not None
            else f"  True-premise rows:      {report.n_true_premise}"
        )
        if report.negative_rejection_rate is not None:
            click.echo(
                f"  False-premise rows:     {report.n_false_premise} "
                f"(NR rate: {report.negative_rejection_rate:.3f})"
            )
    if report.mean_set_f1 is not None:
        click.echo(
            f"  Set-enum rows:          {report.n_set_records} "
            f"(P: {report.mean_set_precision:.3f}  "
            f"R: {report.mean_set_recall:.3f}  "
            f"F1: {report.mean_set_f1:.3f})"
        )
    if report.n_tier3_records > 0:
        click.echo(
            f"  Tier-3 rows:            {report.n_tier3_records}"
        )
        click.echo(
            f"    Cap-hit rate:         {report.cap_hit_rate:.1%}"
            if report.cap_hit_rate is not None
            else "    Cap-hit rate:         n/a"
        )
        click.echo(
            f"    Tool-error rate:      {report.tool_error_rate:.1%}"
            if report.tool_error_rate is not None
            else "    Tool-error rate:      n/a"
        )
        click.echo(
            f"    No-final-answer rate: {report.no_final_answer_rate:.1%}"
            if report.no_final_answer_rate is not None
            else "    No-final-answer rate: n/a"
        )
        click.echo(
            f"    Mean tool calls:      {report.mean_tool_calls:.1f}"
            if report.mean_tool_calls is not None
            else "    Mean tool calls:      n/a"
        )
    if report.sc_agreement_rate is not None:
        click.echo(
            f"  SC agreement rate:      {report.sc_agreement_rate:.3f} "
            f"(over {report.n_sc_records} SC records)"
        )

    # MLflow logging (default on). --no-mlflow skips; --parent-run-id
    # nests the eval run under an existing model-side run.
    if no_mlflow:
        click.echo("MLflow logging skipped (--no-mlflow).")
    else:
        init_experiment()
        config_params = {
            "alpha": alpha,
            "default_delta_distance_km": default_delta_distance_km,
            "default_delta_area_km2": default_delta_area_km2,
            "use_scale_from_magnitude": not no_scale_from_magnitude,
            "polygon_iou_threshold": polygon_iou_threshold,
            "polar_angle_tol_degrees": polar_angle_tol_degrees,
        }
        run_id = log_eval_run(
            report,
            output_dir=output_dir,
            config_params=config_params,
            parent_run_id=parent_run_id,
            predictions_path=predictions_path,
            benchmark_path=benchmark_path,
            alias_store_path=aliases_path,
        )
        if run_id:
            click.echo(f"MLflow eval run id: {run_id}")


# ============================================================================
# pairwise-agreement
# ============================================================================


@cli.command("pairwise-agreement")
@click.option(
    "--records",
    "records_specs",
    multiple=True,
    required=True,
    metavar="NAME=PATH",
    help="Score-records source in ``NAME=PATH`` form, where PATH is a "
    "``score_records.jsonl`` and NAME identifies the model in the "
    "output. Repeat for every model you want to compare (>= 2). "
    "Example: ``--records gemini=results/gem/score_records.jsonl "
    "--records qwen=results/qwen/score_records.jsonl``.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the pair list as JSON to this path. When omitted, the "
    "table is printed but no JSON is written.",
)
def pairwise_agreement_cmd(
    records_specs: tuple[str, ...],
    output_path: Path | None,
) -> None:
    """Compute pairwise model-agreement across score-records files.

    Joins on ``question_id`` and reports per-pair: agreement rate,
    both-right / both-wrong / A-only / B-only rates, and Cohen's kappa.
    Useful for ensemble candidacy (low kappa → models disagree on
    different questions, ensemble may help) and for redundancy
    detection (high kappa → models are saying the same thing).

    \b
    Example:
      python -m benchmark_eval pairwise-agreement \\
          --records gemini=results/gemini-3-flash-plain/score_records.jsonl \\
          --records qwen=results/qwen35-a3b-plain/score_records.jsonl \\
          --records claude=results/claude-sonnet-4.6-plain/score_records.jsonl \\
          --output results/agreement.json
    """
    from benchmark_eval.pairwise_agreement import (
        compute_all_pairs,
        format_table,
        load_em_by_question_id,
        write_report,
    )

    if len(records_specs) < 2:
        raise click.ClickException(
            "--records must be passed at least twice (need >= 2 models "
            "to pair). Got "
            f"{len(records_specs)}."
        )

    em_by_model: dict[str, dict[str, int]] = {}
    for spec in records_specs:
        if "=" not in spec:
            raise click.ClickException(
                f"--records value must be ``NAME=PATH``; got {spec!r}."
            )
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not path.is_file():
            raise click.ClickException(
                f"Score-records file not found: {path} (for model {name!r})."
            )
        if name in em_by_model:
            raise click.ClickException(
                f"Duplicate model name {name!r} in --records list."
            )
        em_by_model[name] = load_em_by_question_id(path)
        click.echo(
            f"  loaded {len(em_by_model[name]):,} records for model {name!r}"
        )

    pairs = compute_all_pairs(em_by_model)
    click.echo("")
    click.echo(format_table(pairs))
    if output_path is not None:
        write_report(pairs, output_path)
        click.echo("")
        click.echo(f"Wrote {len(pairs)} pair(s) to {output_path}")


if __name__ == "__main__":
    cli()
