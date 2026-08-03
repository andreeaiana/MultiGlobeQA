"""Click-based CLI for benchmark_models.

Entry point: ``python -m benchmark_models <subcommand> ...``

Subcommands:

* ``smoke`` — ad-hoc small run on a single variant. The quickest
  path to validating the pipeline end-to-end.
* ``list-variants`` — print every entry from ``models.yaml``, grouped
  by backend and tier.

The template registry is generated out-of-band via
``scripts/build_template_registry_from_benchmark.py``; there is no
CLI subcommand for it.
"""

from __future__ import annotations

from pathlib import Path

import click

from benchmark_models.config import load_models_catalogue
from benchmark_models.mlflow_logging import init_experiment, log_run
from benchmark_models.registry import create_model
from benchmark_models.runner import load_benchmark, run
from benchmark_models.schemas import load_template_registry
from benchmark_models.snapshot import build_paths_only_snapshot_id


@click.group()
def cli() -> None:
    """benchmark_models — systems-under-evaluation for MultiGlobeQA."""


def _assert_registry_covers_examples(
    examples: list, registry, *, preview_count: int = 8,
) -> None:
    """Fail-fast on missing-tid configuration drift.

    The template registry must cover every TID present in the loaded
    benchmark slice. If it doesn't, the OpenAI/vLLM schema-build path
    raises ``unknown tid 'X.Y'`` per row, and every row on an
    uncovered TID lands as a silent ``error_class=config`` record, so
    a registry that does not match the benchmark can void a large
    share of a run. Catching this at startup turns a partially-corrupt
    run into a loud, actionable error before any model call.

    Raises ``click.ClickException`` on missing TIDs so the CLI exits
    cleanly with the regen command in the error message.
    """
    benchmark_tids = {
        ex.tid for ex in examples if getattr(ex, "tid", None) is not None
    }
    known_tids = set(registry.templates.keys())
    unknown = sorted(benchmark_tids - known_tids)
    if not unknown:
        return
    preview = unknown[:preview_count]
    suffix = (
        f" (+{len(unknown) - preview_count} more)"
        if len(unknown) > preview_count
        else ""
    )
    raise click.ClickException(
        f"{len(unknown)} TID(s) in the loaded benchmark are not in the "
        f"template registry: {preview}{suffix}. The runtime prompt + "
        f"schema build would fail for these. Regenerate with:\n"
        f"  uv run python -m scripts.build_template_registry_from_benchmark "
        f"--benchmark <path-to-current-benchmark-dir> "
        f"--output benchmark_models/schemas/template_registry.json\n"
        f"and verify the new registry has all {len(benchmark_tids)} benchmark TIDs."
    )


# ============================================================================
# smoke
# ============================================================================


@cli.command()
@click.option(
    "--variant",
    "-v",
    required=True,
    help="Variant name from models.yaml (e.g., 'gemini-3-flash-plain').",
)
@click.option(
    "--benchmark",
    "-b",
    "benchmark_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("benchmark/final/large/"),
    show_default=True,
    help="Path to the benchmark. Accepts (1) a single JSONL file, "
    "(2) a leaf ``benchmark.jsonl/`` shard directory, or (3) a "
    "higher-level share root (e.g. ``benchmark/final/large/``) containing "
    "the nested ``<kg>/<sfid>/<lang>/benchmark.jsonl/`` layout. "
    "Walked by ``discover_benchmark_shards`` when given a root; pass "
    "``--language`` to restrict the walk to one language.",
)
@click.option(
    "--n",
    "max_examples",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of examples to run. Pass ``0`` to process "
    "every example in the loaded shard (right for SLURM jobs that "
    "want to chew through one cell entirely).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to write predictions JSONL. Default: results/smoke/<variant>/predictions.jsonl",
)
@click.option(
    "--api-key",
    default=None,
    help="API key override (passed to the backend constructor). "
    "If not given, the backend reads from the appropriate env var: "
    "GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY (OpenAI-compatible).",
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Language code override for the examples. The post-edited "
    "share is laid out as benchmark/final/<size>/<kg>/<sfid>/<lang>/ using "
    "ISO 639-3 codes (e.g. 'eng', 'deu', 'ita', 'zho_CN'); pass the "
    "matching code to --language. The override (a) restricts the "
    "share-root walk to that language, and (b) is set on every row "
    "regardless of any per-row language field; rows whose language "
    "field disagrees produce a one-time warning. Defaults to 'eng' "
    "when neither flag nor row carries a language.",
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="Do not resume from existing predictions; start fresh.",
)
@click.option(
    "--retry-errors",
    is_flag=True,
    help="Re-run only questions whose previous prediction had "
    "final_state='error' (transient API failures, etc.). "
    "The existing predictions.jsonl is first copied to a timestamped "
    ".bak file, then rewritten without the errored rows. Non-errored "
    "predictions (answered / abstained / cap_hit) are preserved.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for predict_batch. Defaults to a per-backend "
    "value when unset: vLLM uses its YAML config.batch_size (default "
    "32); tier-1/2 API backends use 16 (activates ThreadPoolExecutor "
    "batching); tier-3 uses 1 (agent loop is per-example). Explicit "
    "values override.",
)
@click.option(
    "--no-mlflow",
    is_flag=True,
    help="Skip MLflow logging for this run.",
)
@click.option(
    "--no-retry",
    is_flag=True,
    help="Disable in-flight retry on transient API errors. Each "
    "failure lands as final_state='error' immediately; recover "
    "later with --retry-errors. Recommended for tier 1/2 (single "
    "LLM call per question, cheap to re-run); not recommended for "
    "tier 3 (each agent loop has irrecoverable mid-run state).",
)
@click.option(
    "--use-kg",
    is_flag=True,
    help="Use real KG access via Spark/Sedona (tier 3a/3c). "
    "Requires Java installed and JAVA_HOME set. Without this "
    "flag, tier 3a/3c use mock KG data.",
)
@click.option(
    "--kg-root",
    "kg_root",
    type=click.Path(path_type=Path),
    default=None,
    help="Root directory containing per-KG Delta tables (N from the "
    "round-2 plan). Defaults to ``benchmark/kgs/`` (or the value of "
    "the BENCHMARK_KGS_ROOT env var). The runner resolves "
    "``{kg_root}/{kg_source}`` and passes it to "
    "``EntityService(TripleBridge(...))``. Only used when "
    "--use-kg is set.",
)
@click.option(
    "--kg-source",
    "kg_source",
    type=str,
    default=None,
    help="Which KG to query in tier 3a/3c (e.g. ``kwg``, ``osmh3kg``, "
    "``worldkg``). The SLURM generator emits one runner invocation "
    "per (lang, kg) cell, so this is typically populated by the "
    "generator. When unset, the bridge falls back to the "
    "OSMH3KG_NODES_PATH env var or the test fixture.",
)
@click.option(
    "--oracle-mode",
    type=click.Choice(["structured", "raw", "verbalized", "t3"]),
    default=None,
    help="Run the oracle baseline. Gold triples for each question are "
    "injected into the prompt. 'structured'/'raw'/'verbalized' are "
    "tier-1 variants (LLM reads the triples and answers directly); "
    "'t3' is a tier-3 variant (Python sandbox, no KG/web tools). "
    "Requires a tier-1 variant (for structured/raw/verbalized) or a "
    "tier-3 variant (for t3). Incompatible with tier 2.",
)
@click.option(
    "--gold-triples",
    "gold_triples_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("benchmark/final/large/"),
    show_default=True,
    help="Path to the gold triples — optional; defaults to "
    "``benchmark/final/large/``, which auto-discovers gold via the "
    "``<kg>/<sfid>/gold_triples.jsonl/`` layout. Accepts (1) a "
    "single JSONL file, (2) a leaf ``gold_triples.jsonl/`` shard "
    "directory, or (3) any higher-level root containing that "
    "nested layout (walked by ``discover_gold_shards``). Only read "
    "when --oracle-mode is set; otherwise ignored regardless of "
    "value.",
)
@click.option(
    "--wkt-simplify-tolerance",
    "wkt_simplify_tolerance",
    type=click.FloatRange(min=0.0),
    default=0.001,
    show_default=True,
    help="Douglas-Peucker tolerance for WKT simplification of gold "
    "triples in oracle contexts (lat/lon degrees; 0.001 ≈ 111 m on "
    "the equator). Set to 0 to disable simplification (oracle uses "
    "verbatim gold WKT — only feasible on closed-source backends "
    "with very large context windows). Applied UNIFORMLY to every "
    "backend so the cross-model oracle comparison stays at a single "
    "retrieval-fidelity point. Only consulted when --oracle-mode "
    "is set; ignored otherwise.",
)
@click.option(
    "--max-steps",
    "max_steps",
    type=click.IntRange(min=1),
    default=None,
    help="Override the tier-3 agent's max_steps (a.k.a. "
    "max_tool_calls). Defaults to the variant's models.yaml setting "
    "(itself defaulting to DEFAULT_MAX_TOOL_CALLS=10, matching the "
    "project plan's documented headline-run cap). Tier-3 only — "
    "rejected on tier-1 / tier-2 variants.",
)
@click.option(
    "--max-tokens",
    "max_tokens",
    type=click.IntRange(min=1),
    default=None,
    help="Override the LLM max_tokens cap. For tier-1/2 variants this "
    "applies to the single-call response budget (variant default 512); "
    "for tier-3 it applies to each agent step's LLM call (variant "
    "default 4096 across 3a/3b/3c). The CLI dispatches on the "
    "variant's tier; no separate flag is needed.",
)
@click.option(
    "--temperature",
    "temperature",
    type=click.FloatRange(min=0.0, max=2.0),
    default=None,
    help="Override the variant's sampling temperature (default 0.0 "
    "for greedy). Required to be > 0 when self-consistency N > 1 is "
    "in use; pydantic validates the combination after the override is "
    "applied.",
)
@click.option(
    "--top-p",
    "top_p",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help="Override the nucleus-sampling top_p (default 1.0). Only "
    "meaningful when temperature > 0.",
)
@click.option(
    "--seed",
    "seed",
    type=int,
    default=None,
    help="Override the variant's random seed (default 42). Recorded "
    "on every Prediction. Best-effort across backends: OpenAI honors "
    "it; Google does not expose a seed parameter.",
)
@click.option(
    "--self-consistency-n",
    "self_consistency_n",
    type=click.IntRange(min=1),
    default=None,
    help="Override the tier-2 self-consistency sample count. Variant "
    "default is 1 (no SC); workhorse tier-2 variants ship with 3 in "
    "models.yaml. Pass 1 to opt out of SC for a cheaper ablation run, "
    "or higher to test SC@5 / SC@7. Requires temperature > 0 when N > 1.",
)
@click.option(
    "--max-concurrent",
    "max_concurrent",
    type=click.IntRange(min=1),
    default=None,
    help="Override the tier-3 predict_batch concurrency. Defaults to "
    "the variant's models.yaml setting, falling back to "
    "_DEFAULT_MAX_CONCURRENT (4). Tier-3 only — rejected on "
    "tier-1 / tier-2 variants.",
)
@click.option(
    "--web-search-timeout",
    "web_search_timeout",
    type=click.FloatRange(min=0.5),
    default=None,
    help="Override the tier-3b / 3c WebSearchTool per-request "
    "timeout (seconds). Defaults to the variant's models.yaml "
    "setting, falling back to _DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS "
    "(15.0). Tier-3b / 3c only — rejected on other tiers.",
)
@click.option(
    "--multimodal-mode",
    "multimodal_mode",
    type=click.Choice(["off", "image", "name"]),
    default=None,
    help="Override the variant's multimodal_mode. 'image' packs the "
    "extracted image bytes into a provider content block; 'name' "
    "substitutes the entity's display name into [IMAGE_<slot>] "
    "placeholders for a text-only baseline; 'off' ignores images. "
    "Defaults to the variant's models.yaml setting (itself defaulting "
    "to 'off'). Pair 'image' / 'name' with the post-edited multimodal "
    "share at benchmark/final/<size>/multimodal/<kg>/.",
)
def smoke(
    variant: str,
    benchmark_path: Path,
    max_examples: int,
    output_path: Path | None,
    api_key: str | None,
    language: str | None,
    no_resume: bool,
    retry_errors: bool,
    batch_size: int | None,
    no_mlflow: bool,
    no_retry: bool,
    use_kg: bool,
    kg_root: Path | None,
    kg_source: str | None,
    oracle_mode: str | None,
    gold_triples_path: Path,
    wkt_simplify_tolerance: float,
    max_steps: int | None,
    max_tokens: int | None,
    max_concurrent: int | None,
    web_search_timeout: float | None,
    multimodal_mode: str | None,
    temperature: float | None,
    top_p: float | None,
    seed: int | None,
    self_consistency_n: int | None,
) -> None:
    """Run a quick smoke test on a single variant.

    Loads a benchmark file, takes the first N examples, runs them
    through the specified variant, and writes predictions to a JSONL
    file. No experiment YAML needed — everything is specified via CLI
    flags.

    \b
    Examples:
      # 10 questions through Gemini, API key from env var
      python -m benchmark_models smoke -v gemini-3-flash-plain

      # 5 questions, explicit API key, custom output
      python -m benchmark_models smoke -v gemini-3-flash-plain \\
          --api-key AIza... --n 5 -o my_test.jsonl

      # DummyModel (no API needed)
      python -m benchmark_models smoke -v dummy
    """
    # ``--no-resume --retry-errors`` is contradictory: ``--no-resume``
    # truncates the predictions file before the run begins, leaving
    # nothing for ``--retry-errors`` to operate on. Surface the
    # conflict so a typo doesn't silently wipe a partial run.
    if no_resume and retry_errors:
        raise click.ClickException(
            "--no-resume and --retry-errors are mutually exclusive: "
            "--no-resume truncates the predictions file at the start "
            "of the run, leaving no errored rows for --retry-errors "
            "to retry. Drop one of the two flags."
        )

    # Apply --no-retry before any backend constructs — the retry
    # toggle is read inside SDK call wrappers, so it must be set
    # before workers spawn.
    if no_retry:
        from benchmark_models.backends._retry import set_retry_enabled

        set_retry_enabled(False)
        click.echo("Retry disabled (--no-retry).")

    # Load the template registry + models catalogue.
    registry = load_template_registry()
    catalogue = load_models_catalogue()

    kg_backend = None  # initialized here, possibly set below for tier 3a/3c
    oracle_context = None  # initialized here, possibly set below for oracle runs
    # Hoist so the variable is always bound. The non-dummy branch
    # overwrites with paths-only or versioned form; the dummy branch
    # keeps the empty default (DummyModel doesn't write to MLflow).
    snapshot_id: str = ""

    # Handle the special "dummy" variant that isn't in models.yaml.
    if variant == "dummy":
        if oracle_mode is not None:
            raise click.ClickException(
                "--oracle-mode is not supported with the dummy variant."
            )
        from benchmark_models.dummy import DummyModel, DummyStrategy

        model = DummyModel(
            strategy=DummyStrategy.TYPE_DEFAULT,
            registry=registry,
        )
        click.echo("Using DummyModel (strategy=TYPE_DEFAULT)")
    else:
        if variant not in catalogue:
            available = ", ".join(catalogue.list_variants()[:10])
            raise click.ClickException(
                f"Unknown variant {variant!r}. Available: [{available}, ...]"
            )
        config = catalogue.get(variant)
        snapshot_id = build_paths_only_snapshot_id()

        # Create KG backend for tier 3a/3c if requested. Skip for
        # oracle runs — T3-oracle has no KG tools by design.
        kg_backend = None
        if (
            use_kg
            and oracle_mode is None
            and config.tier == "tool_agent"
            and config.sub_tier in ("3a", "3c")
        ):
            from benchmark_models.kg_tools.entity_service import EntityService
            from benchmark_models.kg_tools.executor_bridge import resolve_kg_path
            from benchmark_models.kg_tools.schemas import schema_for
            from benchmark_models.kg_tools.triple_bridge import TripleBridge

            # Resolve the per-runner KG path from CLI args. When
            # --kg-source is not set, ``resolve_kg_path`` returns None
            # and we error rather than fall back: the triple-native
            # bridge needs a KG path AND a matching ``KGSchema``, and
            # an env-var or fixture path carries no schema to pair
            # with it.
            resolved_kg_path = resolve_kg_path(
                kg_source=kg_source,
                kg_root=str(kg_root) if kg_root is not None else None,
            )
            if resolved_kg_path is None or kg_source is None:
                raise click.ClickException(
                    "--use-kg requires --kg-source <kwg|osmh3kg|worldkg>. "
                    "The triple-native KG bridge cannot fall back to an "
                    "OSMH3KG_NODES_PATH / fixture path without knowing "
                    "which KG's predicate vocabulary to apply."
                )
            click.echo(
                f"Starting Spark/Sedona for KG access (kg_source={kg_source}, "
                f"kg_path={resolved_kg_path})..."
            )
            kg_backend = EntityService(
                bridge=TripleBridge(
                    kg_path=resolved_kg_path,
                    schema=schema_for(kg_source),
                ),
                schema=schema_for(kg_source),
            )

            # Upgrade the paths-only snapshot id to the versioned
            # tier-3 form ``<paths_hash>@<versions_hash>``. The
            # versioned form pins the exact Delta-table versions used
            # for the run, so tier-3 results stay reproducible when
            # the underlying KG is re-ingested.
            from benchmark_models.snapshot import (
                SnapshotDriftError,
                upgrade_to_versioned,
            )

            try:
                snapshot_id = upgrade_to_versioned(snapshot_id, kg_backend)
                click.echo(
                    f"Upgraded tier-3 snapshot_id to versioned form: "
                    f"{snapshot_id}"
                )
            except SnapshotDriftError as exc:
                raise click.ClickException(str(exc)) from exc

        # Build the oracle context if --oracle-mode is set. Validate
        # the mode against the variant's tier; fail loud if mismatched.
        if oracle_mode is not None:
            from benchmark_models.oracle import (
                T1_ORACLE_MODES,
                OracleContext,
                load_gold_triples,
            )

            if config.tier == "reasoning":
                raise click.ClickException(
                    f"--oracle-mode is not supported for tier-2 variants "
                    f"(got {variant!r}). Use a tier-1 or tier-3 variant."
                )
            if config.tier == "plain" and oracle_mode not in T1_ORACLE_MODES:
                raise click.ClickException(
                    f"--oracle-mode={oracle_mode!r} is for tier-3 variants. "
                    f"For {variant!r} (tier 1), use "
                    f"--oracle-mode={{structured,raw,verbalized}}."
                )
            if config.tier == "tool_agent" and oracle_mode != "t3":
                raise click.ClickException(
                    f"--oracle-mode={oracle_mode!r} is for tier-1 variants. "
                    f"For {variant!r} (tier 3), use --oracle-mode=t3."
                )

            click.echo(f"Loading gold triples from {gold_triples_path}...")
            gold_triples_map = load_gold_triples(gold_triples_path)
            oracle_context = OracleContext(
                mode=oracle_mode,  # type: ignore[arg-type]
                gold_triples_map=gold_triples_map,
                wkt_simplify_tolerance=wkt_simplify_tolerance,
            )

        # Tier-3-only flags: reject misuse loudly so a
        # silently-ignored override doesn't taint a headline run.
        tier3_only_flags: list[tuple[str, object]] = [
            ("--max-steps", max_steps),
            ("--max-concurrent", max_concurrent),
            ("--web-search-timeout", web_search_timeout),
        ]
        if config.tier != "tool_agent":
            for flag, value in tier3_only_flags:
                if value is not None:
                    raise click.ClickException(
                        f"{flag} is a tier-3 flag; got {value!r} for "
                        f"variant {variant!r} (tier {config.tier!r})."
                    )
        # ``--web-search-timeout`` only applies to sub-tiers with web
        # access (3b, 3c). Reject on 3a where the tool isn't loaded.
        if (
            config.tier == "tool_agent"
            and config.sub_tier == "3a"
            and web_search_timeout is not None
        ):
            raise click.ClickException(
                "--web-search-timeout requires sub-tier 3b or 3c "
                f"(WebSearchTool is not loaded for {variant!r}, "
                f"sub-tier 3a)."
            )

        # Build the config-overrides dict from the sampling-knob CLI
        # flags. ``--max-tokens`` is tier-dispatched: tier 1/2 set the
        # base ``max_tokens`` (per-call response budget, default 512);
        # tier 3 sets ``tier3_max_tokens`` (per-step LLM call,
        # sub-tier-specific default). Pydantic validates the resulting
        # combination via ``model_copy(update=...)`` inside create_model
        # — e.g., self_consistency_n>1 with temperature=0 is rejected.
        config_overrides: dict[str, object] = {}
        if temperature is not None:
            config_overrides["temperature"] = temperature
        if top_p is not None:
            config_overrides["top_p"] = top_p
        if seed is not None:
            config_overrides["seed"] = seed
        if self_consistency_n is not None:
            config_overrides["self_consistency_n"] = self_consistency_n
        if max_tokens is not None:
            if config.tier == "tool_agent":
                # Keep using the tier-3 *_override kwarg path so the
                # existing tier-3 resolution logic stays one code path.
                pass
            else:
                config_overrides["max_tokens"] = max_tokens

        model = create_model(
            variant,
            config,
            registry=registry,
            api_key=api_key,
            snapshot_id=snapshot_id,
            kg_backend=kg_backend,
            oracle_context=oracle_context,
            max_steps_override=max_steps,
            max_tokens_override=max_tokens if config.tier == "tool_agent" else None,
            max_concurrent_override=max_concurrent,
            web_search_timeout_override=web_search_timeout,
            multimodal_mode_override=multimodal_mode,  # type: ignore[arg-type]
            config_overrides=config_overrides or None,
        )
        click.echo(f"Created {type(model).__name__} for variant '{variant}'")
        if oracle_context is not None:
            click.echo(f"  Oracle mode: {oracle_mode}")

    # Load benchmark.
    examples = load_benchmark(benchmark_path, language=language)
    if max_examples < 0:
        raise click.ClickException(
            f"--n must be >= 0 (0 means 'all rows'); got {max_examples}."
        )
    if max_examples and len(examples) > max_examples:
        click.echo(f"Using first {max_examples} of {len(examples)} examples")
        examples = examples[:max_examples]
    else:
        click.echo(f"Using all {len(examples)} examples")

    # Fail-fast if the registry doesn't cover every TID in scope.
    # See ``_assert_registry_covers_examples`` for the rationale.
    _assert_registry_covers_examples(examples, registry)

    # For oracle runs, verify every example's question_id is covered
    # BEFORE the first predict call; a gap raises immediately.
    if oracle_context is not None:
        from benchmark_models.oracle import assert_coverage

        assert_coverage(
            oracle_context.gold_triples_map,
            [ex.question_id for ex in examples],
        )

    # Output path. Oracle runs get a mode-specific subdir so different
    # oracle modes on the same variant don't collide in one file.
    if output_path is None:
        if oracle_mode:
            output_path = Path(
                f"results/smoke/{variant}/oracle_{oracle_mode}/predictions.jsonl"
            )
        else:
            output_path = Path(f"results/smoke/{variant}/predictions.jsonl")
    click.echo(f"Output: {output_path}")

    # Per-question progress callback.
    def _progress(current: int, total: int, pred: object) -> None:
        from benchmark_models.base import Prediction

        assert isinstance(pred, Prediction)
        status = "ok" if not pred.refusal else "refused"
        if pred.final_state == "error":
            status = "ERROR"
        answer_preview = repr(pred.answer)
        if len(answer_preview) > 40:
            answer_preview = answer_preview[:37] + "..."
        click.echo(
            f"  [{current}/{total}] {pred.question_id[:12]}... "
            f"→ {status} | answer={answer_preview} | "
            f"{pred.latency_ms:.0f}ms"
        )

    # Run.
    predictions = run(
        model,
        examples,
        output_path,
        batch_size=batch_size,
        resume=not no_resume,
        retry_errors=retry_errors,
        progress_callback=_progress,
    )

    # Summary.
    n_answered = sum(1 for p in predictions if not p.refusal)
    n_refused = sum(1 for p in predictions if p.refusal)
    n_errors = sum(
        1 for p in predictions if p.final_state == "error"
    )
    error_breakdown: dict[str, int] = {}
    for p in predictions:
        if p.final_state == "error":
            key = p.error_class or "unknown"
            error_breakdown[key] = error_breakdown.get(key, 0) + 1
    total_latency = sum(p.latency_ms for p in predictions)
    click.echo("")
    click.echo(f"Done. {len(predictions)} predictions in {total_latency / 1000:.1f}s:")
    click.echo(f"  answered: {n_answered}")
    click.echo(f"  refused:  {n_refused}")
    click.echo(f"  errors:   {n_errors}")
    if error_breakdown:
        for ec in sorted(error_breakdown):
            click.echo(f"    {ec}: {error_breakdown[ec]}")

    # Log to MLflow (skip for DummyModel and when --no-mlflow is set).
    if not no_mlflow and predictions and variant != "dummy":
        init_experiment()
        run_config = catalogue.get(variant)
        # Pass the local ``snapshot_id`` (already upgraded to the
        # versioned form for tier-3 runs above) so per-prediction and
        # MLflow-run provenance agree.
        log_run(
            variant_name=variant,
            config=run_config,
            predictions=predictions,
            output_path=output_path,
            snapshot_id=snapshot_id,
        )
    elif no_mlflow:
        click.echo("MLflow logging skipped (--no-mlflow).")

    # Close the model and KG backend (releases resources).
    model.close()
    if kg_backend is not None:
        kg_backend.close()


# ============================================================================
# list-variants
# ============================================================================


@cli.command("list-variants")
def list_variants_cmd() -> None:
    """Print every variant from models.yaml, grouped by backend and tier."""
    catalogue = load_models_catalogue()
    if not catalogue:
        click.echo("models.yaml is empty.")
        return

    # Group by (backend, tier).
    from collections import defaultdict

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in catalogue.list_variants():
        config = catalogue.get(name)
        groups[(config.backend, config.tier)].append(name)

    for (backend, tier), names in sorted(groups.items()):
        click.echo(f"\n[{backend}] tier={tier}")
        for name in names:
            click.echo(f"  {name}")
    click.echo(f"\nTotal: {len(catalogue)} variants")


# ============================================================================
# cost-estimate
# ============================================================================


@cli.command("cost-estimate")
@click.option(
    "--variant",
    "-v",
    "variant_spec",
    required=True,
    help="One variant name, a comma-separated list, or ``all`` for every "
    "variant in models.yaml.",
)
@click.option(
    "--benchmark",
    "-b",
    "benchmark_path",
    type=click.Path(path_type=Path),
    default=Path("benchmark/final/large/"),
    show_default=True,
    help="Benchmark path used to count rows. Same shape as the smoke "
    "command's --benchmark. The path is read only when ``--n`` is "
    "omitted; with ``--n`` you can estimate against a hypothetical row "
    "count without the data on disk.",
)
@click.option(
    "--language",
    "-l",
    default=None,
    help="Restrict the row count to one language. Without this flag the "
    "walker concatenates every language under the share root.",
)
@click.option(
    "--oracle-mode",
    type=click.Choice(["structured", "raw", "verbalized", "t3"]),
    default=None,
    help="Apply the oracle-mode prompt-bloat factor (gold triples "
    "injected) to the estimate.",
)
@click.option(
    "--n",
    "n_override",
    type=int,
    default=None,
    help="Bypass the benchmark row count and estimate for ``n`` rows "
    "directly. Useful when the benchmark layout isn't yet on disk.",
)
def cost_estimate_cmd(
    variant_spec: str,
    benchmark_path: Path,
    language: str | None,
    oracle_mode: str | None,
    n_override: int | None,
) -> None:
    """Print an approximate API cost estimate before launching a run.

    \b
    Token counts are per-tier rules of thumb; provider prices come
    from a baked-in table that needs manual updating as rate cards
    change. The estimate is for catching order-of-magnitude surprises,
    not for billing.

    \b
    Examples:
      # Single variant, full share at the chosen language.
      python -m benchmark_models cost-estimate \\
          -v gemini-3-flash-plain --language eng

      # Compare multiple variants on a fixed row count (no disk read).
      python -m benchmark_models cost-estimate \\
          -v gemini-3-flash-plain,claude-sonnet-4.6-plain --n 100000

      # Estimate every variant in the catalogue.
      python -m benchmark_models cost-estimate -v all --n 65000
    """
    from benchmark_models.cost_estimator import (
        _PROVIDER_PRICES,
        estimate_cost,
        format_estimate_table,
    )

    catalogue = load_models_catalogue()
    if variant_spec == "all":
        variant_names = catalogue.list_variants()
    else:
        variant_names = [v.strip() for v in variant_spec.split(",") if v.strip()]
    unknown = [v for v in variant_names if v not in catalogue]
    if unknown:
        raise click.ClickException(
            f"Unknown variant(s): {unknown}. Run ``list-variants`` to see "
            f"every catalogue entry."
        )

    if n_override is not None:
        if n_override < 0:
            raise click.ClickException("--n must be >= 0.")
        n_rows = n_override
    else:
        # Count rows by streaming the benchmark; cheap with sharded
        # JSONL and avoids loading full Example records into memory.
        from benchmark_models.io import (
            discover_benchmark_shards,
            iter_jsonl_rows,
        )

        shards = discover_benchmark_shards(benchmark_path, language=language)
        n_rows = sum(1 for _ in (row for shard in shards for row in iter_jsonl_rows(shard)))
        click.echo(f"Counted {n_rows:,} rows across {len(shards)} shard(s).")

    estimates = [
        estimate_cost(
            v, catalogue.get(v), n_rows, oracle_mode=oracle_mode,
        )
        for v in variant_names
    ]
    click.echo("")
    click.echo(format_estimate_table(estimates))
    click.echo("")
    click.echo("Provider price table (USD per 1M tokens; verify before launch):")
    for backend, pricing in _PROVIDER_PRICES.items():
        click.echo(
            f"  {backend:<10} in=${pricing.input_per_million:.2f}  "
            f"out=${pricing.output_per_million:.2f}"
        )
    click.echo("")
    click.echo("Notes:")
    click.echo(
        "  - Token counts are per-tier rules of thumb, not measured. "
        "Real runs vary ±30%."
    )
    click.echo(
        "  - Prompt caching (Gemini) can cut input cost up to 50% "
        "on shared prefixes — not modelled here."
    )
    click.echo(
        "  - vLLM variants show $0 API cost (self-hosted); GPU-hour spend "
        "is separate."
    )
