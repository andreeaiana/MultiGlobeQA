"""Main entry point for Spatial KG RAG Benchmark Generator.

Orchestrates plan-driven benchmark generation:
1. Initialize Spark with Sedona and Delta Lake
2. Load spatial entity data from Delta tables
3. Execute plan-driven pipeline (strata x country x entity pool x SQL)
4. Validate and export results
"""


import pyspark.sql.functions as F
from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from question_generator.config import (
    SEDONA_PACKAGES,
    SPARK_CHECKPOINT_DIR,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_MASTER,
    SPARK_TEMP_DIR,
    MULTIGLOBEQA_CHECKPOINT_ENABLED,
)
from question_generator.tracker import (
    setup_mlflow,
)

try:
    from sedona.spark import SedonaContext

    HAS_SEDONA = True
except ImportError:
    HAS_SEDONA = False


def _auto_shuffle_partitions(default_parallelism: int) -> int:
    """Scale spark.sql.shuffle.partitions with available parallelism.

    Default of 200 oversubscribes a local[16] run (12 tasks per core per
    shuffle stage → scheduling overhead dominates). Rule of thumb: 4× cores,
    floored at 16 so tiny test sessions still parallelize reads.
    """
    return max(16, default_parallelism * 4)


def init_spark() -> SparkSession:
    """Initialize SparkSession with Sedona and Delta Lake."""
    builder = (
        SparkSession.builder.appName("Spatial_KG_RAG_Benchmark_Generator")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Adaptive Query Execution — auto-coalesce small shuffle partitions,
        # handle skew, and re-plan after stage stats are known. Off by default
        # in older Spark, on by default in 3.2+; we set explicitly so the
        # settings aren't silently lost on version bumps.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # 256MB broadcast threshold — small reference tables like
        # countries_view (258 rows) and strata slices always broadcast;
        # removes a shuffle from every TID's country-scoped join.
        .config("spark.sql.autoBroadcastJoinThreshold", str(256 * 1024 * 1024))
        # Parquet-level bloom filter on the gold-Delta `subject` column.
        # Set as `spark.hadoop.*` so Hadoop conf picks it up — applies to
        # both the initial Delta write and OPTIMIZE-driven rewrites.
        # Spark's parquet reader uses these for filter pushdown on equality
        # and IN predicates, which is the access pattern of our gold SQL
        # (random 100-URI lookups defeat Z-ORDER min/max but bloom prunes).
        .config("spark.hadoop.parquet.bloom.filter.enabled#subject", "true")
        .config("spark.hadoop.parquet.bloom.filter.expected.ndv#subject", "100000000")
        .config("spark.hadoop.parquet.bloom.filter.fpp#subject", "0.05")
        .config("spark.sedona.global.index", "true")
        .config("spark.sedona.global.indextype", "rtree")
        .config("spark.sedona.join.autoBroadcastJoinThreshold", str(256 * 1024 * 1024))
        # Heartbeat / RPC timeouts bumped well above default so a long GC
        # pause or a compute-heavy task on the driver doesn't trip the
        # HeartbeatReceiver → kill-executor → SparkContext-shutdown cycle.
        # Observed ~7-min silences in local-mode planet-scale cells.
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.network.timeout", "1200s")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.dir", "spark-warehouse/tmp")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true",
        )
    )

    if SPARK_TEMP_DIR:
        builder = builder.config("spark.local.dir", SPARK_TEMP_DIR)

    builder = builder.config("spark.jars.packages", SEDONA_PACKAGES)

    if HAS_SEDONA:
        spark = builder.getOrCreate()
        spark = SedonaContext.create(spark)
    else:
        spark = builder.getOrCreate()
        logger.warning(
            "Apache Sedona is not installed. "
            "Spatial functions will fail. Install with: pip install apache-sedona"
        )

    # defaultParallelism is only available after getOrCreate, so set
    # shuffle.partitions via runtime conf instead of on the builder.
    shuffle_partitions = _auto_shuffle_partitions(
        int(spark.sparkContext.defaultParallelism)
    )
    spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))
    logger.info(
        "Spark tuned: defaultParallelism={}, shuffle.partitions={}, AQE=on, broadcast=256MB",
        spark.sparkContext.defaultParallelism,
        shuffle_partitions,
    )

    if MULTIGLOBEQA_CHECKPOINT_ENABLED:
        import os as _os
        _os.makedirs(SPARK_CHECKPOINT_DIR, exist_ok=True)
        spark.sparkContext.setCheckpointDir(SPARK_CHECKPOINT_DIR)
        logger.info("Spark checkpoint dir: {}", SPARK_CHECKPOINT_DIR)

    return spark


def _load_question_templates(tid: str) -> tuple[str, str]:
    """Return ``(sub_template, stid)`` for sub-template ``tid``.

    Reads the English template table shipped at
    ``data/templates/templates_eng.tsv``. The shape pipeline attaches this text
    to every benchmark row before export so the exporter's
    ``_interpolate_questions`` can substitute ``{{entity_A}}`` / ``{{entity_B}}``.
    """
    import csv

    from question_generator.config import TEMPLATES_TSV

    with open(TEMPLATES_TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["ftid"] == tid:
                return row["sub_template_eng"], row["vid"].split(".")[2]
    raise ValueError(f"no template row for sub-template {tid!r} in {TEMPLATES_TSV}")


def generate_from_run_spec(run_spec_path: str) -> DataFrame | None:
    """Orchestrate benchmark generation from a ``runs/*.yaml`` run spec.

    Reads the per-sub-template files under
    ``question_generator/sql_shapes/<sfid_x>/<tid>.*`` and the shared
    ``config/kg_prefixes.yaml``.
    """
    from pathlib import Path

    import yaml

    from question_generator.exporter import BenchmarkExporter
    from question_generator.shape_contract import check_shape
    from question_generator.shape_executor import (
        close_kg_context,
        execute_tid_kg,
        load_kg_context,
    )
    from question_generator.shape_loader import (
        compute_run_sha,
        compute_shape_sha,
        load_prefix_map,
        load_run_spec,
        load_tid_shape,
        parse_tid_ref,
    )
    from question_generator.validator import BenchmarkValidator

    run = load_run_spec(run_spec_path)
    project_root = Path(__file__).resolve().parent.parent
    shapes_root = project_root / "question_generator" / "sql_shapes"
    prefix_map = load_prefix_map(project_root / "config" / "kg_prefixes.yaml")

    setup_mlflow()
    spark = init_spark()

    import json as _json
    import os as _os
    import shutil as _shutil

    # Per-cell staging dir. Writing each (TID, KG, stratum) result to parquet
    # immediately truncates the Catalyst logical plan so the optimizer never
    # sees a N-way union of deep sub-plans. Without this, a 36-cell smoke run
    # spends 10+ min in driver-side planning before submitting the first stage
    # (Spark UI shows nothing during that window).
    staging_root = _os.getenv(
        "MULTIGLOBEQA_SHAPE_STAGING_ROOT",
        _os.path.join("spark-warehouse", f"staging_{run.run_name}"),
    )
    # Gold-flat staging mirrors the main staging tree but lives in a sibling
    # root so the read-back can pick up gold rows separately (different schema
    # — flat triples — from main rows).
    gold_staging_root = staging_root + "_gold"
    _os.makedirs(staging_root, exist_ok=True)
    _os.makedirs(gold_staging_root, exist_ok=True)

    # Resume semantics: per-cell marker files encode the shape + run hashes
    # that produced the staged parquet. A re-submission with unchanged shapes
    # and run params skips already-done cells; any edit to a shape (sql, gold
    # sql, meta) or to (run_seed, total_sample, per_bucket_cap) flips the
    # relevant hash and forces that cell to re-run. Set
    # MULTIGLOBEQA_SHAPE_RESUME=0 to force a full re-run (wipes staging roots).
    resume_enabled = _os.getenv("MULTIGLOBEQA_SHAPE_RESUME", "1") != "0"
    if not resume_enabled:
        for root in (staging_root, gold_staging_root):
            if _os.path.exists(root):
                logger.info("Resume disabled; wiping staging root {}", root)
                _shutil.rmtree(root)
                _os.makedirs(root, exist_ok=True)

    try:
        # Strata loop is explicit. An empty `run.strata` means "pool across
        # all strata and execute once" (the old behaviour). A non-empty list
        # means "one execution per stratum; total_sample is a per-cell cap
        # so every requested stratum gets up to that many rows independently".
        stratum_iter: list[str | None] = (
            list(run.strata) if run.strata else [None]
        )

        # Pre-load every shape once so the KG outer loop doesn't re-parse them
        # for each KG. check_shape also runs here so contract errors fail
        # fast, before any Spark work. Each element of run.tids is a reference
        # of the form "<tid>" or "<tid>:<variant>"; the canonical key used
        # throughout the orchestration is the resolved ref "<tid>:<variant>".
        from question_generator.shape_loader import TidShape as _TidShape
        shapes: dict[str, _TidShape] = {}
        shape_shas: dict[str, str] = {}
        templates: dict[str, str] = {}
        for tid_ref in run.tids:
            tid, variant_id = parse_tid_ref(tid_ref)
            shape = load_tid_shape(tid, shapes_root, variant_id=variant_id)
            check_shape(shape, prefix_map)
            key = shape.ref  # canonical "<tid>:<variant>"
            shapes[key] = shape
            shape_shas[key] = compute_shape_sha(shape)
            base_template, _stid = _load_question_templates(tid)
            templates[key] = base_template
        shape_keys = list(shapes.keys())

        run_sha = compute_run_sha(
            run.run_seed,
            run.total_sample,
            # All TIDs in a run share the same run-level caps here; per-TID
            # sampling knobs live inside the shape hash.
            per_bucket_cap=shapes[shape_keys[0]].meta.sampling.per_bucket_cap,
            countries_per_cell=run.countries_per_cell,
            prefix_map=prefix_map,
        )
        logger.info(
            "Resume: enabled={}, run_sha={}, shape_shas={}",
            resume_enabled, run_sha, shape_shas,
        )

        staged_count = 0
        skipped_count = 0

        # Loop order: KG outermost so the strata + entity + triples handles
        # are loaded and cached once per KG, reused across TIDs × strata.
        for kg in run.kgs:
            kg_ctx = load_kg_context(spark, kg, prefix_map=prefix_map)
            # Register population raster + grid + per-KG cells coverage so
            # raster-dependent TIDs (e.g. 21.1 — radius_population_buffer)
            # find the pop_tiles / pop_by_cell / h3_cells_df temp views at
            # SQL analysis time. Idempotent + per-KG (h3_cells_df is
            # KG-specific). No-ops if the raster file is absent.
            from question_generator.raster_setup import register_population_views
            register_population_views(spark, kg)

            # Register the per-KG aggregated road view so road-based TIDs
            # (sfid_d Wave F1 — TIDs 39-47) find `osmh3kg_roads_aggregated`
            # at SQL analysis time. Currently osmh3kg-only; raises
            # NotImplementedError for kwg/worldkg, so guard the call.
            if kg == "osmh3kg":
                from question_generator.road_setup import register_road_aggregation_views
                register_road_aggregation_views(spark, kg)

            # Register the KG-agnostic country_polygons view so
            # country-aware ftids (sfid_a 7.1: "country bordering the
            # country containing entity_A") find Natural Earth's country
            # geometry at SQL analysis time. Idempotent — only registers
            # on first kg of the loop. Source data is auto-downloaded
            # if missing (one-time, ~5 MB).
            from question_generator.country_setup import register_country_polygons_view
            register_country_polygons_view(spark)
            kg_staging = _os.path.join(staging_root, kg)
            kg_gold_staging = _os.path.join(gold_staging_root, kg)
            marker_dir = _os.path.join(kg_staging, "_done")
            _os.makedirs(marker_dir, exist_ok=True)
            _os.makedirs(kg_gold_staging, exist_ok=True)

            try:
                for key in shape_keys:
                    shape = shapes[key]
                    tid = shape.meta.tid
                    variant_id = shape.variant_id
                    if kg not in shape.applicable_kgs:
                        logger.info(
                            "TID {} variant {} skips KG {} "
                            "(not in applicable_kgs)",
                            tid, variant_id, kg,
                        )
                        continue
                    base_template = templates[key]
                    shape_sha = shape_shas[key]

                    for stratum in stratum_iter:
                        stratum_label = (
                            stratum if stratum is not None else "all"
                        )
                        label = (
                            f"TID {tid} variant {variant_id} KG {kg} "
                            f"stratum {stratum_label}"
                        )
                        # Cell name encodes variant so poi/region cells for
                        # the same (tid, kg, stratum) stage side-by-side.
                        cell_name = (
                            f"{tid.replace('.', '_')}"
                            f"__{variant_id}__{stratum_label}"
                        )
                        cell_path = _os.path.join(kg_staging, cell_name)
                        gold_cell_path = _os.path.join(
                            kg_gold_staging, cell_name
                        )
                        marker_path = _os.path.join(
                            marker_dir, f"{cell_name}.json"
                        )

                        # Resume: if the marker exists AND matches current
                        # (shape_sha, run_sha), skip the cell. Any mismatch
                        # invalidates the stale marker + its parquet dir.
                        if resume_enabled and _os.path.exists(marker_path):
                            try:
                                with open(marker_path) as mf:
                                    prev = _json.load(mf)
                            except Exception as e:
                                logger.debug(
                                    "Could not read marker {}: {}",
                                    marker_path, e,
                                )
                                prev = {}
                            if (
                                prev.get("shape_sha") == shape_sha
                                and prev.get("run_sha") == run_sha
                            ):
                                logger.info(
                                    "Skip {} (marker matches, rows={})",
                                    label, prev.get("row_count", "?"),
                                )
                                skipped_count += 1
                                continue
                            # Stale marker: wipe parquet + marker, re-run.
                            if _os.path.exists(cell_path):
                                _shutil.rmtree(cell_path)
                            if _os.path.exists(gold_cell_path):
                                _shutil.rmtree(gold_cell_path)
                            _os.remove(marker_path)

                        logger.info("Executing {}", label)
                        main_df, gold_flat_df = execute_tid_kg(
                            spark, shape, kg, run, prefix_map,
                            stratum=stratum, kg_context=kg_ctx,
                        )
                        main_df = (
                            main_df.withColumn("question", F.lit(base_template))
                            .withColumn("base_template", F.lit(base_template))
                        )
                        main_df = main_df.cache()
                        row_count = main_df.count()
                        main_df.write.mode("overwrite").parquet(cell_path)
                        # Cache+count gold so the parquet write and the
                        # marker counter share a single execution. Without
                        # the cache the count() would re-execute the gold
                        # SQL.
                        gold_flat_df = gold_flat_df.cache()
                        gold_count = gold_flat_df.count()
                        gold_flat_df.write.mode("overwrite").parquet(
                            gold_cell_path
                        )
                        gold_flat_df.unpersist()
                        main_df.unpersist()
                        # Write marker *after* successful parquet write so a
                        # crash between the two doesn't claim completion.
                        with open(marker_path, "w") as mf:
                            _json.dump(
                                {
                                    "shape_sha": shape_sha,
                                    "run_sha": run_sha,
                                    "row_count": row_count,
                                    "gold_row_count": gold_count,
                                    "tid": tid,
                                    "variant_id": variant_id,
                                    "kg": kg,
                                    "stratum": stratum_label,
                                },
                                mf,
                            )
                        staged_count += 1
                        logger.info(
                            "Staged {} rows={} gold_rows={} -> {} / {}",
                            label, row_count, gold_count,
                            cell_path, gold_cell_path,
                        )
            finally:
                close_kg_context(kg_ctx)

        if staged_count == 0 and skipped_count == 0:
            logger.warning("No (TID, KG) combinations executed.")
            return None
        logger.info(
            "Cell summary: {} newly staged, {} resumed from marker",
            staged_count, skipped_count,
        )

        # Read all staged cells back as one DataFrame.
        #   recursiveFileLookup=true — staging subdirs ("<kg>/<tid>__<stratum>")
        #     are NOT hive-style col=val partitions, so Spark would otherwise
        #     raise UNABLE_TO_INFER_SCHEMA trying to derive partition columns.
        #     This also walks past the per-KG _done/ marker dirs (no parquet
        #     inside them so they're harmless to include).
        #   mergeSchema=true — future multi-TID runs with differing column
        #     sets still union cleanly.
        combined = (
            spark.read
            .option("recursiveFileLookup", "true")
            .option("mergeSchema", "true")
            .parquet(staging_root)
        )
        # Gold is staged in a sibling tree with a different (flat) schema.
        # Read it separately and hand to the exporter so it doesn't have to
        # re-derive provenance from a JSON column.
        try:
            combined_gold = (
                spark.read
                .option("recursiveFileLookup", "true")
                .option("mergeSchema", "true")
                .parquet(gold_staging_root)
            )
        except Exception:
            # Empty staging tree (e.g. all cells produced 0 gold rows) — skip.
            combined_gold = None

        validator = BenchmarkValidator()
        validated = validator.validate(combined)

        exporter = BenchmarkExporter(output_dir=run.output_dir)
        exporter.export(
            validated,
            formats=["jsonl", "parquet"],
            gold_triples_df=combined_gold,
        )

        # Emit the fully-resolved run for reproducibility.
        _os.makedirs(run.output_dir, exist_ok=True)
        with open(Path(run.output_dir) / "plan.resolved.yaml", "w") as f:
            yaml.safe_dump(run.model_dump(), f, sort_keys=False)

        logger.info("Shape-based run complete: {}", run.run_name)
        # Staging dir is retained across runs to support resume. Set
        # MULTIGLOBEQA_SHAPE_RESUME=0 to force a full re-run, or delete
        # the directory manually.
        return validated
    finally:
        spark.stop()
