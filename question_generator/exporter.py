"""
Benchmark Exporter for standardized output formats.

Provides JSONL and Parquet export with schema normalization and
question interpolation.
"""

import os

import pyspark.sql.functions as F
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

from question_generator.config import DATA_DIR

# Default output directory
OUTPUT_BENCHMARK_DIR: str = os.getenv("OUTPUT_BENCHMARK_DIR", str(DATA_DIR / "benchmark_output"))

# Canonical output schema column ordering
CANONICAL_COLUMNS = [
    "question_id",
    "tid",
    "sfid",
    "vid",
    "variant_stid",
    "question",
    "base_template",
    "ground_truth",
    "ground_truth_type",
    "entity_a_id",
    "entity_a_name",
    "entity_b_id",
    "entity_b_name",
    "entity_c_id",
    "entity_c_name",
    "entity_a_subcategory",
    "entity_b_subcategory",
    "entity_c_subcategory",
    "entity_d_id",
    "entity_d_name",
    # Event-instance column family (separate from entity_*; populated by
    # ftids whose question template uses {{event_A}}/{{event_B}} as event
    # INSTANCE references, e.g. sfid_i 74.1, 77.1, 81.1). When the per-ftid
    # SQL doesn't project these, they're added as NULL by the canonical
    # column padding below — exporter substitution falls back to the
    # placeholder fallback string.
    "event_a_id",
    "event_a_name",
    "event_b_id",
    "event_b_name",
    "parameters",
    "kg_source",
    "strata_cell",
    "is_true_premise",
    "is_valid",
    "validation_errors",
]


class BenchmarkExporter:
    """
    Export benchmark DataFrames in standardized JSONL and Parquet formats.

    Normalizes column names, interpolates question templates with entity names,
    and writes to the configured output directory.
    """

    def __init__(self, output_dir: str = OUTPUT_BENCHMARK_DIR) -> None:
        self.output_dir = output_dir

    def _standardize_schema(self, df: DataFrame) -> DataFrame:
        """
        Normalize column names to the canonical benchmark schema.

        Maps the internal column names (TID, SFID, entity_A_id, etc.) to
        lowercase canonical names and adds missing columns as NULL.

        Args:
            df: Raw benchmark DataFrame from template execution.

        Returns:
            DataFrame with canonical column names.
        """
        # Column rename mapping (source -> canonical)
        rename_map = {
            "TID": "tid",
            "SFID": "sfid",
            "Base_Template": "base_template",
            "Question_Variant": "question",
            "entity_A_id": "entity_a_id",
            "entity_A_name": "entity_a_name",
            "entity_B_id": "entity_b_id",
            "entity_B_name": "entity_b_name",
            "event_A_id": "event_a_id",
            "event_A_name": "event_a_name",
            "event_B_id": "event_b_id",
            "event_B_name": "event_b_name",
            "entity_A_category": "entity_a_category",
            "entity_B_category": "entity_b_category",
            "entity_C_id": "entity_c_id",
            "entity_C_name": "entity_c_name",
            "entity_C_category": "entity_c_category",
            "entity_A_subcategory": "entity_a_subcategory",
            "entity_B_subcategory": "entity_b_subcategory",
        }

        # Apply renames for columns that exist
        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.withColumnRenamed(old_name, new_name)

        # Add question_id if not present
        if "question_id" not in df.columns:
            df = df.withColumn("question_id", F.expr("uuid()"))

        # Ensure ground_truth is cast to string for uniform output
        if "ground_truth" in df.columns:
            df = df.withColumn("ground_truth", F.col("ground_truth").cast(StringType()))

        # Determine ground_truth_type: prefer per-TID value from YAML registry
        # (emitted by TemplateManager), fall back to SFID-level default.
        if "sfid" in df.columns:
            sfid_fallback = (
                F.when(F.col("sfid") == "A", "distance_meters")
                .when(F.col("sfid") == "B", "count")
                .when(F.col("sfid").isin("C", "K"), "boolean")
                .when(F.col("sfid") == "D", "path_json")
                .when(F.col("sfid") == "L", "geometry_metric")
                .when(F.col("sfid") == "P", "coordinates")
                .when(F.col("sfid").isin("E", "H"), "entity_reference")
                .when(F.col("sfid") == "F", "ranked_entity")
                .when(F.col("sfid") == "M", "qualitative")
                .when(F.col("sfid") == "O", "grid_cell_id")
                .when(F.col("sfid") == "G", "coordinates")
                .when(F.col("sfid") == "J", "temporal")
                .otherwise("unknown")
            )
            if "ground_truth_type" not in df.columns:
                df = df.withColumn("ground_truth_type", sfid_fallback)
            else:
                df = df.withColumn(
                    "ground_truth_type",
                    F.coalesce(F.col("ground_truth_type"), sfid_fallback),
                )

        # Add missing canonical columns as NULL
        for col_name in CANONICAL_COLUMNS:
            if col_name not in df.columns:
                df = df.withColumn(col_name, F.lit(None).cast(StringType()))

        # Add kg_source from the data if identifiable
        if "kg_source" in df.columns:
            df = df.withColumn(
                "kg_source",
                F.coalesce(F.col("kg_source"), F.lit("unknown")),
            )

        return df

    def _interpolate_questions(self, df: DataFrame) -> DataFrame:
        """
        Replace {{...}} placeholders with actual entity names and metadata.

        Supported placeholders:
          - {{entity_A}}, {{entity_B}}, {{entity_C}} — entity names
          - {{entity_A_category}}, {{entity_B_category}}, {{entity_C_category}}
            — entity subcategory / node_type (e.g. "wildfire", "hospital")
          - {{time_period}} — from the parameters column if present

        Args:
            df: DataFrame with entity name/subcategory columns.

        Returns:
            DataFrame with interpolated question text.
        """
        if "question" not in df.columns:
            return df

        # Java's Matcher (used by Spark's regexp_replace) treats `$N` in the
        # REPLACEMENT string as a backreference and `\` as an escape. When a
        # KG entity name happens to contain those characters (osmh3kg has
        # names like "Entrance - 5$"), Matcher throws "Illegal group
        # reference: group index is missing". Escape `\` then `$` so the
        # replacement is treated as literal text.
        def _lit_repl(c):
            return F.regexp_replace(
                F.regexp_replace(c, r"\\", r"\\\\"),
                r"\$",
                r"\\\$",
            )

        # Entity names. Event placeholders ({{event_A}}, {{event_B}}) get
        # their own column family (event_a_name / event_b_name) rather than
        # collapsing onto entity_a_name / entity_b_name, so a template that
        # uses BOTH placeholder families in one row (e.g. sfid_i 77.1 /
        # 81.1: "When did {{event_A}} hit {{entity_A}}?") resolves each
        # slot independently. The event_* columns are projected
        # directly by the per-ftid SQL when the slot is an event INSTANCE
        # (vs a category LABEL — those use meta.parameters substitution
        # via the executor, not column substitution here).
        #
        # Order matters: we substitute {{event_A}} BEFORE {{entity_A}}.
        # If a row has only entity_a_name (no event_a_name column), the
        # event_A pattern's coalesce falls through to its fallback string
        # and {{entity_A}} resolves normally on the next pass.
        placeholder_col_map = [
            (r"\{\{event_A\}\}", "event_a_name", "Event A"),
            (r"\{\{event_B\}\}", "event_b_name", "Event B"),
            (r"\{\{entity_A\}\}", "entity_a_name", "Entity A"),
            (r"\{\{entity_B\}\}", "entity_b_name", "Entity B"),
            (r"\{\{entity_C\}\}", "entity_c_name", "Entity C"),
            (r"\{\{entity_D\}\}", "entity_d_name", "Entity D"),
        ]
        # Entity categories (subcategory / node_type)
        placeholder_col_map += [
            (r"\{\{entity_A_category\}\}", "entity_a_subcategory", "entity A"),
            (r"\{\{entity_B_category\}\}", "entity_b_subcategory", "entity B"),
            (r"\{\{entity_C_category\}\}", "entity_c_subcategory", "entity C"),
            (r"\{\{grid_cell_ID\}\}", "grid_cell_id", "grid cell"),
        ]

        for pattern, col_name, fallback in placeholder_col_map:
            if col_name in df.columns:
                df = df.withColumn(
                    "question",
                    F.regexp_replace(
                        F.col("question"),
                        pattern,
                        _lit_repl(F.coalesce(F.col(col_name), F.lit(fallback))),
                    ),
                )

        # Time period — from a direct time_period column or JSON parameters
        if "parameters" in df.columns:
            # Build time period from period_start/period_end if time_period not set
            time_period_expr = F.coalesce(
                F.col("time_period") if "time_period" in df.columns else F.lit(None),
                F.get_json_object(F.col("parameters"), "$.time_period"),
                # Construct from period_start + period_end
                F.when(
                    F.get_json_object(F.col("parameters"), "$.period_start").isNotNull(),
                    F.concat(
                        F.get_json_object(F.col("parameters"), "$.period_start"),
                        F.lit(" to "),
                        F.get_json_object(F.col("parameters"), "$.period_end"),
                    ),
                ),
                F.lit("the time period"),
            )
            df = df.withColumn(
                "question",
                F.regexp_replace(
                    F.col("question"),
                    r"\{\{time_period\}\}",
                    _lit_repl(time_period_expr),
                ),
            )
        elif "time_period" in df.columns:
            df = df.withColumn(
                "question",
                F.regexp_replace(
                    F.col("question"),
                    r"\{\{time_period\}\}",
                    _lit_repl(F.coalesce(F.col("time_period"), F.lit("the time period"))),
                ),
            )

        # Indexed time periods ({{time_period_1}}, {{time_period_2}}) — column wins over params
        for idx, json_keys in (("1", ["time_period_1", "period_start"]),
                                ("2", ["time_period_2", "period_end"])):
            sources = []
            col_name = f"time_period_{idx}"
            if col_name in df.columns:
                sources.append(F.col(col_name).cast("string"))
            if "parameters" in df.columns:
                for k in json_keys:
                    sources.append(F.get_json_object(F.col("parameters"), f"$.{k}"))
            sources.append(F.lit(f"period {idx}"))
            df = df.withColumn(
                "question",
                F.regexp_replace(
                    F.col("question"),
                    rf"\{{\{{time_period_{idx}\}}\}}",
                    _lit_repl(F.coalesce(*sources)),
                ),
            )

        # Attribute name ({{attribute_A}}) — column wins over parameters JSON
        attr_sources = []
        if "attribute_a" in df.columns:
            attr_sources.append(F.col("attribute_a").cast("string"))
        if "parameters" in df.columns:
            attr_sources.append(F.get_json_object(F.col("parameters"), "$.attribute_key"))
        attr_sources.append(F.lit("attribute"))
        df = df.withColumn(
            "question",
            F.regexp_replace(F.col("question"), r"\{\{attribute_A\}\}", _lit_repl(F.coalesce(*attr_sources))),
        )

        # Radius/value parameter — unified coalesce: column wins over parameters
        value_sources = []
        if "radius_km" in df.columns:
            value_sources.append(F.col("radius_km").cast("string"))
        if "value" in df.columns:
            value_sources.append(F.col("value").cast("string"))
        if "parameters" in df.columns:
            value_sources.append(F.get_json_object(F.col("parameters"), "$.value"))
            value_sources.append(F.get_json_object(F.col("parameters"), "$.radius_km"))
            value_sources.append(F.get_json_object(F.col("parameters"), "$.resolution"))
        value_sources.append(F.lit("N/A"))

        df = df.withColumn(
            "question",
            F.regexp_replace(
                F.col("question"),
                r"\{\{value\}\}",
                _lit_repl(F.coalesce(*value_sources)),
            ),
        )

        # Indexed values ({{value_1}}, {{value_2}}) — column wins over parameters
        for idx in ("1", "2"):
            sources = []
            col_name = f"value_{idx}"
            if col_name in df.columns:
                sources.append(F.col(col_name).cast("string"))
            if "parameters" in df.columns:
                sources.append(F.get_json_object(F.col("parameters"), f"$.value_{idx}"))
            sources.append(F.lit("N/A"))
            df = df.withColumn(
                "question",
                F.regexp_replace(
                    F.col("question"),
                    rf"\{{\{{value_{idx}\}}\}}",
                    _lit_repl(F.coalesce(*sources)),
                ),
            )

        # Safety net: strip any remaining {{...}} parameter placeholders
        df = df.withColumn(
            "question",
            F.regexp_replace(F.col("question"), r"\{\{[a-zA-Z_]+\}\}", F.lit("")),
        )

        # Collapse nested double-quotes from entity names that contain quotes
        # e.g. 'Ristorante ""Landhaus""' → 'Ristorante "Landhaus"'
        df = df.withColumn(
            "question",
            F.regexp_replace(F.col("question"), '""', '"'),
        )

        return df

    def export(
        self,
        df: DataFrame,
        formats: list[str] | None = None,
        output_dir: str | None = None,
        append: bool = False,
        gold_triples_df: DataFrame | None = None,
    ) -> dict[str, str]:
        """
        Export benchmark DataFrame in the requested formats.

        Args:
            df: Benchmark DataFrame to export.
            formats: List of output formats ("jsonl", "parquet"). Defaults to both.
            output_dir: Override output directory.
            append: If True, append to existing output instead of overwriting.
            gold_triples_df: Optional flat gold-triples DataFrame with columns
                ``(question_id, entity_role, subject, predicate, object)``.
                When supplied, the exporter writes it directly as
                ``gold_triples.parquet`` (and ``gold_triples.jsonl``) after
                inner-joining on the surviving question_ids in ``df`` — so
                gold rows tied to dropped questions become orphans and are
                excluded. When ``None``, gold is derived instead from a
                ``gold_triples`` JSON column on ``df`` (the path direct
                exporter tests use).

        Returns:
            Dictionary mapping format names to their output file paths.
        """
        if formats is None:
            formats = ["jsonl", "parquet"]

        target_dir = output_dir or self.output_dir
        os.makedirs(target_dir, exist_ok=True)

        # Standardize and interpolate
        df = self._standardize_schema(df)
        df = self._interpolate_questions(df)

        # Materialize to fix question_id: uuid() is non-deterministic,
        # so separate reads from the lazy df would produce different IDs.
        # Caching ensures export_df and gold_triples_df share the same IDs.
        df.cache()
        try:
            total_input = df.count()

            # Hard gate: a row is either valid or it does not ship. The
            # validator has already populated is_valid upstream; absence of
            # the column (callers that skip validation) is treated as True
            # so direct exporter tests keep working.
            if "is_valid" in df.columns:
                df = df.filter(
                    F.coalesce(F.col("is_valid"), F.lit(True)) == True  # noqa: E712
                )
            after_gate = df.count()

            # Select only canonical columns that exist
            select_cols = [c for c in CANONICAL_COLUMNS if c in df.columns]
            export_df = df.select(*select_cols)

            # Deduplicate questions with identical text and ground truth
            export_df = export_df.dropDuplicates(["question", "ground_truth"])
            after_dedup = export_df.count()

            logger.info(
                "Export gate: input={} valid={} dedup={} "
                "(dropped {} invalid, {} duplicate)",
                total_input, after_gate, after_dedup,
                total_input - after_gate, after_gate - after_dedup,
            )

            # Split gold triples into a separate DataFrame. Derive from
            # export_df (post-gate, post-dedup) via inner join so every gold
            # row has a matching benchmark row. Taking gold from the pre-dedup
            # df emits orphan gold-triple rows whose question never ships.
            #
            # Two paths:
            #   * gold_triples_df supplied — flat triples keyed by entity_id.
            #     Join each role column (entity_a_id, entity_b_id, …) on the
            #     post-dedup df to attach question_id and emit one flat row
            #     per (question, role, triple).
            #   * gold_triples_df not supplied — gold comes from the
            #     ``gold_triples`` JSON column on df, the path that direct
            #     exporter-only tests use.
            gold_export_df: DataFrame | None = None
            if gold_triples_df is not None:
                # Discover role columns by convention: anything matching
                # ``entity_<role>_id`` on the benchmark side. Works for the
                # standard TID 1.1 (entity_a_id, entity_b_id) shape and any
                # future TID that follows the same naming.
                role_cols = [
                    c for c in export_df.columns
                    if c.startswith("entity_") and c.endswith("_id")
                ]
                if role_cols:
                    pieces: list[DataFrame] = []
                    for role_col in role_cols:
                        role_label = (
                            role_col.removeprefix("entity_").removesuffix("_id")
                        )
                        per_role = (
                            export_df.select("question_id", role_col)
                            .join(
                                gold_triples_df,
                                export_df[role_col]
                                == gold_triples_df["entity_id"],
                                "inner",
                            )
                            .select(
                                "question_id",
                                F.lit(role_label).alias("entity_role"),
                                "subject",
                                "predicate",
                                "object",
                            )
                        )
                        pieces.append(per_role)
                    gold_export_df = pieces[0]
                    for p in pieces[1:]:
                        gold_export_df = gold_export_df.unionByName(p)
            elif "gold_triples" in df.columns:
                gold_source = df.select("question_id", "gold_triples")
                gold_export_df = (
                    export_df.select("question_id")
                    .join(gold_source, on="question_id", how="inner")
                    .filter(F.col("gold_triples").isNotNull())
                )

            # Reduce partition count to avoid many small files on networked storage.
            num_partitions = max(1, export_df.rdd.getNumPartitions() // 10)
            export_df = export_df.coalesce(num_partitions)

            output_paths: dict[str, str] = {}
            write_mode = "append" if append else "overwrite"

            if "jsonl" in formats:
                jsonl_dir = os.path.join(target_dir, "benchmark.jsonl")
                logger.info("Writing JSONL to {}", jsonl_dir)
                export_df.coalesce(1).write.mode(write_mode).json(jsonl_dir)
                output_paths["jsonl"] = jsonl_dir

            if "parquet" in formats:
                parquet_path = os.path.join(target_dir, "benchmark.parquet")
                logger.info("Writing Parquet to {}", parquet_path)
                export_df.write.mode(write_mode).parquet(parquet_path)
                output_paths["parquet"] = parquet_path

            # Write gold triples as a separate file (joinable via question_id)
            if gold_export_df is not None:
                if "jsonl" in formats:
                    gt_jsonl = os.path.join(target_dir, "gold_triples.jsonl")
                    logger.info("Writing gold triples JSONL to {}", gt_jsonl)
                    gold_export_df.coalesce(1).write.mode(write_mode).json(gt_jsonl)
                    output_paths["gold_triples_jsonl"] = gt_jsonl
                if "parquet" in formats:
                    gt_parquet = os.path.join(target_dir, "gold_triples.parquet")
                    logger.info("Writing gold triples Parquet to {}", gt_parquet)
                    gold_export_df.write.mode(write_mode).parquet(gt_parquet)
                    output_paths["gold_triples_parquet"] = gt_parquet

            logger.info("Export complete in {} format(s)", ", ".join(formats))
            return output_paths
        finally:
            df.unpersist()
