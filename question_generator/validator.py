"""
Benchmark Answer Validation Pipeline.

Validates ground truth values based on ground_truth_type-specific rules to catch
incorrect or implausible answers before they enter the benchmark.
"""


import pyspark.sql.functions as F
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType

# Earth circumference in meters (upper bound for distance)
EARTH_CIRCUMFERENCE_M = 40_075_000


class BenchmarkValidator:
    """
    Validates benchmark ground truth values against per-type rules.

    Dispatches validation by ``ground_truth_type`` when that column is
    present, and by SFID-based rules when it is not.

    Adds 'is_valid' (boolean) and 'validation_errors' (string) columns
    to the benchmark DataFrame.
    """

    def validate(self, df: DataFrame) -> DataFrame:
        """
        Apply validation rules to each row based on its ground_truth_type.

        Args:
            df: Benchmark DataFrame with at least 'ground_truth' column and
                either 'ground_truth_type' or 'SFID'/'sfid'.

        Returns:
            DataFrame with added 'is_valid' and 'validation_errors' columns.
        """
        # Determine dispatch column: prefer ground_truth_type, fall back to SFID
        has_gt_type = "ground_truth_type" in df.columns
        sfid_col = "sfid" if "sfid" in df.columns else "SFID"
        has_sfid = sfid_col in df.columns

        if not has_gt_type and not has_sfid:
            logger.warning("No ground_truth_type or SFID column found; marking all rows as valid.")
            return df.withColumn("is_valid", F.lit(True)).withColumn(
                "validation_errors", F.lit(None).cast(StringType())
            )

        # Check for error column — rows with errors are automatically invalid
        has_error_col = "error" in df.columns

        # Coerce ground_truth to STRING up-front so the per-type validators
        # (try_cast as double, startswith, contains, length, etc.) can be
        # analyzed cleanly even when the source SQL produced a complex type
        # like ARRAY<STRING> (e.g. TID 62.x's SORT_ARRAY(COLLECT_LIST(...))).
        # Without this coercion, Spark's analyzer fails the entire is_valid
        # CASE expression because one of the WHEN branches references a cast
        # that's invalid for the input type, even though that branch wouldn't
        # be selected at runtime for this row's ground_truth_type.
        if "ground_truth" in df.columns:
            df = df.withColumn("ground_truth", F.col("ground_truth").cast(StringType()))

        # Build validation expressions for each ground_truth_type
        gt = F.col("ground_truth")

        # Use try_cast for safe numeric conversion (returns NULL instead of
        # throwing on non-numeric strings like entity names).
        gt_double = F.expr("try_cast(ground_truth as double)")
        gt_long = F.expr("try_cast(ground_truth as long)")

        # --- distance_meters: numeric > 0, < Earth circumference ---
        distance_valid = (
            gt.isNotNull()
            & gt_double.isNotNull()
            & (gt_double > 0)
            & (gt_double < EARTH_CIRCUMFERENCE_M)
        )
        distance_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            gt_double.isNull(), "ground_truth is not numeric"
        ).when(
            gt_double <= 0, "distance must be > 0"
        ).when(
            gt_double >= EARTH_CIRCUMFERENCE_M, "distance exceeds Earth circumference"
        ).otherwise(None)

        # --- count: integer >= 0 ---
        count_valid = (
            gt.isNotNull()
            & gt_long.isNotNull()
            & (gt_long >= 0)
        )
        count_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            gt_long.isNull(), "ground_truth is not an integer"
        ).when(
            gt_long < 0, "count must be >= 0"
        ).otherwise(None)

        # --- boolean: "true" or "false" ---
        boolean_valid = (
            gt.isNotNull()
            & F.lower(gt.cast(StringType())).isin("true", "false")
        )
        boolean_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            ~F.lower(gt.cast(StringType())).isin("true", "false"),
            "ground_truth must be true or false",
        ).otherwise(None)

        # --- path_json: contains total_distance_m and hop_count ---
        path_valid = (
            gt.isNotNull()
            & gt.cast(StringType()).contains("total_distance_m")
            & gt.cast(StringType()).contains("hop_count")
        )
        path_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            ~gt.cast(StringType()).contains("total_distance_m"),
            "path JSON missing total_distance_m",
        ).when(
            ~gt.cast(StringType()).contains("hop_count"),
            "path JSON missing hop_count",
        ).otherwise(None)

        # --- geometry_metric: numeric >= 0 ---
        shape_valid = (
            gt.isNotNull()
            & gt_double.isNotNull()
            & (gt_double >= 0)
        )
        shape_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            gt_double.isNull(), "ground_truth is not numeric"
        ).when(
            gt_double < 0, "geometry metric must be >= 0"
        ).otherwise(None)

        # --- coordinates: non-null ---
        coord_valid = gt.isNotNull()
        coord_errors = F.when(gt.isNull(), "ground_truth is NULL").otherwise(None)

        # --- entity_reference / ranked_entity: non-null ---
        entity_valid = gt.isNotNull()
        entity_errors = F.when(gt.isNull(), "ground_truth is NULL").otherwise(None)

        # --- qualitative: non-empty string ---
        qualitative_valid = gt.isNotNull() & (F.length(gt) > 0)
        qualitative_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            F.length(gt) == 0, "qualitative ground_truth must not be empty"
        ).otherwise(None)

        # --- entity_list / category_list: non-null, starts with [ ---
        list_valid = gt.isNotNull() & gt.startswith("[")
        list_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            ~gt.startswith("["), "ground_truth must be a list (start with [)"
        ).otherwise(None)

        # --- temporal: non-null, non-empty ---
        temporal_valid = gt.isNotNull() & (F.length(gt) > 0)
        temporal_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            F.length(gt) == 0, "temporal ground_truth must not be empty"
        ).otherwise(None)

        # --- geometry_wkt: non-null, starts with WKT prefix ---
        wkt_prefixes = ("POINT", "POLYGON", "LINESTRING", "MULTI", "GEOMETRYCOLLECTION")
        wkt_check = F.lit(False)
        for prefix in wkt_prefixes:
            wkt_check = wkt_check | gt.startswith(prefix)
        geometry_wkt_valid = gt.isNotNull() & wkt_check
        geometry_wkt_errors = F.when(gt.isNull(), "ground_truth is NULL").when(
            ~wkt_check, "ground_truth must be valid WKT"
        ).otherwise(None)

        # --- grid_cell_*: non-null ---
        grid_valid = gt.isNotNull()
        grid_errors = F.when(gt.isNull(), "ground_truth is NULL").otherwise(None)

        # --- default: non-null ---
        default_valid = gt.isNotNull()
        default_errors = F.when(gt.isNull(), "ground_truth is NULL").otherwise(None)

        # Dispatch by ground_truth_type (primary) or SFID (fallback)
        if has_gt_type:
            gtt = F.col("ground_truth_type")

            is_valid_expr = (
                F.when(gtt.isin("distance_meters", "distance_kilometers"), distance_valid)
                .when(gtt.isin("count", "numeric_integer"), count_valid)
                .when(gtt == "boolean", boolean_valid)
                .when(gtt == "path_json", path_valid)
                .when(gtt == "geometry_metric", shape_valid)
                .when(gtt == "numeric", shape_valid)
                .when(gtt == "coordinates", coord_valid)
                .when(gtt.isin("entity_reference", "ranked_entity", "entity_single_name"), entity_valid)
                .when(gtt == "qualitative", qualitative_valid)
                .when(gtt.isin("entity_list", "category_list", "set_enumeration"), list_valid)
                .when(gtt == "temporal", temporal_valid)
                .when(gtt == "geometry_wkt", geometry_wkt_valid)
                .when(gtt.isin("grid_cell_id", "grid_cell_h3", "grid_cell_s2", "grid_cell_geohash"), grid_valid)
                .otherwise(default_valid)
            )

            validation_errors_expr = (
                F.when(gtt.isin("distance_meters", "distance_kilometers"), distance_errors)
                .when(gtt.isin("count", "numeric_integer"), count_errors)
                .when(gtt == "boolean", boolean_errors)
                .when(gtt == "path_json", path_errors)
                .when(gtt == "geometry_metric", shape_errors)
                .when(gtt == "numeric", shape_errors)
                .when(gtt == "coordinates", coord_errors)
                .when(gtt.isin("entity_reference", "ranked_entity", "entity_single_name"), entity_errors)
                .when(gtt == "qualitative", qualitative_errors)
                .when(gtt.isin("entity_list", "category_list", "set_enumeration"), list_errors)
                .when(gtt == "temporal", temporal_errors)
                .when(gtt == "geometry_wkt", geometry_wkt_errors)
                .when(gtt.isin("grid_cell_id", "grid_cell_h3", "grid_cell_s2", "grid_cell_geohash"), grid_errors)
                .otherwise(default_errors)
            )
        else:
            # SFID-based dispatch: no ground_truth_type column on this frame
            sfid = F.col(sfid_col)

            is_valid_expr = (
                F.when(sfid == "A", distance_valid)
                .when(sfid == "B", count_valid)
                .when(sfid.isin("C", "K"), boolean_valid)
                .when(sfid == "D", path_valid)
                .when(sfid == "L", shape_valid)
                .when(sfid == "P", coord_valid)
                .when(sfid.isin("E", "H", "F"), entity_valid)
                .when(sfid == "I", default_valid)
                .when(sfid == "J", temporal_valid)
                .when(sfid == "M", qualitative_valid)
                .otherwise(default_valid)
            )

            validation_errors_expr = (
                F.when(sfid == "A", distance_errors)
                .when(sfid == "B", count_errors)
                .when(sfid.isin("C", "K"), boolean_errors)
                .when(sfid == "D", path_errors)
                .when(sfid == "L", shape_errors)
                .when(sfid == "P", coord_errors)
                .when(sfid.isin("E", "H", "F"), entity_errors)
                .when(sfid == "I", default_errors)
                .when(sfid == "J", temporal_errors)
                .when(sfid == "M", qualitative_errors)
                .otherwise(default_errors)
            )

        # Rows with errors are always invalid
        if has_error_col:
            is_valid_expr = F.when(
                F.col("error").isNotNull(), F.lit(False)
            ).otherwise(is_valid_expr)

            validation_errors_expr = F.when(
                F.col("error").isNotNull(), F.col("error")
            ).otherwise(validation_errors_expr)

        # False-premise rows: NULL ground_truth is expected and valid.
        # Covers both explicit "false" and NULL (premise unknown/not applicable).
        if "is_true_premise" in df.columns:
            is_not_true = (F.col("is_true_premise") == "false") | F.col("is_true_premise").isNull()
            is_valid_expr = F.when(
                is_not_true & gt.isNull(), F.lit(True)
            ).otherwise(is_valid_expr)

            validation_errors_expr = F.when(
                is_not_true & gt.isNull(), F.lit(None).cast(StringType())
            ).otherwise(validation_errors_expr)

        df = df.withColumn("is_valid", is_valid_expr)
        df = df.withColumn("validation_errors", validation_errors_expr)

        return df
