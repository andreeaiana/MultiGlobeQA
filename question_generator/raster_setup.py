"""Shared raster + population grid view registration.

The question-generation pipelines need population-related Spark temp views
(``pop_tiles``, ``pop_by_cell``, ``h3_cells_df``) for raster-dependent TIDs
(e.g. TID 21.1, radius population buffer). This module is the single source
of truth that every call site delegates to.

Idempotent: calling ``register_population_views`` repeatedly per session is
safe — already-present views short-circuit before any expensive raster parse.
"""

from __future__ import annotations

import os

from loguru import logger
from pyspark.sql import SparkSession

from question_generator.config import (
    POPULATION_RASTER_PATH,
    REFERENCE_DATA_PATH,
    cells_path,
)


def _table_exists(spark: SparkSession, name: str) -> bool:
    """Best-effort catalog lookup; treats lookup failure as 'not present'."""
    try:
        return bool(spark.catalog.tableExists(name))
    except Exception:
        return False


def register_population_views(spark: SparkSession, kg_label: str) -> None:
    """Register population raster + pre-aggregated grid + per-KG cells coverage.

    Registers up to three Spark temp views, each independently:
      * ``pop_tiles`` — GHS-POP GeoTIFF tiled into 256x256 chunks for
        ``RS_ZonalStats`` over arbitrary entity geometries (TID 21.1 et al.).
      * ``pop_by_cell`` — pre-aggregated H3-res-3 population grid
        (``data/reference/population_h3_res3``) for temporal/aggregate TIDs.
      * ``h3_cells_df`` — per-KG H3 cell coverage Delta (``cells_path(kg)``)
        used by population spatial joins.

    Each registration is wrapped in its own try/except so a missing/failing
    optional source does not block the others. Idempotent: if a view is
    already registered in this Spark session, it is left alone (saves the
    ~30s TIFF re-parse on subsequent calls per (session, kg)).

    Args:
        spark: Active SparkSession (Sedona-enabled for ``pop_tiles`` to be
            useful, but the registration itself does not require Sedona).
        kg_label: KG identifier used to resolve ``cells_path(kg_label)`` for
            the per-KG ``h3_cells_df`` view.

    No-ops gracefully if a source is missing — production deploys without the
    raster are valid for non-raster TIDs (logged at INFO, not WARNING).
    """
    # ------------------------------------------------------------------ #
    # pop_tiles — GHS-POP GeoTIFF
    # ------------------------------------------------------------------ #
    if _table_exists(spark, "pop_tiles"):
        # Silent no-op: same session already paid the raster-parse cost.
        pass
    else:
        raster_path = POPULATION_RASTER_PATH
        if not os.path.exists(raster_path):
            logger.info(
                "Population raster not found at {} -- raster-dependent TIDs "
                "disabled",
                raster_path,
            )
        else:
            try:
                raw = spark.read.format("binaryFile").load(raster_path)
                raw.createOrReplaceTempView("_pop_raw_tif")
                tiled = spark.sql("""
                    SELECT tile AS rast
                    FROM (
                        SELECT RS_TileExplode(RS_FromGeoTiff(content), 256, 256)
                               AS (x, y, tile)
                        FROM _pop_raw_tif
                    )
                """)
                # The binaryFile reader produces a single-row DataFrame
                # (one GeoTIFF blob), so the RS_FromGeoTiff →
                # RS_TileExplode chain materializes ALL tiles in ONE
                # partition. A downstream CROSS JOIN against pop_tiles
                # then inherits the single-partition layout and collapses
                # to a single task evaluating RS_ZonalStats sequentially
                # across every (anchor x tile) pair, which can stall a
                # raster-dependent TID for tens of minutes.
                #
                # Repartition by defaultParallelism (= spark.master's
                # local[N] thread count, e.g. 16 in our SLURM config) so
                # the zonal-stats step parallelises across tile slices.
                # The repartition is metadata-only until an action runs;
                # cost is paid once on the first downstream query.
                tiled = tiled.repartition(
                    spark.sparkContext.defaultParallelism
                )
                tiled.createOrReplaceTempView("pop_tiles")
                logger.info(
                    "Registered pop_tiles view from {} "
                    "(repartitioned to {} for parallel CROSS JOIN)",
                    raster_path,
                    spark.sparkContext.defaultParallelism,
                )
            except Exception as e:
                logger.warning(
                    "Could not load population raster: {} -- population "
                    "ranking disabled",
                    e,
                )

    # ------------------------------------------------------------------ #
    # pop_by_cell — pre-aggregated H3 population grid (KG-agnostic)
    # ------------------------------------------------------------------ #
    if not _table_exists(spark, "pop_by_cell"):
        pop_grid_path = str(REFERENCE_DATA_PATH / "population_h3_res3")
        try:
            pop_grid = spark.read.format("delta").load(pop_grid_path)
            pop_grid.createOrReplaceTempView("pop_by_cell")
            logger.info("Registered pop_by_cell view from {}", pop_grid_path)
        except Exception:
            logger.debug(
                "pop_by_cell not available -- temporal population TIDs "
                "disabled"
            )

    # ------------------------------------------------------------------ #
    # h3_cells_df — per-KG H3 coverage table (KG-specific!)
    # ------------------------------------------------------------------ #
    # NB: h3_cells_df is per-KG, so a session that has previously seen one KG
    # must re-register for a different KG. We therefore only short-circuit if
    # the view is already registered AND its source matches the requested KG.
    # Spark's catalog API doesn't expose source paths cheaply, so the simpler
    # contract is: always re-register (createOrReplaceTempView is cheap on a
    # Delta read — it's metadata-only until an action runs).
    try:
        cells_df = spark.read.format("delta").load(cells_path(kg_label))
        cells_df.createOrReplaceTempView("h3_cells_df")
        logger.info("Registered h3_cells_df view for {}", kg_label)
    except Exception:
        logger.debug("h3_cells_df not available for {}", kg_label)
