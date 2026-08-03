"""Register a session-time aggregated road view for osmh3kg.

Discovery context (see plan 2026-05-04-osmh3kg-road-aggregation-view.md):
- osmh3kg has 4.78M road rows; 51.4% of named roads span multiple OSM ways
- Each row is one OSM way (id 1:1 with row); geometry is MULTILINESTRING (99.4%)
- sfid_d Wave F1 SQL semantics ("which regions does named road X traverse?")
  require a single merged geometry per named road

This module materializes ``osmh3kg_roads_aggregated`` once per session by
joining ``osmh3kg_nodes`` (filtered to roads) to ``osmh3kg_strata`` for
``country_iso3`` enrichment, then GROUP BY ``(country_iso3, name, node_type)``
with ``ST_LineMerge(ST_Union_Aggr(geometry))``. Mirrors the
``raster_setup.register_population_views`` pattern.

KG scope: osmh3kg only. kwg has no road sub-category labels (only
RoadSegment/Trail buckets); worldkg coverage TBD.

Edge case: ``ST_LineMerge`` returns ``GEOMETRYCOLLECTION EMPTY`` for some
single-row aggregations. The SQL uses a CASE branch (COUNT=1 → original WKT
untouched) plus a COALESCE/NULLIF fallback on the merged path so the
unmerged geometry survives if LineMerge produces an empty collection.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger
from pyspark.sql import SparkSession


def _table_exists(spark: SparkSession, name: str) -> bool:
    """Best-effort catalog lookup; treats lookup failure as 'not present'."""
    try:
        return bool(spark.catalog.tableExists(name))
    except Exception:
        return False


def register_road_aggregation_views(
    spark: SparkSession,
    kg_label: str,
    nodes_path: Optional[str] = None,
    strata_path_override: Optional[str] = None,
) -> None:
    """Register ``<kg_label>_roads_aggregated`` as a temp view.

    Args:
        spark: Sedona-enabled SparkSession.
        kg_label: Must be ``'osmh3kg'`` for the initial release.
        nodes_path: Optional override of the nodes Delta path
            (default: ``data/{kg_label}/{kg_label}_nodes``).
        strata_path_override: Optional override of the strata Delta path
            (default: ``question_generator.config.strata_path(kg_label)``).
            Tests pass ``tmp_path`` Delta locations.

    Raises:
        NotImplementedError: kg_label is not ``'osmh3kg'``.
    """
    if kg_label != "osmh3kg":
        raise NotImplementedError(
            f"Road aggregation only supported for osmh3kg (got kg_label={kg_label!r}). "
            "kwg has no road sub-category labels; worldkg TBD."
        )

    view_name = f"{kg_label}_roads_aggregated"
    if _table_exists(spark, view_name):
        logger.info("View {} already registered; skipping", view_name)
        return

    # Resolve paths
    if nodes_path is None:
        nodes_path = f"data/{kg_label}/{kg_label}_nodes"
    if strata_path_override is None:
        from question_generator.config import strata_path
        strata_path_override = strata_path(kg_label)

    try:
        nodes = spark.read.format("delta").load(nodes_path)
        strata = spark.read.format("delta").load(strata_path_override)
    except Exception as e:
        logger.warning(
            "Cannot load Deltas for {} aggregation (nodes={}, strata={}): {} -- "
            "view {} will not be available",
            kg_label, nodes_path, strata_path_override, e, view_name,
        )
        return

    # Inline a SQL view registration so ST_Union_Aggr/ST_LineMerge can run
    # natively without DataFrame round-trips. Use temp views for the inputs.
    nodes.createOrReplaceTempView("__road_setup_nodes")
    strata.createOrReplaceTempView("__road_setup_strata")

    sql = """
    SELECT
      s.country_iso3,
      n.name,
      n.node_type,
      CASE
        WHEN COUNT(*) = 1
          THEN MIN(n.geometry)
        ELSE COALESCE(
          NULLIF(ST_AsText(ST_LineMerge(ST_Union_Aggr(ST_GeomFromWKT(n.geometry)))),
                 'GEOMETRYCOLLECTION EMPTY'),
          ST_AsText(ST_Union_Aggr(ST_GeomFromWKT(n.geometry)))
        )
      END AS geometry,
      COUNT(*) AS source_way_count,
      MIN(n.id) AS first_osm_id,
      MD5(CONCAT_WS('|', s.country_iso3, n.name, n.node_type)) AS id,
      'road_network_feature_category' AS category,
      MIN(n.entity_uri) AS entity_uri
    FROM __road_setup_nodes n
    JOIN __road_setup_strata s ON n.id = s.entity_id
    WHERE n.category = 'road_network_feature_category'
      AND n.name IS NOT NULL
      AND n.name != ''
    GROUP BY s.country_iso3, n.name, n.node_type
    """

    aggregated = spark.sql(sql)
    # Drop the helper temp views to keep the catalog clean
    spark.catalog.dropTempView("__road_setup_nodes")
    spark.catalog.dropTempView("__road_setup_strata")

    aggregated.createOrReplaceTempView(view_name)
    logger.info(
        "Registered {} from {} + {}",
        view_name, nodes_path, strata_path_override,
    )
