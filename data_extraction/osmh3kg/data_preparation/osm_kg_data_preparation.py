"""
OSM Knowledge Graph data preparation.

Creates an RDF-style Knowledge Graph from OpenStreetMap data using Apache Sedona.
The KG includes:
- Tag-based class/subclass triples
- Geometry triples (hasGeometry, asWKT)
- H3 grid spatial relations (intersects, contains, crosses, etc.)
"""

import h3_pyspark
# NOTE: delta.configure_spark_with_delta_pip in delta-spark 4.1.0 computes an
# incorrect Maven coordinate (io.delta:delta-spark_4.1_2.13:4.1.0) which does
# not exist on Maven Central. We set spark.jars.packages manually below with
# the canonical io.delta:delta-spark_2.13:4.1.0 coordinate instead.
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    concat,
    explode,
    expr,
    from_json,
    ifnull,
    lit,
)
from pyspark.sql.types import MapType, StringType
from pyspark import StorageLevel
from sedona.spark import SedonaContext
from sedona.sql.st_functions import ST_AsText

from data_extraction.osmh3kg.helper.constants import (
    geometry_file_path,
    grid_compaction,
    grid_level,
    grid_parquet_path,
    kg_output_path,
    osm_main_tags_parquet_path,
    sedona_packages,
    spark_driver_memory,
    spark_executor_memory,
    spark_master,
    spark_temp_directory,
)

# RDF predicates used in triple construction
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
GEO_HAS_GEOMETRY = "http://www.opengis.net/ont/geosparql#hasGeometry"
GEO_AS_WKT = "http://www.opengis.net/ont/geosparql#asWKT"


def _init_spark_session() -> SparkSession:
    """Initialize Spark with Delta Lake and Sedona."""
    builder = (
        SparkSession.builder.master(spark_master)
        .appName("kg_creation")
        .config("spark.driver.memory", spark_driver_memory)
        .config("spark.executor.memory", spark_executor_memory)
        .config("spark.worker.cleanup.enabled", "true")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # AQE: coalesce small shuffle partitions + split skewed ones at runtime.
        # Geoparquet files written by write_sedona_geoparquet() follow OSM
        # layer × geographic-tile layout — dense urban partitions dwarf rural
        # ones, and a static shuffle partition count can't balance both.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
        .config(
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes",
            "256MB",
        )
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB")
    )

    if spark_temp_directory:
        builder = builder.config("spark.local.dir", spark_temp_directory)

    all_packages = ",".join(
        list(sedona_packages) + ["io.delta:delta-spark_2.13:4.1.0"]
    )
    builder = builder.config("spark.jars.packages", all_packages)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _create_tag_triples(
    osm_data: DataFrame,
    sedona: SparkSession,
    tags_parquet_path: str = osm_main_tags_parquet_path,
) -> DataFrame:
    """
    Create RDF triples from OSM tag key-value pairs.

    Commonly used tags become rdf:type and rdfs:subClassOf triples.
    Other tags become direct predicate-object triples.

    Args:
        osm_data: DataFrame with osm_id, all_tags, date columns.
        sedona: Active SparkSession.
        tags_parquet_path: Path to the OSM main tags parquet file.

    Returns:
        DataFrame with subject, predicate, object, date columns.
    """
    osm_data_tags = osm_data.select(
        "osm_id",
        from_json(col("all_tags"), MapType(StringType(), StringType())),
        "date",
    )
    osm_data_tags = osm_data_tags.select("osm_id", explode("entries"), "date")

    logger.info("Reading OSM main tags from {}", tags_parquet_path)
    osm_key_value = sedona.read.format("parquet").load(tags_parquet_path)
    osm_key_value = osm_key_value.withColumn("Transform", lit(True))

    osm_joined_tags = osm_data_tags.join(
        osm_key_value, ["key", "value"], how="left"
    ).select(
        osm_data_tags.osm_id,
        osm_data_tags.key,
        osm_data_tags.value,
        osm_data_tags.date,
        ifnull(osm_key_value.Transform, lit(False)).alias("Transform"),
    )

    return (
        osm_joined_tags.select(
            col("osm_id").alias("subject"),
            lit(RDF_TYPE).alias("predicate"),
            col("value").alias("object"),
            col("date"),
        )
        .where(osm_joined_tags.Transform == True)  # noqa: E712
        .unionAll(
            osm_joined_tags.select(
                col("value").alias("subject"),
                lit(RDFS_SUBCLASS_OF).alias("predicate"),
                col("key").alias("object"),
                col("date"),
            ).where(osm_joined_tags.Transform == True)  # noqa: E712
        )
        .unionAll(
            osm_joined_tags.select(
                col("osm_id").alias("subject"),
                col("key").alias("predicate"),
                col("value").alias("object"),
                col("date"),
            ).where(osm_joined_tags.Transform == False)  # noqa: E712
        )
    )


def _create_geometry_triples(osm_data: DataFrame) -> DataFrame:
    """
    Create geometry-related RDF triples (hasGeometry, asWKT).

    Args:
        osm_data: DataFrame with osm_id, geometry, date columns.

    Returns:
        DataFrame with subject, predicate, object, date columns.
    """
    osm_data_geometry = osm_data.select("osm_id", "geometry", "date")
    return osm_data_geometry.select(
        col("osm_id").alias("subject"),
        lit(GEO_HAS_GEOMETRY).alias("predicate"),
        concat(lit("geo"), col("osm_id")).alias("object"),
        col("date"),
    ).unionAll(
        osm_data_geometry.select(
            concat(lit("geo"), col("osm_id")).alias("subject"),
            lit(GEO_AS_WKT).alias("predicate"),
            ST_AsText(col("geometry")).alias("object"),
            col("date"),
        )
    )


def _create_h3_grid_relations(
    h3_grid: DataFrame, osm_data: DataFrame
) -> DataFrame:
    """
    Create H3 grid RDF triples including spatial relations to OSM features.

    Args:
        h3_grid: DataFrame with h3_id, geometry columns.
        osm_data: DataFrame with osm_id, geometry, date columns.

    Returns:
        DataFrame with subject, predicate, object, date columns.
    """
    if grid_compaction:
        h3_grid = h3_grid.withColumn(
            "resolution", h3_pyspark.h3_get_resolution(col("h3_id"))
        )
    else:
        h3_grid = h3_grid.withColumn("resolution", lit(grid_level))

    # H3 grid geometry triples
    h3_grid_geometry = h3_grid.select(
        col("h3_id").alias("subject"),
        lit(GEO_HAS_GEOMETRY).alias("predicate"),
        concat(lit("geoH3"), "h3_id").alias("object"),
    ).unionAll(
        h3_grid.select(
            concat(lit("geoH3"), "h3_id").alias("subject"),
            lit(GEO_AS_WKT).alias("predicate"),
            ST_AsText(col("geometry")).alias("object"),
        )
    )
    h3_grid.createOrReplaceTempView("h3_grid")

    # H3 neighbor relations
    h3_grid_neighbor = (
        h3_grid.alias("leftH3")
        .join(
            h3_grid.alias("rightH3"),
            on=expr(
                "ST_INTERSECTS(leftH3.geometry, rightH3.geometry) "
                "AND leftH3.h3_id != rightH3.h3_id "
                "AND leftH3.resolution = rightH3.resolution"
            ),
        )
        .select(
            col("leftH3.h3_id").alias("subject"),
            lit("hcf:isAdjacentTo").alias("predicate"),
            col("rightH3.h3_id").alias("object"),
        )
    )

    if grid_compaction:
        # Parent-child relations for compacted grids
        h3_grid_parent = (
            h3_grid.alias("leftH3")
            .join(
                h3_grid.alias("rightH3"),
                on=expr(
                    "leftH3.h3_id != rightH3.h3_id "
                    "AND leftH3.resolution < rightH3.resolution "
                    "AND ST_INTERSECTS(leftH3.geometry, rightH3.geometry)"
                ),
            )
            .select(
                col("leftH3.h3_id").alias("subject"),
                lit("isParentCellOf").alias("predicate"),
                col("rightH3.h3_id").alias("object"),
            )
        )
        h3_grid_child = h3_grid_parent.select(
            col("object").alias("subject"),
            lit("isChildCellOf").alias("predicate"),
            col("subject").alias("object"),
        )
        h3_grid_relation = (
            h3_grid_geometry.unionAll(h3_grid_child)
            .unionAll(h3_grid_parent)
            .unionAll(h3_grid_neighbor)
        )
    else:
        h3_grid_relation = h3_grid_geometry.unionAll(h3_grid_neighbor)

    # Replicate H3 grid relations across all dates
    unique_dates = (
        osm_data.select("date").distinct().toPandas()["date"].astype("str").tolist()
    )
    h3_grid_relation = h3_grid_relation.withColumn("date", explode(lit(unique_dates)))

    return h3_grid_relation


def _create_spatial_relations(
    osm_data: DataFrame, h3_grid: DataFrame
) -> DataFrame:
    """
    Create spatial relation triples between H3 cells and OSM features.

    Computes: sfIntersects, sfContains, sfCrosses, sfEquals,
    sfOverlaps, sfTouches, sfWithin, ehCovers, ehCoveredBy.

    GeometryCollection handling: JTS's relate-family predicates
    (ST_Intersects / ST_Contains / ST_Crosses / ...) reject
    GeometryCollection inputs ("Operation does not support
    GeometryCollection arguments"). We split the input by geometry type:

    * Non-GC rows (Point, Multi*, Polygon, LineString, …) take the full
      9-predicate path unchanged.
    * GC rows are ST_Dump-exploded into their constituent primitive
      geometries, and then only sfIntersects is computed — because only
      Intersects has union-safe semantics under per-part evaluation
      ("any part intersects" ⇔ "whole intersects"). The other 8
      predicates (Contains, Within, Covers, CoveredBy, Equals, Overlaps,
      Crosses, Touches) do NOT survive piecewise decomposition, so we
      deliberately skip them for GC entities rather than emitting wrong
      triples. distinct() at the end collapses the duplicate
      (subject, sfIntersects, object) rows produced when multiple parts
      of the same GC entity intersect the same H3 cell.

    Args:
        osm_data: DataFrame with osm_id, geometry, date columns.
        h3_grid: DataFrame with h3_id, geometry columns.

    Returns:
        DataFrame with subject, predicate, object, date columns.
    """
    # ST_GeometryType returns canonical 'ST_<Type>' strings (e.g.
    # 'ST_GeometryCollection'). Do NOT use the bare `GeometryType()` —
    # its return-value casing is version-inconsistent in Sedona 1.8.1
    # and silently mismatched our 'GeometryCollection' literal, leaking
    # GC rows into the native-predicate path and crashing ST_Contains.
    is_gc = expr("ST_GeometryType(geometry) = 'ST_GeometryCollection'")
    # Vertex-count-conditional simplification. Brazil diagnostic showed the
    # JTS fat-tail cost is driven by a tiny cohort of ultra-vertex geoms
    # (distribution p99.9=1286, p99.99=385k — 7 rows with >100k vertices,
    # 784 rows with >10k vertices out of 15M). These are mostly large
    # reservoirs (natural=water), forests (natural=wood), and admin
    # polygons. Tag-based filtering can't cleanly isolate them; vertex
    # count can. Apply tolerance proportional to vertex density:
    #   >100k verts  → 1e-2 (~1 km) — 7 monsters (continent-scale
    #                   reservoirs, brazil country poly). ~1 km rim fuzz
    #                   is irrelevant for features spanning thousands of km.
    #   >10k verts   → 3e-3 (~300 m) — 777 big polys (state polygons,
    #                   major rivers/forests). Still well under H3 res 7
    #                   cell diameter (~1.2 km).
    #   else         → 1e-4 (~10 m)  — 99.99% of data; keeps POI/building
    #                   geometry pristine.
    # ST_MakeValid after simplify repairs any self-intersections introduced
    # when adjacent edges collapse below the tolerance.
    simplified_osm = osm_data.withColumn(
        "geometry",
        expr(
            "CASE "
            "WHEN ST_NPoints(geometry) > 100000 "
            "  THEN ST_MakeValid(ST_SimplifyPreserveTopology(geometry, 1e-2)) "
            "WHEN ST_NPoints(geometry) > 10000 "
            "  THEN ST_MakeValid(ST_SimplifyPreserveTopology(geometry, 3e-3)) "
            "ELSE ST_MakeValid(ST_SimplifyPreserveTopology(geometry, 1e-4)) "
            "END"
        ),
    )
    native_osm = simplified_osm.filter(~is_gc)
    gc_osm = simplified_osm.filter(is_gc).select(
        "osm_id",
        "date",
        explode(expr("ST_Dump(geometry)")).alias("geometry"),
    )

    # --- Path 1: non-GC rows, single-pass 9-predicate evaluation ---------
    # Sedona best-practice for multi-predicate spatial joins: ONE candidate
    # join + refine per predicate. Prior attempts (persist, then Delta
    # checkpoint) both failed: persist() replayed the join 9x under
    # Catalyst, and the checkpoint blew up to 905 GB because every
    # candidate row carried full h3+osm WKB. Solution: flatten all 9
    # predicate branches into a single SQL expression via
    # `array(CASE WHEN … END, …)` + `explode`. Spark plan has ONE
    # reference to `candidates`, so there is nothing for Catalyst to
    # replay. Same JTS work (N rows × 9 relate() calls), zero disk
    # materialisation.
    logger.info("Computing spatial join candidates...")
    candidates = (
        native_osm.alias("osm_data")
        .join(
            h3_grid.alias("h3_grid"),
            on=expr("ST_Intersects(h3_grid.geometry, osm_data.geometry)"),
        )
        .select(
            col("h3_grid.h3_id").alias("h3_id"),
            col("h3_grid.geometry").alias("h3_geometry"),
            col("osm_data.osm_id").alias("osm_id"),
            col("osm_data.geometry").alias("osm_geometry"),
            col("osm_data.date").alias("date"),
        )
    )

    # Intersects is unconditionally true (candidates come from the
    # ST_Intersects join). Remaining 8 predicates run via CASE WHEN; null
    # rows drop after explode. Arg order (h3_geometry, osm_geometry)
    # matches the prior per-predicate loop — semantics preserved.
    #
    # ST_Equals dropped: Sedona implements it as symDifference().isEmpty(),
    # which runs the JTS overlay engine on every (h3_cell, osm_geom) pair.
    # On continent-scale admin multipolygons (brazil states, india states)
    # this is O(n²) on vertex count and pegs tasks for hours. Semantically
    # H3 hexagon == OSM geometry is provably false by bounding-box
    # disagreement alone, so the predicate contributes zero signal.
    #
    # ST_IsValid guard dropped: ST_MakeValid ran at geometry-read time and
    # again after ST_SimplifyPreserveTopology above. Running IsValid per
    # pair re-executes PolygonTopologyAnalyzer's noding validator — pure
    # overhead with no new information. Downstream relate() calls are
    # already safe on the MakeValid output.
    predicates_array = expr(
        "array("
        "'geo:sfIntersects',"
        "CASE WHEN ST_Contains(h3_geometry, osm_geometry) THEN 'geo:sfContains' END,"
        "CASE WHEN ST_Crosses(h3_geometry, osm_geometry) THEN 'geo:sfCrosses' END,"
        "CASE WHEN ST_Overlaps(h3_geometry, osm_geometry) THEN 'geo:sfOverlaps' END,"
        "CASE WHEN ST_Touches(h3_geometry, osm_geometry) THEN 'geo:sfTouches' END,"
        "CASE WHEN ST_Within(h3_geometry, osm_geometry) THEN 'geo:sfWithin' END,"
        "CASE WHEN ST_Covers(h3_geometry, osm_geometry) THEN 'geo:ehCovers' END,"
        "CASE WHEN ST_CoveredBy(h3_geometry, osm_geometry) THEN 'geo:ehCoveredBy' END"
        ")"
    )
    native_relations = (
        candidates.withColumn("preds", predicates_array)
        .select(
            col("h3_id").alias("subject"),
            explode("preds").alias("predicate"),
            col("osm_id").alias("object"),
            col("date"),
        )
        .filter(col("predicate").isNotNull())
    )

    # --- Path 2: GC rows, exploded parts, Intersects-only -------------
    # distinct() scoped to this branch only — GC explode produces legitimate
    # dupes (one Intersects row per constituent geom of the same osm_id).
    # Native predicate branches above are already unique on
    # (h3_id, predicate, osm_id, date), so applying distinct() to the full
    # union would force a wasteful global shuffle across all triples.
    gc_intersects = (
        gc_osm.alias("osm_data")
        .join(
            h3_grid.alias("h3_grid"),
            on=expr("ST_Intersects(h3_grid.geometry, osm_data.geometry)"),
        )
        .select(
            col("h3_grid.h3_id").alias("subject"),
            lit("geo:sfIntersects").alias("predicate"),
            col("osm_data.osm_id").alias("object"),
            col("osm_data.date").alias("date"),
        )
        .distinct()
    )

    return native_relations.unionAll(gc_intersects)


def kg_creation() -> None:
    """
    Create the OSM H3 Knowledge Graph.

    Uses Apache Sedona to:
    1. Transform OSM tags into RDF-style triples
    2. Create geometry triples
    3. Generate H3 grid relations
    4. Compute spatial relations between H3 cells and OSM features
    5. Write the combined KG to Delta format
    """
    spark = _init_spark_session()
    sedona = SedonaContext.create(spark)

    logger.info("Reading OSM data from {}", geometry_file_path)
    # Geometry is now a Delta table written by write_sedona_geoparquet() with
    # ST_MakeValid(ST_Force2D(...)) applied + OPTIMIZE ZORDER BY geohash run.
    # Delta ZORDER-compacted files are already geohash-locality-sorted, so
    # we skip the hash repartition(200, osm_id) that we needed when the
    # input was raw geoparquet.
    osm_data = (
        sedona.read.format("delta")
        .load(geometry_file_path)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    logger.info("Reading H3 grid from {}", grid_parquet_path)
    h3_grid = (
        sedona.read.format("geoparquet")
        .load(grid_parquet_path)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    # Create all triple types
    logger.info("Creating tag triples...")
    tag_triples = _create_tag_triples(osm_data, sedona)

    logger.info("Creating geometry triples...")
    geometry_triples = _create_geometry_triples(osm_data)

    logger.info("Creating H3 grid relations...")
    h3_relations = _create_h3_grid_relations(h3_grid, osm_data)

    logger.info("Creating spatial relations...")
    spatial_relations = _create_spatial_relations(osm_data, h3_grid)

    # Combine all triple types.
    # Final repartition(400) before Delta write: unionAll inherits the max
    # partition count from all branches (each with independent skew),
    # producing lopsided output files. Hash-repartition on (subject,
    # predicate) gives uniformly-sized Delta files and balanced write tasks.
    osm_kg_data = (
        tag_triples.unionAll(geometry_triples)
        .unionAll(spatial_relations)
        .unionAll(h3_relations)
        .repartition(400, "subject", "predicate")
    )

    logger.info("Writing KG to {}", kg_output_path)
    osm_kg_data.write.mode("overwrite").format("delta").save(kg_output_path)

    # OPTIMIZE ZORDER BY (subject, predicate): compacts small parquet files
    # (40+ tasks each writing one file = many small files) into larger
    # ones, and co-locates rows with the same subject/predicate so
    # downstream entity lookups scan fewer files.
    logger.info("Running OPTIMIZE ZORDER on {}", kg_output_path)
    sedona.sql(
        f"OPTIMIZE delta.`{kg_output_path}` ZORDER BY (subject, predicate)"
    )
    logger.info("KG creation complete.")


if __name__ == "__main__":
    kg_creation()
