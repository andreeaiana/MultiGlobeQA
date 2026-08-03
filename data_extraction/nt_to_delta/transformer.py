"""Stage 2: Transform triples Delta table to entities Delta table.

This module provides PySpark transformations to convert the raw triples
produced by the Rust parser into the entity-centric format expected by
the benchmark question generator's sampler.
"""

import logging
import re
from pathlib import Path

import h3
from delta import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast, coalesce, col, first, lit
from pyspark.sql.types import StringType
from sedona.spark import SedonaContext

from data_extraction.nt_to_delta.config import NtToDeltaConfig

# Optional dependencies
from question_generator.config import (
    SEDONA_PACKAGES,
    SPARK_MASTER,
    SPARK_TEMP_DIR,
)

logger = logging.getLogger(__name__)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

# Salt bucket count for the 2-stage geometry aggregation. Bounds the
# collect_list size per task to ~max_parts_per_entity / SALT_BUCKETS, so a
# mega-multipolygon (e.g. a planet-scale admin boundary with millions of
# exploded parts) fans out across 256 tasks instead of piling onto a
# single partition. A lower count (e.g. 64) still leaves enough of one
# entity's exploded parts in a single bucket for one task to write 12.2 GiB
# against a 135 MiB median.
SALT_BUCKETS = 256


# ---------------------------------------------------------------------------
# Pure Python utility functions (used by tests and external scripts,
# NOT as Spark UDFs — the transformer uses native SQL expressions instead)
# ---------------------------------------------------------------------------


def extract_local_name(uri: str) -> str:
    """Extract the local name from a URI.

    Examples:
        http://example.org/entity/Paris -> Paris
        http://www.w3.org/1999/02/22-rdf-syntax-ns#type -> type
    """
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def extract_coords_from_wkt(wkt: str) -> tuple[float, float] | None:
    """Extract lat/lon coordinates from WKT geometry.

    Args:
        wkt: WKT geometry string (e.g., "POINT(2.3522 48.8566)")

    Returns:
        Tuple of (lat, lon) or None if parsing fails.
    """
    if not wkt:
        return None
    if "^^" in wkt:
        wkt = wkt.split("^^")[0].strip('"')

    point_match = re.search(
        r"POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)", wkt, re.IGNORECASE
    )
    if point_match:
        lon = float(point_match.group(1))
        lat = float(point_match.group(2))
        return (lat, lon)

    coord_match = re.search(r"([-\d.]+)\s+([-\d.]+)", wkt)
    if coord_match:
        lon = float(coord_match.group(1))
        lat = float(coord_match.group(2))
        return (lat, lon)

    return None


def compute_h3_index(lat: float, lon: float, resolution: int) -> str | None:
    """Compute H3 index from coordinates.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        resolution: H3 resolution (0-15).

    Returns:
        H3 index string or None if computation fails.
    """
    try:
        if lat is None or lon is None:
            return None
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return h3.latlng_to_cell(lat, lon, resolution)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Native Spark SQL expression helpers (no Python UDFs)
# ---------------------------------------------------------------------------


def _extract_local_name_expr(uri_col: F.Column) -> F.Column:
    """Extract the local name from a URI column using native Spark SQL.

    Equivalent to the ``extract_local_name`` Python helper above but runs
    entirely in the JVM, avoiding per-row Python serialization overhead.

    Examples::

        http://example.org/entity/Paris  →  Paris
        http://www.w3.org/1999/02/22-rdf-syntax-ns#type  →  type
    """
    stripped = F.regexp_replace(uri_col, "/+$", "")
    return F.when(
        F.instr(uri_col, "#") > 0,
        F.substring_index(uri_col, "#", -1),
    ).otherwise(
        F.substring_index(stripped, "/", -1),
    )


def _clean_wkt_expr(raw_wkt_col: F.Column) -> F.Column:
    """Strip RDF typed literal wrapping from a WKT column using native Spark SQL.

    Converts:
      ``"POINT(2.29 48.86)"^^<...wktLiteral>`` → ``POINT(2.29 48.86)``
      ``"<http://.../CRS84>POINT(-80 36)"^^<...wktLiteral>`` → ``POINT(-80 36)``

    KWG uses GeoSPARQL CRS-prefixed WKT (a ``<CRS-URI>`` prepended to the
    geometry). Sedona's ``ST_GeomFromWKT`` rejects the ``<...>`` prefix, so
    this helper strips it along with the datatype suffix and quotes.
    """
    # Split on '^^' and take the first part (drop the datatype annotation)
    without_datatype = F.split(raw_wkt_col, r"\^\^").getItem(0)
    # Strip leading/trailing quotes and whitespace
    unquoted = F.trim(F.regexp_replace(without_datatype, """^["']+|["']+$""", ""))
    # Drop a leading GeoSPARQL CRS URI like '<http://.../CRS84>' if present
    return F.trim(F.regexp_replace(unquoted, r"^<[^>]*>\s*", ""))


def _literal_date_expr(raw_literal_col: F.Column) -> F.Column:
    """Parse an RDF literal object column into a DATE.

    The Rust parser serialises typed literals verbatim:
      ``"1953-05-02"^^<http://www.w3.org/2001/XMLSchema#date>``
      ``"2020-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>``
      ``"2020-01-01"@en``
      ``"2020-01-01"``

    This helper strips the language/datatype tail, removes the surrounding
    quotes, truncates to the ``yyyy-MM-dd`` prefix so xsd:dateTime values
    work without a separate format path, and uses ``try_to_date`` so rows
    whose inner value is not a valid date become NULL instead of throwing
    in ANSI mode (which would abort the whole stage).
    """
    # Strip '^^<...>' or '@lang' tail
    without_tail = F.split(raw_literal_col, r"\^\^|@").getItem(0)
    # Strip surrounding quotes + whitespace
    inner = F.trim(F.regexp_replace(without_tail, """^["']+|["']+$""", ""))
    # First 10 chars = yyyy-MM-dd prefix (works for both xsd:date and xsd:dateTime)
    prefix = F.substring(inner, 1, 10)
    return F.try_to_date(prefix, "yyyy-MM-dd")


def _h3_index_expr(geometry_col: F.Column, resolution: int) -> F.Column:
    """Compute H3 cell ID from a WKT geometry column using Sedona's ST_H3CellIDs.

    Computes the index as a pure JVM expression (no Python UDF chain through
    extract_coords_from_wkt / compute_h3_index). Uses ST_GeomFromWKT to parse WKT
    and ST_H3CellIDs to compute the covering H3 cells; takes the first cell as the
    primary index.

    Returns NULL for invalid or empty geometries.
    """
    return F.expr(
        f"CAST(ST_H3CellIDs(ST_GeomFromWKT({geometry_col._jc.toString()}), {resolution}, true)[0] AS STRING)"
    )


class TriplesTransformer:
    """Transform triples Delta table to entities Delta table.

    Performs a 4-join entity assembly using the GeoSPARQL 2-hop pattern:
      entity → geometry_link_predicate → geo_node → geo:asWKT → wkt_literal

    All transformations use native Spark SQL expressions (no Python UDFs)
    for optimal performance on large datasets.
    """

    def __init__(self, spark: SparkSession, config: NtToDeltaConfig):
        self.spark = spark
        self.config = config

    def transform(
        self, triples_path: str | None = None, entities_path: str | None = None
    ) -> DataFrame:
        """Transform triples to entities.

        Args:
            triples_path: Path to triples Delta table (uses config if not provided).
            entities_path: Path for output entities Delta table (uses config if not provided).

        Returns:
            DataFrame with entity schema.
        """
        triples_path = triples_path or str(self.config.triples_delta_path)
        entities_path = entities_path or str(self.config.entities_delta_path)

        logger.info(f"Loading triples from: {triples_path}")
        triples_df = self.spark.read.format("delta").load(triples_path)

        # osmh3kg_triples is a stack of yearly OSM snapshots in one Delta
        # table (e.g. 7 yearly snapshots from 2018-01-01 to 2024-01-01,
        # ~27 B rows total — the latest snapshot alone is only ~22% of that).
        # When we don't model temporal dynamics (temporal_mode=None), keeping
        # all snapshots inflates every downstream filter, join and shuffle by
        # ~5x for nothing. Filtering to the latest snapshot drops the
        # redundant rows and replaces "soup of historical states" with
        # "OSM as of latest snapshot" — which is what a non-temporal benchmark
        # actually wants. Gated on temporal_mode is None so kwg/worldkg paths
        # (which use predicate-mode dates) are untouched.
        if "date" in triples_df.columns and self.config.temporal_mode is None:
            latest_date = triples_df.agg(F.max("date")).collect()[0][0]
            logger.info(
                f"Filtering triples to latest snapshot date: {latest_date}"
            )
            triples_df = (
                triples_df.filter(col("date") == latest_date).drop("date")
            )

            # CHECKPOINT 1: materialise the date-filtered triples to fast
            # node-local parquet so the 6 downstream predicate filters in
            # _build_entity_df read from a small ~70 GB parquet instead of
            # re-scanning the 487 GB source Delta six times. With the date
            # filter dropping ~78% of rows, the materialised table fits
            # comfortably in node-local /tmp scratch.
            #
            # sortWithinPartitions("predicate") (NOT repartition): the
            # triples table has ~30 distinct predicate values but very
            # uneven distribution — geo:asWKT alone is ~600M rows while
            # other predicates are orders of magnitude smaller. A hash
            # repartition by predicate creates ~30 non-empty partitions
            # out of 4000, one of which dominates (~5 GiB / 635M rows in
            # a single task, 3.7 min duration while other tasks idle).
            # sortWithinPartitions skips the shuffle entirely — each
            # input Spark task sorts its own rows by predicate, writes
            # one output file with row groups clustered by predicate.
            # Downstream filter(predicate == X) still gets row-group-level
            # pruning via parquet min/max stats inside each file.
            triples_dated_path = (
                Path(SPARK_TEMP_DIR.split(",")[0])
                / f"triples_dated_{self.config.kg_source}"
            )
            logger.info(
                f"Materialising date-filtered triples to: {triples_dated_path}"
            )
            (
                triples_df
                .sortWithinPartitions("predicate")
                .write.mode("overwrite")
                .parquet(str(triples_dated_path))
            )
            triples_df = self.spark.read.parquet(str(triples_dated_path))

        # Do NOT .cache() the raw triples DataFrame here.
        # The entity build path filters triples_df 14+ times (name, label,
        # geometry, category, dates, address), and for planet-scale KGs
        # (e.g. osmh3kg at ~487 GB) the cached RDD blows the Spark storage
        # fraction of the driver heap and spills every block to disk, making
        # the whole pipeline run at disk-cache speed.
        # Delta predicate pushdown + parquet column pruning + min/max file
        # skipping make the repeated filter scans cheaper than a failing
        # cache; each filter only reads the parquet files whose predicate
        # range matches.
        entities_df = self._build_entity_df(triples_df)

        logger.info(f"Writing entities to: {entities_path}")
        write_mode = "overwrite" if self.config.overwrite else "errorIfExists"
        entities_df.write.format("delta").mode(write_mode).save(entities_path)
        DeltaTable.forPath(self.spark, entities_path).optimize().executeZOrderBy(
            "geometry_type"
        )

        logger.info(f"Transformation complete: {entities_df.count()} entities created")
        return entities_df

    def _build_entity_df(self, triples_df: DataFrame) -> DataFrame:
        """Build entity-centric DataFrame using 4-join assembly.

        Steps:
          a) Seed from real entities (subjects with rdf:type)
          b) Extract labels with nameEn priority over rdfs:label
          c) Resolve geometry via 2-hop join (entity → geo_node → wkt)
          d) Resolve category via class_lookup broadcast join
          e) Final assembly with filter on name+geometry presence
        """
        # a) Seed: only subjects that have a type triple are real entities.
        # Uses config.type_predicates so kg-specific presets (e.g. osmh3kg's
        # short-form "rdf:type" vs the default full URI) take effect.
        entity_uris = (
            triples_df.filter(col("predicate").isin(self.config.type_predicates))
            .select(col("subject").alias("entity_uri"))
            .distinct()
        )

        # b) Labels: nameEn takes priority over fallback label predicates.
        name_en_df = (
            triples_df.filter(col("predicate") == self.config.name_en_predicate)
            .groupBy("subject")
            .agg(first("object").alias("name_en"))
        )

        rdfs_label_df = (
            triples_df.filter(col("predicate").isin(self.config.label_predicates))
            .groupBy("subject")
            .agg(first("object").alias("rdfs_label"))
        )

        labels_df = name_en_df.join(rdfs_label_df, "subject", "full_outer").select(
            "subject", coalesce("name_en", "rdfs_label").alias("name")
        )

        # b2) Address disambiguation: build "Name, Street Nr" or "Name (Suburb)"
        # Only for KGs with address predicates (e.g., OSMH3KG)
        if self.config.address_street_predicate:
            labels_df = self._disambiguate_names(triples_df, labels_df)

        # d) Category: broadcast join with class_lookup CSV filtered by kg_source
        category_df = self._build_category_df(triples_df, self.config.kg_source)

        # d2) Temporal: extract start_date and end_date per entity
        dates_df = self._build_dates_df(triples_df)

        # CHECKPOINT 2: entity backbone (entity_uris + labels + category + dates,
        # filtered to require name and category not-null). Materialised so the
        # geometry build runs as its own independent Spark action — bounded
        # failure radius, accurate stats for the final inner join, and the
        # filter is applied BEFORE the materialise so the parquet only stores
        # surviving entities (~30% of the entity_uris seed for OSMH3KG).
        backbone = (
            entity_uris.join(
                labels_df, entity_uris["entity_uri"] == labels_df["subject"], "left"
            )
            .drop("subject")
            .join(
                category_df, entity_uris["entity_uri"] == category_df["subject"], "left"
            )
            .drop("subject")
            .join(
                dates_df, entity_uris["entity_uri"] == dates_df["subject"], "left"
            )
            .drop("subject")
            .filter(col("name").isNotNull() & col("category").isNotNull())
        )
        backbone_path = (
            Path(SPARK_TEMP_DIR.split(",")[0])
            / f"entity_backbone_{self.config.kg_source}"
        )
        logger.info(f"Materialising entity backbone to: {backbone_path}")
        backbone.write.mode("overwrite").parquet(str(backbone_path))
        backbone = self.spark.read.parquet(str(backbone_path))

        # c) Geometry: 2-hop join via geometry node
        # Hop 1: entity_uri → geo_node
        geo_links = triples_df.filter(
            col("predicate").isin(self.config.geometry_link_predicates)
        ).select(col("subject").alias("entity_uri"), col("object").alias("geo_node"))

        # Hop 2: geo_node → raw WKT literal, cleaned via native Spark SQL
        wkt_vals = triples_df.filter(
            col("predicate").isin(self.config.geometry_predicates)
        ).select(col("subject").alias("geo_node"), col("object").alias("raw_wkt"))
        # Force ShuffleHashJoin (build hash table from geo_links, the smaller
        # 141.5 GiB side; probe with the 805 GiB wkt_vals side) instead of
        # the planner's default SortMergeJoin. SMJ has to globally sort
        # ~947 GiB by geo_node before merging, with spill into spark.local.dir
        # — catastrophic when that lives on CephFS. SHJ skips the sort
        # entirely: per shuffle partition, build a small hash table from
        # geo_links and stream wkt_vals through it, no spill, no global sort.
        geometry_parts = (
            geo_links.hint("shuffle_hash")
            .join(wkt_vals, on="geo_node", how="inner")
            .select("entity_uri", _clean_wkt_expr(col("raw_wkt")).alias("wkt_clean"))
        )

        # Aggregate exploded multi-geometry parts into one Multi* per entity.
        # ogr2ogr's -explodecollections splits MultiPolygons into individual
        # Polygons, producing multiple rows per entity_uri — rejoin them here.
        #
        # Two-stage salted collect → ST_Collect:
        #   Stage 1 groups by (entity_uri, salt) so a hot entity (e.g. a
        #     planet-scale admin boundary with thousands of exploded parts)
        #     fans out across SALT_BUCKETS sub-partitions, bounding per-task
        #     collect_list size. ST_Collect is used here rather than
        #     ST_Union_Aggr: ST_Union_Aggr is a streaming aggregate but crashes
        #     on JTS non-noded intersection errors in OverlayOp on pathological
        #     OSM vertex pairs. ST_Collect does no JTS overlay math, so that
        #     crash path cannot arise.
        #   Stage 2 merges the ≤SALT_BUCKETS partials per entity into one
        #     MultiGeometry (memory-bounded by construction).
        # rand seed is fixed so stage retries are idempotent across failures.
        # ST_SimplifyPreserveTopology(..., 1e-4) caps per-partial vertex count.
        # Without this, mega-entities (planet-scale admin boundaries with
        # millions of exploded polygon parts) produce ST_AsText WKT strings
        # that are multi-GB per row even after salting — enough to stall a
        # single task for hours and spill 100+ GiB to local disk during the
        # collect_list aggregate. 1e-4 deg ≈ 10 m tolerance, which is at
        # the edge of OSM's nominal 1-5 m accuracy and well below the
        # granularity that H3-res-7 indexing (~5 km edge) or downstream
        # spatial-join metrics care about. A tolerance of 1e-6 (about 10 cm)
        # is too fine to materially shrink mega-entity WKT.
        # PreserveTopology variant prevents self-intersection on concave
        # inputs; pays a constant factor in simplification cost vs plain
        # ST_Simplify but is the only safe option for polygon data.
        stage1 = (
            geometry_parts
            .withColumn("salt", (F.rand(seed=42) * SALT_BUCKETS).cast("int"))
            .groupBy("entity_uri", "salt")
            .agg(
                F.expr(
                    "ST_SimplifyPreserveTopology("
                    "  ST_Collect(collect_list(ST_GeomFromWKT(wkt_clean))), 1e-4)"
                ).alias("partial_geom")
            )
        )
        geometry_df = stage1.groupBy("entity_uri").agg(
            F.expr(
                "ST_AsText(ST_Collect(collect_list(partial_geom)))"
            ).alias("geometry")
        )

        # CHECKPOINT 3: materialise the geometry result. The salted ST_Collect
        # is the slowest single operation in the pipeline — every task parses
        # ~150k WKT strings into JTS Geometry objects via ST_GeomFromWKT, which
        # is JTS-CPU-bound and ~1.5 min/task at planet OSM scale. Materialising
        # forces this to complete as ONE Spark action with a parquet output;
        # the final assembly then reads from disk and any retry skips the WKT
        # parse entirely.
        geometry_path = (
            Path(SPARK_TEMP_DIR.split(",")[0])
            / f"geometry_result_{self.config.kg_source}"
        )
        logger.info(f"Materialising geometry result to: {geometry_path}")
        geometry_df.write.mode("overwrite").parquet(str(geometry_path))
        geometry_df = self.spark.read.parquet(str(geometry_path))

        # e) Final assembly: inner join backbone × geometry on entity_uri.
        # Backbone is already filtered (name + category not-null), and inner
        # join with geometry implicitly drops entities with no geometry.
        # Both inputs are materialised parquet — Spark plans this final stage
        # from accurate file footprints, and the whole final assembly runs as
        # one short Spark action that's pure compute (WKT parse → centroid →
        # H3) over ~100M rows, no shuffles.
        h3_resolution = self.config.h3_resolution
        result = (
            backbone.join(geometry_df, "entity_uri", "inner")
            .filter(col("geometry").isNotNull())
            .withColumn("geom", F.expr("ST_GeomFromWKT(geometry)"))
            # Drop entities whose parsed geometry is NULL or empty. Sedona's
            # ST_H3CellIDs crashes with NullPointerException on POINT EMPTY
            # (e.g. from KWG entities carrying "POINT EMPTY" literals).
            .filter(
                col("geom").isNotNull()
                & ~F.expr("ST_IsEmpty(geom)")
            )
            .withColumn("centroid", F.expr("ST_Centroid(geom)"))
            .select(
                _extract_local_name_expr(col("entity_uri")).alias("id"),
                col("entity_uri"),
                col("name"),
                col("geometry"),
                F.expr("GeometryType(geom)").alias("geometry_type"),
                col("wkg_class").alias("node_type"),
                col("category"),
                col("subcategory"),
                # Second guard: if the centroid itself is empty (defensive —
                # ST_Centroid of valid non-empty input shouldn't return empty,
                # but float edge cases happen), return NULL instead of
                # invoking ST_H3CellIDs which dereferences a null coordinate.
                F.expr(
                    "CASE WHEN ST_IsEmpty(centroid) THEN NULL "
                    f"ELSE CAST(ST_H3CellIDs(centroid, {h3_resolution}, false)[0] AS STRING) END"
                ).alias("h3_index"),
                F.expr(
                    "CASE WHEN ST_IsEmpty(centroid) THEN NULL "
                    "ELSE CAST(ST_H3CellIDs(centroid, 3, false)[0] AS STRING) END"
                ).alias("h3_res3"),
                col("start_date"),
                col("end_date"),
            )
        )

        return result

    def _build_dates_df(self, triples_df: DataFrame) -> DataFrame:
        """Extract (subject, start_date, end_date) based on temporal_mode config.

        - "column":    OSMH3KG quadruples — MIN/MAX of the `date` column per subject
        - "predicate": KWG triples — date literals from temporal predicates
        - None:        no temporal data, returns empty frame with NULL dates
        """
        from pyspark.sql.types import DateType, StructField, StructType

        empty_schema = StructType(
            [
                StructField("subject", StringType()),
                StructField("start_date", DateType()),
                StructField("end_date", DateType()),
            ]
        )

        mode = self.config.temporal_mode

        if mode is None:
            return self.spark.createDataFrame([], empty_schema)

        if mode == "column":
            # OSMH3KG: triples have a 4th `date` column
            if "date" not in triples_df.columns:
                logger.warning(
                    "temporal_mode='column' but triples have no 'date' column"
                )
                return self.spark.createDataFrame([], empty_schema)

            return (
                triples_df.filter(col("date").isNotNull())
                .groupBy(col("subject"))
                .agg(
                    F.min("date").alias("start_date"),
                    F.max("date").alias("end_date"),
                )
            )

        if mode == "predicate":
            # KWG: temporal data is behind multi-hop paths:
            #   entity → hasTemporalScope → ?time
            #   entity → sosa:isFeatureOfInterestOf → ?obs → sosa:phenomenonTime → ?time
            #   ?time  → time:inXSDDateTime | time:inXSDDate → date literal
            #
            # Also support direct predicates (prov:generatedAtTime) as fallback.

            # Hop 1: entity → temporal scope node (via hasTemporalScope)
            temporal_scope_preds = [
                "http://stko-kwg.geog.ucsb.edu/lod/ontology/hasTemporalScope",
            ]
            scope_df = (
                triples_df.filter(col("predicate").isin(temporal_scope_preds))
                .select(
                    col("subject").alias("entity_uri"),
                    col("object").alias("time_node"),
                )
            )

            # Hop 1 (alt): entity → sosa:isFeatureOfInterestOf → obs → sosa:phenomenonTime → time
            sosa_foi_df = (
                triples_df.filter(
                    col("predicate")
                    == "http://www.w3.org/ns/sosa/isFeatureOfInterestOf"
                )
                .select(
                    col("subject").alias("entity_uri"),
                    col("object").alias("obs_node"),
                )
            )
            sosa_time_df = (
                triples_df.filter(
                    col("predicate") == "http://www.w3.org/ns/sosa/phenomenonTime"
                )
                .select(
                    col("subject").alias("obs_node"),
                    col("object").alias("time_node"),
                )
            )
            sosa_df = sosa_foi_df.join(sosa_time_df, on="obs_node").select(
                "entity_uri", "time_node"
            )

            # Union both paths to time nodes
            all_time_links = scope_df.unionByName(sosa_df).distinct()

            # Hop 2: time node → date literal (works when hasTemporalScope
            # or sosa paths point directly to a time:Instant with inXSDDate).
            date_preds = [
                "http://www.w3.org/2006/time#inXSDDateTime",
                "http://www.w3.org/2006/time#inXSDDate",
            ]
            date_literals = (
                triples_df.filter(col("predicate").isin(date_preds))
                .select(
                    col("subject").alias("time_node"),
                    _literal_date_expr(col("object")).alias("date_val"),
                )
            )

            # Direct join: time_link → date literal (2-hop total)
            entity_dates_direct = all_time_links.join(
                date_literals, on="time_node"
            ).select(col("entity_uri").alias("subject"), col("date_val"))

            # KWG 3-hop path: hasTemporalScope → interval →
            #   fireDiscoveryTime/fireOutTime → time instant → inXSDDate.
            # The interval node doesn't carry inXSDDate itself; it has
            # begin/end predicates pointing to the actual time instants.
            begin_end_preds = (
                self.config.start_date_predicates
                + self.config.end_date_predicates
            )
            interval_to_instant = (
                triples_df.filter(col("predicate").isin(begin_end_preds))
                .select(
                    col("subject").alias("time_node"),
                    col("object").alias("instant_node"),
                )
            )
            # interval → instant → inXSDDate
            entity_dates_3hop = (
                all_time_links.join(interval_to_instant, on="time_node")
                .join(
                    date_literals.withColumnRenamed("time_node", "instant_node"),
                    on="instant_node",
                )
                .select(col("entity_uri").alias("subject"), col("date_val"))
            )

            # "Direct" predicates on entity (fireDiscoveryTime, fireOutTime)
            # may point to either a date literal or a time-instant URI.
            # Try literal parse first; when it returns NULL the object is a
            # URI that needs one more hop through date_literals.
            direct_start_raw = triples_df.filter(
                col("predicate").isin(self.config.start_date_predicates)
            ).select(col("subject"), col("object"))

            direct_end_raw = triples_df.filter(
                col("predicate").isin(self.config.end_date_predicates)
            ).select(col("subject"), col("object"))

            # Path A: object is already a date literal
            direct_start_lit = direct_start_raw.select(
                col("subject"),
                _literal_date_expr(col("object")).alias("date_val"),
            ).filter(col("date_val").isNotNull())

            direct_end_lit = direct_end_raw.select(
                col("subject"),
                _literal_date_expr(col("object")).alias("date_val"),
            ).filter(col("date_val").isNotNull())

            # Path B: object is a time-instant URI → follow to inXSDDate
            direct_start_hop = (
                direct_start_raw.join(
                    date_literals,
                    direct_start_raw["object"] == date_literals["time_node"],
                )
                .select(direct_start_raw["subject"], col("date_val"))
            )
            direct_end_hop = (
                direct_end_raw.join(
                    date_literals,
                    direct_end_raw["object"] == date_literals["time_node"],
                )
                .select(direct_end_raw["subject"], col("date_val"))
            )

            # Combine all date extraction paths
            all_dates = (
                entity_dates_direct
                .unionByName(entity_dates_3hop)
                .unionByName(direct_start_lit)
                .unionByName(direct_end_lit)
                .unionByName(direct_start_hop)
                .unionByName(direct_end_hop)
            )

            # MIN = start_date, MAX = end_date per entity
            result = all_dates.groupBy("subject").agg(
                F.min("date_val").alias("start_date"),
                F.max("date_val").alias("end_date"),
            )

            return result

        logger.warning("Unknown temporal_mode='%s', skipping date extraction", mode)
        return self.spark.createDataFrame([], empty_schema)

    def _disambiguate_names(
        self, triples_df: DataFrame, labels_df: DataFrame
    ) -> DataFrame:
        """Append address info to entities whose name is non-unique.

        Only entities whose ``name`` appears more than once get address
        enrichment.  Unique names are kept as-is.

        Priority: "Name, Street Nr" > "Name (Suburb)" > original "Name".
        """
        cfg = self.config

        # Identify non-unique names (count > 1)
        name_counts = labels_df.groupBy("name").agg(
            F.count("*").alias("_name_cnt")
        )
        labels_with_cnt = labels_df.join(name_counts, on="name", how="left")

        # Extract address components per subject
        street_df = (
            triples_df.filter(col("predicate") == cfg.address_street_predicate)
            .groupBy("subject")
            .agg(first("object").alias("addr_street"))
        )

        parts = [street_df]
        has_housenumber = cfg.address_housenumber_predicate is not None
        has_suburb = cfg.address_suburb_predicate is not None

        if has_housenumber:
            hn_df = (
                triples_df.filter(col("predicate") == cfg.address_housenumber_predicate)
                .groupBy("subject")
                .agg(first("object").alias("addr_housenumber"))
            )
            parts.append(hn_df)

        if has_suburb:
            suburb_df = (
                triples_df.filter(col("predicate") == cfg.address_suburb_predicate)
                .groupBy("subject")
                .agg(first("object").alias("addr_suburb"))
            )
            parts.append(suburb_df)

        # Join all address parts together
        addr_df = parts[0]
        for p in parts[1:]:
            addr_df = addr_df.join(p, on="subject", how="full_outer")

        # Join with labels and build disambiguated name
        enriched = labels_with_cnt.join(addr_df, on="subject", how="left")

        # Priority: "Name, Street Nr" > "Name (Suburb)" > "Name"
        street_name = F.concat(
            col("name"), lit(", "),
            col("addr_street"),
            F.when(col("addr_housenumber").isNotNull(), F.concat(lit(" "), col("addr_housenumber")))
            .otherwise(lit("")),
        )

        suburb_name = F.concat(col("name"), lit(" ("), col("addr_suburb"), lit(")"))

        # Only disambiguate when the name appears more than once
        is_duplicate = col("_name_cnt") > 1
        disambiguated = (
            F.when(
                is_duplicate & col("addr_street").isNotNull(),
                street_name,
            )
            .when(
                is_duplicate & col("addr_suburb").isNotNull(),
                suburb_name,
            )
            .otherwise(col("name"))
        )

        result = enriched.select(
            "subject",
            disambiguated.alias("name"),
        )

        return result

    def _build_category_df(self, triples_df: DataFrame, graph: str) -> DataFrame:
        """Build (subject, wkg_class, category, subcategory) via class_lookup broadcast join."""
        class_lookup_path = str(self.config.class_lookup_path)
        try:
            class_lookup_df = self.spark.read.csv(
                class_lookup_path, header=True
            ).filter(col("source") == graph)
        except Exception as e:
            logger.warning(
                f"Could not load class_lookup from {class_lookup_path}: {e}. category will be NULL."
            )
            from pyspark.sql.types import StructField, StructType

            schema = StructType(
                [
                    StructField("subject", StringType()),
                    StructField("wkg_class", StringType()),
                    StructField("category", StringType()),
                    StructField("subcategory", StringType()),
                ]
            )
            return self.spark.createDataFrame([], schema)

        category_df = (
            triples_df.filter(col("predicate").isin(self.config.type_predicates))
            .join(
                broadcast(class_lookup_df),
                triples_df["object"] == class_lookup_df["class"],
                "left",
            )
            .groupBy("subject")
            .agg(
                first(_extract_local_name_expr(col("object"))).alias("wkg_class"),
                first("category").alias("category"),
                first("key").alias("subcategory"),
            )
        )
        return category_df


if __name__ == "__main__":
    from data_extraction.nt_to_delta.config import NtToDeltaConfig

    builder = (
        SparkSession.builder.appName("PopulationAggregation")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sedona.global.index", "true")
        .config("spark.sedona.global.indextype", "rtree")
        .config("spark.sedona.join.autoBroadcastJoinThreshold", "10485760")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true -Djts.overlay=ng",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true -Djts.overlay=ng",
        )
    )
    if SPARK_TEMP_DIR:
        builder = builder.config("spark.local.dir", SPARK_TEMP_DIR)
    builder = builder.config("spark.jars.packages", SEDONA_PACKAGES)
    spark = builder.getOrCreate()
    spark = SedonaContext.create(spark)
    config = {
        "triples_delta_path": "data/sample_data/osmh3kgkg_delta",
        "entities_delta_path": "data/sample_data/test_entities",
        "name_en_predicate": "http://www.worldkg.org/schema/nameEn",
        "geometry_link_predicates": "http://www.opengis.net/ont/geosparql#hasGeometry",
        "geometry_predicates": "http://www.opengis.net/ont/geosparql#asWKT",
        "class_lookup_path": "entity_constraints/class_lookup.csv",
        "h3_resolution": 7,
        "overwrite": True,
    }
    initiated_config = NtToDeltaConfig(
        input_path=None,
        triples_delta_path=Path("data/sample_data/osmh3kg_kg_delta"),
        entities_delta_path=Path("data/sample_data/osmh3kg_entities_test"),
        name_en_predicate="name",
        overwrite=True,
        kg_source="osmh3kg",
        temporal_mode="column",
    )

    transformer_class = TriplesTransformer(spark=spark, config=initiated_config)
    transformer_class.transform()
