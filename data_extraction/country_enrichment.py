"""Country enrichment and strata assignment for entities.

Spatial-joins entities with country polygons, looks up World Bank income
levels, joins population density from the pre-aggregated H3 grid, and
computes the strata cell (income × density) for stratified sampling.

Usage:
    uv run python -m data_extraction.country_enrichment
"""

import pyspark.sql.functions as F
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col, lit, regexp_extract, row_number, when
from pyspark.sql.window import Window

# US-domestic data sources whose entities, when geometry falls outside any
# country polygon (offshore islands, coastal edge cases), should be
# attributed to USA. KWG-specific — these prefixes don't appear in
# osmh3kg or worldkg entity ids, so the check is a no-op there.
# Coverage rationale per prefix:
#   bphcSite                 — HRSA Bureau of Primary Health Care (US)
#   zipCodeArea              — US Census ZIP Code Tabulation Areas
#   statisticalArea          — US Census Statistical Areas
#   climateDivision          — NOAA NCEI US Climate Divisions
#   federalJudicialDistrict  — US federal court districts
#   nifcWildlandFireIncident — National Interagency Fire Center (US)
#   mtbsfire                 — Monitoring Trends in Burn Severity (US)
#   blueSkyWildfire          — BlueSky modeling system (US wildfires)
#   rxOpenFacility           — DEA-registered US pharmacies
#   roadSegment              — TIGER road segments (US Census)
_KWG_US_DOMESTIC_PREFIXES = frozenset({
    "bphcSite",
    "zipCodeArea",
    "statisticalArea",
    "climateDivision",
    "federalJudicialDistrict",
    "nifcWildlandFireIncident",
    "mtbsfire",
    "blueSkyWildfire",
    "rxOpenFacility",
    "roadSegment",
})

from delta.tables import DeltaTable
from sedona.spark import SedonaContext

from question_generator.config import (
    REFERENCE_DATA_PATH,
    SEDONA_PACKAGES,
    SPARK_MASTER,
    SPARK_TEMP_DIR,
)


def init_spark_session() -> SparkSession:
    """Create a SparkSession with Delta Lake and Sedona packages."""
    builder = (
        SparkSession.builder.appName("MultiGlobeQA_CountryEnrichment")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.local.dir", "spark-warehouse/tmp")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sedona.global.index", "true")
        .config("spark.sedona.global.indextype", "rtree")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.jars.packages", SEDONA_PACKAGES)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    if SPARK_TEMP_DIR:
        builder = builder.config("spark.local.dir", SPARK_TEMP_DIR)

    return builder.getOrCreate()


def enrich_with_strata(
    entity_df_path: str,
    population_grid_path: str,
    output_path: str,
    spark: SparkSession,
) -> None:
    """Enrich entities with country, income level, population density, and strata cell.

    Steps:
        1. Load country polygons as Sedona geometries, broadcast join with entity centroids
        2. Look up World Bank income level from country ISO3 code
        3. Join with population grid on h3_res3 for density classification
        4. Compute strata_cell = income_level × population_density
        5. Write strata assignment table to Delta
    """
    from data_extraction.reference_data.downloader import (
        load_countries_geodataframe,
        load_income_groups,
    )

    # ── 1. Load entities ─────────────────────────────────────────────
    logger.info("Loading entities from {}", entity_df_path)
    entities = (
        spark.read.format("delta")
        .load(entity_df_path)
        .select("id", "geometry", "h3_res3")
    )

    # ── 2. Load country polygons as Spark DataFrame ──────────────────
    logger.info("Loading country polygons")
    countries_gdf = load_countries_geodataframe()
    # Use the downloader-normalized `iso3` column, NOT raw ISO_A3.
    # Natural Earth encodes "-99" as a placeholder in ISO_A3 for several
    # real countries (Norway, France, Kosovo) — the actual ISO3 lives in
    # ISO_A3_EH. The downloader writes the resolved value to `iso3`.
    iso_col = "iso3" if "iso3" in countries_gdf.columns else "ISO_A3"
    countries_pdf = countries_gdf[[iso_col, "geometry"]].rename(
        columns={iso_col: "country_iso3"}
    )
    countries_pdf["country_wkt"] = countries_pdf["geometry"].apply(
        lambda g: g.wkt if g is not None else None
    )
    countries_pdf = countries_pdf.drop(columns=["geometry"])

    countries_df = broadcast(
        spark.createDataFrame(countries_pdf)
        .withColumn("country_geom", F.expr("ST_GeomFromWKT(country_wkt)"))
        .drop("country_wkt")
    )

    # ── 3. Spatial join: entity centroid within country polygon ───────
    logger.info("Spatial join: entities → countries")
    entities_with_geom = entities.withColumn(
        "entity_point", F.expr("ST_Centroid(ST_GeomFromWKT(geometry))")
    )

    entity_country = (
        entities_with_geom.join(
            countries_df,
            F.expr("ST_Contains(country_geom, entity_point)"),
            "left",
        )
        .select("id", "h3_res3", "country_iso3")
        .withColumn(
            "country_iso3",
            when(col("country_iso3").isNull(), lit("UNKNOWN")).otherwise(
                col("country_iso3")
            ),
        )
    )

    # Deduplicate: one country per entity (overlapping polygons at borders).
    # Push "-99" (Natural Earth placeholder for unresolved disputed
    # territories) and "UNKNOWN" to the bottom so a real country wins on
    # ties — otherwise a border entity that intersects e.g. Kosovo (-99)
    # AND Serbia would be tagged -99 because "-" sorts before letters.
    dedup_w = Window.partitionBy("id").orderBy(
        (col("country_iso3") == "-99").cast("int"),
        (col("country_iso3") == "UNKNOWN").cast("int"),
        col("country_iso3"),
    )
    entity_country = (
        entity_country.withColumn("_rn", row_number().over(dedup_w))
        .filter(col("_rn") == 1)
        .drop("_rn")
    )

    # Layer 1 fallback: KWG entities from US-domestic data sources whose
    # geometries fell outside the 10m country polygons (e.g. coastal
    # offshore facilities, edge-of-polygon roadSegments) get assigned USA.
    # Also catch administrativeRegion.USA.* explicitly. Harmless for
    # osmh3kg/worldkg since their entity ids never use these prefixes.
    us_prefixes = list(_KWG_US_DOMESTIC_PREFIXES)
    entity_country = entity_country.withColumn(
        "_prefix", regexp_extract(col("id"), r"^([^.]+)", 1)
    ).withColumn(
        "country_iso3",
        when(
            (col("country_iso3") == "UNKNOWN") &
            (col("_prefix").isin(us_prefixes) |
             col("id").startswith("administrativeRegion.USA.")),
            lit("USA"),
        ).otherwise(col("country_iso3")),
    ).drop("_prefix")

    # Layer 2 fallback: marine events (USGS earthquake epicentres, NOAA
    # hurricane track segments, smoke plumes) live in open water and
    # won't intersect any land country polygon. Marine Regions World
    # EEZ v12 assigns every 200NM territorial zone to a sovereign ISO3
    # (ISO_SOV1) — reusing that here attributes e.g. a Hawaiian quake
    # to USA, a Philippine Sea typhoon segment to PHL/JPN.
    # Only entities still tagged UNKNOWN after Layer 1 go through this
    # second spatial join (cheap subset).
    try:
        from data_extraction.reference_data.downloader import load_eez_geodataframe

        logger.info("Layer 2: EEZ fallback for UNKNOWN entities")
        eez_gdf = load_eez_geodataframe()
        eez_pdf = eez_gdf[["ISO_SOV1", "geometry"]].rename(
            columns={"ISO_SOV1": "eez_iso"}
        )
        eez_pdf["eez_wkt"] = eez_pdf["geometry"].apply(
            lambda g: g.wkt if g is not None else None
        )
        eez_pdf = eez_pdf.drop(columns=["geometry"]).dropna(subset=["eez_wkt"])

        eez_df = broadcast(
            spark.createDataFrame(eez_pdf)
            .withColumn("eez_geom", F.expr("ST_GeomFromWKT(eez_wkt)"))
            .drop("eez_wkt")
        )

        # Re-join UNKNOWN entities against their original geometries,
        # then spatial-join against the EEZ sovereign polygons.
        unknown_to_eez = (
            entity_country.filter(col("country_iso3") == "UNKNOWN")
            .select("id")
            .join(
                entities_with_geom.select("id", "entity_point"),
                "id",
                "left",
            )
            .join(
                eez_df,
                F.expr("ST_Contains(eez_geom, entity_point)"),
                "left",
            )
            # One EEZ per entity even when claims overlap at the margin.
            .groupBy("id")
            .agg(F.first("eez_iso", ignorenulls=True).alias("eez_iso"))
        )

        entity_country = (
            entity_country.join(unknown_to_eez, "id", "left")
            .withColumn(
                "country_iso3",
                when(
                    (col("country_iso3") == "UNKNOWN") & col("eez_iso").isNotNull(),
                    col("eez_iso"),
                ).otherwise(col("country_iso3")),
            )
            .drop("eez_iso")
        )
    except FileNotFoundError as exc:
        # EEZ file not provisioned — skip Layer 2 silently rather than
        # breaking the pipeline. Downstream only loses the marine
        # attribution refinement.
        logger.warning("Skipping Layer 2 EEZ fallback: %s", exc)

    # ── 4. Look up World Bank income level ───────────────────────────
    logger.info("Looking up World Bank income levels")
    income_map = load_income_groups()
    income_rows = [(iso3, level) for iso3, level in income_map.items()]
    income_df = broadcast(
        spark.createDataFrame(income_rows, ["country_iso3", "income_level"])
    )

    entity_income = entity_country.join(
        income_df, on="country_iso3", how="left"
    ).withColumn(
        "income_level",
        when(col("income_level").isNull(), lit("UNKNOWN")).otherwise(
            col("income_level")
        ),
    )

    # ── 5. Join with population grid for density class ───────────────
    logger.info("Joining with population grid at {}", population_grid_path)
    pop_grid = (
        spark.read.format("delta")
        .load(population_grid_path)
        .select(
            col("h3_res3").alias("pop_h3_res3"),
            "density_class",
        )
    )

    entity_strata = (
        entity_income.join(
            pop_grid,
            entity_income["h3_res3"] == pop_grid["pop_h3_res3"],
            "left",
        )
        .drop("pop_h3_res3")
        .withColumn(
            "population_density",
            when(col("density_class").isNull(), lit("unknown")).otherwise(
                col("density_class")
            ),
        )
        .drop("density_class")
    )

    # ── 6. Compute strata cell key ───────────────────────────────────
    result = entity_strata.withColumn(
        "strata_cell",
        F.concat_ws("_", col("income_level"), col("population_density")),
    ).select(
        col("id").alias("entity_id"),
        "country_iso3",
        "income_level",
        "population_density",
        "strata_cell",
    )
    # ── 7. Write to Delta + optimize ─────────────────────────────────
    logger.info("Writing strata assignment to {}", output_path)
    result.write.partitionBy("income_level").format("delta").mode("overwrite").save(
        output_path
    )

    DeltaTable.forPath(spark, output_path).optimize().executeZOrderBy(
        "country_iso3", "population_density"
    )

    # ── Summary ──────────────────────────────────────────────────────
    total = result.count()
    logger.info("Strata assignment: {} entities", total)
    result.groupBy("strata_cell").count().orderBy(F.desc("count")).show(
        20, truncate=False
    )
    result.groupBy("income_level").count().orderBy(F.desc("count")).show()
    result.groupBy("population_density").count().orderBy(F.desc("count")).show()

    logger.info("Done.")


if __name__ == "__main__":
    spark = init_spark_session()
    sedona_spark = SedonaContext.create(spark)
    enrich_with_strata(
        entity_df_path="data/sample_data/osmh3kg_entities_test",
        population_grid_path=str(REFERENCE_DATA_PATH / "population_h3_res3"),
        output_path="data/sample_data/osmh3kg_strata_test",
        spark=sedona_spark,
    )
    spark.stop()
