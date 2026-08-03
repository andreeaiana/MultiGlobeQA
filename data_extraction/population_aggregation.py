"""Aggregate a global population raster to H3 resolution-3 cells.

Reads a GeoTIFF (e.g. GHS-POP 1 km), tiles it, computes zonal sums
per H3 cell, classifies into density tiers, and writes a static
Delta table at data/reference/population_h3_res3/.

Usage:
    uv run python -m data_extraction.population_aggregation
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from loguru import logger

from question_generator.config import (
    SPARK_MASTER,
    SPARK_TEMP_DIR,
    SEDONA_PACKAGES,
    H3_DENSITY_RESOLUTION,
    REFERENCE_DATA_PATH,
)
from sedona.spark import SedonaContext
from delta.tables import DeltaTable

RASTER_PATH = "data/population/global_pop_2025.tif"
OUTPUT_PATH = str(REFERENCE_DATA_PATH / "population_h3_res3")
TILE_SIZE = 256
H3_RES = H3_DENSITY_RESOLUTION  # 3


def init_spark_session() -> SparkSession:
    """Create a SparkSession with Delta Lake and Sedona packages."""
    builder = (
        SparkSession.builder.appName("PopulationAggregation")
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
        .config("spark.sedona.join.autoBroadcastJoinThreshold", "10485760")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dsun.java2d.d3d=false -Djava.awt.headless=true",
        )
        .config("spark.jars.packages", SEDONA_PACKAGES)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    if SPARK_TEMP_DIR:
        builder = builder.config("spark.local.dir", SPARK_TEMP_DIR)

    return builder.getOrCreate()


def aggregate_population(
    raster_path: str, output_path: str, spark: SparkSession
) -> None:
    """Aggregate population raster to H3 res 3 cells with density classification."""

    logger.info("Loading raster from {}", raster_path)
    raw = spark.read.format("binaryFile").load(raster_path)
    raw.createOrReplaceTempView("raw_tif")

    logger.info("Tiling raster into {}x{} chunks", TILE_SIZE, TILE_SIZE)
    tiled = spark.sql(f"""
        SELECT tile AS rast, x, y, path
        FROM (
            SELECT RS_TileExplode(RS_FromGeoTiff(content), {TILE_SIZE}, {TILE_SIZE}) AS (x, y, tile),
                   path
            FROM raw_tif
        )
    """)

    num_partitions = spark.sparkContext.defaultParallelism
    tiled = tiled.repartition(num_partitions)
    tiled.createOrReplaceTempView("tiles")

    tile_count = spark.sql("SELECT COUNT(*) FROM tiles").collect()[0][0]
    logger.info("Created {} tiles", tile_count)

    logger.info("Computing zonal stats per H3 res {} cell", H3_RES)
    spark.sql(f"""
        WITH tile_h3 AS (
            SELECT rast,
                   CAST(h3_id AS STRING) AS h3_res3
            FROM tiles
            LATERAL VIEW explode(ST_H3CellIDs(RS_Envelope(rast), {H3_RES}, true)) AS h3_id
        ),
        tile_h3_geom AS (
            SELECT rast, h3_res3,
                   ST_H3ToGeom(array(CAST(h3_res3 AS BIGINT)))[0] AS h3_geom
            FROM tile_h3
        ),
        tile_stats AS (
            SELECT h3_res3,
                   RS_ZonalStats(rast, h3_geom, 1, 'sum', true) AS pop_sum
            FROM tile_h3_geom
        )
        SELECT h3_res3,
               SUM(pop_sum) AS population
        FROM tile_stats
        WHERE pop_sum IS NOT NULL AND pop_sum > 0
        GROUP BY h3_res3
    """).createOrReplaceTempView("pop_by_cell")

    cell_count = spark.sql("SELECT COUNT(*) FROM pop_by_cell").collect()[0][0]
    logger.info("Aggregated population for {} H3 cells", cell_count)

    logger.info("Classifying density tiers (low / medium / high)")
    pop_df = spark.table("pop_by_cell")

    quantiles = pop_df.approxQuantile("population", [0.33, 0.67], 0.05)
    low_threshold = quantiles[0]
    high_threshold = quantiles[1]
    logger.info(
        "Density thresholds — low: <= {:.0f}, medium: <= {:.0f}, high: > {:.0f}",
        low_threshold,
        high_threshold,
        high_threshold,
    )

    result = pop_df.withColumn(
        "density_class",
        F.when(F.col("population") <= low_threshold, "low")
        .when(F.col("population") <= high_threshold, "medium")
        .otherwise("high"),
    )

    logger.info("Writing to Delta table at {}", output_path)
    result.coalesce(1).write.format("delta").mode("overwrite").save(output_path)

    logger.info("Optimizing Delta table")
    DeltaTable.forPath(spark, output_path).optimize().executeCompaction()

    final = spark.read.format("delta").load(output_path)
    total = final.count()
    for cls in ["low", "medium", "high"]:
        cnt = final.filter(F.col("density_class") == cls).count()
        logger.info("  {}: {} cells ({:.1f}%)", cls, cnt, cnt / total * 100)

    logger.info("Done. {} total H3 cells written to {}", total, output_path)


if __name__ == "__main__":
    spark = init_spark_session()
    sedona_spark = SedonaContext.create(spark)
    aggregate_population(
        raster_path=RASTER_PATH,
        output_path=OUTPUT_PATH,
        spark=sedona_spark,
    )
    spark.stop()
