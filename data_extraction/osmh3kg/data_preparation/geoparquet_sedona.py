"""
Convert OSM GeoParquet data with Sedona spatial processing.

Reads raw OSM GeoParquet files, adds date and geohash columns,
validates geometries, and writes to a cleaned GeoParquet output.
"""

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat,
    input_file_name,
    lit,
    regexp_extract,
    to_date,
)
from pyspark.sql.functions import expr
from sedona.spark import SedonaContext
from sedona.sql.st_functions import ST_GeoHash

from data_extraction.osmh3kg.helper.constants import (
    geo_hash_level,
    geometry_file_path,
    sedona_packages,
    osm_parquet_path,
    spark_driver_memory,
    spark_executor_memory,
    spark_master,
    spark_temp_directory,
)


def write_sedona_geoparquet() -> None:
    """Read raw OSM GeoParquet, add metadata columns, and write cleaned output."""
    # Delta-spark on classpath: geometry output is now a Delta table (not
    # plain geoparquet) so downstream kg_creation can benefit from Delta
    # OPTIMIZE compaction + statistics pushdown.
    all_packages = ",".join(
        list(sedona_packages) + ["io.delta:delta-spark_2.13:4.1.0"]
    )
    builder = (
        SparkSession.builder.master(spark_master)
        .appName("osmparquet")
        .config("spark.driver.memory", spark_driver_memory)
        .config("spark.executor.memory", spark_executor_memory)
        .config("spark.jars.packages", all_packages)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    if spark_temp_directory:
        builder = builder.config("spark.local.dir", spark_temp_directory)

    sedona = SedonaContext.create(builder.getOrCreate())

    logger.info("Reading OSM GeoParquet from {}", osm_parquet_path)
    data = (
        sedona.read.option("recursiveFileLookup", "true")
        .format("geoparquet")
        .load(osm_parquet_path)
    )
    data = data.withColumn("date", concat(lit("20260301")))
    data = data.withColumns(
        {
            "date": to_date("date", "yyyyMMdd"),
            "geohash": ST_GeoHash("geometry", geo_hash_level),
        }
    )
    data = data.select(
        col("osm_id"), col("all_tags"), col("geometry"), col("date"), col("geohash")
    )
    data = data.na.drop()
    # ST_Force2D strips NaN Z-components that some OSM sources inject into
    # otherwise-2D geometries — those NaN Z values trigger JTS
    # TopologyException "side location conflict" at predicate evaluation time.
    # ST_MakeValid repairs self-intersecting rings and other degenerate
    # topology that relate-family predicates cannot handle.
    data = data.withColumn(
        "geometry", expr("ST_MakeValid(ST_Force2D(geometry))")
    )

    logger.info("Writing cleaned Delta to {}", geometry_file_path)
    (
        data.write.mode("overwrite")
        .format("delta")
        .save(geometry_file_path)
    )

    # Delta OPTIMIZE with ZORDER BY geohash: compacts small parquet files
    # into ~1 GB chunks and sorts geometries by geohash — spatial locality
    # means ST_Intersects joins downstream can prune files aggressively on
    # geohash min/max statistics instead of scanning everything.
    logger.info("Running OPTIMIZE ZORDER BY geohash on {}", geometry_file_path)
    sedona.sql(
        f"OPTIMIZE delta.`{geometry_file_path}` ZORDER BY (geohash)"
    )


if __name__ == "__main__":
    write_sedona_geoparquet()
