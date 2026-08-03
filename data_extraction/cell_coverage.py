from delta import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col
from sedona.spark import SedonaContext

from question_generator.config import (
    SEDONA_PACKAGES,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_MASTER,
    SPARK_TEMP_DIR,
)


def init_spark_session():
    """Create a SparkSession with Delta Lake and Sedona packages."""
    builder = (
        SparkSession.builder.appName("MultiGlobeQA_CellCoverage")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.jars.packages", SEDONA_PACKAGES)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Disable the SparkUI port (4040+) and bind driver to loopback —
        # avoids 4040→404N port-walk warnings when other Spark sessions
        # share the SLURM node, and prevents driver-binding failures on
        # nodes whose external interface refuses arbitrary binds.
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )

    if SPARK_TEMP_DIR:
        builder = builder.config("spark.local.dir", SPARK_TEMP_DIR)

    return builder.getOrCreate()


def cell_coverage_calculation(
    entity_df_path: str, output_path: str, spark: SparkSession
):
    """Build the EntityCells junction table for non-POINT geometries.

    For each non-POINT entity, explodes ST_H3CellIDs into all H3 res 7 cells
    the geometry covers, computes the res 3 parent, and marks the centroid cell.
    """
    # Stage 1: drop corrupt WKTs and persist the cleaned source to a temp
    # Delta path. Spark Catalyst will otherwise pipeline the rlike filter
    # into the same task as ST_H3CellIDs (despite the intervening
    # repartition) and the H3 call still gets evaluated for rows the
    # filter would reject — KWG has ~9 corrupt polygon WKTs (e.g. an
    # Alberta smokePlumeSnapshot with coordinate "28 2145") that crash
    # Sedona's H3Utils with ClassCastException. The write-then-read
    # boundary forces the filter to materialize first.
    #
    # Regex matches a 4+ digit INTEGER part of a coordinate. Lookbehind
    # excludes `.` so we don't match decimal precision like `35.174429`.
    # Optional `\.[0-9]+` covers `12345.678` (corrupt integer with
    # decimal). Valid lat/lon coords never exceed 180 (3 digits).
    cleaned_path = output_path.rstrip("/") + "__cleaned_src"
    (
        spark.read.format("delta")
        .load(entity_df_path)
        .filter(col("geometry_type") != "POINT")
        .filter(col("geometry").isNotNull())
        .filter(~col("geometry").rlike(r"(?<![\d.])-?[0-9]{4,}(?:\.[0-9]+)?(?![\d])"))
        .select("id", "geometry", "h3_index")
        .repartition(spark.sparkContext.defaultParallelism)
        .write.format("delta").mode("overwrite").save(cleaned_path)
    )

    relevant_entities = spark.read.format("delta").load(cleaned_path)

    cells_df = (
        relevant_entities.withColumn("geom", F.expr("ST_GeomFromWKT(geometry)"))
        .withColumn("h3_cells", F.expr("ST_H3CellIDs(geom, 7, true)"))
        .select(
            col("id").alias("entity_id"),
            F.explode("h3_cells").alias("h3_res7"),
            col("h3_index"),
        )
        .withColumn(
            "h3_res3",
            F.expr(
                "CAST(ST_H3CellIDs(ST_H3ToGeom(array(h3_res7))[0], 3, false)[0] AS STRING)"
            ),
        )
        .withColumn("is_centroid", col("h3_res7") == col("h3_index"))
        .drop("h3_index")
        .repartition("h3_res3")
    )
    cells_df.write.format("delta").mode("overwrite").save(output_path)
    DeltaTable.forPath(spark, output_path).optimize().executeZOrderBy(
        "h3_res7", "entity_id"
    )


if __name__ == "__main__":
    spark = init_spark_session()
    sedona_spark = SedonaContext.create(spark)
    cell_coverage_calculation(
        entity_df_path="data/sample_data/osmh3kg_entities_test",
        output_path="data/sample_data/osmh3kg_cell_coverage_test",
        spark=sedona_spark,
    )
