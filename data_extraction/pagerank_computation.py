"""Standalone PageRank computation for spatial knowledge graphs.

Reads the raw triples Delta table for each KG, builds an entity-to-entity
edge graph, runs PageRank (igraph for graphs that fit in memory, GraphFrames
for planet-scale distributed graphs), and writes the results as a sibling
Delta table.

Usage (run one KG):
    python -m data_extraction.pagerank_computation --kg worldkg

Usage (run all three KGs):
    python -m data_extraction.pagerank_computation

The script prefers igraph (C-based, fast) when the graph fits in memory.
Falls back to GraphFrames (Spark-distributed) for very large graphs.
"""

import argparse
import logging
import os
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from question_generator.config import (
    KWG_ENTITIES_PATH,
    OSMH3KG_NODES_PATH,
    PAGERANK_DAMPING,
    PAGERANK_MAX_ITER,
    SEDONA_PACKAGES,
    SPARK_DRIVER_MEMORY,
    SPARK_EXECUTOR_MEMORY,
    SPARK_MASTER,
    SPARK_TEMP_DIR,
    WORLDKG_NODES_PATH,
    WORLDKG_TRIPLES_PATH,
)

logger = logging.getLogger(__name__)

# igraph is used for graphs below this edge count (fast, single-machine).
# Above this threshold, GraphFrames (distributed) is used.
# Default 500M edges — fits comfortably in ~130 GB RAM.
PAGERANK_IGRAPH_MAX_EDGES: int = int(
    os.getenv("PAGERANK_IGRAPH_MAX_EDGES", "500000000")
)

# ---------------------------------------------------------------------------
# KG registry — maps short name to (triples_path, entities_path)
# ---------------------------------------------------------------------------
_KG_REGISTRY: dict[str, dict[str, str]] = {
    "worldkg": {
        "triples_path": WORLDKG_TRIPLES_PATH,
        "entities_path": WORLDKG_NODES_PATH,
        "pagerank_path": str(
            __import__("pathlib").Path(WORLDKG_NODES_PATH).parent / "worldkg_pagerank"
        ),
    },
    "osmh3kg": {
        "triples_path": str(
            __import__("pathlib").Path(OSMH3KG_NODES_PATH).parent / "osmh3kg_triples"
        ),
        "entities_path": OSMH3KG_NODES_PATH,
        "pagerank_path": str(
            __import__("pathlib").Path(OSMH3KG_NODES_PATH).parent / "osmh3kg_pagerank"
        ),
    },
    "kwg": {
        "triples_path": str(
            __import__("pathlib").Path(KWG_ENTITIES_PATH).parent / "knowwheregraph_triples"
        ),
        "entities_path": KWG_ENTITIES_PATH,
        "pagerank_path": str(
            __import__("pathlib").Path(KWG_ENTITIES_PATH).parent / "knowwheregraph_pagerank"
        ),
    },
}


# ---------------------------------------------------------------------------
# Spark initialisation
# ---------------------------------------------------------------------------


def init_spark() -> SparkSession:
    """Create a SparkSession with Delta Lake and GraphFrames packages."""
    builder = (
        SparkSession.builder.appName("MultiGlobeQA_PageRank")
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # GraphFrames/GraphX checkpoint support — checkpointing every 2 iterations
        # realigns vertex partition indexes.
        .config("spark.graphx.pregel.checkpointInterval", "2")
        .config("spark.jars.packages", SEDONA_PACKAGES)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    if SPARK_TEMP_DIR:
        builder = builder.config("spark.local.dir", SPARK_TEMP_DIR)

    spark = builder.getOrCreate()
    spark.sparkContext.setCheckpointDir("spark-warehouse/checkpoints")
    return spark


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_local_name_expr(col: F.Column) -> F.Column:
    """Extract local name from a URI using native Spark SQL."""
    stripped = F.regexp_replace(col, "/+$", "")
    return F.when(
        F.instr(col, "#") > 0,
        F.substring_index(col, "#", -1),
    ).otherwise(
        F.substring_index(stripped, "/", -1),
    )


def _build_edges(spark: SparkSession, triples_path: str) -> DataFrame | None:
    """Load triples and extract entity-to-entity edges.

    Returns:
        DataFrame[src, dst] of distinct edges, or None if empty/unreachable.
    """
    try:
        triples_df = spark.read.format("delta").load(triples_path)
    except Exception as exc:
        logger.warning("Could not load triples from %s: %s", triples_path, exc)
        return None

    if triples_df.limit(1).count() == 0:
        logger.warning("Triples table at %s is empty — skipping PageRank.", triples_path)
        return None

    edges_df = (
        triples_df.select(
            _extract_local_name_expr(F.col("subject")).alias("src"),
            _extract_local_name_expr(F.col("object")).alias("dst"),
        )
        .filter((F.col("src") != "") & (F.col("dst") != ""))
        .distinct()
    )
    return edges_df


# ---------------------------------------------------------------------------
# igraph-based PageRank (fast, single-machine)
# ---------------------------------------------------------------------------


def _compute_pagerank_igraph(
    spark: SparkSession,
    edges_df: DataFrame,
    damping: float = PAGERANK_DAMPING,
) -> DataFrame:
    """Compute PageRank using igraph (C-based, very fast).

    Collects edges to the driver, builds an igraph graph, runs PageRank,
    and returns a Spark DataFrame with (id, pagerank).
    """
    import igraph as ig

    logger.info("Collecting edges to driver for igraph PageRank...")
    edges_local = edges_df.select("src", "dst").collect()
    logger.info("Collected %d edges", len(edges_local))

    # Build vertex mapping (igraph needs integer vertex IDs)
    vertex_set: set[str] = set()
    for row in edges_local:
        vertex_set.add(row["src"])
        vertex_set.add(row["dst"])

    vertex_list = sorted(vertex_set)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    logger.info("Graph has %d vertices", len(vertex_list))

    # Build igraph
    edge_tuples = [(vertex_to_idx[r["src"]], vertex_to_idx[r["dst"]]) for r in edges_local]
    g = ig.Graph(n=len(vertex_list), edges=edge_tuples, directed=True)

    logger.info("Running igraph PageRank (damping=%.2f)...", damping)
    scores = g.pagerank(damping=damping)
    logger.info("igraph PageRank complete")

    # Convert back to Spark DataFrame
    result_rows = [(vertex_list[i], float(scores[i])) for i in range(len(vertex_list))]
    schema = StructType(
        [
            StructField("id", StringType(), False),
            StructField("pagerank", DoubleType(), False),
        ]
    )
    return spark.createDataFrame(result_rows, schema)


# ---------------------------------------------------------------------------
# GraphFrames-based PageRank (distributed, for planet-scale)
# ---------------------------------------------------------------------------


def _compute_pagerank_graphframes(
    spark: SparkSession,
    edges_df: DataFrame,
    reset_probability: float = 1.0 - PAGERANK_DAMPING,
    max_iter: int = PAGERANK_MAX_ITER,
) -> DataFrame:
    """Compute PageRank using GraphFrames (Spark-distributed)."""
    vertices_df = (
        edges_df.select(F.col("src").alias("id"))
        .union(edges_df.select(F.col("dst").alias("id")))
        .distinct()
    )

    num_partitions = max(spark.sparkContext.defaultParallelism, 8)
    vertices_df = vertices_df.repartition(num_partitions, "id")
    edges_df = edges_df.repartition(num_partitions, "src")
    edges_df.cache()

    try:
        from graphframes import GraphFrame  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "graphframes Python package not found. "
            "Install with: pip install graphframes-py==0.10.0"
        ) from exc

    logger.info("Running GraphFrames PageRank (maxIter=%d)...", max_iter)
    graph = GraphFrame(vertices_df, edges_df)
    results = graph.pageRank(resetProbability=reset_probability, maxIter=max_iter)

    edges_df.unpersist()

    return results.vertices.select(
        F.col("id"),
        F.col("pageRank").cast(DoubleType()).alias("pagerank"),
    )


# ---------------------------------------------------------------------------
# Unified PageRank entry point
# ---------------------------------------------------------------------------


def compute_pagerank(
    spark: SparkSession,
    triples_path: str,
    damping: float = PAGERANK_DAMPING,
    max_iter: int = PAGERANK_MAX_ITER,
) -> DataFrame | None:
    """Compute PageRank, auto-selecting igraph or GraphFrames.

    Uses igraph (fast, single-machine) for graphs below PAGERANK_IGRAPH_MAX_EDGES.
    Falls back to GraphFrames (distributed) for larger graphs.

    Args:
        spark: Active SparkSession.
        triples_path: Path to the raw triples Delta table.
        damping: PageRank damping factor (default 0.85).
        max_iter: Maximum iterations (GraphFrames only).

    Returns:
        DataFrame with columns (id STRING, pagerank DOUBLE), or None.
    """
    edges_df = _build_edges(spark, triples_path)
    if edges_df is None:
        return None

    edge_count = edges_df.count()
    logger.info("Edge count: %d (igraph threshold: %d)", edge_count, PAGERANK_IGRAPH_MAX_EDGES)

    if edge_count <= PAGERANK_IGRAPH_MAX_EDGES:
        try:
            import igraph  # noqa: F401

            logger.info("Using igraph for PageRank (graph fits in memory)")
            return _compute_pagerank_igraph(spark, edges_df, damping=damping)
        except ImportError:
            logger.warning(
                "igraph not installed — falling back to GraphFrames. "
                "Install with: pip install igraph"
            )

    logger.info("Using GraphFrames for PageRank (distributed)")
    return _compute_pagerank_graphframes(
        spark, edges_df, reset_probability=1.0 - damping, max_iter=max_iter
    )


# ---------------------------------------------------------------------------
# Delta write helper
# ---------------------------------------------------------------------------


def write_pagerank_table(pagerank_df: DataFrame, pagerank_path: str) -> None:
    """Write PageRank scores to a standalone Delta table (overwrite)."""
    pagerank_df.write.format("delta").mode("overwrite").save(pagerank_path)
    logger.info("PageRank written to %s", pagerank_path)


# ---------------------------------------------------------------------------
# Per-KG orchestration
# ---------------------------------------------------------------------------


def run_pagerank_for_kg(spark: SparkSession, kg_name: str) -> bool:
    """Run the full PageRank pipeline for a single KG.

    Returns:
        True on success, False if skipped or failed.
    """
    if kg_name not in _KG_REGISTRY:
        logger.error("Unknown KG '%s'. Valid options: %s", kg_name, list(_KG_REGISTRY.keys()))
        return False

    cfg = _KG_REGISTRY[kg_name]
    triples_path = cfg["triples_path"]
    pagerank_path = cfg["pagerank_path"]

    logger.info("=== PageRank for %s ===", kg_name)
    logger.info("  Triples : %s", triples_path)
    logger.info("  Output  : %s", pagerank_path)

    pagerank_df = compute_pagerank(spark, triples_path)
    if pagerank_df is None:
        logger.warning("PageRank skipped for %s (no data).", kg_name)
        return False

    write_pagerank_table(pagerank_df, pagerank_path)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute PageRank for spatial KG entities and write back to Delta tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kg",
        choices=list(_KG_REGISTRY.keys()),
        nargs="+",
        default=list(_KG_REGISTRY.keys()),
        help="Which knowledge graph(s) to process. Defaults to all three.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=PAGERANK_MAX_ITER,
        help="Maximum PageRank iterations (GraphFrames only).",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=PAGERANK_DAMPING,
        help="PageRank damping factor.",
    )
    parser.add_argument(
        "--force-graphframes",
        action="store_true",
        help="Force GraphFrames even if graph fits in memory.",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = build_parser().parse_args()

    if args.force_graphframes:
        global PAGERANK_IGRAPH_MAX_EDGES
        PAGERANK_IGRAPH_MAX_EDGES = 0
        logger.info("Forcing GraphFrames mode (igraph disabled)")

    spark = init_spark()
    logger.info("Spark session started: %s", spark.sparkContext.appName)

    success_count = 0
    for kg_name in args.kg:
        ok = run_pagerank_for_kg(spark, kg_name)
        if ok:
            success_count += 1

    spark.stop()

    if success_count == 0 and args.kg:
        logger.error("All KGs failed or were skipped.")
        return 1

    logger.info("PageRank computation complete: %d/%d KGs processed.", success_count, len(args.kg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
