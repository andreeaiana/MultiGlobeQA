"""
Configuration module for Spatial KG RAG Benchmark Generator.

All configuration is centralized here and can be overridden via environment variables.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = Path(os.getenv("MULTIGLOBEQA_DATA_DIR", str(PROJECT_ROOT / "data")))

# Delta Table paths - can be overridden via environment variables
WORLDKG_TRIPLES_PATH: str = os.getenv(
    "WORLDKG_TRIPLES_PATH", str(DATA_DIR / "worldkg" / "worldkg_table")
)
WORLDKG_NODES_PATH: str = os.getenv(
    "WORLDKG_NODES_PATH", str(DATA_DIR / "worldkg" / "worldkg_nodes")
)
OSMH3KG_NODES_PATH: str = os.getenv(
    "OSMH3KG_NODES_PATH", str(DATA_DIR / "osmh3kg" / "osmh3kg_nodes")
)
OSMH3KG_TRIPLES_PATH: str = os.getenv(
    "OSMH3KG_TRIPLES_PATH", str(DATA_DIR / "osmh3kg" / "osmh3kg_triples_subgraph")
)
KWG_ENTITIES_PATH: str = os.getenv(
    "KWG_ENTITIES_PATH", str(DATA_DIR / "kwg" / "knowwheregraph_entities")
)
KWG_TRIPLES_PATH: str = os.getenv(
    "KWG_TRIPLES_PATH", str(DATA_DIR / "kwg" / "knowwheregraph_triples")
)

# Entity category constraints
CLASS_LOOKUP_PATH: str = os.getenv(
    "CLASS_LOOKUP_PATH", str(PROJECT_ROOT / "entity_constraints" / "class_lookup.csv")
)

# Templates and Output
TEMPLATES_TSV: str = os.getenv(
    "TEMPLATES_TSV", str(DATA_DIR / "templates" / "templates_eng.tsv")
)
OUTPUT_BENCHMARK_DELTA_PATH: str = os.getenv(
    "OUTPUT_BENCHMARK_DELTA_PATH", str(DATA_DIR / "benchmark_output_delta")
)

# MLflow Configuration
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "file://" + str(PROJECT_ROOT / "logs" / "mlruns")
)
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "Spatial_KG_RAG_Benchmark")


# Spark Configuration
SPARK_MASTER: str = os.getenv("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY: str = os.getenv("SPARK_DRIVER_MEMORY", "8g")
SPARK_EXECUTOR_MEMORY: str = os.getenv("SPARK_EXECUTOR_MEMORY", "8g")
SPARK_TEMP_DIR: str | None = os.getenv("SPARK_TEMP_DIR")

# Default entities per template (override with ENTITIES_PER_TEMPLATE)
DEFAULT_ENTITIES_PER_TEMPLATE: int = int(os.getenv("ENTITIES_PER_TEMPLATE", "30"))

# Planet-scale safety rails: cap per-pool row count before temp-view register
# and filter oversized WKT strings out of gold-triples CTEs. Both overridable.
# Note: there is NO cap on the number of gold triples per entity — aggregate
# queries (e.g. "how many wildfires in region X") need complete provenance
# across all matching facts, not a sampled subset.
MAX_POOL_SIZE: int = int(os.getenv("MULTIGLOBEQA_MAX_POOL_SIZE", "500000"))
MAX_WKT_BYTES: int = int(os.getenv("MULTIGLOBEQA_MAX_WKT_BYTES", "1048576"))
GOLD_TRIPLES_UNRECOVERABLE_ERROR_THRESHOLD: float = float(
    os.getenv("MULTIGLOBEQA_GOLD_TRIPLES_ERROR_THRESHOLD", "0.05")
)

# Spark checkpointing for long-running planet-scale jobs: trim the DAG after
# each TID/strata SQL so resume + retry don't re-execute the entire pipeline
# from scratch on JVM restart. Default ON; set MULTIGLOBEQA_CHECKPOINT=false
# for short smoke runs where the extra Parquet write cost outweighs benefit.
MULTIGLOBEQA_CHECKPOINT_ENABLED: bool = os.getenv(
    "MULTIGLOBEQA_CHECKPOINT", "true"
).lower() == "true"
SPARK_CHECKPOINT_DIR: str = os.getenv(
    "SPARK_CHECKPOINT_DIR",
    str((DATA_DIR / "_spark_checkpoints").resolve()) if DATA_DIR else "spark-warehouse/_checkpoints",
)

# H3 Configuration (used by data extraction / enrichment)
H3_DENSITY_RESOLUTION: int = int(os.getenv("H3_DENSITY_RESOLUTION", "3"))

# Pathfinding Configuration
# Max bounding box span in degrees for OSMnx road network queries (~0.5° ≈ 55km)
PATHFINDING_MAX_BBOX_DEG: float = float(os.getenv("PATHFINDING_MAX_BBOX_DEG", "0.5"))

# Reference data (Natural Earth countries + World Bank income groups)
REFERENCE_DATA_PATH: Path = Path(os.getenv("REFERENCE_DATA_PATH", str(DATA_DIR / "reference")))
COUNTRIES_GEOJSON_PATH: Path = REFERENCE_DATA_PATH / "countries.geojson"
WB_INCOME_GROUPS_JSON_PATH: Path = REFERENCE_DATA_PATH / "wb_income_groups.json"
POPULATION_RASTER_PATH: str = os.getenv(
    "POPULATION_RASTER_PATH", str(DATA_DIR / "population" / "global_pop_2025.tif")
)



def strata_path(kg_label: str) -> str:
    """Path to the strata assignment Delta table for a given KG."""
    return str(DATA_DIR / kg_label / f"{kg_label}_strata")


def cells_path(kg_label: str) -> str:
    """Path to the EntityCells Delta table for a given KG."""
    return str(DATA_DIR / kg_label / f"{kg_label}_cells")

# Error handling
FAIL_ON_TEMPLATE_ERROR: bool = os.getenv("MULTIGLOBEQA_FAIL_ON_ERROR", "false").lower() == "true"

# TIDs to skip during template processing (comma-separated).
# Excluded TIDs need infrastructure not yet wired into the pipeline:
#   37, 38       — GraphFrames betweenness centrality (precompute step pending)
#   45, 46       — pathfinding via Overpass API (UDF not implemented)
#   48, 49.1-3   — OSRM/Valhalla route pre-materialization pending
#   50, 51       — same router pre-materialization
# TIDs 47.1-3 are not excluded: their linear referencing
# (ST_LineLocatePoint) is pure Sedona SQL and needs no extra infrastructure.
EXCLUDED_TIDS: frozenset[str] = frozenset(
    t.strip()
    for t in os.getenv(
        "MULTIGLOBEQA_EXCLUDED_TIDS",
        "37,38,45,46,48,49.1,49.2,49.3,50,51",
    ).split(",")
    if t.strip()
)

# TIDs to include (comma-separated). When set, ONLY these TIDs are processed.
# Empty string (default) means all TIDs are included. Useful for debugging.
# e.g. "1,13,64" to only generate distance, intersection, and grid-cell questions.
_included_raw = os.getenv("MULTIGLOBEQA_INCLUDED_TIDS", "")
INCLUDED_TIDS: frozenset[str] | None = (
    frozenset(t.strip() for t in _included_raw.split(",") if t.strip())
    if _included_raw.strip()
    else None
)

# PageRank Computation Configuration
PAGERANK_MAX_ITER: int = int(os.getenv("PAGERANK_MAX_ITER", "20"))
PAGERANK_DAMPING: float = float(os.getenv("PAGERANK_DAMPING", "0.85"))

# GraphFrames package for PySpark 4.x / Scala 2.13
GRAPHFRAMES_PACKAGE: str = os.getenv(
    "GRAPHFRAMES_PACKAGE",
    "io.graphframes:graphframes-spark4_2.13:0.10.0",
)

# Sedona Package Versions
# For PySpark 4.x, use sedona-spark-shaded-4.0_2.13 (covers Spark 4.0 and 4.1)
# GeoTools 1.8.1-33.1 is required for raster operators (RS_FromGeoTiff, RS_Value, etc.)
# Include delta-spark for Delta Lake support and GraphFrames for PageRank
SEDONA_PACKAGES: str = os.getenv(
    "SEDONA_PACKAGES",
    "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.8.1,"
    "org.datasyslab:geotools-wrapper:1.8.1-33.1,"
    # commons-collections 3.x is required by GeoTools' internal iterators;
    # Spark 4.x dropped it from its tree, and the geotools-wrapper does not
    # bundle it, so executors hit `NoClassDefFoundError:
    # org/apache/commons/collections/iterators/FilterIterator` whenever a
    # Sedona/GeoTools call walks a FilteredIterator (sfid_f shapes that
    # operate on POLYGON / GEOMETRYCOLLECTION inputs trip this on
    # worldkg + osmh3kg).
    "commons-collections:commons-collections:3.2.2,"
    "io.delta:delta-spark_2.13:4.1.0," + GRAPHFRAMES_PACKAGE,
)

# Spatial Function IDs (SFIDs) mapping to executor types
# A, B, C, G, I, J, L, M -> Sedona (spatial/geometry operations)
# H, E, P, D, F, K, O -> Standard Spark (hierarchy/index lookups)
SEDONA_SFIDS: frozenset[str] = frozenset({"A", "B", "C", "G", "I", "J", "L", "M"})
STANDARD_SPARK_SFIDS: frozenset[str] = frozenset({"H", "E", "P", "D", "F", "K", "O"})


def get_spark_packages() -> str:
    """Returns the Spark packages string for session configuration.

    Includes Sedona, GeoTools wrapper, Delta Spark, and GraphFrames.
    """
    return SEDONA_PACKAGES


def validate_config() -> dict[str, bool]:
    """
    Validates that required paths and configurations exist.

    Returns:
        Dictionary mapping config names to their validation status.
    """
    validations = {
        "DATA_DIR_EXISTS": DATA_DIR.exists(),
        "TEMPLATES_TSV_EXISTS": Path(TEMPLATES_TSV).exists(),
        "REFERENCE_DATA_DIR_EXISTS": REFERENCE_DATA_PATH.exists(),
        "COUNTRIES_GEOJSON_EXISTS": COUNTRIES_GEOJSON_PATH.exists(),
        "WB_INCOME_GROUPS_JSON_EXISTS": WB_INCOME_GROUPS_JSON_PATH.exists(),
    }
    return validations
