"""
Configuration constants for OSM H3 Knowledge Graph generation.

All values can be overridden via environment variables.
"""

import os
from pathlib import Path

# Base paths — project-relative defaults instead of hardcoded /ceph/ paths
_HELPER_DIR = Path(__file__).resolve().parent
_DATA_PREP_DIR = _HELPER_DIR.parent / "data_preparation"
_OSMH3KG_DIR = _HELPER_DIR.parent
_PROJECT_ROOT = _OSMH3KG_DIR.parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# Data gathering variables
helper_path_directory: str = os.getenv(
    "OSMH3_HELPER_PATH", str(_HELPER_DIR)
)
osm_data_path: str = os.getenv(
    "OSMH3_RAW_PATH", str(_DATA_DIR / "osm_extract" / "raw_eng")
)
osm_parquet_path: str = os.getenv(
    "OSMH3_BRONZE_PATH", str(_DATA_DIR / "osm_extract" / "bronze")
)
osm_start_date: str = os.getenv("OSMH3_START_DATE", "2010-01-01")
osm_end_date: str = os.getenv("OSMH3_END_DATE", "2022-12-31")
osm_area: str = os.getenv("OSMH3_AREA", "California, United States")
osm_clipping: bool = os.getenv("OSMH3_CLIPPING", "false").lower() == "true"
ogr_temporary: str = os.getenv(
    "OSMH3_OGR_TEMP", str(_DATA_DIR / "osm_extract" / "ogrTemp")
)

# Data preparation variables — portable defaults
cpu_cores: int = int(os.getenv("OSMH3_CPU_CORES", str(os.cpu_count() or 4)))
spark_master: str = os.getenv("SPARK_MASTER", f"local[{cpu_cores}]")
spark_temp_directory: str | None = os.getenv("SPARK_TEMP_DIR")
spark_driver_memory: str = os.getenv("SPARK_DRIVER_MEMORY", "4g")
spark_executor_memory: str = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
kg_output_path: str = os.getenv(
    "OSMH3_KG_OUTPUT", str(_DATA_DIR / "osmh3kg" / "knowledge_graph")
)

# Grid configuration
grid_clipping: bool = os.getenv("OSMH3_GRID_CLIPPING", "false").lower() == "true"
grid_level: int = int(os.getenv("OSMH3_GRID_LEVEL", "7"))
grid_compaction: bool = os.getenv("OSMH3_GRID_COMPACTION", "false").lower() == "true"
grid_parquet_path: str = os.getenv(
    "OSMH3_GRID_PATH", str(_DATA_DIR / "osmh3kg" / "h3_grid.parquet")
)
geo_hash_level: int = int(os.getenv("OSMH3_GEOHASH_LEVEL", "3"))

# Sedona packages - compatible with Spark 4.x / PySpark 4.1 (Scala 2.13).
# Previous value pinned sedona-spark-shaded-3.5_2.12 which pulls Scala 2.12
# artifacts and crashes at SedonaContext.create with
# NoClassDefFoundError: scala/collection/GenTraversableOnce (removed in 2.13).
# Aligned with question_generator/config.py which already uses the 4.0_2.13
# variant in its SEDONA_PACKAGES list.
sedona_packages: list[str] = [
    "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.8.1",
    "org.datasyslab:geotools-wrapper:1.8.1-33.1",
    # Sedona's DuplicatesFilter (invoked in spatial-join materialization
    # after .persist()) references commons-collections 3.x's FilterIterator,
    # which the sedona-spark-shaded jar does NOT bundle. Missing on classpath
    # → NoClassDefFoundError at write time. Pin 3.2.2 explicitly.
    "commons-collections:commons-collections:3.2.2",
]

geometry_file_path: str = os.getenv(
    "OSMH3_GEOMETRY_PATH", str(_DATA_DIR / "osm_extract" / "osm_geometry")
)

# World data parquet (used by h3_generation)
world_data_parquet_path: str = os.getenv(
    "OSMH3_WORLD_DATA_PATH", str(_HELPER_DIR / "worlddata" / "world_data.parquet")
)

# OSM main tags parquet (used by osm_kg_data_preparation)
osm_main_tags_parquet_path: str = os.getenv(
    "OSMH3_MAIN_TAGS_PATH", str(_HELPER_DIR / "osm_main_tags.parquet")
)


def get_sedona_packages_string() -> str:
    """Get Sedona packages as a comma-separated string for Spark config."""
    return ",".join(sedona_packages)
