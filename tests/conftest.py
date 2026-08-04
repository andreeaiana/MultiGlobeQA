"""
Pytest fixtures for MultiGlobeQA tests.
"""

from pathlib import Path

import pytest
from pyspark.sql import SparkSession

BERLIN_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "berlin"


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """
    Create a SparkSession for testing.

    Uses local mode with minimal resources for fast test execution.
    """
    spark = (
        SparkSession.builder
        .appName("MultiGlobeQA-Tests")
        .master("local[2]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield spark

    spark.stop()


# ---------------------------------------------------------------------------
# Berlin city snapshot fixtures (built via: make build-fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def berlin_worldkg_entities(spark: SparkSession):
    """Load the Berlin WorldKG entities from the committed Parquet fixture."""
    path = BERLIN_FIXTURES_DIR / "worldkg_entities.parquet"
    if not path.exists():
        pytest.skip("Berlin fixtures not built. Run: make build-fixtures")
    return spark.read.parquet(str(path))


@pytest.fixture(scope="session")
def berlin_worldkg_triples(spark: SparkSession):
    """Load the Berlin WorldKG triples from the committed Parquet fixture."""
    path = BERLIN_FIXTURES_DIR / "worldkg_triples.parquet"
    if not path.exists():
        pytest.skip("Berlin fixtures not built. Run: make build-fixtures")
    return spark.read.parquet(str(path))


@pytest.fixture(scope="session")
def berlin_osmh3kg_entities(spark: SparkSession):
    """Load the Berlin OSMH3KG entities from the committed Parquet fixture."""
    path = BERLIN_FIXTURES_DIR / "osmh3kg_entities.parquet"
    if not path.exists():
        pytest.skip("Berlin fixtures not built. Run: make build-fixtures")
    return spark.read.parquet(str(path))


@pytest.fixture(scope="session")
def berlin_osmh3kg_triples(spark: SparkSession):
    """Load the Berlin OSMH3KG triples from the committed Parquet fixture."""
    path = BERLIN_FIXTURES_DIR / "osmh3kg_triples.parquet"
    if not path.exists():
        pytest.skip("Berlin fixtures not built. Run: make build-fixtures")
    return spark.read.parquet(str(path))


# ---------------------------------------------------------------------------
# KWG fixtures (built via: python scripts/build_kwg_fixtures.py)
# Contains NIFC_Wildfire + AdministrativeRegion_2 globally.
# ---------------------------------------------------------------------------

KWG_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "kwg"


@pytest.fixture(scope="session")
def kwg_entities(spark: SparkSession):
    """Load the KWG entities from the committed Parquet fixture."""
    path = KWG_FIXTURES_DIR / "kwg_entities.parquet"
    if not path.exists():
        pytest.skip("KWG fixtures not built. Run: python scripts/build_kwg_fixtures.py")
    return spark.read.parquet(str(path))


@pytest.fixture(scope="session")
def kwg_triples(spark: SparkSession):
    """Load the KWG triples from the committed Parquet fixture."""
    path = KWG_FIXTURES_DIR / "kwg_triples.parquet"
    if not path.exists():
        pytest.skip("KWG triples not built. Run: python scripts/build_kwg_fixtures.py")
    return spark.read.parquet(str(path))
