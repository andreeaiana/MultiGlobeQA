"""Register a session-time `country_polygons` view from Natural Earth.

Used by ftids that need country-level geometry (e.g. sfid_a 7.1's
"country bordering the country containing entity_A" semantics).
Mirrors the `road_setup.register_road_aggregation_views` pattern.

Source data: Natural Earth 110m admin_0_countries (already cached at
`data/reference/countries.geojson` per
`data_extraction.reference_data.downloader`). The KG-agnostic loader
ensures the file is downloaded if missing.

Schema of the registered view `country_polygons`:
    iso_a3        STRING   -- ISO 3-letter code (per Natural Earth's iso3 column)
    name          STRING   -- country common name
    geom          GEOMETRY -- Sedona Geometry, materialized once

KG-agnostic — register once per session regardless of which KG run
follows. Cheap (only ~250 country rows) so re-registration is also
cheap; we still short-circuit if the view already exists.

Open caveat: `ST_Touches` over country polygons captures LAND-shared
borders only; maritime-only borders (e.g. Australia ↔ Papua New
Guinea) are NOT detected. This matches the typical reading of
"bordering country" as a shared land boundary; revisit if the
benchmark needs maritime-borders semantics.
"""
from __future__ import annotations

from loguru import logger
from pyspark.sql import SparkSession


def _table_exists(spark: SparkSession, name: str) -> bool:
    """Best-effort catalog lookup; treats lookup failure as 'not present'."""
    try:
        return bool(spark.catalog.tableExists(name))
    except Exception:
        return False


def register_country_polygons_view(spark: SparkSession) -> None:
    """Register the `country_polygons` Spark temp view.

    Idempotent — short-circuits if the view already exists. KG-agnostic
    (no `kg_label` argument; the country data is the same regardless
    of which KG the benchmark targets).

    Side effects:
      * If `data/reference/countries.geojson` is absent,
        `ensure_reference_data` downloads it (one-time, ~5 MB).
      * Creates `country_polygons` Spark temp view with a Geometry
        column suitable for Sedona spatial joins.
    """
    view_name = "country_polygons"
    if _table_exists(spark, view_name):
        logger.info(f"View {view_name} already registered; skipping")
        return

    # Lazy import — avoids forcing geopandas on test contexts that don't
    # need country polygons.
    from data_extraction.reference_data import load_countries_geodataframe

    try:
        gdf = load_countries_geodataframe()
    except Exception as e:
        logger.warning(
            f"Cannot load country polygons ({type(e).__name__}: {e}); "
            f"view {view_name!r} will not be available — ftids that need "
            f"it will fail at SQL analysis time."
        )
        return

    # Convert GeoDataFrame to a Spark DataFrame via WKT round-trip.
    # GeoPandas-to-Spark direct conversion needs Sedona's type adapters
    # which are version-fragile; WKT is the safe lingua-franca.
    rows = []
    for _, r in gdf.iterrows():
        iso3 = r.get("iso3") or r.get("ISO_A3")
        name = r.get("NAME") or r.get("ADMIN") or ""
        if not iso3 or len(str(iso3)) != 3:
            continue
        wkt = r["geometry"].wkt if r.get("geometry") is not None else None
        if not wkt:
            continue
        rows.append((str(iso3), str(name), wkt))

    if not rows:
        logger.warning(
            f"load_countries_geodataframe returned 0 rows with valid "
            f"(iso3, geometry); {view_name!r} not registered"
        )
        return

    raw_df = spark.createDataFrame(rows, schema=["iso_a3", "name", "geometry_wkt"])
    raw_df.createOrReplaceTempView("__country_setup_raw")

    # Compose Geometry-typed DataFrame, then register that DataFrame
    # directly (NOT a CREATE TEMP VIEW ... AS SELECT, which is a lazy
    # SQL view that re-resolves __country_setup_raw on each downstream
    # use and fails after long-running sessions evict the underlying
    # plan from cache). The cached DataFrame survives across queries.
    geom_df = spark.sql(
        """
        SELECT iso_a3, name,
               ST_GeomFromWKT(geometry_wkt) AS geom
        FROM __country_setup_raw
        """
    ).cache()
    geom_df.count()  # force materialization
    geom_df.createOrReplaceTempView(view_name)
    # __country_setup_raw is intentionally NOT dropped: even though
    # geom_df is cached and re-registered as a fresh DataFrame view,
    # Spark's analyzer occasionally re-resolves the underlying logical
    # plan and walks back to the source TempView. Keeping it alive
    # through the session is harmless (a few hundred small WKT rows).
    logger.info(
        f"Registered {view_name} from Natural Earth ({len(rows)} country polygons)"
    )
