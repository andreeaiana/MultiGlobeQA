"""
Generate H3 grid with flexible configuration options.

Input:
    - grid_level: Level of H3 grid cell as integer in range 0 to 15 (Default: 9).
      Background on grid level: https://h3geo.org/docs/core-library/restable
    - grid_compaction: Boolean value whether the selection of grid cells should be
      compacted or not. Background: https://h3geo.org/docs/highlights/indexing/
"""

import json
from multiprocessing import Pool
from typing import Any

import geopandas as gpd
import h3
import pandas as pd
from loguru import logger
from pyspark.sql.functions import col
from sedona.spark import SedonaContext
from sedona.sql.st_functions import ST_GeoHash, ST_MakeValid
from shapely.geometry import Polygon, box
from tqdm import tqdm

from data_extraction.osmh3kg.helper.constants import (
    cpu_cores,
    geo_hash_level,
    get_sedona_packages_string,
    grid_clipping,
    grid_compaction,
    grid_level,
    grid_parquet_path,
    osm_area,
    spark_driver_memory,
    spark_executor_memory,
    spark_master,
    world_data_parquet_path,
)


def extract_h3_grid(world_data_feature: dict[str, Any]) -> dict[str, Any]:
    """
    Extract H3 grid cells for a single country/region feature.

    Uses the H3 v4 API (polygon_to_cells) to generate grid cells aligned
    to the Uber H3 Discrete Global Grid system.

    Args:
        world_data_feature: Dictionary containing country properties
            and the respective geometry.

    Returns:
        Dictionary with keys: country, continent, subregion, h3_id, land_geometry.
    """
    properties = world_data_feature["properties"]
    geometry = world_data_feature["geometry"]
    # H3 v4 API: polygon_to_cells replaces polyfill_geojson
    h3_ids = h3.geo_to_cells(geometry, res=grid_level)
    if grid_compaction:
        h3_ids = h3.compact_cells(h3_ids)
    row_dict = {
        "country": properties["admin"],
        "continent": properties["continent"],
        "subregion": properties["subregion"],
        "h3_id": list(h3_ids),
        "land_geometry": str(geometry),
    }
    return row_dict


def create_h3_grid(world_data_path: str = world_data_parquet_path) -> None:
    """
    Create an H3 grid from world data and write to GeoParquet.

    Args:
        world_data_path: Path to the world_data.parquet file.
    """
    logger.info("Reading world data from {}", world_data_path)
    world_data = gpd.read_parquet(world_data_path)
    world_data = world_data.explode(ignore_index=True)

    if grid_clipping:
        try:
            from OSMPythonTools.nominatim import Nominatim

            nominatim = Nominatim()
            area_parameter = nominatim.query(osm_area).toJSON()[0].get("boundingbox")
            area_bbox = box(
                miny=float(area_parameter[0]),
                maxy=float(area_parameter[1]),
                minx=float(area_parameter[2]),
                maxx=float(area_parameter[3]),
            )
            world_data = world_data.clip(area_bbox)
        except ImportError:
            logger.error(
                "OSMPythonTools is required for grid_clipping. "
                "Install with: pip install OSMPythonTools"
            )
            raise

    world_data_json = json.loads(world_data.to_json())
    features = world_data_json["features"]

    logger.info("Generating H3 grid cells at level {} for {} features", grid_level, len(features))
    with Pool(cpu_cores) as pool:
        row_list = list(
            tqdm(
                pool.imap_unordered(extract_h3_grid, features),
                total=len(features),
            )
        )
    data = pd.DataFrame.from_records(row_list)
    data = data.explode(column="h3_id")
    data = data.drop("land_geometry", axis=1)
    data = data.dropna()

    # H3 v4 API: cell_to_boundary replaces h3_to_geo_boundary.
    # cell_to_boundary returns (lat, lon) tuples, but Shapely/WKT/GeoParquet
    # expect (lon, lat) = (x, y) ordering.  Swap to avoid a CRS mismatch
    # that silently breaks all downstream ST_Intersects / ST_Contains joins.
    data["geometry"] = data["h3_id"].apply(
        lambda h3_id: [(lon, lat) for lat, lon in h3.cell_to_boundary(h3_id)]
    )
    data["geometry"] = data["geometry"].apply(Polygon)
    gdf_data = gpd.GeoDataFrame(data, geometry="geometry")
    gdf_data = gdf_data.reset_index(drop=True)

    config = (
        SedonaContext.builder()
        .master(spark_master)
        .appName("h3generation")
        .config("spark.driver.memory", spark_driver_memory)
        .config("spark.executor.memory", spark_executor_memory)
        .config("spark.jars.packages", get_sedona_packages_string())
        .getOrCreate()
    )
    sedona = SedonaContext.create(config)
    sedona_df = sedona.createDataFrame(gdf_data)
    sedona_df = sedona_df.withColumn(
        "geohash", ST_GeoHash("geometry", geo_hash_level)
    ).orderBy("geohash")
    sedona_df = sedona_df.withColumn("geometry", ST_MakeValid(col("geometry")))

    logger.info("Writing H3 grid to {}", grid_parquet_path)
    sedona_df.write.mode("overwrite").format("geoparquet").save(grid_parquet_path)


if __name__ == "__main__":
    create_h3_grid()
