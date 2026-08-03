"""
Executors for Spatial Benchmark Template Processing.

This module implements the Factory Pattern for routing templates to appropriate
execution engines based on their Spatial Function IDs (SFIDs).

- SedonaExecutor: Handles spatial/geometry operations (SFIDs: A, B, C, L)
- StandardSparkExecutor: Handles hierarchy/index lookups (SFIDs: H, E, P, D, F, K, O)
"""

import json
import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pyspark.sql.functions as F
import yaml
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType, StringType
from shapely import wkt as shapely_wkt
from shapely.geometry import box as shapely_box

from question_generator.config import (
    PATHFINDING_MAX_BBOX_DEG,
    SEDONA_SFIDS,
    STANDARD_SPARK_SFIDS,
)

# Valid OSMnx network types for transport-mode pathfinding (TID 49)
_OSMNX_NETWORK_TYPES: frozenset[str] = frozenset({"walk", "bike", "drive"})


def _calc_fractal_dimension(geom_wkt: str) -> float | None:
    """
    Calculate the fractal dimension of a geometry using box-counting method.

    Uses 8 box scales from max_extent/2 to max_extent/256. For each scale,
    counts the number of boxes that intersect the geometry, then computes
    fractal dimension via log-log linear regression.

    Args:
        geom_wkt: Well-Known Text representation of the geometry.

    Returns:
        Approximate fractal dimension, or None if calculation fails.
    """
    if not geom_wkt:
        return None
    try:
        geom = shapely_wkt.loads(geom_wkt)
        if geom.is_empty:
            return None

        minx, miny, maxx, maxy = geom.bounds
        extent_x = maxx - minx
        extent_y = maxy - miny
        max_extent = max(extent_x, extent_y)

        if max_extent <= 0:
            # Point geometry — fractal dimension is 0
            return 0.0

        # Generate 8 box scales: max_extent/2 down to max_extent/256
        scales = [max_extent / (2**i) for i in range(1, 9)]

        log_scales = []
        log_counts = []

        for box_size in scales:
            count = 0
            # Count boxes that intersect the geometry
            x = minx
            while x < maxx:
                y = miny
                while y < maxy:
                    cell = shapely_box(x, y, x + box_size, y + box_size)
                    if geom.intersects(cell):
                        count += 1
                    y += box_size
                x += box_size

            if count > 0:
                log_scales.append(math.log(1.0 / box_size))
                log_counts.append(math.log(count))

        if len(log_scales) < 2:
            return None

        # Linear regression on log-log data: fractal_dim = slope
        coeffs = np.polyfit(log_scales, log_counts, 1)
        fractal_dim = float(coeffs[0])

        # Clamp to valid range [0, 2] for 2D geometries
        return max(0.0, min(2.0, fractal_dim))

    except Exception:
        return None


def _calc_pathfinding(start_geom: str, end_geom: str) -> str | None:
    """
    Calculate shortest path between two geometries using OSMnx road network.

    Retrieves the local road network around both entities, finds the nearest
    network nodes, and computes the shortest path weighted by distance.

    Args:
        start_geom: WKT of starting geometry.
        end_geom: WKT of ending geometry.

    Returns:
        JSON string with total_distance_m, hop_count, path_node_ids,
        and route_wkt, or None if no path found.
    """
    if not start_geom or not end_geom:
        return None
    try:
        import osmnx as ox
        from shapely.geometry import LineString

        start_point = shapely_wkt.loads(start_geom)
        end_point = shapely_wkt.loads(end_geom)

        start_lon, start_lat = start_point.centroid.x, start_point.centroid.y
        end_lon, end_lat = end_point.centroid.x, end_point.centroid.y

        # Build a bounding box around both points with a buffer (~2km)
        buffer_deg = 0.02  # ~2km at mid-latitudes
        north = max(start_lat, end_lat) + buffer_deg
        south = min(start_lat, end_lat) - buffer_deg
        east = max(start_lon, end_lon) + buffer_deg
        west = min(start_lon, end_lon) - buffer_deg

        # Guard: skip Overpass query if bbox is too large
        lat_span = north - south
        lon_span = east - west
        if lat_span > PATHFINDING_MAX_BBOX_DEG or lon_span > PATHFINDING_MAX_BBOX_DEG:
            return json.dumps({
                "error": "bbox_too_large",
                "lat_span_deg": round(lat_span, 4),
                "lon_span_deg": round(lon_span, 4),
                "max_allowed_deg": PATHFINDING_MAX_BBOX_DEG,
            })

        # Enable caching to avoid re-downloading the same bbox
        ox.settings.use_cache = True

        # Retrieve road network
        graph = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type="drive",
        )

        # Find nearest nodes
        start_node = ox.nearest_nodes(graph, start_lon, start_lat)
        end_node = ox.nearest_nodes(graph, end_lon, end_lat)

        # Compute shortest path
        route = ox.shortest_path(graph, start_node, end_node, weight="length")

        if route is None:
            return json.dumps({"error": "no_path_found"})

        # Calculate total distance
        total_distance = 0.0
        route_coords = []
        for u, v in zip(route[:-1], route[1:], strict=True):
            edge_data = graph[u][v][0]
            total_distance += edge_data.get("length", 0.0)

        for node_id in route:
            node_data = graph.nodes[node_id]
            route_coords.append((node_data["x"], node_data["y"]))

        route_line = LineString(route_coords)

        result = {
            "total_distance_m": round(total_distance, 2),
            "hop_count": len(route) - 1,
            "path_node_ids": route,
            "route_wkt": route_line.wkt,
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})


def _calc_pathfinding_by_mode(
    start_geom: str, end_geom: str, network_type: str
) -> str | None:
    """
    Calculate shortest path between two geometries using a specific transport mode.

    Like ``_calc_pathfinding`` but accepts an OSMnx ``network_type`` parameter
    to select the road network graph (walk, bike, or drive).

    Args:
        start_geom: WKT of starting geometry.
        end_geom: WKT of ending geometry.
        network_type: One of "walk", "bike", "drive".

    Returns:
        JSON string with total_distance_m, hop_count, path_node_ids,
        route_wkt, and network_type, or None if no path found.
    """
    if not start_geom or not end_geom:
        return None
    if network_type not in _OSMNX_NETWORK_TYPES:
        return json.dumps({"error": f"unsupported_network_type: {network_type}"})
    try:
        import osmnx as ox
        from shapely.geometry import LineString

        start_point = shapely_wkt.loads(start_geom)
        end_point = shapely_wkt.loads(end_geom)

        start_lon, start_lat = start_point.centroid.x, start_point.centroid.y
        end_lon, end_lat = end_point.centroid.x, end_point.centroid.y

        buffer_deg = 0.02
        north = max(start_lat, end_lat) + buffer_deg
        south = min(start_lat, end_lat) - buffer_deg
        east = max(start_lon, end_lon) + buffer_deg
        west = min(start_lon, end_lon) - buffer_deg

        lat_span = north - south
        lon_span = east - west
        if lat_span > PATHFINDING_MAX_BBOX_DEG or lon_span > PATHFINDING_MAX_BBOX_DEG:
            return json.dumps({
                "error": "bbox_too_large",
                "lat_span_deg": round(lat_span, 4),
                "lon_span_deg": round(lon_span, 4),
                "max_allowed_deg": PATHFINDING_MAX_BBOX_DEG,
            })

        ox.settings.use_cache = True

        graph = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type=network_type,
        )

        start_node = ox.nearest_nodes(graph, start_lon, start_lat)
        end_node = ox.nearest_nodes(graph, end_lon, end_lat)

        route = ox.shortest_path(graph, start_node, end_node, weight="length")
        if route is None:
            return json.dumps({"error": "no_path_found"})

        total_distance = 0.0
        route_coords = []
        for u, v in zip(route[:-1], route[1:], strict=True):
            edge_data = graph[u][v][0]
            total_distance += edge_data.get("length", 0.0)

        for node_id in route:
            node_data = graph.nodes[node_id]
            route_coords.append((node_data["x"], node_data["y"]))

        route_line = LineString(route_coords)

        result = {
            "total_distance_m": round(total_distance, 2),
            "hop_count": len(route) - 1,
            "path_node_ids": route,
            "route_wkt": route_line.wkt,
            "network_type": network_type,
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})


def _calc_pathfinding_extended(start_geom: str, end_geom: str) -> str | None:
    """
    Extended pathfinding with additional route analysis fields.

    Computes the shortest path like ``_calc_pathfinding`` and enriches the
    result with:
    - ``bottleneck_node_id``: The node on the path with the highest
      betweenness centrality (among path sub-graph nodes).
    - ``route_wkt``: WKT LineString of the route geometry.
    - ``road_names``: List of unique road names along the path edges.

    Args:
        start_geom: WKT of starting geometry.
        end_geom: WKT of ending geometry.

    Returns:
        JSON string with enriched route data, or None if no path found.
    """
    if not start_geom or not end_geom:
        return None
    try:
        import networkx as nx
        import osmnx as ox
        from shapely.geometry import LineString

        start_point = shapely_wkt.loads(start_geom)
        end_point = shapely_wkt.loads(end_geom)

        start_lon, start_lat = start_point.centroid.x, start_point.centroid.y
        end_lon, end_lat = end_point.centroid.x, end_point.centroid.y

        buffer_deg = 0.02
        north = max(start_lat, end_lat) + buffer_deg
        south = min(start_lat, end_lat) - buffer_deg
        east = max(start_lon, end_lon) + buffer_deg
        west = min(start_lon, end_lon) - buffer_deg

        lat_span = north - south
        lon_span = east - west
        if lat_span > PATHFINDING_MAX_BBOX_DEG or lon_span > PATHFINDING_MAX_BBOX_DEG:
            return json.dumps({
                "error": "bbox_too_large",
                "lat_span_deg": round(lat_span, 4),
                "lon_span_deg": round(lon_span, 4),
                "max_allowed_deg": PATHFINDING_MAX_BBOX_DEG,
            })

        ox.settings.use_cache = True

        graph = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type="drive",
        )

        start_node = ox.nearest_nodes(graph, start_lon, start_lat)
        end_node = ox.nearest_nodes(graph, end_lon, end_lat)

        route = ox.shortest_path(graph, start_node, end_node, weight="length")
        if route is None:
            return json.dumps({"error": "no_path_found"})

        # Compute distance and collect road names
        total_distance = 0.0
        route_coords = []
        road_names: list[str] = []
        for u, v in zip(route[:-1], route[1:], strict=True):
            edge_data = graph[u][v][0]
            total_distance += edge_data.get("length", 0.0)
            road_name = edge_data.get("name")
            if road_name and isinstance(road_name, str):
                road_names.append(road_name)

        for node_id in route:
            node_data = graph.nodes[node_id]
            route_coords.append((node_data["x"], node_data["y"]))

        route_line = LineString(route_coords)

        # Compute betweenness centrality on the subgraph induced by path nodes
        # to identify the bottleneck node.
        sub = graph.subgraph(route).copy()
        bc = nx.betweenness_centrality(sub, weight="length")
        bottleneck_node = max(bc, key=bc.get) if bc else route[0]

        result = {
            "total_distance_m": round(total_distance, 2),
            "hop_count": len(route) - 1,
            "path_node_ids": route,
            "route_wkt": route_line.wkt,
            "road_names": list(dict.fromkeys(road_names)),
            "bottleneck_node_id": int(bottleneck_node),
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})


def _calc_pathfinding_avoid(
    start_geom: str, end_geom: str, avoid_geom: str
) -> str | None:
    """
    Calculate shortest path that avoids a specified geometry.

    Removes edges whose midpoint falls within the avoidance geometry
    before computing the shortest path.

    Args:
        start_geom: WKT of starting geometry.
        end_geom: WKT of ending geometry.
        avoid_geom: WKT geometry to avoid (edges overlapping this are removed).

    Returns:
        JSON string with route data, or None if no path found.
    """
    if not start_geom or not end_geom or not avoid_geom:
        return None
    try:
        import osmnx as ox
        from shapely.geometry import LineString, Point

        start_point = shapely_wkt.loads(start_geom)
        end_point = shapely_wkt.loads(end_geom)
        avoid_polygon = shapely_wkt.loads(avoid_geom)

        start_lon, start_lat = start_point.centroid.x, start_point.centroid.y
        end_lon, end_lat = end_point.centroid.x, end_point.centroid.y

        buffer_deg = 0.02
        north = max(start_lat, end_lat) + buffer_deg
        south = min(start_lat, end_lat) - buffer_deg
        east = max(start_lon, end_lon) + buffer_deg
        west = min(start_lon, end_lon) - buffer_deg

        lat_span = north - south
        lon_span = east - west
        if lat_span > PATHFINDING_MAX_BBOX_DEG or lon_span > PATHFINDING_MAX_BBOX_DEG:
            return json.dumps({
                "error": "bbox_too_large",
                "lat_span_deg": round(lat_span, 4),
                "lon_span_deg": round(lon_span, 4),
                "max_allowed_deg": PATHFINDING_MAX_BBOX_DEG,
            })

        ox.settings.use_cache = True

        graph = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            network_type="drive",
        )

        # Remove edges whose midpoint falls inside the avoidance geometry
        edges_to_remove = []
        for u, v, _key in graph.edges(keys=True):
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            mid_x = (u_data["x"] + v_data["x"]) / 2
            mid_y = (u_data["y"] + v_data["y"]) / 2
            if avoid_polygon.contains(Point(mid_x, mid_y)):
                edges_to_remove.append((u, v, _key))

        graph.remove_edges_from(edges_to_remove)

        start_node = ox.nearest_nodes(graph, start_lon, start_lat)
        end_node = ox.nearest_nodes(graph, end_lon, end_lat)

        route = ox.shortest_path(graph, start_node, end_node, weight="length")
        if route is None:
            return json.dumps({"error": "no_path_found_avoiding_barrier"})

        total_distance = 0.0
        route_coords = []
        for u, v in zip(route[:-1], route[1:], strict=True):
            edge_data = graph[u][v][0]
            total_distance += edge_data.get("length", 0.0)

        for node_id in route:
            node_data = graph.nodes[node_id]
            route_coords.append((node_data["x"], node_data["y"]))

        route_line = LineString(route_coords)

        result = {
            "total_distance_m": round(total_distance, 2),
            "hop_count": len(route) - 1,
            "path_node_ids": route,
            "route_wkt": route_line.wkt,
            "edges_removed": len(edges_to_remove),
        }
        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": str(e)})


class BaseExecutor(ABC):
    """Abstract base class for template executors."""

    def __init__(self, spark: SparkSession) -> None:
        """
        Initialize the executor.

        Args:
            spark: Active SparkSession instance.
        """
        self.spark = spark

    @abstractmethod
    def execute(
        self,
        template_config: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any],
    ) -> DataFrame:
        """
        Execute a template against the provided entities.

        Args:
            template_config: Configuration dict from the SQL registry.
            entities_df: DataFrame containing sampled entities.
            params: Runtime parameters for query interpolation.

        Returns:
            DataFrame with ground truth results.
        """
        pass

    def _create_temp_view(self, df: DataFrame, prefix: str = "entities") -> str:
        """
        Create a temporary view for SQL execution.

        Args:
            df: DataFrame to register as a view.
            prefix: Prefix for the view name.

        Returns:
            The generated view name.
        """
        view_name = f"{prefix}_{id(self)}_{id(df)}"
        df.createOrReplaceTempView(view_name)
        return view_name

    def _create_geometry_view(self, df: DataFrame, prefix: str) -> str:
        """
        Create a temp view with all WKT geometry columns converted to Sedona types.

        Detects columns ending in ``_geometry`` or named ``geometry`` and wraps
        them with ``ST_GeomFromWKT``.  Falls back to a raw view on error.
        """
        raw_view = self._create_temp_view(df, f"{prefix}_raw")
        geom_cols = [c for c in df.columns if c.endswith("_geometry") or c == "geometry"]
        if not geom_cols:
            return raw_view

        view_name = f"{raw_view}_geom"
        non_geom_cols = [c for c in df.columns if c not in geom_cols]
        non_geom_select = ", ".join(f"`{c}`" for c in non_geom_cols)
        geom_select = ", ".join(f"ST_GeomFromWKT(`{c}`) AS `{c}`" for c in geom_cols)
        select_clause = f"{non_geom_select}, {geom_select}" if non_geom_select else geom_select

        try:
            self.spark.sql(
                f"CREATE OR REPLACE TEMP VIEW {view_name} AS "
                f"SELECT {select_clause} FROM {raw_view}"
            )
            return view_name
        except Exception:
            return raw_view


class SedonaExecutor(BaseExecutor):
    """
    Executor for spatial/geometry operations using Apache Sedona.

    Handles SFIDs: A (Distance/Proximity), B (Metric Containment),
    C (Topological), L (Shape/Geometry).
    """

    SUPPORTED_SFIDS: frozenset[str] = SEDONA_SFIDS

    def execute(
        self,
        template_config: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any],
    ) -> DataFrame:
        """
        Execute a Sedona-based spatial query.

        Args:
            template_config: Must contain 'type' (SQL/UDF) and 'sql' or 'udf_name'.
            entities_df: DataFrame with geometry columns.
            params: Parameters like entity_A_id, entity_B_id, etc.

        Returns:
            DataFrame with spatial computation results.
        """
        execution_type = template_config.get("type", "SQL")

        if execution_type == "SQL":
            return self._execute_sql(template_config, entities_df, params)
        elif execution_type == "UDF":
            return self._execute_udf(template_config, entities_df, params)
        else:
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"Unknown execution type: {execution_type}"))

    def _execute_sql(
        self,
        template_config: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any],
    ) -> DataFrame:
        """Execute a parameterized SQL query using Sedona spatial functions."""
        sql_template = template_config.get("sql", "")
        if not sql_template:
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit("No SQL template provided"))

        view_name = self._create_geometry_view(entities_df, "sedona_entities")
        format_params = {"view_name": view_name, **params}

        try:
            formatted_sql = sql_template.format(**format_params)
            return self.spark.sql(formatted_sql)
        except KeyError as e:
            logger.warning("Missing parameter in SQL template: {}", e)
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"Missing parameter: {e}"))
        except Exception as e:
            logger.error("SQL execution error: {}", e)
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"SQL execution error: {str(e)}"))

    def _execute_udf(
        self,
        template_config: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any],
    ) -> DataFrame:
        """Execute a custom UDF for complex spatial operations."""
        udf_name = template_config.get("udf_name", "")

        if udf_name == "calculate_fractal_dimension":
            fractal_udf = F.udf(_calc_fractal_dimension, DoubleType())
            return entities_df.withColumn(
                "ground_truth",
                fractal_udf(F.col("entity_A_geometry")).cast(StringType()),
            )
        else:
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"Unknown UDF: {udf_name}"))


class StandardSparkExecutor(BaseExecutor):
    """
    Executor for hierarchy and index-based lookups using standard Spark SQL.

    Handles SFIDs: H (Hierarchy), E (Administrative), P (Coordinates),
    D (Network), F (Ordering), K (Directional).
    """

    SUPPORTED_SFIDS: frozenset[str] = STANDARD_SPARK_SFIDS

    def execute(
        self,
        template_config: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any],
    ) -> DataFrame:
        """
        Execute a standard Spark SQL query.

        Args:
            template_config: Must contain 'type' and 'sql' or 'udf_name'.
            entities_df: DataFrame with entity data.
            params: Parameters for query interpolation.

        Returns:
            DataFrame with query results.
        """
        execution_type = template_config.get("type", "SQL")

        if execution_type == "SQL":
            return self._execute_sql(template_config, entities_df, params)
        elif execution_type == "UDF":
            return self._execute_udf(template_config, entities_df, params)
        else:
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"Unknown execution type: {execution_type}"))

    def _execute_sql(
        self,
        template_config: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any],
    ) -> DataFrame:
        """Execute a parameterized SQL query."""
        sql_template = template_config.get("sql", "")
        if not sql_template:
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit("No SQL template provided"))

        view_name = self._create_geometry_view(entities_df, "spark_entities")
        format_params = {"view_name": view_name, **params}

        try:
            formatted_sql = sql_template.format(**format_params)
            return self.spark.sql(formatted_sql)
        except KeyError as e:
            logger.warning("Missing parameter in SQL template: {}", e)
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"Missing parameter: {e}"))
        except Exception as e:
            logger.error("SQL execution error: {}", e)
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"SQL execution error: {str(e)}"))

    def _execute_udf(
        self,
        template_config: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any],
    ) -> DataFrame:
        """Execute a custom UDF for network/path operations."""
        udf_name = template_config.get("udf_name", "")

        if udf_name == "calculate_pathfinding":
            pathfinding_udf = F.udf(_calc_pathfinding, StringType())
            return entities_df.withColumn(
                "ground_truth",
                pathfinding_udf(
                    F.col("entity_A_geometry"),
                    F.col("entity_B_geometry"),
                ),
            )
        elif udf_name == "calculate_pathfinding_extended":
            extended_udf = F.udf(_calc_pathfinding_extended, StringType())
            return entities_df.withColumn(
                "ground_truth",
                extended_udf(
                    F.col("entity_A_geometry"),
                    F.col("entity_B_geometry"),
                ),
            )
        elif udf_name == "calculate_pathfinding_by_mode":
            network_type = params.get("network_type", "drive")
            by_mode_udf = F.udf(
                lambda s, e: _calc_pathfinding_by_mode(s, e, network_type),
                StringType(),
            )
            return entities_df.withColumn(
                "ground_truth",
                by_mode_udf(
                    F.col("entity_A_geometry"),
                    F.col("entity_B_geometry"),
                ),
            )
        elif udf_name == "calculate_pathfinding_avoid":
            avoid_udf = F.udf(_calc_pathfinding_avoid, StringType())
            return entities_df.withColumn(
                "ground_truth",
                avoid_udf(
                    F.col("entity_A_geometry"),
                    F.col("entity_B_geometry"),
                    F.col("entity_C_geometry"),
                ),
            )
        else:
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"Unknown UDF: {udf_name}"))


class RegistryExecutor:
    """
    Main executor that routes templates to appropriate executors based on SFID.

    Implements the Factory Pattern to dispatch execution to either
    SedonaExecutor or StandardSparkExecutor based on the template's SFID.
    """

    def __init__(self, spark: SparkSession, registry_path: str) -> None:
        """
        Initialize the registry executor.

        Args:
            spark: Active SparkSession instance.
            registry_path: Path to the YAML registry file.
        """
        self.spark = spark
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self._global_view_name: str | None = None
        self._triples_view_name: str | None = None

        # Initialize specialized executors
        self.sedona_executor = SedonaExecutor(spark)
        self.standard_executor = StandardSparkExecutor(spark)

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        """
        Load the TID-to-execution mapping from the YAML registry.

        Returns:
            Dictionary mapping TIDs to their execution configurations.
        """
        try:
            with open(self.registry_path, encoding="utf-8") as file:
                data = yaml.safe_load(file)
                return data.get("templates", {})
        except FileNotFoundError:
            logger.warning("Registry file not found at {}", self.registry_path)
            return {}
        except yaml.YAMLError as e:
            logger.error("Error parsing YAML registry: {}", e)
            return {}

    def register_global_entities(self, df: DataFrame) -> None:
        """Register the full entity table as a temp view for aggregation queries."""
        self._global_view_name = f"global_entities_{id(self)}"
        # Convert geometry columns to Sedona types in the global view
        geom_cols = [c for c in df.columns if c == "geometry"]
        if geom_cols:
            raw_name = f"{self._global_view_name}_raw"
            df.createOrReplaceTempView(raw_name)
            non_geom = [c for c in df.columns if c != "geometry"]
            select = ", ".join(f"`{c}`" for c in non_geom)
            select += ", ST_GeomFromWKT(geometry) AS geometry"
            try:
                self.spark.sql(
                    f"CREATE OR REPLACE TEMP VIEW {self._global_view_name} AS "
                    f"SELECT {select} FROM {raw_name}"
                )
            except Exception:
                df.createOrReplaceTempView(self._global_view_name)
        else:
            df.createOrReplaceTempView(self._global_view_name)

    def register_triples(self, df: DataFrame) -> None:
        """Register the triples table as a temp view for temporal queries.

        Temporal templates (SFID I/J) join the entity view with this triples
        view at query time.  The view is referenced via the ``{triples_view}``
        placeholder in SQL templates.

        Args:
            df: Triples DataFrame (subject, predicate, object[, date]).
        """
        self._triples_view_name = f"triples_{id(self)}"
        df.createOrReplaceTempView(self._triples_view_name)

    def get_executor_for_sfid(self, sfid: str) -> BaseExecutor:
        """
        Get the appropriate executor for a given SFID.

        Args:
            sfid: Spatial Function ID (A, B, C, D, E, F, H, K, L, O, P).

        Returns:
            The appropriate executor instance.
        """
        if sfid in SEDONA_SFIDS:
            return self.sedona_executor
        else:
            return self.standard_executor

    def execute(
        self,
        template_row: dict[str, Any],
        entities_df: DataFrame,
        params: dict[str, Any] | None = None,
    ) -> DataFrame:
        """
        Execute a template against the provided entities.

        Routes to the appropriate executor based on the template's SFID.

        Args:
            template_row: Template metadata including TID and SFID.
            entities_df: DataFrame containing sampled entities.
            params: Optional runtime parameters for query interpolation.

        Returns:
            DataFrame with ground truth results.
        """
        tid = str(template_row.get("TID", "")).strip()
        sfid = str(template_row.get("SFID", "")).strip()

        # Get template configuration from registry
        config = self.registry.get(tid)

        if not config:
            return entities_df.withColumn(
                "ground_truth", F.lit(None).cast(StringType())
            ).withColumn("error", F.lit(f"TID {tid} not found in registry"))

        # Use SFID from registry if not provided in template_row
        if not sfid:
            sfid = config.get("sfid", "")

        # Get appropriate executor and execute
        executor = self.get_executor_for_sfid(sfid)

        # Merge params with any defaults
        execution_params = params or {}
        if self._global_view_name:
            execution_params["global_view"] = self._global_view_name
        if self._triples_view_name:
            execution_params["triples_view"] = self._triples_view_name

        return executor.execute(config, entities_df, execution_params)

    def get_supported_tids(self) -> list[str]:
        """
        Get list of all TIDs supported by the registry.

        Returns:
            List of template IDs.
        """
        return list(self.registry.keys())

    def get_sfid_for_tid(self, tid: str) -> str | None:
        """
        Get the SFID for a given TID.

        Args:
            tid: Template ID.

        Returns:
            The SFID, or None if TID not found.
        """
        config = self.registry.get(str(tid))
        if config:
            return config.get("sfid")
        return None
