-- TID 25.1 — Count: how many regions of the variant's category does
-- a given named road traverse?
--
-- Pure-Sedona two-pool TID:
--   pool_a = osmh3kg_roads_aggregated session view
--            (one row per (country_iso3, name, node_type) named road, with the
--             merged geometry produced by ST_LineMerge(ST_Union_Aggr(...))).
--            Registered by question_generator/road_setup.py at init_spark
--            time. Columns:
--              country_iso3, name, node_type, geometry, source_way_count,
--              first_osm_id, id (MD5(country|name|node_type)), category,
--              entity_uri (first OSM way's URI per aggregated road).
--   pool_b = region entities filtered to the active variant's category
--            (e.g. node_type = 'district' / 'county' / 'state' /
--            'neighbourhood').
--
-- Region-traversal predicate: ST_Intersects(road_geom, region_geom) +
-- same_country guard. Aggregate the count of intersecting regions per
-- anchor road via COUNT(*).
--
-- Sampling: per-country ROW_NUMBER hash diversity on the road anchor
-- (deterministic in {rand_seed_a}) rather than an unordered random
-- sample, so anchors stay balanced across countries.
--
-- Sedona spatial-join pattern:
-- pre-CTE materialize WKT → Geometry, then put ST_Intersects in the JOIN
-- ON clause. Activates RangeJoinExec, avoids USA-skew hangs.
--
-- Geometry-type filters:
--   - pool_a: NO LINESTRING-only filter — the aggregated view's geometry
--     is already merged (LineMerge → LineString or MultiLineString).
--   - pool_b: keep POLYGON / MULTIPOLYGON guard — the regions are areal
--     entities and ST_Intersects against a non-areal would be undefined.
WITH anchored_roads AS (
    SELECT
      r.id, r.name, r.node_type, r.country_iso3, r.geometry,
      ROW_NUMBER() OVER (
        PARTITION BY r.country_iso3
        ORDER BY hash(r.id, {rand_seed_a})
      ) AS country_rank
    FROM {pool_a} r
  ),
  sampled_roads AS (
    SELECT *
    FROM anchored_roads
    WHERE country_rank <= {per_country_cap}
  ),
  road_geom_cte AS (
    SELECT
      sr.id          AS road_id,
      sr.name        AS road_name,
      sr.node_type   AS road_type,
      sr.country_iso3,
      ST_GeomFromWKT(sr.geometry) AS road_geom
    FROM sampled_roads sr
  ),
  region_geom_cte AS (
    SELECT
      reg.id        AS region_id,
      reg.name      AS region_name,
      reg.country_iso3,
      ST_GeomFromWKT(reg.geometry) AS region_geom
    FROM {pool_b} reg
    WHERE reg.geometry_type IN ('POLYGON', 'MULTIPOLYGON')
  )
SELECT
  -- Category label for {{entity_A}} (the counted region category).
  '{entity_a_label}' AS entity_a_name,
  rgeo.road_id     AS entity_b_id,
  rgeo.road_name   AS entity_b_name,
  CAST(COUNT(*) AS STRING) AS ground_truth
FROM road_geom_cte rgeo
JOIN region_geom_cte reggeo
  ON rgeo.country_iso3 = reggeo.country_iso3
 AND ST_Intersects(rgeo.road_geom, reggeo.region_geom)
GROUP BY rgeo.road_id, rgeo.road_name
HAVING COUNT(*) >= 2
LIMIT {total_sample};
