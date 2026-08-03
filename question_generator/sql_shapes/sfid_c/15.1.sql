-- TID 15.1 — Intersection point count between two LINESTRING geometries.
-- Question: "In how many points do {{entity_A}} and {{entity_B}} intersect?"
-- Ground truth: integer count of POINT components in
--               ST_Intersection(a, b).
--
-- LINESTRING × LINESTRING ONLY (deliverable section "intersection_point_count"):
-- POLYGON intersections return area geometries, not points; restricting to
-- linear geometries lets ST_CollectionExtract(..., 1) extract just the
-- point components. Both pool filters in meta.yaml require
-- geometry_type IN ('LINESTRING', 'MULTILINESTRING') — enforced at pool
-- view registration time so the SQL can assume the constraint holds.
--
-- Sedona spatial-join pattern:
-- pre-CTE materialize WKT -> Geometry, ST_Intersects in JOIN ON. This
-- activates RangeJoinExec instead of CartesianProduct. The
-- `WHERE a.id < b.id` symmetric-pair guard lives in the join_condition
-- (rendered into JOIN ON via {join_condition}) so it composes with the
-- spatial predicate without breaking the optimizer's pattern recognition.
--
-- Hash-diversity per-country anchor sampling on pool_b (same pattern as
-- 10.1). pool_a is the candidate set the spatial join probes; pool_b
-- carries the per-country diversity ROW_NUMBER. Without per-country caps,
-- USA-dominated KGs (osmh3kg, kwg) would draw most pairs from USA cells.
--
-- Returns a NUMERIC ground_truth (integer point count) — Numeric
-- ground-truth-type per registry. ST_NumGeometries on an empty collection
-- returns 0 via COALESCE; the LEFT JOIN preserves zero-intersection pairs
-- when desired (here we use INNER because a non-intersecting pair has
-- nothing to count by definition).
WITH pool_a_geom AS (
    -- Materialize geometry as Geometry-typed (not WKT) so Sedona's
    -- optimizer recognises the spatial join in the JOIN ON clause and
    -- triggers RangeJoinExec. Inline ST_GeomFromWKT(...) in JOIN ON
    -- often blocks recognition.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
  pool_b_capped AS (
    -- Per-country diversity sampling on pool_b. PARTITION BY country_iso3
    -- caps each country's anchor count at {per_country_cap}; the global
    -- LIMIT {total_sample} bounds total questions.
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM pool_b_geom
  ),
  anchors AS (
    SELECT * FROM pool_b_capped WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CAST(
    COALESCE(
      ST_NumGeometries(
        ST_CollectionExtract(
          ST_Intersection(a.geom, b.geom),
          1
        )
      ),
      0
    ) AS STRING
  ) AS ground_truth
FROM pool_a_geom a
JOIN anchors b
  ON ST_Intersects(a.geom, b.geom)
 AND {join_condition}
LIMIT {total_sample};
