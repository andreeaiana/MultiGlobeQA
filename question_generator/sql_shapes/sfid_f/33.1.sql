-- TID 33.1 — Directional extreme: NORTHernmost POI within a region.
-- Question: "Which {{entity_A}} lies furthest north within {{entity_B}}?"
-- Ground truth: pool_a entity name with the maximum centroid latitude
--               among all entities ST_Contains'd by the region anchor.
--
-- Shape: 2-pool. pool_a is the POI category being ranked; pool_b is the
-- region anchor (one question per region). For each region we:
--   1. Find every pool_a entity ST_Contains'd by the region.
--   2. Rank by ST_Y(centroid) DESC (north = larger Y), tiebreak on name.
--   3. Pick rank = 1 — the northernmost POI in that region.
--
-- One row per region (not per global pool_a) lets the benchmark generate
-- many independent directional-extreme questions. Outer LIMIT
-- {total_sample} caps the total. Without the PARTITION BY the ranking
-- would emit ONE global answer (a single LIMIT 1 row), which doesn't
-- scale to a benchmark, so it partitions per region anchor instead.
--
-- All four 33.x sub-templates share this SQL; only the ORDER BY direction
-- changes (33.1 = ST_Y DESC, 33.2 = ST_Y ASC, 33.3 = ST_X ASC, 33.4 =
-- ST_X DESC). See sibling .sql files for the analogous comments.
WITH pool_b_geom AS (
    -- Materialize region geom with ST_MakeValid repair and geometry_type
    -- filter. Some osmh3kg region polygons pass ST_IsValid but still
    -- crash JTS ST_Contains with "side location conflict" — ST_MakeValid
    -- repairs them. The CTE forces materialization so Catalyst can't
    -- pipeline the repair past the spatial join.
    SELECT /*+ REPARTITION(16) */
      id, name, ST_MakeValid(ST_GeomFromWKT(geometry)) AS geom
    FROM {pool_b}
    WHERE geometry IS NOT NULL
      AND NOT geometry RLIKE 'NaN'
      AND geometry_type IN ('POLYGON', 'MULTIPOLYGON')
      AND ST_IsValid(ST_GeomFromWKT(geometry))
  ),
  ranked AS (
    SELECT
      a.id    AS entity_a_id,
      a.name  AS entity_a_name,
      b.id    AS entity_b_id,
      b.name  AS entity_b_name,
      a.name  AS ground_truth,
      ROW_NUMBER() OVER (
        PARTITION BY b.id
        ORDER BY ST_Y(ST_Centroid(ST_GeomFromWKT(a.geometry))) DESC,
                 a.name ASC
      ) AS rank_in_region
    FROM {pool_a} a
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, ST_GeomFromWKT(a.geometry))
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  ground_truth
FROM ranked
WHERE rank_in_region = 1
LIMIT {total_sample};
