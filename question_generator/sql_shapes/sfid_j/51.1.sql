-- TID 51.1 — Upstream/downstream along a shared waterway.
-- Question: "Does {{entity_A}} lie upstream or downstream of {{entity_B}}
--            along {{entity_river}}?"
-- Ground truth: 'upstream' | 'downstream' | NULL (same offset)
--
-- Method:
--   1. Cross-product POI A × POI B × candidate rivers, restricted to
--      same h3 cell + same country.
--   2. Project A and B onto each river via ST_LineLocatePoint (returns
--      a normalized 0..1 offset along the linestring).
--   3. Keep rows where BOTH A and B are within 1 km of the river
--      (ST_DistanceSphere ≤ 1000 m).
--   4. For each (A, B) pair, pick the river with the smallest combined
--      proximity (rank 1 by dist_a + dist_b).
--   5. Compare offsets: lower offset = upstream (OSM convention: way
--      direction = source → mouth = downstream).
--
-- NULL guards on every geometry surface missing-geometry rows as NULL
-- ground_truth so they're dropped.
--
-- Sedona pattern: ST_DistanceSphere predicates live in the JOIN ON
-- clauses (not WHERE) so RangeJoinExec activates and the cross product
-- doesn't materialize fully before filtering.
WITH pool_a_capped AS (
    SELECT
      a.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_a})) AS unique_id
    FROM {pool_a} a
  ),
  pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_b})) AS unique_id
    FROM {pool_b} b
  ),
  pool_river_capped AS (
    SELECT
      w.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_river})) AS unique_id
    FROM {pool_river} w
  ),
  -- Materialize Sedona Geometry once per row so the spatial JOIN ON
  -- predicates don't re-parse WKT for every candidate pair.
  pool_a_geom AS (
    SELECT
      a.id, a.name, a.country_iso3, a.h3_res3, a.unique_id, a.geometry,
      ST_Centroid(ST_GeomFromWKT(a.geometry)) AS a_pt
    FROM pool_a_capped a
    WHERE a.geometry IS NOT NULL
  ),
  pool_b_geom AS (
    SELECT
      b.id, b.name, b.country_iso3, b.h3_res3, b.unique_id, b.geometry,
      ST_Centroid(ST_GeomFromWKT(b.geometry)) AS b_pt
    FROM pool_b_capped b
    WHERE b.geometry IS NOT NULL
  ),
  pool_river_geom AS (
    SELECT
      w.id, w.name, w.country_iso3, w.h3_res3, w.unique_id, w.geometry,
      ST_GeomFromWKT(w.geometry) AS w_geom
    FROM pool_river_capped w
    WHERE w.geometry IS NOT NULL
  ),
  projections AS (
    SELECT
      a.id    AS entity_a_id,
      a.name  AS entity_a_name,
      b.id    AS entity_b_id,
      b.name  AS entity_b_name,
      w.id    AS river_id,
      w.name  AS river_name,
      ST_LineLocatePoint(w.w_geom, a.a_pt) AS offset_a,
      ST_LineLocatePoint(w.w_geom, b.b_pt) AS offset_b,
      ST_DistanceSphere(a.a_pt, ST_ClosestPoint(w.w_geom, a.a_pt))
        AS dist_a_river,
      ST_DistanceSphere(b.b_pt, ST_ClosestPoint(w.w_geom, b.b_pt))
        AS dist_b_river
    FROM pool_a_geom a
    INNER JOIN pool_b_geom b
      ON {join_condition}
    INNER JOIN pool_river_geom w
      ON a.country_iso3 = w.country_iso3
     AND a.h3_res3 = w.h3_res3
     AND ST_DistanceSphere(a.a_pt, ST_ClosestPoint(w.w_geom, a.a_pt)) <= 1000
     AND ST_DistanceSphere(b.b_pt, ST_ClosestPoint(w.w_geom, b.b_pt)) <= 1000
  ),
  ranked AS (
    SELECT
      entity_a_id, entity_a_name,
      entity_b_id, entity_b_name,
      river_id, river_name,
      offset_a, offset_b,
      ROW_NUMBER() OVER (
        PARTITION BY entity_a_id, entity_b_id
        ORDER BY (dist_a_river + dist_b_river) ASC
      ) AS river_rank
    FROM projections
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  river_id,
  river_name,
  CASE
    WHEN offset_a < offset_b THEN 'upstream'
    WHEN offset_a > offset_b THEN 'downstream'
    ELSE NULL  -- same offset on the river — registry specifies up/down only
  END AS ground_truth
FROM ranked
WHERE river_rank = 1
LIMIT {total_sample};
