-- TID 2.1 — directional distance: north component.
-- Rank-equality structural-pair sampler (mirror of 1.1) with the SELECT
-- projecting the NORTHWARD (meridional) component of A→B distance: the
-- distance from B's centroid to a point at (B.lon, A.lat). This isolates
-- the N-S separation and ignores E-W offset.
--
-- Direction filter lives in meta.join_condition:
--   ST_Y(ST_Centroid(a.geometry)) > ST_Y(ST_Centroid(b.geometry))
-- i.e. A is north of B. Pairs failing this WHERE are dropped before
-- LIMIT, so the answer set is non-trivially direction-constrained.
--
-- For 2.5–2.8 (intercardinal NE/NW/SE/SW) the SELECT uses 1.1's
-- straight-line ST_ClosestPoint distance instead — the quadrant filter
-- in join_condition already constrains direction, so no axis projection
-- is needed.
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
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CAST(
    CASE
      WHEN ST_Intersects(ST_GeomFromWKT(a.geometry),
                         ST_GeomFromWKT(b.geometry))
        THEN 0.0
      -- Project A's lat onto B's lon, then measure to B's centroid.
      -- Pure N-S distance; ignores how far east/west A is from B.
      ELSE ST_DistanceSphere(
             ST_Point(
               ST_X(ST_Centroid(ST_GeomFromWKT(b.geometry))),
               ST_Y(ST_Centroid(ST_GeomFromWKT(a.geometry)))
             ),
             ST_Centroid(ST_GeomFromWKT(b.geometry))
           ) / 1000.0
    END AS STRING
  )                                     AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
