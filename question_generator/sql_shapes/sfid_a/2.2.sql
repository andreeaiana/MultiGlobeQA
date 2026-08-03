-- TID 2.2 — directional distance: south component.
-- Mirror of 2.1 (north) with the direction inverted. SELECT projects the
-- SOUTHWARD meridional component: distance from a point at (B.lon, A.lat)
-- to B's centroid, computed only when A is SOUTH of B (filter in
-- meta.join_condition: ST_Y(centroid(a)) < ST_Y(centroid(b))).
--
-- Distance formula is identical to 2.1 — the magnitude of the meridional
-- separation is direction-agnostic. Only the quadrant filter differs.
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
