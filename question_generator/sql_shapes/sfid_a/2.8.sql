-- TID 2.8 — directional distance: southwest quadrant.
-- Intercardinal sibling. Same straight-line ST_ClosestPoint distance as
-- 2.5/2.6/2.7; meta.join_condition adds the SW quadrant filter
-- (A south AND west of B).
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
             ST_ClosestPoint(ST_GeomFromWKT(a.geometry),
                             ST_GeomFromWKT(b.geometry)),
             ST_ClosestPoint(ST_GeomFromWKT(b.geometry),
                             ST_GeomFromWKT(a.geometry))
           ) / 1000.0
    END AS STRING
  )                                     AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
