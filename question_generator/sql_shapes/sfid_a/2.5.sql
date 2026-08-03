-- TID 2.5 — directional distance: northeast quadrant.
-- Intercardinal sibling of 2.1-2.4. Unlike the cardinals, intercardinals
-- DON'T project a single-axis distance — there's no natural "NE-aligned"
-- meridian to measure along. Instead the SELECT uses 1.1's straight-line
-- ST_ClosestPoint distance, and meta.join_condition adds a quadrant
-- filter (A north AND east of B) so only NE pairs land in the answer set.
--
-- Filter (in meta.join_condition):
--   ST_Y(centroid(a)) > ST_Y(centroid(b))  -- A north of B
--   ST_X(centroid(a)) > ST_X(centroid(b))  -- A east of B
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
