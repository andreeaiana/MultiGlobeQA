-- TID 63.1 — Relative position (polar coordinates).
-- Question: "What is the relative position of {{entity_A}} in the
--            coordinate system of {{entity_B}}?"
-- Ground truth: "bearing_degrees,distance_km" — bearing is clockwise
-- from north (0°=N, 90°=E, 180°=S, 270°=W), distance is geodesic km.
--
-- Two-pool rank-equality structural-pair sampler (mirror of 1.1) with
-- the same-h3-cell constraint added to the join (mirror of 49.1) so
-- pairs are co-located, matching the registry's spatial_scope: same_cell.
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
  CASE
    -- NULL guard: missing geometry → NULL ground_truth → validator drops.
    WHEN a.geometry IS NULL OR b.geometry IS NULL THEN NULL
    ELSE CONCAT(
      CAST(
        DEGREES(
          ST_Azimuth(
            ST_PointOnSurface(ST_GeomFromWKT(b.geometry)),
            ST_PointOnSurface(ST_GeomFromWKT(a.geometry))
          )
        ) AS STRING
      ),
      ',',
      CAST(
        ST_DistanceSphere(
          ST_PointOnSurface(ST_GeomFromWKT(a.geometry)),
          ST_PointOnSurface(ST_GeomFromWKT(b.geometry))
        ) / 1000.0 AS STRING
      )
    )
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
