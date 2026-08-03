-- TID 52.1 — Bearing classification (8-way categorical).
-- Question: "What direction would you travel from {{entity_A}} to reach
--            {{entity_B}}?"
-- Ground truth: 'North' | 'Northeast' | 'East' | 'Southeast'
--               | 'South' | 'Southwest' | 'West' | 'Northwest'
--
-- Method: compute the initial bearing (forward azimuth) from A to B
-- using spherical trigonometry on centroid lat/lon, then classify the
-- normalized [0, 360) bearing into 8 sectors of 45° each.
--
--   bearing = atan2(
--     sin(Δlon) * cos(lat_B),
--     cos(lat_A) * sin(lat_B) - sin(lat_A) * cos(lat_B) * cos(Δlon)
--   )
--   Normalized via MOD(degrees + 360, 360).
--
-- Same rank-equality + INNER JOIN pattern as 90.x; the spatial
-- computation is per-row in the SELECT, not a join predicate. The
-- variant (poi vs region) controls which categories the pools draw
-- from and whether the join_condition restricts to same h3 cell or
-- just same country (declared in meta.yaml's variant block).
--
-- NULL guard: missing geometry → NULL ground_truth (validator drops).
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
  paired AS (
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      a.geometry AS a_geom,
      b.geometry AS b_geom
    FROM pool_a_capped a
    INNER JOIN pool_b_capped b
      ON {join_condition}
  ),
  bearings AS (
    SELECT
      entity_a_id,
      entity_a_name,
      entity_b_id,
      entity_b_name,
      a_geom,
      b_geom,
      CASE
        WHEN a_geom IS NULL OR b_geom IS NULL THEN NULL
        ELSE MOD(
          DEGREES(
            ATAN2(
              SIN(RADIANS(
                ST_X(ST_Centroid(ST_GeomFromWKT(b_geom)))
                - ST_X(ST_Centroid(ST_GeomFromWKT(a_geom)))
              ))
              * COS(RADIANS(ST_Y(ST_Centroid(ST_GeomFromWKT(b_geom))))),
              COS(RADIANS(ST_Y(ST_Centroid(ST_GeomFromWKT(a_geom)))))
              * SIN(RADIANS(ST_Y(ST_Centroid(ST_GeomFromWKT(b_geom)))))
              - SIN(RADIANS(ST_Y(ST_Centroid(ST_GeomFromWKT(a_geom)))))
              * COS(RADIANS(ST_Y(ST_Centroid(ST_GeomFromWKT(b_geom)))))
              * COS(RADIANS(
                  ST_X(ST_Centroid(ST_GeomFromWKT(b_geom)))
                  - ST_X(ST_Centroid(ST_GeomFromWKT(a_geom)))
              ))
            )
          ) + 360, 360
        )
      END AS bearing
    FROM paired
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  CASE
    WHEN bearing IS NULL THEN NULL
    WHEN bearing >= 337.5 OR bearing < 22.5  THEN 'North'
    WHEN bearing >= 22.5  AND bearing < 67.5  THEN 'Northeast'
    WHEN bearing >= 67.5  AND bearing < 112.5 THEN 'East'
    WHEN bearing >= 112.5 AND bearing < 157.5 THEN 'Southeast'
    WHEN bearing >= 157.5 AND bearing < 202.5 THEN 'South'
    WHEN bearing >= 202.5 AND bearing < 247.5 THEN 'Southwest'
    WHEN bearing >= 247.5 AND bearing < 292.5 THEN 'West'
    WHEN bearing >= 292.5 AND bearing < 337.5 THEN 'Northwest'
  END AS ground_truth
FROM bearings
LIMIT {total_sample};
