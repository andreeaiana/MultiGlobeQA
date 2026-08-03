-- TID 50.1 — Left/right of route (categorical).
-- Question: "Is {{entity_A}} to the left or right of the route from
--            {{entity_B}} to {{entity_C}}?"
-- Ground truth: 'left' | 'right' | NULL (collinear)
--
-- Method: 2D cross product on centroid coordinates of A relative to the
-- B→C straight-line route:
--   cross = (C.x - B.x) * (A.y - B.y) - (C.y - B.y) * (A.x - B.x)
--   cross > 0 → left, cross < 0 → right, cross = 0 → collinear (NULL)
--
-- Three-pool rank-equality + same-cell + same-country join. The collinear
-- branch returns NULL (not 'collinear') so the validator drops the row —
-- the registry contract specifies left/right ground truth only.
--
-- NULL guards on every geometry surface missing-geometry rows as NULL
-- ground_truth so they're dropped, rather than silently classified.
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
  pool_c_capped AS (
    SELECT
      c.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_c})) AS unique_id
    FROM {pool_c} c
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  c.id   AS entity_c_id,
  c.name AS entity_c_name,
  CASE
    -- NULL guard: any missing geometry → NULL ground_truth (validator drops).
    WHEN a.geometry IS NULL OR b.geometry IS NULL OR c.geometry IS NULL THEN NULL
    WHEN (
      (ST_X(ST_Centroid(ST_GeomFromWKT(c.geometry)))
         - ST_X(ST_Centroid(ST_GeomFromWKT(b.geometry))))
      * (ST_Y(ST_Centroid(ST_GeomFromWKT(a.geometry)))
         - ST_Y(ST_Centroid(ST_GeomFromWKT(b.geometry))))
      - (ST_Y(ST_Centroid(ST_GeomFromWKT(c.geometry)))
         - ST_Y(ST_Centroid(ST_GeomFromWKT(b.geometry))))
      * (ST_X(ST_Centroid(ST_GeomFromWKT(a.geometry)))
         - ST_X(ST_Centroid(ST_GeomFromWKT(b.geometry))))
    ) > 0 THEN 'left'
    WHEN (
      (ST_X(ST_Centroid(ST_GeomFromWKT(c.geometry)))
         - ST_X(ST_Centroid(ST_GeomFromWKT(b.geometry))))
      * (ST_Y(ST_Centroid(ST_GeomFromWKT(a.geometry)))
         - ST_Y(ST_Centroid(ST_GeomFromWKT(b.geometry))))
      - (ST_Y(ST_Centroid(ST_GeomFromWKT(c.geometry)))
         - ST_Y(ST_Centroid(ST_GeomFromWKT(b.geometry))))
      * (ST_X(ST_Centroid(ST_GeomFromWKT(a.geometry)))
         - ST_X(ST_Centroid(ST_GeomFromWKT(b.geometry))))
    ) < 0 THEN 'right'
    ELSE NULL  -- collinear (cross == 0) — registry specifies left/right only
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
INNER JOIN pool_c_capped c
  ON a.country_iso3 = c.country_iso3
 AND a.id != c.id
 AND b.id != c.id
 AND a.unique_id = c.unique_id
 AND a.h3_res3 = c.h3_res3
LIMIT {total_sample};
