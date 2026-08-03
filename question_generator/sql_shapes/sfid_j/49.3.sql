-- TID 49.3 — Cardinal direction boolean (east).
-- Question: "Is {{entity_A}} east of {{entity_B}}?"
-- Ground truth: 'true' if A's centroid longitude > B's centroid longitude.
--
-- Cardinal sibling of 49.1/49.2: same rank-equality + same-cell, ST_X
-- comparison instead of ST_Y.
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
    WHEN a.geometry IS NULL OR b.geometry IS NULL THEN NULL
    WHEN ST_X(ST_Centroid(ST_GeomFromWKT(a.geometry)))
       > ST_X(ST_Centroid(ST_GeomFromWKT(b.geometry)))
    THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
