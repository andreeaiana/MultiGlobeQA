-- TID 49.1 — Cardinal direction boolean (north).
-- Question: "Is {{entity_A}} north of {{entity_B}}?"
-- Ground truth: 'true' if A's centroid latitude > B's centroid latitude.
--
-- Same rank-equality + same-cell structure as TID 1.1, but the SELECT
-- computes a boolean string instead of a distance. Variant scope:
-- in-country POIs within the same h3_res3 cell.
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
    -- NULL guard: surfaces missing geometry as NULL ground_truth so the
    -- validator drops the row, rather than silently labelling it 'false'.
    WHEN a.geometry IS NULL OR b.geometry IS NULL THEN NULL
    WHEN ST_Y(ST_Centroid(ST_GeomFromWKT(a.geometry)))
       > ST_Y(ST_Centroid(ST_GeomFromWKT(b.geometry)))
    THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
