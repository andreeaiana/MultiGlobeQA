-- TID 53.1 — Area comparison boolean (cross-country, polygons only).
-- Question: "Is {{entity_A}} larger in area than {{entity_B}}?"
-- Ground truth: 'true' iff A's spheroidal area > B's spheroidal area.
--
-- The polygon-only filter lives in meta.applicable_kgs[*].pool_*.filter,
-- so the SQL itself is just rank-equality + area comparison.
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
    WHEN ST_AreaSpheroid(ST_GeomFromWKT(a.geometry))
       > ST_AreaSpheroid(ST_GeomFromWKT(b.geometry))
    THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
