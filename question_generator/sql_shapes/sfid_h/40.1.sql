-- TID 40.1 — H3 grid cell shared lookup.
-- Question: "Which grid cell contains both {{entity_A}} and {{entity_B}}?"
-- Ground truth: the h3_res3 cell ID (string) that both POIs share. The
-- join_condition pins both entities to the same cell, so a.h3_res3 ==
-- b.h3_res3 always — projecting either is correct.
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
  -- a.h3_res3 = b.h3_res3 in join_condition already drops NULL-h3 rows
  -- (NULL = NULL is NULL, filtered by INNER JOIN), so no NULL guard
  -- needed in the SELECT.
  CAST(a.h3_res3 AS STRING) AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
