-- TID 46.1 — Event temporal overlap (boolean).
-- Question: "Did {{event_A}} and {{event_B}} overlap in time?"
-- Ground truth: 'true' if [a.start_date, a.end_date] ∩ [b.start_date, b.end_date] ≠ ∅,
-- 'false' otherwise. Both events must share country.
--
-- Two-pool rank-equality structural-pair sampler (mirror of 1.1) with
-- distinct seeds so symmetric pools don't collapse to the same entity.
-- Same-country in join_condition; no h3 cell constraint (events can
-- span large regions and we want temporally co-occurring pairs even if
-- spatially distant).
--
-- Template uses {{event_A}} and {{event_B}} which the exporter maps to
-- entity_a_name and entity_b_name respectively — clean placeholder
-- mapping, no slot collision.
WITH pool_a_capped AS (
    SELECT a.*, ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_a})) AS unique_id
    FROM {pool_a} a
  ),
  pool_b_capped AS (
    SELECT b.*, ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_b})) AS unique_id
    FROM {pool_b} b
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  a.name AS event_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  b.name AS event_b_name,
  CASE
    WHEN a.start_date IS NULL OR a.end_date IS NULL
      OR b.start_date IS NULL OR b.end_date IS NULL THEN NULL
    WHEN a.start_date <= b.end_date AND b.start_date <= a.end_date THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
