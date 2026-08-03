-- TID 31.1 — Full containment boolean with class-balanced sampling.
-- Question: "Does the administrative boundary of {{entity_A}} fully
-- contain the area of {{entity_B}}?"
-- Ground truth: 'true' iff ST_Contains(A, B) — the FULL geometry of B
-- lies inside A (vs 28.1 which only tests B's centroid).
--
-- DESIGN: same Sedona spatial-join + class-balanced sampling pattern as
-- 52.1. The only semantic difference is that ST_Contains is applied to
-- B's full geometry instead of its centroid, so this is a stricter
-- containment test (a partially-overlapping region pair returns false
-- here but might return true under 28.1's centroid-only check).
--
-- See 52.1.sql for the long-form Sedona / class-balance rationale.
WITH pool_a_geom AS (
    -- Materialize geometry as a Geometry-typed column (not a WKT string)
    -- so Sedona's optimizer can recognize the spatial join in the JOIN ON
    -- clause below and trigger its RangeJoin (R-tree) strategy.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
  true_sample AS (
    -- Sedona spatial join: ST_Contains in JOIN ON triggers RangeJoinExec.
    -- Note A on the OUTER side of ST_Contains here: question semantics is
    -- "does A contain B" so A must be the container.
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      'true' AS ground_truth
    FROM pool_a_geom a
    JOIN pool_b_geom b
      ON ST_Contains(a.geom, b.geom)
     AND {join_condition}
    LIMIT {target_true_count}
  ),
  false_sample AS (
    -- FALSE class can't use spatial join (no R-tree pattern for NOT-contains).
    -- ~98% of pairs are NOT-contained, so plain CROSS JOIN + LIMIT finds
    -- matches trivially in the first ~25 pairs scanned per partition.
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      'false' AS ground_truth
    FROM pool_a_geom a
    CROSS JOIN pool_b_geom b
    WHERE {join_condition}
      AND NOT ST_Contains(a.geom, b.geom)
    LIMIT {target_false_count}
  ),
  combined AS (
    SELECT entity_a_id, entity_a_name, entity_b_id, entity_b_name, ground_truth
    FROM true_sample
    UNION ALL
    SELECT entity_a_id, entity_a_name, entity_b_id, entity_b_name, ground_truth
    FROM false_sample
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  ground_truth
FROM combined
LIMIT {total_sample};
