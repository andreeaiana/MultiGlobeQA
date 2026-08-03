-- TID 28.1 — Admin containment boolean with class-balanced sampling.
-- Question: "Is {{entity_A}} administratively part of {{entity_B}}?"
-- Ground truth: 'true' iff B's polygon contains A's polygon.
--
-- DESIGN: TRUE class via Sedona spatial join, FALSE class via plain
-- CROSS JOIN. The two classes have different characteristics that
-- justify the asymmetric SQL shape:
--
--   - TRUE class (~2% of pairs): needs to find the rare ST_Contains
--     matches across millions of admin polygon pairs. Without an
--     index, USA's 3K × 3K = 9M-pair country-keyed cross product
--     concentrates onto 1-2 input partitions and each task evaluates
--     ~625K ST_Contains calls on heavy state polygons → hangs.
--     Solution: put ST_Contains in JOIN ON (not WHERE) on a
--     Geometry-typed column (not WKT). Sedona detects this and
--     swaps in RangeJoinExec which builds an R-tree over one side
--     and probes with the other → O(N log M) instead of O(N×M).
--
--   - FALSE class (~98% of pairs): no R-tree pattern exists for
--     `NOT ST_Contains`. But because 98% of pairs ARE not-contained,
--     CROSS JOIN + WHERE NOT ST_Contains + LIMIT 25 finds matches
--     trivially in the first ~25 pairs scanned per partition. The
--     REPARTITION(16) hint on the input CTEs ensures the cross-join
--     spawns 16+ tasks instead of collapsing to 1-2.
--
-- WHY the geometry must be materialized in a CTE:
--   Sedona only recognizes the spatial join pattern when the JOIN ON
--   predicate sees Geometry-typed columns. Inline ST_GeomFromWKT(...)
--   inside the JOIN ON often blocks recognition and Spark falls back
--   to CartesianProduct. The pool_a_geom / pool_b_geom CTEs convert
--   WKT → Geometry once, upstream of the join.
WITH pool_a_geom AS (
    -- Materialize geometry as a Geometry-typed column (not a WKT string)
    -- so Sedona's optimizer can recognize the spatial join in the JOIN ON
    -- clause below and trigger its RangeJoin (R-tree) strategy. Without
    -- this CTE, inline ST_GeomFromWKT(...) inside the JOIN ON often
    -- prevents recognition and Spark falls back to CartesianProduct.
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
    -- Sedona spatial join: ST_Contains in JOIN ON triggers RangeJoinExec
    -- (broadcast R-tree of one side, probe with the other). For 3K×3K
    -- pools this is O(N log M) instead of O(N×M)=9M nested-loop pairs.
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      'true' AS ground_truth
    FROM pool_a_geom a
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, a.geom)
     AND {join_condition}
    LIMIT {target_true_count}
  ),
  false_sample AS (
    -- FALSE class can't use spatial join (no R-tree pattern for NOT-contains).
    -- But FALSE pairs are ~98% of the cross product, so plain CROSS JOIN +
    -- LIMIT finds 25 trivially in the first ~25 pairs scanned per partition.
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      'false' AS ground_truth
    FROM pool_a_geom a
    CROSS JOIN pool_b_geom b
    WHERE {join_condition}
      AND NOT ST_Contains(b.geom, a.geom)
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
