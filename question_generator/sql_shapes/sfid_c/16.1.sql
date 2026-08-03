-- TID 16.1 — Intersection category count.
-- Question: "How many {{entity_A}} intersect {{entity_B}}?"
-- Ground truth: integer count of pool_a entities (the queried CATEGORY,
--               e.g. highways) that intersect each pool_b anchor.
--
-- Category-as-slot pattern (mirrors sfid_h 70.x and sfid_b 10.1):
-- pool_a is the UNSAMPLED full category pool; the question asks about
-- ALL entities of that category intersecting B, not a sampled subset.
-- pool_b is the anchor (sampled with per-country hash diversity); each
-- row in the result corresponds to one anchor entity B with its
-- intersection count over pool_a.
--
-- Variant differentiation: meta.parameters.entity_a_label pins the
-- PLURAL display label (e.g. "highways") rendered into entity_a_name
-- so the question template's {{entity_A}} substitution produces
-- grammatical output. The single deliverable variant for ftid 16.1 is
-- road_network_feature_category × road_network_feature_instance; the
-- one node_type bucket under road_network_feature_category in
-- entity_constraints/class_lookup.csv is `highway`, so the variant
-- pins that node_type via the pool filter and labels accordingly.
--
-- Aggregate gold (10.1-style): the count depends on every pool_a entity
-- matching the spatial predicate, so the gold extension UNIONs anchor
-- URIs with all matching pool_a entity URIs (see 26.1.gold.sql).
--
-- Sedona spatial-join pattern (memory: feedback_sedona_spatial_join):
-- pre-CTE materialize WKT -> Geometry, ST_Intersects in JOIN ON. This
-- activates RangeJoinExec for the LEFT JOIN; without the materialization,
-- inline ST_GeomFromWKT(...) inside JOIN ON often falls back to
-- CartesianProduct on USA-heavy KGs.
--
-- LEFT JOIN preserves zero-match anchors (ground_truth='0') — same as
-- 10.1's design. Do NOT switch to INNER JOIN: that silently drops
-- zero-match anchors and biases the benchmark toward anchors with at
-- least one intersector.
WITH pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
  pool_b_capped AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM pool_b_geom
  ),
  anchors AS (
    SELECT * FROM pool_b_capped WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  -- entity_a_name carries the PLURAL category label so the exporter's
  -- {{entity_A}} placeholder renders grammatically ("How many highways
  -- intersect ..." rather than "How many highway intersect ...").
  '{entity_a_label}' AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CAST(COUNT(a.id) AS STRING) AS ground_truth
FROM anchors b
LEFT JOIN pool_a_geom a
  ON {join_condition}
GROUP BY b.id, b.name
LIMIT {total_sample};
