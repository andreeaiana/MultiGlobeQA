-- TID 17.1 — Intersection category enumeration.
-- Question: "Which {{entity_A}} intersect {{entity_B}}?"
-- Ground truth: SORTED set of pool_a entity NAMES that intersect each
--               pool_b anchor.
--
-- Same shape as 16.1 (intersection_category_count) except the aggregate
-- swaps COUNT(*) for SORT_ARRAY(COLLECT_LIST(a.name)) — set enumeration
-- rather than scalar count. pool_a is UNSAMPLED (the question asks
-- about ALL category entities); pool_b is the per-country diversity-
-- sampled anchor set.
--
-- Variant differentiation: meta.parameters.entity_a_label pins the
-- PLURAL display label. Single deliverable variant for ftid 17.1 is
-- road_network_feature_category × road_network_feature_instance, which
-- resolves to node_type='highway' per entity_constraints/class_lookup.csv.
--
-- Aggregate gold (10.1-style): the enumeration depends on every pool_a
-- entity matching the spatial predicate, so the gold extension UNIONs
-- anchor URIs with all matching pool_a entity URIs (see 27.1.gold.sql).
-- A model retrieving the gold should see every entity name appearing in
-- the answer set — without the extension, only the anchor's triples
-- would be available.
--
-- Sedona spatial-join pattern (memory: feedback_sedona_spatial_join):
-- pre-CTE materialize WKT -> Geometry, ST_Intersects in JOIN ON.
-- Without the materialization, the GROUP BY + COLLECT_LIST aggregate
-- can stall on USA-heavy KGs as Spark falls back to CartesianProduct.
--
-- LEFT JOIN preserves zero-match anchors (ground_truth=empty array).
-- Acceptable: enumerating "no intersector" is a meaningful answer.
WITH pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
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
  -- {{entity_A}} placeholder renders grammatically.
  '{entity_a_label}' AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CAST(SORT_ARRAY(COLLECT_LIST(a.name)) AS STRING) AS ground_truth
FROM anchors b
LEFT JOIN pool_a_geom a
  ON {join_condition}
GROUP BY b.id, b.name
LIMIT {total_sample};
