-- TID 13.1 — Intersection boolean (general, POINT-allowed in slot A).
-- Question: "Does {{entity_A}} intersect {{entity_B}}?"
-- Ground truth: 'true' iff ST_Intersects(a.geometry, b.geometry).
--
-- Sampling notes:
--   - {pool_a} / {pool_b} are substituted with the run's entity pools.
--   - Per-country ROW_NUMBER hash diversity bounds candidate pools
--     before the cross-join — avoids USA-skew driving the LIMIT to
--     pull mostly USA pairs.
--   - Pre-CTE ST_GeomFromWKT materialization so Sedona can recognize
--     the spatial predicate, and so the join and the ST_Intersects
--     projection see a Geometry column rather than re-parsing WKT on
--     every comparison.
--   - LIMIT {total_sample} caps the output at the run-spec sample size.
--
-- Symmetric predicate (`a.id < b.id` for de-duplication) and shared
-- pool/category in symmetric variants is handled by independent shuffle
-- seeds (rand_seed_a != rand_seed_b), so a pool reused on both sides
-- still yields distinct ranks.
WITH pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      a.id, a.name, a.country_iso3,
      ST_GeomFromWKT(a.geometry) AS geom,
      ROW_NUMBER() OVER (
        PARTITION BY a.country_iso3
        ORDER BY hash(a.id, {rand_seed_a})
      ) AS bucket_rank
    FROM {pool_a} a
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      b.id, b.name, b.country_iso3,
      ST_GeomFromWKT(b.geometry) AS geom,
      ROW_NUMBER() OVER (
        PARTITION BY b.country_iso3
        ORDER BY hash(b.id, {rand_seed_b})
      ) AS bucket_rank
    FROM {pool_b} b
  ),
  pool_a_capped AS (
    SELECT * FROM pool_a_geom WHERE bucket_rank <= {per_country_cap}
  ),
  pool_b_capped AS (
    SELECT * FROM pool_b_geom WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CAST(ST_Intersects(a.geom, b.geom) AS STRING) AS ground_truth
FROM pool_a_capped a
JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
