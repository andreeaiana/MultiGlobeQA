-- TID 19.1 — Overlap boolean (POLYGON × POLYGON).
-- Question: "Does {{entity_A}} overlap {{entity_B}}?"
-- Ground truth: 'true' iff ST_Overlaps(a.geometry, b.geometry).
--
-- ST_Overlaps requires SAME-DIMENSION geometries to be meaningful: it
-- evaluates to TRUE iff the geometries share some interior points but
-- neither is contained in the other. POINT × any returns FALSE
-- vacuously; POINT-allowed pools would produce a benchmark of all-false
-- answers. Variant filters enforce POLYGON / MULTIPOLYGON on both
-- pools.
--
-- Translation conventions identical to 13.1 / 14.1: see those headers
-- for Pre-CTE Geometry materialization, per-country ROW_NUMBER hash
-- diversity, and total_sample LIMIT rationale.
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
  CAST(ST_Overlaps(a.geom, b.geom) AS STRING) AS ground_truth
FROM pool_a_capped a
JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
