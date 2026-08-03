-- TID 14.1 — Intersection boolean (geometries-emphasized, BOTH slots non-POINT).
-- Question: "Do the geometries of {{entity_A}} and {{entity_B}} intersect?"
-- Ground truth: 'true' iff ST_Intersects(a.geometry, b.geometry).
--
-- Same SQL body as 13.1 (intersection_boolean shape). 14.1 differs from
-- 13.1 ONLY in entity-type constraints declared in meta.variants (both
-- slots restricted to LINESTRING / MULTILINESTRING / POLYGON /
-- MULTIPOLYGON, no POINT). The slot pairings are road x road,
-- road x region and region x region -- never POI.
--
-- Translation conventions identical to 13.1: see 22.1.sql header for
-- the rationale on Pre-CTE Geometry materialization, per-country
-- ROW_NUMBER hash diversity, and total_sample LIMIT.
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
