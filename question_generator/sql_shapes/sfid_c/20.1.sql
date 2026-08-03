-- TID 20.1 — Boundary touch boolean.
-- Question: "Does the boundary of {{entity_A}} touch {{entity_B}}?"
-- Ground truth: 'true' iff ST_Touches(a.geometry, b.geometry).
--
-- ST_Touches: TRUE iff geometries share at least one boundary point but
-- have no interior points in common. Symmetric predicate. Defined for
-- any geometry type combination including POINT × POLYGON (POINT
-- touches polygon iff it lies exactly on the boundary).
--
-- Conventions identical to the rest of the sfid_c topology
-- shapes (13.1, 14.1, 19.1, 22.1, 23.1): see those headers for Pre-CTE
-- Geometry materialization, per-country ROW_NUMBER hash diversity, and
-- total_sample LIMIT rationale.
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
  CAST(ST_Touches(a.geom, b.geom) AS STRING) AS ground_truth
FROM pool_a_capped a
JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
