-- TID 23.1 — Containment boolean (DIRECTIONAL: A contains B).
-- Question: "Does {{entity_A}} contain {{entity_B}}?"
-- Ground truth: 'true' iff ST_Contains(a.geometry, b.geometry).
--
-- DIRECTIONAL predicate, unlike the symmetric topology shapes. This
-- shape uses `WHERE a.id != b.id` (not `a.id < b.id`) because
-- both orderings are semantically distinct questions:
--   "Does Bavaria contain Munich?"  → true
--   "Does Munich contain Bavaria?"  → false
-- join_condition therefore keeps both orderings via `a.id != b.id`.
--
-- pool-A must be areal (POLYGON / MULTIPOLYGON) — only polygons can
-- "contain" anything in the strict ST_Contains sense (point and line
-- containment of the same point/line is degenerate). pool-B can be
-- any geometry type (POI / region / natfeat).
--
-- Translation conventions identical to the other sfid_c topology shapes.
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
  CAST(ST_Contains(a.geom, b.geom) AS STRING) AS ground_truth
FROM pool_a_capped a
JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
