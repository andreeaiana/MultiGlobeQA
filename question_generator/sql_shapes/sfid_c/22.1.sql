-- TID 22.1 — Disjointness boolean.
-- Question: "Is {{entity_A}} disjoint from {{entity_B}}?"
-- Ground truth: 'true' iff ST_Disjoint(a.geometry, b.geometry).
--
-- ST_Disjoint is the logical negation of ST_Intersects: TRUE iff the
-- two geometries share NO points at all. Symmetric predicate. Defined
-- for any geometry type combination.
--
-- NOTE on result skew: in geographically large pools, the vast majority
-- of pairs ARE disjoint (most random POIs in different cities don't
-- intersect), so the natural distribution skews ~95%+ TRUE. The
-- per-country guard in join_condition narrows the cross-join to
-- co-country pairs, reducing the trivial-disjoint rate. False-premise
-- injection (handled at exporter level) supplements with engineered
-- TRUE/FALSE balance if needed; at SQL level we just emit the natural
-- per-pair predicate value.
--
-- Translation conventions identical to the rest of sfid_c.
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
  CAST(ST_Disjoint(a.geom, b.geom) AS STRING) AS ground_truth
FROM pool_a_capped a
JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
