-- TID 35.7 — Ranked attribute lookup: longest entity in region.
-- Question: "What is the longest {{entity_A}} in {{entity_B}}?"
-- Ground truth: rank-1 pool_a entity per region anchor, ordered
--               by length (DESC).
--
-- Shape: 2-pool. pool_a is the candidate (road_network_feature_category, filtered by
-- category and geometry type); pool_b is the region anchor (one
-- question per region). PARTITION BY b.id + WHERE rank_in_region = 1
-- yields one rank-1 entity per region, scaling to {total_sample}
-- questions rather than a single global result.
-- Tiebreak on a.name for determinism.
--
-- Siblings 35.8 and 35.9 share this template; only the selected rank
-- changes (1 / 2 / 3). All three keep the same ORDER BY metric and
-- direction.
WITH pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM {pool_b} b
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      b.*, ST_GeomFromWKT(b.geometry) AS geom
    FROM pool_b_capped b
    WHERE bucket_rank <= {per_country_cap}
  ),
  anchor_countries AS (
    SELECT DISTINCT country_iso3 FROM pool_b_geom
  ),
  pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      a.*, ST_GeomFromWKT(a.geometry) AS geom
    FROM {pool_a} a
    JOIN anchor_countries ac ON a.country_iso3 = ac.country_iso3
  ),
  ranked AS (
    SELECT /*+ BROADCAST(b) */
      a.id    AS entity_a_id,
      a.name  AS entity_a_name,
      b.id    AS entity_b_id,
      b.name  AS entity_b_name,
      ST_Length(ST_GeomFromWKT(a.geometry)) AS length_deg,
      a.name  AS ground_truth,
      ROW_NUMBER() OVER (
        PARTITION BY b.id
        ORDER BY ST_Length(ST_GeomFromWKT(a.geometry)) DESC,
                 a.name ASC
      ) AS rank_in_region
    FROM pool_a_geom a
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, a.geom)
    WHERE a.country_iso3 = b.country_iso3
      AND a.geometry_type IN ('LINESTRING', 'MULTILINESTRING')
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  CAST(length_deg AS STRING) AS length_deg,
  ground_truth
FROM ranked
WHERE rank_in_region = 1
LIMIT {total_sample};
