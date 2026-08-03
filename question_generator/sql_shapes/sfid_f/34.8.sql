-- TID 34.8 — Attribute ranking: smallest by compactness entity within a region.
-- Question: "What is the smallest {{entity_A}} in
--            {{entity_B}} by compactness?"
-- Ground truth: rank-1 pool_a entity per region anchor, by compactness.
--
-- Shape: 2-pool. pool_a is the candidate sub-region (region or natural
-- feature category) being ranked; pool_b is the encompassing region
-- anchor (one question per region). PARTITION BY b.id + WHERE
-- rank_in_region = 1 yields one rank-1 pool_a per region, scaling to
-- {total_sample} questions rather than a single global rank-1 row.
-- Tiebreak on a.name for determinism.
--
-- Geometry filter: POLYGON, MULTIPOLYGON.
-- Metric expression: compactness.
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
      (4 * 3.141592653589793 * ST_AreaSpheroid(ST_GeomFromWKT(a.geometry)))
        / POWER(ST_Length(ST_Boundary(ST_GeomFromWKT(a.geometry))), 2) AS compactness,
      a.name  AS ground_truth,
      ROW_NUMBER() OVER (
        PARTITION BY b.id
        ORDER BY (4 * 3.141592653589793 * ST_AreaSpheroid(ST_GeomFromWKT(a.geometry)))
               / POWER(ST_Length(ST_Boundary(ST_GeomFromWKT(a.geometry))), 2) ASC,
                 a.name ASC
      ) AS rank_in_region
    FROM pool_a_geom a
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, a.geom)
    WHERE a.country_iso3 = b.country_iso3
      AND a.geometry_type IN ('POLYGON', 'MULTIPOLYGON')
      AND ST_Length(ST_Boundary(ST_GeomFromWKT(a.geometry))) > 0
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  CAST(compactness AS STRING) AS compactness,
  ground_truth
FROM ranked
WHERE rank_in_region = 1
LIMIT {total_sample};
