-- TID 64.3 — Grid conversion to Geohash (precision 6).
-- Question: "Convert the location of {{entity_A}} to geohash at precision
--            {{value}} characters."
-- Ground truth: geohash string at precision 6 (≈ 1.2km × 600m cells).
--
-- Sibling of 64.1 / 111.2. ST_GeoHash returns a string directly — no
-- array indexing needed.
WITH pool_a_capped AS (
    SELECT
      a.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS bucket_rank
    FROM {pool_a} a
  ),
  anchors AS (
    SELECT id, name, geometry
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  CAST({geohash_precision} AS STRING) AS geohash_precision,
  -- Project precision for the {{value}} placeholder in the question template.
  CAST({geohash_precision} AS STRING) AS value,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    ELSE ST_GeoHash(
      ST_PointOnSurface(ST_GeomFromWKT(a.geometry)),
      {geohash_precision}
    )
  END AS ground_truth
FROM anchors a
LIMIT {total_sample};
