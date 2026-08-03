-- TID 47.1 — Event area total (single-pool projection).
-- Question: "What is the total area of {{event_A}}?"
-- Ground truth: total area of event A's geometry in km² (ST_AreaSpheroid
-- on WGS84, divided by 1e6).
--
-- Single-pool unary projection (mirror of 37.1 / 12.1) with per-country
-- ROW_NUMBER hash diversity sampling.
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
  a.name AS event_a_name,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    ELSE CAST(
      ST_AreaSpheroid(ST_GeomFromWKT(a.geometry)) / 1000000.0
      AS STRING
    )
  END AS ground_truth
FROM anchors a
LIMIT {total_sample};
