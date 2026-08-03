-- TID 64.1 — Grid conversion to H3 (resolution 7).
-- Question: "Convert the location of {{entity_A}} to H3 at resolution {{value}}."
-- Ground truth: H3 cell ID at resolution 7 covering the entity's centroid.
--
-- Single-pool unary projection (mirror of 37.1) with per-country
-- ROW_NUMBER hash diversity sampling.
--
-- ST_H3CellIDs returns an array; [0] picks the centroid cell. ST_PointOnSurface
-- yields a guaranteed-interior representative point for polygons (identity
-- for points/lines), so the cell assignment is deterministic regardless of
-- geometry type.
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
  CAST({h3_resolution} AS STRING) AS h3_resolution,
  -- Project resolution for the {{value}} placeholder in the question template.
  CAST({h3_resolution} AS STRING) AS value,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    ELSE CAST(
      ST_H3CellIDs(
        ST_PointOnSurface(ST_GeomFromWKT(a.geometry)),
        {h3_resolution},
        false
      )[0] AS STRING
    )
  END AS ground_truth
FROM anchors a
LIMIT {total_sample};
