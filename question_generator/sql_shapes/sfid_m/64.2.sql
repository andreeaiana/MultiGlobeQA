-- TID 64.2 — Grid conversion to S2 (level 15).
-- Question: "Convert the location of {{entity_A}} to S2 at level {{value}}."
-- Ground truth: S2 cell ID at level 15 covering the entity's centroid.
--
-- Sibling of 64.1 (H3) — same single-pool projection, different cell
-- system. ST_S2CellIDs returns an array of longs; [0] is the centroid cell.
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
  CAST({s2_level} AS STRING) AS s2_level,
  -- Project level for the {{value}} placeholder in the question template.
  CAST({s2_level} AS STRING) AS value,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    ELSE CAST(
      ST_S2CellIDs(
        ST_PointOnSurface(ST_GeomFromWKT(a.geometry)),
        {s2_level}
      )[0] AS STRING
    )
  END AS ground_truth
FROM anchors a
LIMIT {total_sample};
