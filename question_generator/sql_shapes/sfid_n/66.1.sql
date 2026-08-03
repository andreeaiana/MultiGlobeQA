-- TID 66.1 — Bounding box of an entity (axis-aligned envelope).
-- Question: "What is the bounding box of {{entity_A}}?"
-- Ground truth: "min_lat,min_lon,max_lat,max_lon" composite string —
-- single column so the validator's coordinates rule applies cleanly
-- (mirror sfid_o/63.1's composite-coordinate convention).
--
-- ST_Envelope returns the axis-aligned minimum bounding rectangle as a
-- POLYGON for non-degenerate geometries; ST_XMin / ST_YMin / ST_XMax /
-- ST_YMax extract the corner coordinates. For POINT entities the
-- envelope is the point itself (min == max), which is mathematically
-- correct (per sfid_p_spatial_reference.md design note on degenerate
-- bboxes). ROUND to 6 dp matches sibling 112/113/114 precision.
--
-- Single-pool unary projection — no inter-pool join, no spatial JOIN ON.
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
    SELECT id, name, node_type, geometry_type, subcategory, country_iso3, geometry
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id            AS entity_a_id,
  a.name          AS entity_a_name,
  a.node_type     AS entity_a_node_type,
  a.geometry_type AS entity_a_geometry_type,
  a.subcategory   AS entity_a_subcategory,
  a.country_iso3  AS entity_a_country,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    ELSE CONCAT(
      CAST(ROUND(ST_YMin(ST_Envelope(ST_GeomFromWKT(a.geometry))), 6) AS STRING),
      ',',
      CAST(ROUND(ST_XMin(ST_Envelope(ST_GeomFromWKT(a.geometry))), 6) AS STRING),
      ',',
      CAST(ROUND(ST_YMax(ST_Envelope(ST_GeomFromWKT(a.geometry))), 6) AS STRING),
      ',',
      CAST(ROUND(ST_XMax(ST_Envelope(ST_GeomFromWKT(a.geometry))), 6) AS STRING)
    )
  END AS ground_truth
FROM anchors a
LIMIT {total_sample};
