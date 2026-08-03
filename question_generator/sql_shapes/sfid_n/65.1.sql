-- TID 65.1 — Centroid coordinate retrieval (region polygon).
-- Question: "What are the latitude and longitude of the centroid of {{entity_A}}?"
-- Ground truth: "lat,lon" composite string. ST_Centroid(WKT) gives the
-- polygon's geometric centroid; ST_Y / ST_X extract the coordinates.
--
-- Same single-pool structure as the plain coordinate-retrieval shapes:
-- only the SELECT's coordinate-extraction expression differs
-- (ST_Centroid wraps the geometry before ST_X / ST_Y).
--
-- ST_Centroid returns the GEOMETRIC centroid (NOT a representative
-- point). For non-convex polygons the centroid may fall outside the
-- polygon, but for benchmark purposes this is the canonical "center"
-- definition. ROUND to 6 dp avoids floating-point noise.
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
    SELECT id, name, node_type, subcategory, country_iso3, geometry
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id          AS entity_a_id,
  a.name        AS entity_a_name,
  a.node_type   AS entity_a_node_type,
  a.subcategory AS entity_a_subcategory,
  a.country_iso3 AS entity_a_country,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    ELSE CONCAT(
      CAST(ROUND(ST_Y(ST_Centroid(ST_GeomFromWKT(a.geometry))), 6) AS STRING),
      ',',
      CAST(ROUND(ST_X(ST_Centroid(ST_GeomFromWKT(a.geometry))), 6) AS STRING)
    )
  END AS ground_truth
FROM anchors a
LIMIT {total_sample};
