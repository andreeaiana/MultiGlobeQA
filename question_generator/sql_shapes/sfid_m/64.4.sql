-- TID 64.4 — Grid conversion to latitude-longitude grid (resolution {value}
-- degrees).
-- Question: "Convert the location of {{entity_A}} to latitude-longitude grid
--            at resolution {{value}} degrees."
-- Ground truth: a regular lat/lon grid cell label of the form
--   "<lat_bucket>_<lon_bucket>"
-- where each coordinate is snapped to the nearest multiple of
-- {latlon_resolution_deg}. e.g. for resolution 0.01°, Berlin (52.52, 13.40)
-- bucks to "52.52_13.40".
--
-- Sibling of 64.1 / 64.2 / 64.3 — same single-pool projection,
-- different cell system. Pure arithmetic on the centroid lat/lon: no Sedona
-- gridding function is needed (or available) for plain regular grids.
--   bucket = FLOOR(coord / {value}) * {value}
-- ROUND on the result avoids floating-point trailing digits like
-- "52.519999999..." that otherwise leak from the FLOOR/multiply.
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
  ),
  centroids AS (
    SELECT
      id,
      name,
      geometry,
      ST_X(ST_PointOnSurface(ST_GeomFromWKT(geometry))) AS lon,
      ST_Y(ST_PointOnSurface(ST_GeomFromWKT(geometry))) AS lat
    FROM anchors
  )
SELECT
  c.id   AS entity_a_id,
  c.name AS entity_a_name,
  CAST({latlon_resolution_deg} AS STRING) AS latlon_resolution_deg,
  -- Project resolution for the {{value}} placeholder in the question template.
  CAST({latlon_resolution_deg} AS STRING) AS value,
  CASE
    WHEN c.geometry IS NULL OR c.lon IS NULL OR c.lat IS NULL THEN NULL
    ELSE CONCAT(
      CAST(ROUND(FLOOR(c.lat / {latlon_resolution_deg}) * {latlon_resolution_deg}, 6) AS STRING),
      '_',
      CAST(ROUND(FLOOR(c.lon / {latlon_resolution_deg}) * {latlon_resolution_deg}, 6) AS STRING)
    )
  END AS ground_truth
FROM centroids c
LIMIT {total_sample};
