-- TID 5.1 — Cross-country resolved-anchor distance comparison.
-- Question: "Which is closer to the country that borders the country
--            containing {{entity_A}}: {{entity_B}} or {{entity_C}}?"
-- Ground truth: name of B or C — whichever is closer to the (singular)
--               bordering country's boundary.
--
-- DESIGN — the "resolved" anchor is computed entirely in SQL:
--
--   1. anchor_a       — sample a POI (pool_a) whose containing country has
--                       EXACTLY ONE land-border neighbor (so "the country
--                       that borders" is unambiguous; multi-neighbor
--                       countries are excluded by design)
--   2. bordering_poly — the single neighbor's country polygon (joined via
--                       the country_polygons session view registered in
--                       country_setup.py)
--   3. dist_b, dist_c — geodesic distance from each candidate (b, c) to
--                       the bordering country's polygon boundary
--   4. SELECT         — CASE picks the closer candidate as ground_truth
--
-- COUNTRIES WITH EXACTLY ONE LAND NEIGHBOR (Natural Earth via
-- ST_Touches): ~12-18 countries including Ireland (UK), Lesotho (ZAF),
-- Portugal (ESP), Denmark (DEU), South Korea (PRK), Brunei (MYS),
-- Qatar (SAU), Gambia (SEN), East Timor (IDN), PNG (IDN), Monaco (FRA),
-- San Marino + Vatican (ITA), Andorra (FRA), Liechtenstein (DEU/CHE).
-- Maritime-only borders are NOT counted (Australia ↔ PNG is sea).
--
-- B/C SCOPE — both candidates are sampled from the SAME COUNTRY as
-- anchor_a. This avoids a global cross-join blowup (POIs × POIs over
-- the whole world) and keeps the comparison meaningful (questions like
-- "Which is closer ... Berlin or Munich?" make sense; "... Tokyo or
-- Paris?" doesn't). Per-variant filters narrow the candidate type.
--
-- KG SCOPE — all 3 KGs declare the variant; the country_polygons view
-- is KG-agnostic. Per-KG vocabulary differences in pool_b/pool_c
-- filters are encoded in meta.applicable_kgs.
--
-- Sampling — per-country ROW_NUMBER hash diversity on anchor_a. With
-- ~12-18 eligible countries, per_country_cap=10 yields up to ~150-180
-- candidate anchors before the (b, c) cross-join + LIMIT trim.
WITH country_neighbors AS (
    -- Adjacency from Natural Earth via ST_Touches. KG-agnostic; uses
    -- the country_polygons session view registered at init_spark.
    SELECT
      a.iso_a3 AS country,
      b.iso_a3 AS neighbor
    FROM country_polygons a
    JOIN country_polygons b
      ON ST_Touches(a.geom, b.geom)
     AND a.iso_a3 != b.iso_a3
  ),
  countries_with_one_neighbor AS (
    -- Filter to countries whose "the bordering country" is unambiguous.
    -- HAVING COUNT(*) = 1 enforces the question's singular determiner.
    SELECT
      country,
      MIN(neighbor) AS sole_neighbor
    FROM country_neighbors
    GROUP BY country
    HAVING COUNT(*) = 1
  ),
  pool_a_geom AS (
    -- Pre-CTE Geometry materialization for any downstream ST_* call
    -- (per feedback_sedona_spatial_join memory).
    SELECT
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  anchor_a_capped AS (
    -- Per-country ROW_NUMBER hash diversity (per project_sampling_rewrite
    -- memory). Restrict to anchor's country having EXACTLY ONE neighbor.
    SELECT
      a.id, a.name, a.country_iso3, a.geom,
      cn.sole_neighbor AS border_iso3,
      ROW_NUMBER() OVER (
        PARTITION BY a.country_iso3
        ORDER BY hash(a.id, {rand_seed_a})
      ) AS country_rank
    FROM pool_a_geom a
    JOIN countries_with_one_neighbor cn ON a.country_iso3 = cn.country
  ),
  anchor_a AS (
    SELECT * FROM anchor_a_capped WHERE country_rank <= {per_country_cap}
  ),
  bordering_poly AS (
    -- Single bordering country polygon per anchor — join on the
    -- pre-computed sole_neighbor.
    SELECT
      a.id AS anchor_id, a.country_iso3 AS anchor_country,
      a.border_iso3,
      cp.geom AS border_geom
    FROM anchor_a a
    JOIN country_polygons cp ON cp.iso_a3 = a.border_iso3
  ),
  pool_b_geom AS (
    SELECT
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
  pool_c_geom AS (
    SELECT
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_c}
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  c.id   AS entity_c_id,
  c.name AS entity_c_name,
  CASE
    WHEN ST_DistanceSphere(b.geom, bp.border_geom)
       < ST_DistanceSphere(c.geom, bp.border_geom)
      THEN b.name
    ELSE c.name
  END AS ground_truth
FROM anchor_a a
JOIN bordering_poly bp ON bp.anchor_id = a.id
JOIN pool_b_geom b ON b.country_iso3 = a.country_iso3 AND b.id != a.id
JOIN pool_c_geom c ON c.country_iso3 = a.country_iso3 AND c.id != a.id AND c.id != b.id
LIMIT {total_sample};
