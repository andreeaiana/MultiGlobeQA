-- TID 38.1 — Centroid of the region containing a POI (two-pool).
-- Question: "What is the most central point of {{entity_A}} where
--            {{entity_B}} is located?"
--   e.g.    "What is the most central point of the country where the
--            Everest is located?"
-- Ground truth: "lat,lon" string from ST_PointOnSurface of the smallest
-- containing region's polygon (per registry: ground_truth_type =
-- coordinates).
--
-- SHAPE — two-pool, ST_Contains spatial join, ROW_NUMBER per-POI ranking.
-- Structurally similar to 64.1 but the containment direction is REVERSED:
-- the region (pool_a) contains the POI (pool_b), so the JOIN ON predicate
-- is ST_Contains(a.geom, b.geom) — region geom on the LEFT of
-- ST_Contains. For each POI, ROW_NUMBER ranks containing regions by
-- ST_Area ASC and the outer SELECT keeps rn = 1 — the SMALLEST containing
-- region per POI. This is a deliberate disambiguation: a POI may sit
-- inside multiple region instances of the queried category (e.g. a city
-- inside both a federal state and a country if the variant filters
-- region_category broadly), and the smallest one is the most specific
-- answer to "the region where X is located".
--
-- POIs with ZERO containing regions (oceans, disputed areas) are dropped
-- naturally (INNER JOIN — no row to keep when no region contains the
-- POI; there is no answer to "what is the most central point of the
-- country where X is located" when X is not in any country in the KG).
--
-- PERFORMANCE — Sedona spatial join (mirrors 10.1 / 11.1 / 64.1 pattern).
-- The pool_a_geom / pool_b_geom CTEs materialize `geom` as Geometry-typed
-- columns so ST_Contains in JOIN ON activates Sedona's RangeJoinExec.
-- Inline ST_GeomFromWKT(...) inside JOIN ON would block recognition and
-- degrade to CartesianProduct (see memory: feedback_sedona_spatial_join).
--
-- ANCHOR SAMPLING — per-country ROW_NUMBER hash diversity on pool_b.
-- pool_b (POIs) is the anchor side — one output row per anchor POI.
-- Same per-country bucket pattern as 10.1 / 11.1 / 12.1 / 37.1 / 64.1:
-- PARTITION BY country_iso3, ORDER BY hash(id, {rand_seed_b}), cap at
-- {per_country_cap}. POIs are heavily USA-biased on osmh3kg / kwg, so
-- per-country sampling is essential here.
--
-- CENTROID PROJECTION — ST_PointOnSurface (NOT ST_Centroid).
-- The output is a "lat,lon" pair representing a point on the region's
-- polygon. ST_PointOnSurface guarantees the point lies INSIDE the
-- polygon — important for concave / multipolygon shapes (Indonesia,
-- Chile) where ST_Centroid can fall in the ocean. Same primitive 37.1
-- uses for the same reason. ST_Area for the RANKING is fine either
-- way (ranking by area is monotone in either spheroidal or planar
-- area; ST_Area is cheaper).
WITH pool_a_geom AS (
    -- Materialize region geometry as Geometry-typed column. We also
    -- carry the raw `geometry` WKT through so the SELECT can apply
    -- ST_PointOnSurface to it for the OUTPUT centroid (the rendered
    -- ST_PointOnSurface in the SELECT works on the original WKT, not
    -- the pre-materialized `geom` column — keeping the projection
    -- explicit helps reviewers see what's being projected).
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, geometry,
      ST_MakeValid(ST_GeomFromWKT(geometry)) AS geom
    FROM {pool_a}
    WHERE geometry IS NOT NULL
      AND NOT geometry RLIKE 'NaN'
      AND geometry_type IN ('POLYGON', 'MULTIPOLYGON')
      AND ST_IsValid(ST_GeomFromWKT(geometry))
  ),
  pool_b_geom AS (
    -- POI side: id + name for output, country_iso3 for join, geom for
    -- the spatial predicate. POIs are points, so the WKT itself isn't
    -- carried (we don't need to project anything from the POI geometry
    -- to the output).
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_MakeValid(ST_GeomFromWKT(geometry)) AS geom
    FROM {pool_b}
    WHERE geometry IS NOT NULL
      AND NOT geometry RLIKE 'NaN'
      AND ST_IsValid(ST_GeomFromWKT(geometry))
  ),
  -- Per-country hash-diversity anchor sampling on the POI side.
  pool_b_capped AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM pool_b_geom
  ),
  anchors AS (
    SELECT id, name, country_iso3, geom
    FROM pool_b_capped
    WHERE bucket_rank <= {per_country_cap}
  ),
  ranked AS (
    -- For each anchor POI b, find containing regions a and rank them
    -- by region polygon area ASC. Smallest containing region is the
    -- most specific answer (rn = 1).
    SELECT
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      a.id       AS region_id,
      a.name     AS region_name,
      a.geometry AS region_geometry,
      ROW_NUMBER() OVER (
        PARTITION BY b.id
        ORDER BY ST_Area(a.geom) ASC
      ) AS rn
    FROM anchors b
    INNER JOIN pool_a_geom a
      ON {join_condition}
  )
SELECT
  entity_b_id   AS entity_b_id,
  entity_b_name AS entity_b_name,
  -- Region (entity_A) role columns. The exporter / question template
  -- substitutes the smallest-containing region's name into the
  -- {{entity_A}} slot via entity_a_name.
  region_id     AS entity_a_id,
  region_name   AS entity_a_name,
  -- ground_truth: "lat,lon" CSV from ST_PointOnSurface of the region.
  -- Same NULL-guard + CONCAT pattern 37.1 uses. ST_PointOnSurface (not
  -- ST_Centroid) — see header for rationale.
  CASE
    WHEN region_geometry IS NULL THEN NULL
    ELSE CONCAT(
      CAST(ST_Y(ST_PointOnSurface(ST_GeomFromWKT(region_geometry))) AS STRING),
      ',',
      CAST(ST_X(ST_PointOnSurface(ST_GeomFromWKT(region_geometry))) AS STRING)
    )
  END AS ground_truth
FROM ranked
WHERE rn = 1
-- Outer LIMIT enforces the global cap deterministically; per-country
-- diversity comes from the PARTITION above.
LIMIT {total_sample};
