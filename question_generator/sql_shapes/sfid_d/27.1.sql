-- TID 27.1 — POI-near-road COUNT ("path proximity").
-- Question: "How many {{entity_A}} does {{entity_B}} pass by?"
-- Ground truth: integer count of pool_a (POI) entities of the queried
-- category whose geometry lies within {proximity_buffer_meters} of
-- each pool_b (road) anchor's LINESTRING.
--
-- COUNT sibling of TID 42.1. Identical SQL except the projection: the
-- set TID's `to_json(sort_array(collect_set(a.name)))` becomes
-- `CAST(COUNT(a.id) AS STRING)` here. Same anchor source (per-session
-- aggregated road view), same proximity predicate, same per-country
-- ROW_NUMBER hash diversity, same LEFT JOIN preserve-zero-match
-- semantics. Keep the two files in lock-step — see MAINTENANCE
-- CONTRACT in 42.1.sql header.
--
-- ZERO-MATCH SEMANTICS:
--   With LEFT JOIN, an anchor road with no nearby POIs of the queried
--   category yields a single row whose `a.*` columns are NULL.
--   COUNT(a.id) on a single NULL row evaluates to 0 (COUNT of a column
--   ignores NULLs, unlike COUNT(*) which would return 1). Cast to
--   STRING for validator/exporter consistency. So zero-match anchors
--   render the question "How many cafes does road X pass by?" with
--   ground_truth='0' instead of being silently dropped. Do NOT switch
--   to COUNT(*) here: it would inflate zero-match rows to 1.
--
-- ANCHOR SOURCE — osmh3kg_roads_aggregated.
--   See 42.1.sql header for the full explanation. Briefly: per-session
--   view registered by ``road_setup.register_road_aggregation_views``,
--   one row per (country_iso3, name, node_type) named road with merged
--   geometry, MD5 id, MIN(entity_uri) as the anchor URI. osmh3kg-only.
--
-- PROXIMITY PREDICATE — ST_DWithin in planar degrees.
--   ST_DWithin(b.geom, a.geom, {proximity_buffer_meters} * 0.00001).
--   See 42.1.sql for the metres→degrees conversion rationale.
--
-- PERFORMANCE — Sedona spatial join.
--   Pool_a_geom and pool_b_geom CTEs pre-materialize Geometry-typed
--   columns so Sedona's optimizer recognizes ST_DWithin in JOIN ON
--   as a spatial join and triggers RangeJoinExec (R-tree). Inline
--   ST_GeomFromWKT(...) inside JOIN ON would block recognition (see
--   memory: feedback_sedona_spatial_join).
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity.
--   See 42.1.sql header.
WITH pool_a_geom AS (
    -- POIs: id (counted), country_iso3 + node_type for the join, geom
    -- for the spatial predicate. Drop raw `geometry` WKT and `name`
    -- (the count doesn't enumerate names, unlike 26.1) to keep the
    -- joined frame narrow.
    SELECT /*+ REPARTITION(16) */
      id, country_iso3, node_type,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    -- Roads (aggregated): id (MD5 anchor key), name (rendered into
    -- {{entity_B}}), country_iso3 (PARTITION BY for diversity),
    -- node_type (debug carry-over), geom for the spatial predicate.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, node_type,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
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
    SELECT * FROM pool_b_capped WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  -- entity_a_name carries the PLURAL category label ('cafes') so the
  -- exporter's {{entity_A}} placeholder renders a grammatical question
  -- ("How many cafes does road X pass by?") rather than the singular
  -- ("How many cafe does road X pass by?") which uses the node_type
  -- value needed for the SQL filter.
  '{entity_a_category_label}' AS entity_a_name,
  -- entity_a_subcategory keeps the SINGULAR node_type for technical
  -- reference (matches what the SQL filter actually ran with).
  '{entity_a_category}'       AS entity_a_subcategory,
  b.id                        AS entity_b_id,
  b.name                      AS entity_b_name,
  -- ground_truth: integer count CAST to STRING (validator's count rule
  -- accepts numeric strings). COUNT(a.id) — NOT COUNT(*) — so zero-
  -- match anchors emit '0' rather than '1' (LEFT JOIN preservation
  -- semantics; see ZERO-MATCH SEMANTICS in header).
  CAST(COUNT(a.id) AS STRING)  AS ground_truth
FROM anchors b
LEFT JOIN pool_a_geom a
  -- Same-country prune (the road and POI must share country_iso3 — the
  -- aggregated view's `country_iso3` originates from strata enrichment).
  -- a.id != b.id is technically unreachable (POI ids and road MD5 ids
  -- live in disjoint name-spaces) but kept for symmetry with 26.1 and
  -- as a defensive guard if future merges allow id-collision.
  ON a.country_iso3 = b.country_iso3
  AND a.id != b.id
  AND a.node_type = '{entity_a_category}'
  AND ST_DWithin(b.geom, a.geom, {proximity_buffer_meters} * 0.00001)
GROUP BY b.id, b.name
-- Outer LIMIT: per-country cap × ~150 countries can exceed total_sample.
-- This LIMIT enforces the global cap deterministically.
LIMIT {total_sample};
