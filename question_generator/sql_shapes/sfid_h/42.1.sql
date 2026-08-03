-- TID 42.1 — Multi-index finest single-cell cover (H3 + S2).
-- Question: "Which grid cell contains {{entity_A}} both in h3 and s2 formats?"
-- Ground truth: composite string
--   "H3(res=R): <cell_id> | S2(lvl=L): <cell_id>"
-- where R / L are the FINEST H3 resolution / S2 level whose centroid-cell
-- still fully contains the region's WKT geometry.
--
-- SHAPE — single-pool unary aggregation with two parallel resolution-search
-- CTE families (one for H3 0..15, one for S2 0..20). Mirrors 37.1's
-- single-pool anchor sampling (per-country ROW_NUMBER hash + per_country_cap
-- + outer LIMIT total_sample) — NOT 1.1's rank-equality two-pool pattern.
--
-- Algorithm per anchor:
--   1. Compute centroid of region geometry.
--   2. CROSS JOIN with the H3 resolution array (0..15); for each (entity, res):
--        compute the H3 cell at the centroid, convert it back to a polygon
--        via ST_H3ToGeom, check ST_Contains(cell_polygon, region_geom).
--   3. Take MAX(resolution) where containment holds → finest covering H3 cell.
--   4. Repeat steps 2-3 for S2 with levels 0..20 and ST_S2CellIDs / ST_S2ToGeom.
--   5. JOIN the H3 and S2 results back on entity_a_id, project the composite
--      ground_truth string.
--
-- Cost note: every anchor pays a CROSS JOIN over the resolution array
-- (16 + 21 = 37 candidate cells per anchor), each with a Sedona
-- cell→polygon conversion + ST_Contains. Per_country_cap on the anchor
-- pool is the load brake — keep total anchors modest until the cell-polygon
-- materialization cost is benchmarked end-to-end.
WITH pool_a_capped AS (
    -- Per-country hash-diversity ranks. Mirrors 37.1's anchor pattern:
    -- PARTITION BY country_iso3 + ORDER BY hash(id, seed) gives a
    -- deterministic per-country shuffle, capped at {per_country_cap}
    -- so USA / DEU don't dominate the anchor pool.
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
  entity_geoms AS (
    SELECT
      a.id                                         AS entity_a_id,
      a.name                                       AS entity_a_name,
      ST_GeomFromWKT(a.geometry)                   AS geom,
      ST_Centroid(ST_GeomFromWKT(a.geometry))      AS centroid
    FROM anchors a
  ),
  -- H3 resolution search: 0 through 15 (full Sedona-supported range).
  h3_resolutions AS (
    SELECT explode(array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)) AS res
  ),
  h3_candidates AS (
    SELECT
      e.entity_a_id,
      e.entity_a_name,
      e.geom,
      r.res                                        AS h3_resolution,
      ST_H3CellIDs(e.centroid, r.res, false)[0]    AS h3_cell_id,
      ST_H3ToGeom(
        array(ST_H3CellIDs(e.centroid, r.res, false)[0])
      )[0]                                         AS h3_cell_geom
    FROM entity_geoms e
    CROSS JOIN h3_resolutions r
  ),
  h3_valid AS (
    SELECT entity_a_id, entity_a_name, h3_resolution, h3_cell_id
    FROM h3_candidates
    WHERE ST_Contains(h3_cell_geom, geom)
  ),
  h3_finest AS (
    SELECT entity_a_id, MAX(h3_resolution) AS h3_finest_resolution
    FROM h3_valid
    GROUP BY entity_a_id
  ),
  h3_result AS (
    SELECT
      f.entity_a_id,
      v.entity_a_name,
      f.h3_finest_resolution,
      CAST(v.h3_cell_id AS STRING)               AS h3_cell
    FROM h3_finest f
    JOIN h3_valid v
      ON f.entity_a_id = v.entity_a_id
      AND f.h3_finest_resolution = v.h3_resolution
  ),
  -- S2 level search: 0 through 20 (practical upper bound; S2 supports
  -- up to 30 but cells beyond ~20 are sub-meter and not useful here).
  s2_levels AS (
    SELECT explode(array(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)) AS lvl
  ),
  s2_candidates AS (
    SELECT
      e.entity_a_id,
      e.geom,
      l.lvl                                        AS s2_level,
      ST_S2CellIDs(e.centroid, l.lvl)[0]           AS s2_cell_id,
      ST_S2ToGeom(
        array(ST_S2CellIDs(e.centroid, l.lvl)[0])
      )[0]                                         AS s2_cell_geom
    FROM entity_geoms e
    CROSS JOIN s2_levels l
  ),
  s2_valid AS (
    SELECT entity_a_id, s2_level, s2_cell_id
    FROM s2_candidates
    WHERE ST_Contains(s2_cell_geom, geom)
  ),
  s2_finest AS (
    SELECT entity_a_id, MAX(s2_level) AS s2_finest_level
    FROM s2_valid
    GROUP BY entity_a_id
  ),
  s2_result AS (
    SELECT
      f.entity_a_id,
      f.s2_finest_level,
      CAST(v.s2_cell_id AS STRING)               AS s2_cell
    FROM s2_finest f
    JOIN s2_valid v
      ON f.entity_a_id = v.entity_a_id
      AND f.s2_finest_level = v.s2_level
  )
SELECT
  h.entity_a_id                                  AS entity_a_id,
  h.entity_a_name                                AS entity_a_name,
  -- Composite string GT — multi-column GT support is deferred, so we
  -- pack both indices into a single qualitative string. Downstream
  -- consumers can split on " | " then on ": " if structured access is
  -- needed.
  CONCAT(
    'H3(res=', CAST(h.h3_finest_resolution AS STRING), '): ', h.h3_cell,
    ' | S2(lvl=', CAST(s.s2_finest_level AS STRING), '): ', s.s2_cell
  )                                              AS ground_truth
FROM h3_result h
JOIN s2_result s
  ON h.entity_a_id = s.entity_a_id
LIMIT {total_sample};
