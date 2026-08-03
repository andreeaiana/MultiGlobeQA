-- TID 41.1 — H3 grid cell shared lookup at parameterized resolution.
-- Question: "Which grid cell at resolution {{value}} contains both
--            {{entity_A}} and {{entity_B}}?"
-- Ground truth: the H3 cell ID (string) at {resolution} that both POIs
-- share, or NULL when they don't share a cell at that resolution.
--
-- SHAPE — variant of 40.1 with a parameterized H3 resolution.
-- The structural skeleton is identical to 40.1: rank-equality sampling,
-- INNER JOIN on attribute equality (same coarse h3_res3 cell + same
-- country + ranked-pair). The DIFFERENCE is the SELECT computes the
-- shared cell at a configurable resolution via ST_H3CellIDs(centroid,
-- {resolution}, false), and the ground_truth is NULL when the two
-- entities' centroids fall in different cells at that resolution.
--
-- Resolution intuition:
--   res 3  — ~12,000 km² cells; nearly all same_cell pairs share
--   res 5  — ~253 km² cells;   most pairs share
--   res 7  — ~5.2 km² cells;   some pairs share
--   res 9  — ~0.1 km² cells;   very few pairs share
-- The resolution gradient produces both true (shared) and false-premise
-- (NULL ground_truth) instances naturally, no separate range param needed.
--
-- The {resolution} parameter is a static meta.parameters value (default 7).
-- To run with a different resolution, override via meta.parameters or
-- declare sibling sub-templates that each pin one resolution, the way
-- TID 36's eight direction sub-templates are split. This shape ships as
-- a single sub-template covering every resolution.
WITH pool_a_capped AS (
    SELECT
      a.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_a})) AS unique_id
    FROM {pool_a} a
  ),
  pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_b})) AS unique_id
    FROM {pool_b} b
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  -- Project resolution for the {{value}} placeholder in the question template.
  CAST({resolution} AS STRING) AS value,
  -- Shared cell at the requested resolution. NULL when the centroids
  -- fall in different cells at {resolution} — the false-premise case
  -- the validator drops downstream.
  CASE
    WHEN a.geometry IS NULL OR b.geometry IS NULL THEN NULL
    WHEN ST_H3CellIDs(ST_Centroid(ST_GeomFromWKT(a.geometry)), {resolution}, false)[0]
       = ST_H3CellIDs(ST_Centroid(ST_GeomFromWKT(b.geometry)), {resolution}, false)[0]
    THEN CAST(
           ST_H3CellIDs(ST_Centroid(ST_GeomFromWKT(a.geometry)), {resolution}, false)[0]
         AS STRING)
    ELSE NULL
  END    AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
