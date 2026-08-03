-- TID 57.4 — Boundary roughness index (single-pool, parameterized).
-- Question: "How irregular is {{entity_A}}'s boundary, measured by
--            roughness index at {value} km smoothing scale?"
-- Ground truth: Perimeter(simplified at value/2) / Perimeter(simplified at value*2).
--   Higher ratio → rougher boundary (more detail lost between scales).
--   Smooth boundary → ratio ~ 1.
--
-- Per-country ROW_NUMBER hash diversity (mirror of 37.1 / 99.x).
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
  -- Project smoothing scale for the {{value}} placeholder.
  CAST({value} AS STRING) AS value,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    -- Degenerate-coarse-perimeter guard: heavy simplification can
    -- collapse small polygons to ~0 perimeter. Surface as NULL.
    WHEN ST_Perimeter(
           ST_SimplifyPreserveTopology(
             ST_GeomFromWKT(a.geometry), ({value} * 2.0) / 111.0
           ), true
         ) <= 0 THEN NULL
    ELSE CAST(
      ST_Perimeter(
        ST_SimplifyPreserveTopology(
          ST_GeomFromWKT(a.geometry), ({value} / 2.0) / 111.0
        ), true
      )
      / ST_Perimeter(
        ST_SimplifyPreserveTopology(
          ST_GeomFromWKT(a.geometry), ({value} * 2.0) / 111.0
        ), true
      )
      AS STRING
    )
  END    AS ground_truth
FROM anchors a
LIMIT {total_sample};
