-- TID 57.5 — Vertices per kilometre (single-pool, parameterized).
-- Question: "How irregular is {{entity_A}}, measured by vertices per
--            kilometre at {value} km sampling interval?"
-- Ground truth: ST_NPoints(simplified) / (Perimeter(simplified) / 1000).
--   Higher density = more boundary detail per unit length.
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
  -- Project sampling interval for the {{value}} placeholder.
  CAST({value} AS STRING) AS value,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    -- Degenerate-perimeter guard: a degenerate simplification → 0
    -- perimeter, division by zero. Surface as NULL.
    WHEN ST_Perimeter(
           ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0),
           true
         ) <= 0 THEN NULL
    ELSE CAST(
      CAST(ST_NPoints(
        ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0)
      ) AS DOUBLE)
      / (ST_Perimeter(
        ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0),
        true
      ) / 1000.0)
      AS STRING
    )
  END    AS ground_truth
FROM anchors a
LIMIT {total_sample};
