-- TID 57.3 — Convex hull area ratio (single-pool, parameterized).
-- Question: "How irregular is {{entity_A}}, measured by convex-hull area
--            ratio at {value} km resolution?"
-- Ground truth: Area(simplified) / Area(ConvexHull(simplified)).
--   Convex shape → 1.0; concave / fragmented → < 1.0.
--
-- Both areas computed via ST_AreaSpheroid so the ratio is dimensionless
-- and globally comparable. Same single-pool ROW_NUMBER hash diversity
-- as 37.1 / 57.1.
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
  -- Project simplification tolerance for the {{value}} placeholder.
  CAST({value} AS STRING) AS value,
  CASE
    WHEN a.geometry IS NULL THEN NULL
    -- Degenerate-hull guard: a degenerate (collinear) point set has a
    -- 0-area convex hull, dividing by zero. Surface as NULL.
    WHEN ST_AreaSpheroid(
           ST_ConvexHull(
             ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0)
           )
         ) <= 0 THEN NULL
    ELSE CAST(
      ST_AreaSpheroid(
        ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0)
      )
      / ST_AreaSpheroid(
        ST_ConvexHull(
          ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0)
        )
      )
      AS STRING
    )
  END    AS ground_truth
FROM anchors a
LIMIT {total_sample};
