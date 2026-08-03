-- TID 57.1 — Perimeter-to-area ratio (single-pool, parameterized).
-- Question: "How irregular is the boundary of {{entity_A}}, measured by
--            perimeter-to-area ratio at {value} km resolution?"
-- Ground truth: ST_Perimeter / ST_AreaSpheroid (1/m, after simplification).
--
-- Both metrics computed on a topology-preserving simplified geometry at
-- tolerance ({value} / 111.0) degrees — the /111.0 converts km → degrees
-- at the equator (rough, but applied consistently across shapes).
--
-- SHAPE — single pool, no spatial join. Mirrors 37.1's per-country
-- anchor sampling with a single-pool LIMIT.
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
    -- Degenerate-area guard: ST_AreaSpheroid of a degenerate / line-collapsed
    -- simplification is 0 → division by zero. Surface as NULL.
    WHEN ST_AreaSpheroid(
           ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0)
         ) = 0 THEN NULL
    ELSE CAST(
      ST_Perimeter(
        ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0),
        true
      )
      / ST_AreaSpheroid(
        ST_SimplifyPreserveTopology(ST_GeomFromWKT(a.geometry), {value} / 111.0)
      )
      AS STRING
    )
  END    AS ground_truth
FROM anchors a
LIMIT {total_sample};
