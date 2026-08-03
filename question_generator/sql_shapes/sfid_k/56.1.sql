-- TID 56.1 — Perimeter projection (single-pool unary, polygons only).
-- Question: "What is the perimeter of {{entity_A}}?"
-- Ground truth: ST_Perimeter (geodesic, true) / 1000 → km, as a string.
--
-- ST_Perimeter(..., true) flag selects geodesic (WGS84 spheroid) length
-- in metres, NOT planar degrees — same reasoning as 53.1's
-- ST_AreaSpheroid choice (planar degrees are meaningless globally).
--
-- SHAPE — single pool, no spatial join. Pure single-pool unary TID:
-- no pool_b, no join, no {join_condition}. Mirrors 37.1's structure.
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity (mirror of 12.1 / 37.1).
-- Per-country bucket-rank prevents USA-skew on dense KGs (osmh3kg, kwg)
-- and keeps the geographic distribution of anchors balanced. See
-- 63.1.sql for the full rationale; the per_country_cap and seed
-- defaults match.
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
  CASE
    -- NULL guard: surfaces missing geometry as NULL ground_truth so the
    -- validator drops the row, rather than emitting a broken numeric.
    WHEN a.geometry IS NULL THEN NULL
    ELSE CAST(
      ST_Perimeter(ST_GeomFromWKT(a.geometry), true) / 1000.0 AS STRING
    )
  END    AS ground_truth
FROM anchors a
LIMIT {total_sample};
