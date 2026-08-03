-- TID 54.1 — Compactness comparison boolean (cross-country, polygons only).
-- Question: "Is {{entity_A}} more compact than {{entity_B}}?"
-- Ground truth: 'true' iff A's IPQ > B's IPQ.
--
-- Compactness metric: IPQ = 4π · ST_AreaSpheroid / ST_Perimeter²
-- (Isoperimetric Quotient — dimensionless, ≤ 1, = 1 for a perfect circle).
-- Spheroid area + geodesic perimeter both yield m / m² so the ratio is
-- unit-consistent and comparable across countries / scales.
--
-- Same join shape as 53.1: rank-equality, cross-country (no
-- country_iso3 / h3_res3 in join_condition). The polygon-only filter
-- lives in meta.applicable_kgs[*].pool_*.filter.
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
  CASE
    -- NULL guard: surfaces missing geometry as NULL ground_truth so the
    -- validator drops the row, rather than silently labelling it 'false'.
    WHEN a.geometry IS NULL OR b.geometry IS NULL THEN NULL
    -- Degenerate-perimeter guard: a 0-perimeter geometry would divide
    -- by zero. Surface as NULL so the validator drops the row.
    WHEN ST_Perimeter(ST_GeomFromWKT(a.geometry), true) = 0
      OR ST_Perimeter(ST_GeomFromWKT(b.geometry), true) = 0 THEN NULL
    WHEN (4 * PI() * ST_AreaSpheroid(ST_GeomFromWKT(a.geometry)))
         / POWER(ST_Perimeter(ST_GeomFromWKT(a.geometry), true), 2)
       > (4 * PI() * ST_AreaSpheroid(ST_GeomFromWKT(b.geometry)))
         / POWER(ST_Perimeter(ST_GeomFromWKT(b.geometry), true), 2)
    THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
