-- TID 62.1 — Qualitative probabilistic containment (3-tier categorical).
-- Question: "Is {{entity_A}} plausibly within {{entity_B}}?"
-- Ground truth: 'Yes' (contained), 'Plausibly' (within 500m of boundary),
-- 'No' (more than 500m outside boundary).
--
-- Sedona spatial-join pattern (mirror of 28.1): pre-CTE Geometry
-- materialization + ST_Contains in JOIN ON. Per-country sampling on
-- both pools to keep the cross-product bounded.
--
-- Boundary distance check: the 500m threshold could be approximated in
-- planar degrees as 500/111320 ≈ 0.00449, but we compute the spheroidal
-- distance via ST_DistanceSphere in the SELECT for accuracy at all latitudes.
WITH pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
  pool_a_capped AS (
    SELECT *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS bucket_rank
    FROM pool_a_geom
  ),
  anchors AS (
    SELECT id, name, country_iso3, geom
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  ),
  containment AS (
    -- LEFT JOIN so anchors that hit no container still emit a row
    -- (boundary_distance_km will be NULL → 'No' branch).
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      ST_Contains(b.geom, a.geom) AS is_contained,
      CASE
        WHEN b.geom IS NULL THEN NULL
        WHEN ST_Contains(b.geom, a.geom) THEN 0.0
        WHEN ST_Intersects(a.geom, b.geom) THEN 0.0
        ELSE ST_DistanceSphere(
               ST_ClosestPoint(a.geom, b.geom),
               ST_ClosestPoint(b.geom, a.geom)
             ) / 1000.0
      END AS boundary_distance_km
    FROM anchors a
    LEFT JOIN pool_b_geom b
      ON {join_condition}
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  CASE
    WHEN entity_b_id IS NULL THEN 'No'
    WHEN is_contained THEN 'Yes'
    WHEN boundary_distance_km <= 0.5 THEN 'Plausibly'
    ELSE 'No'
  END AS ground_truth
FROM containment
LIMIT {total_sample};
