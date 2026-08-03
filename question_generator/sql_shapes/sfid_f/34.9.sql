-- TID 34.9 — Attribute ranking: biggest by elongation entity within a region.
-- Question: "What is the biggest {{entity_A}} in
--            {{entity_B}} by elongation?"
-- Ground truth: rank-1 pool_a entity per region anchor, by elongation.
--
-- Shape: 2-pool. pool_a is the candidate sub-region (region or natural
-- feature category) being ranked; pool_b is the encompassing region
-- anchor (one question per region). PARTITION BY b.id + WHERE
-- rank_in_region = 1 yields one rank-1 pool_a per region, scaling to
-- {total_sample} questions instead of a single global LIMIT 1.
-- Tiebreak on a.name for determinism.
--
-- Geometry filter: POLYGON, MULTIPOLYGON (applied in both pool CTE WHERE clauses).
-- Metric expression: elongation.
WITH pool_a_geom AS (
    -- Materialize pool_a (entity) geom with ST_MakeValid + filters.
    SELECT /*+ REPARTITION(16) */
      id, name, geometry,
      ST_MakeValid(ST_GeomFromWKT(geometry)) AS geom
    FROM {pool_a}
    WHERE geometry IS NOT NULL
      AND NOT geometry RLIKE 'NaN'
      AND geometry_type IN ('POLYGON', 'MULTIPOLYGON')
      AND ST_IsValid(ST_GeomFromWKT(geometry))
      AND ST_AreaSpheroid(ST_GeomFromWKT(geometry)) > 0
  ),
  pool_b_geom AS (
    -- Materialize pool_b (region anchor) geom with ST_MakeValid + filters.
    -- Some osmh3kg region polygons pass ST_IsValid but still crash JTS
    -- ST_Contains; ST_MakeValid repairs them and the CTE forces
    -- materialization so Catalyst can't pipeline the repair past the join.
    SELECT /*+ REPARTITION(16) */
      id, name,
      ST_MakeValid(ST_GeomFromWKT(geometry)) AS geom
    FROM {pool_b}
    WHERE geometry IS NOT NULL
      AND NOT geometry RLIKE 'NaN'
      AND geometry_type IN ('POLYGON', 'MULTIPOLYGON')
      AND ST_IsValid(ST_GeomFromWKT(geometry))
  ),
  ranked AS (
    SELECT
      a.id    AS entity_a_id,
      a.name  AS entity_a_name,
      b.id    AS entity_b_id,
      b.name  AS entity_b_name,
      (3.141592653589793 * POWER(ST_MinimumBoundingRadius(a.geom).radius, 2))
        / NULLIF(ST_AreaSpheroid(a.geom), 0) AS elongation,
      a.name  AS ground_truth,
      ROW_NUMBER() OVER (
        PARTITION BY b.id
        ORDER BY (3.141592653589793 * POWER(ST_MinimumBoundingRadius(a.geom).radius, 2))
               / ST_AreaSpheroid(a.geom) DESC,
                 a.name ASC
      ) AS rank_in_region
    FROM pool_a_geom a
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, a.geom)
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  CAST(elongation AS STRING) AS elongation,
  ground_truth
FROM ranked
WHERE rank_in_region = 1
LIMIT {total_sample};
