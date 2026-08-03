-- TID 33.4 — Directional extreme: EASTernmost POI within a region.
-- Question: "Which {{entity_A}} lies furthest east within {{entity_B}}?"
-- Ground truth: pool_a entity name with the rank-1 centroid coordinate
--               (ORDER BY ST_X(ST_Centroid(ST_GeomFromWKT(a.geometry))) DESC) among entities ST_Contains'd by the region.
--
-- Shape: 2-pool. pool_a is the POI category being ranked; pool_b is the
-- region anchor (one question per region). PARTITION BY b.id + WHERE
-- rank_in_region = 1 yields one northernmost/southernmost/etc. POI per
-- region, scaling to {total_sample} questions rather than a single
-- global result. Tiebreak on a.name for determinism.
--
-- The 33.x siblings share this template; only the ORDER BY direction
-- changes across them (north / south / east / west).
WITH pool_b_geom AS (
    -- Materialize region geom with ST_MakeValid repair and geometry_type
    -- filter. Some osmh3kg region polygons pass ST_IsValid but still
    -- crash JTS ST_Contains with "side location conflict" — ST_MakeValid
    -- repairs them. The CTE forces materialization so Catalyst can't
    -- pipeline the repair past the spatial join.
    SELECT /*+ REPARTITION(16) */
      id, name, ST_MakeValid(ST_GeomFromWKT(geometry)) AS geom
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
      a.name  AS ground_truth,
      ROW_NUMBER() OVER (
        PARTITION BY b.id
        ORDER BY ST_X(ST_Centroid(ST_GeomFromWKT(a.geometry))) DESC,
                 a.name ASC
      ) AS rank_in_region
    FROM {pool_a} a
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, ST_GeomFromWKT(a.geometry))
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  ground_truth
FROM ranked
WHERE rank_in_region = 1
LIMIT {total_sample};
