-- TID 21.1 — Adjacency boolean. Three variants share this SQL body, each
-- with a different spatial predicate selected via {predicate_mode}:
--
--   region:    ST_Touches on materialised polygon Geometry (border-touch).
--   grid_cell: H3-index neighbour check via ST_H3CellDistance(...) = 1.
--   poi:       Distance-based proximity within {adjacency_threshold_m}.
--
-- DESIGN NOTE (registry has only 21.1 — single ftid, NOT 3 ftids):
--   Splitting this into three separate template ids isn't an option
--   (sql_registry.yaml has one stub keyed '21.1'). So we encode the 3
--   pool pairings as 3 variants of the SAME ftid and switch the spatial
--   predicate via a per-variant parameter rather than per-variant SQL
--   bodies. All three branches reference columns that exist on every
--   entity row (id, name, geometry, h3_index from main.py:177's entity
--   schema), so each branch compiles regardless of which variant is
--   active — only one is logically selected by the {predicate_mode}
--   string match in CASE WHEN.
--
--   The CASE-on-string-literal pattern is the same per-variant literal
--   substitution used for label projections elsewhere in the shape set,
--   except that here the literal drives a predicate dispatch rather
--   than a projected column.
--
-- SEDONA SPATIAL JOIN — pre-CTE Geometry materialisation for the region
-- (ST_Touches) branch so RangeJoinExec can index the join. The h3 and
-- proximity branches don't need indexed joins at the cross-product
-- volumes we sample (per_country_cap × per_country_cap), but they do
-- benefit from the unified WKT→Geometry conversion happening once
-- upstream.
--
-- ANCHOR DIVERSITY — ROW_NUMBER PARTITION BY country_iso3 ORDER BY hash.
-- Both pools capped per-country to bound the in-country cross-join.
WITH pool_a_geom AS (
    -- Materialise geometry as Geometry-typed (not WKT string) so the
    -- region (ST_Touches) variant's join sees indexable columns. h3_index
    -- comes through unchanged for the grid_cell variant. id/name carry
    -- forward for the projection.
    SELECT /*+ REPARTITION(16) */
      a.id,
      a.name,
      a.country_iso3,
      a.h3_index,
      ST_GeomFromWKT(a.geometry) AS geom,
      ROW_NUMBER() OVER (
        PARTITION BY a.country_iso3
        ORDER BY hash(a.id, {rand_seed_a})
      ) AS bucket_rank
    FROM {pool_a} a
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      b.id,
      b.name,
      b.country_iso3,
      b.h3_index,
      ST_GeomFromWKT(b.geometry) AS geom,
      ROW_NUMBER() OVER (
        PARTITION BY b.country_iso3
        ORDER BY hash(b.id, {rand_seed_b})
      ) AS bucket_rank
    FROM {pool_b} b
  ),
  anchors_a AS (
    SELECT * FROM pool_a_geom WHERE bucket_rank <= {per_country_cap}
  ),
  anchors_b AS (
    SELECT * FROM pool_b_geom WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CAST(
    CASE
      WHEN '{predicate_mode}' = 'touches' THEN ST_Touches(a.geom, b.geom)
      WHEN '{predicate_mode}' = 'h3' THEN
        ST_H3CellDistance(
          CAST(a.h3_index AS BIGINT),
          CAST(b.h3_index AS BIGINT)
        ) = 1
      WHEN '{predicate_mode}' = 'proximity' THEN
        ST_DistanceSphere(a.geom, b.geom) < {adjacency_threshold_m}
      ELSE NULL
    END
    AS STRING
  ) AS ground_truth
FROM anchors_a a
INNER JOIN anchors_b b
  ON {join_condition}
LIMIT {total_sample};
