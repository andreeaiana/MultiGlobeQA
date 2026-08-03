-- TID 54.x — Collection count: how many {{collection_of_entity}} are
-- inside region {{entity_A}}?
-- Ground truth: integer count of pool_b entities (POIs / natural features
-- of the queried collection's node_types) whose centroid lies inside
-- pool_a's polygon.
--
-- SHAPE — Sedona spatial join + LEFT JOIN COUNT (mirror of 10.1).
-- pool_a is the outer "anchor" pool (regions). pool_b is the inner
-- counted pool, narrowed by node_type via {join_condition}. The LEFT
-- JOIN + COUNT/GROUP BY produces one row per anchor region, with the
-- count as ground_truth — anchors with ZERO matches still produce a
-- row whose ground_truth = '0'. Do NOT switch this to INNER JOIN: that
-- would silently drop zero-match anchors and bias the benchmark toward
-- regions that happen to have at least one matching feature.
--
-- Sedona spatial-join requirements (mirror of 10.1 / 28.1):
--   1. Both sides must expose Geometry-typed columns (NOT WKT strings)
--      so Sedona's optimizer recognises ST_Contains in JOIN ON and
--      activates RangeJoinExec (R-tree). pool_a_geom / pool_b_geom CTEs
--      materialize `geom` upstream of the join.
--   2. The spatial predicate (ST_Contains) is a SEPARATE conjunct in
--      JOIN ON — not nested inside another expression — for Sedona to
--      spot it. The non-spatial conjuncts in {join_condition} (country
--      equality, node_type filter) participate naturally.
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity (mirror of 10.1).
-- Per-country bucket ranks; geographically balanced anchor selection.
WITH pool_a_geom AS (
    -- Materialize geometry as a Geometry-typed column upstream of the
    -- spatial JOIN so Sedona can swap in RangeJoinExec.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    -- Project only the columns the downstream COUNT and JOIN read.
    SELECT /*+ REPARTITION(16) */
      id, country_iso3, node_type,
      ST_Centroid(ST_GeomFromWKT(geometry)) AS geom
    FROM {pool_b}
  ),
  -- Hash-diversity anchor sampling (regions). PARTITION BY country_iso3
  -- ensures each country independently produces ranks 1..N from its own
  -- regions, so {per_country_cap} caps anchors per country.
  pool_a_capped AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS bucket_rank
    FROM pool_a_geom
  ),
  anchors AS (
    SELECT * FROM pool_a_capped WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id                  AS entity_a_id,
  a.name                AS entity_a_name,
  -- entity_b_name carries the PLURAL collection label
  -- ("lakes, ponds and reservoirs") so the exporter's
  -- {{collection_of_entity}} placeholder substitutes correctly.
  '{collection_label}'  AS entity_b_name,
  CAST(COUNT(b.id) AS STRING) AS ground_truth
FROM anchors a
LEFT JOIN pool_b_geom b
  ON {join_condition}
GROUP BY a.id, a.name
LIMIT {total_sample};
