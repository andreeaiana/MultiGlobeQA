-- TID 18.1 — Intersection set-difference enumeration.
-- Question: "Which {{entity_A}} intersect {{entity_B}} but not {{entity_C}}?"
-- Ground truth: SORTED set of pool_a entity NAMES that intersect pool_b
--               anchor B but do NOT intersect pool_c anchor C.
--
-- 3-slot template with category-as-slot pattern: pool_a is the queried
-- CATEGORY (unsampled scan); pool_b and pool_c are SAME-TYPE instance
-- anchors (constraint per deliverable design — "Roads intersecting
-- river X but not district Y" is semantically odd). Each result row
-- corresponds to one (B, C) anchor pair with the surviving A-set.
--
-- Variant differentiation: meta.parameters.entity_a_label pins the
-- PLURAL display label per variant. The 4 deliverable pairings are
-- shipped as 4 variants — see meta.yaml.
--
-- Sedona spatial-join pattern (memory: feedback_sedona_spatial_join):
-- pre-CTE materialize WKT -> Geometry; ST_Intersects in JOIN ON for
-- the b-side join. The NOT ST_Intersects predicate (c-side filter)
-- can NOT use a spatial-join index — there's no R-tree pattern for
-- "not contained" — but it executes against the already-shrunk
-- candidate set surviving the b-side spatial join, so it stays cheap.
--
-- Hash-diversity per-country sampling on BOTH pool_b and pool_c with
-- INDEPENDENT seeds. Without distinct seeds the symmetric same-type
-- pairing would deterministically collapse onto the same anchor per
-- country and the b.id != c.id filter would drop everything (mirror
-- 1.1 / 44.1's symmetric-pool pattern).
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
  pool_c_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_c}
  ),
  pool_b_capped AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM pool_b_geom
  ),
  pool_c_capped AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_c})
      ) AS bucket_rank
    FROM pool_c_geom
  ),
  anchors_b AS (
    SELECT * FROM pool_b_capped WHERE bucket_rank <= {per_country_cap}
  ),
  anchors_c AS (
    SELECT * FROM pool_c_capped WHERE bucket_rank <= {per_country_cap}
  ),
  -- Build the (B, C) anchor pairs: same country, distinct ids. Then
  -- LEFT JOIN to pool_a entities that intersect B but NOT C. The
  -- ST_Intersects against B uses the spatial-join optimizer; the
  -- NOT ST_Intersects against C runs as a post-filter on the smaller
  -- surviving candidate set.
  pair_anchors AS (
    SELECT
      b.id              AS b_id,
      b.name            AS b_name,
      b.geom            AS b_geom,
      b.country_iso3    AS country_iso3,
      c.id              AS c_id,
      c.name            AS c_name,
      c.geom            AS c_geom
    FROM anchors_b b
    JOIN anchors_c c
      ON b.country_iso3 = c.country_iso3
     AND b.id != c.id
  ),
  -- Pre-prune pool_a to ONLY the countries that have anchor pairs.
  -- This semi-join is cheap (anchor countries are a small set, ~12
  -- strata countries × 2 sample countries) but eliminates 95%+ of the
  -- pool_a row count BEFORE the spatial index is built. Catalyst can
  -- push this down to a partitioned Delta scan if pool_a is partitioned
  -- by country, otherwise it runs as a hash semi-join — either way,
  -- much cheaper than letting Sedona R-tree the entire pool.
  anchor_countries AS (
    SELECT DISTINCT country_iso3 FROM anchors_b
  ),
  pool_a_pruned AS (
    SELECT /*+ REPARTITION(16) */ a.*
    FROM pool_a_geom a
    JOIN anchor_countries ac ON a.country_iso3 = ac.country_iso3
  ),
  -- Step 1: pure-spatial INNER JOIN with BROADCAST hint on pair_anchors
  -- (small ~100-row driver-side relation). With BROADCAST(pa), every
  -- pool_a partition does its own spatial check against the broadcast
  -- anchor set, parallelizing across all 16 partitions instead of
  -- collapsing into 1-2 SortMerge partitions. Sedona's RangeJoinExec
  -- recognizes the broadcast-spatial pattern and uses an R-tree on the
  -- broadcast side to bound the per-row geometric checks.
  -- country equality stays in WHERE so it doesn't trigger SortMergeJoin
  -- selection (memory: feedback_sedona_spatial_join).
  intersects_b AS (
    SELECT /*+ BROADCAST(pa) */
      pa.b_id, pa.b_name, pa.c_id, pa.c_name,
      pa.country_iso3, pa.c_geom,
      a.id   AS a_id,
      a.name AS a_name,
      a.geom AS a_geom
    FROM pool_a_pruned a
    INNER JOIN pair_anchors pa
      ON ST_Intersects(a.geom, pa.b_geom)
    WHERE a.country_iso3 = pa.country_iso3
      AND a.id != pa.b_id
      AND a.id != pa.c_id
  ),
  -- Step 2: post-filter NOT ST_Intersects(a, C). Now operating on the
  -- already-shrunk candidate set (entities that intersect B), so the
  -- nested ST_Intersects evaluation is bounded by |intersects_b| not
  -- |pool_a × pair_anchors|.
  -- Note: pairs where every candidate also intersects C are dropped
  -- here (no rows survive); this is also the natural drop semantics
  -- for "entities that intersect B but not C" — an empty answer is
  -- a degenerate question and the validator would filter it anyway.
  diffs AS (
    SELECT b_id, b_name, c_id, c_name, a_name
    FROM intersects_b
    WHERE NOT ST_Intersects(a_geom, c_geom)
  )
SELECT
  -- entity_a_name carries the PLURAL category label so the exporter's
  -- {{entity_A}} placeholder renders grammatically.
  '{entity_a_label}' AS entity_a_name,
  b_id   AS entity_b_id,
  b_name AS entity_b_name,
  c_id   AS entity_c_id,
  c_name AS entity_c_name,
  CAST(SORT_ARRAY(COLLECT_LIST(a_name)) AS STRING) AS ground_truth
FROM diffs
GROUP BY b_id, b_name, c_id, c_name
LIMIT {total_sample};
