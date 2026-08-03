-- TID 32.4 — Ranked full containment (shortest by length).
-- Question: "Does the administrative border of {{entity_A}} fully contain
-- the shortest {{entity_B_label}} of {{entity_C}}?"
-- Identical structure to 32.3 with ORDER BY rank_value ASC.
-- See 56.1.sql for design notes.
WITH pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  -- Per-country hash sample on pool_c (containers) FIRST. Cuts the
  -- ranked_features CTE from |pool_b| × |pool_c| spatial-join cost
  -- down to |pool_b| × |anchors_c|. Without this, a USA stratum with
  -- thousands of admin regions × millions of natural features hangs
  -- forever (the original 32.4 stuck for 2:28h on osmh3kg HIC_high).
  pool_c_capped AS (
    SELECT
      c.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_c})
      ) AS bucket_rank
    FROM {pool_c} c
  ),
  anchors_c AS (
    SELECT id, name, country_iso3, ST_GeomFromWKT(geometry) AS geom
    FROM pool_c_capped
    WHERE bucket_rank <= {per_country_cap}
  ),
  -- Country semi-join prune on pool_b. Only need features in countries
  -- that have anchor containers. Cuts pool_b row count by ~95% on a
  -- typical stratum. Defer ST_Centroid + ST_Length until after the
  -- prune so we don't compute them for every row in pool_b.
  anchor_countries AS (
    SELECT DISTINCT country_iso3 FROM anchors_c
  ),
  pool_b_pruned AS (
    SELECT /*+ REPARTITION(16) */
      b.id, b.name, b.country_iso3,
      ST_GeomFromWKT(b.geometry) AS geom,
      ST_Centroid(ST_GeomFromWKT(b.geometry)) AS centroid,
      ST_Length(ST_GeomFromWKT(b.geometry)) AS rank_value
    FROM {pool_b} b
    JOIN anchor_countries ac ON b.country_iso3 = ac.country_iso3
  ),
  -- Pure-spatial JOIN ON (country goes to WHERE) so Sedona's
  -- RangeJoinExec activates instead of SortMergeJoin on country_iso3.
  -- BROADCAST hint on anchors_c (small, ~per_country_cap × N_countries
  -- rows) so each pool_b partition does its own spatial check against
  -- the broadcast anchor set in parallel.
  ranked_features AS (
    SELECT /*+ BROADCAST(c) */
      b.id AS feature_id, b.name AS feature_name,
      b.geom AS feature_geom, b.country_iso3 AS country_iso3,
      c.id AS container_id, c.name AS container_name,
      b.rank_value,
      ROW_NUMBER() OVER (
        PARTITION BY c.id
        ORDER BY b.rank_value ASC
      ) AS rn
    FROM pool_b_pruned b
    JOIN anchors_c c
      ON ST_Contains(c.geom, b.centroid)
    WHERE b.country_iso3 = c.country_iso3
  ),
  rank1 AS (
    SELECT * FROM ranked_features WHERE rn = 1
  ),
  -- Pool_a country pre-prune (mirrors pool_b — only test administrative
  -- regions in countries that have anchor containers).
  pool_a_pruned AS (
    SELECT /*+ REPARTITION(16) */ a.*
    FROM pool_a_geom a
    JOIN anchor_countries ac ON a.country_iso3 = ac.country_iso3
  ),
  -- True samples: a contains the rank-1 feature. Pure-spatial JOIN ON
  -- (country eq + id exclusions in WHERE so SortMergeJoin doesn't
  -- collapse on country_iso3). BROADCAST(rf) since rank1 is small.
  true_sample AS (
    SELECT /*+ BROADCAST(rf) */
      a.id            AS entity_a_id,
      a.name          AS entity_a_name,
      rf.feature_id   AS entity_b_id,
      rf.feature_name AS entity_b_name,
      rf.container_id AS entity_c_id,
      rf.container_name AS entity_c_name,
      'true'          AS ground_truth
    FROM pool_a_pruned a
    JOIN rank1 rf
      ON ST_Contains(a.geom, rf.feature_geom)
    WHERE a.country_iso3 = rf.country_iso3
      AND a.id != rf.feature_id
      AND a.id != rf.container_id
    LIMIT {target_true_count}
  ),
  -- False samples: a does NOT contain the rank-1 feature. NOT spatial
  -- predicates can't use spatial index, so this would be O(N×M) on the
  -- naive CROSS JOIN. Restructure: hash-rank candidate (a, rf) pairs
  -- INSIDE country first, take a small bounded sample, THEN apply
  -- the NOT ST_Contains filter (most random pairs satisfy it
  -- naturally — random region rarely contains a specific feature).
  -- This keeps the false-sample cost bounded by per_country_cap × N
  -- and avoids the full cross-product evaluation.
  false_candidates AS (
    SELECT /*+ BROADCAST(rf) */
      a.id, a.name, a.geom, a.country_iso3,
      rf.feature_id, rf.feature_name, rf.feature_geom,
      rf.container_id, rf.container_name,
      ROW_NUMBER() OVER (
        PARTITION BY rf.feature_id
        ORDER BY hash(a.id, rf.feature_id, {rand_seed_a})
      ) AS sample_rank
    FROM pool_a_pruned a
    JOIN rank1 rf
      ON a.country_iso3 = rf.country_iso3
    WHERE a.id != rf.feature_id
      AND a.id != rf.container_id
  ),
  false_sample AS (
    SELECT
      id              AS entity_a_id,
      name            AS entity_a_name,
      feature_id      AS entity_b_id,
      feature_name    AS entity_b_name,
      container_id    AS entity_c_id,
      container_name  AS entity_c_name,
      'false'         AS ground_truth
    FROM false_candidates
    WHERE sample_rank <= 5  -- bounded per-anchor candidates
      AND NOT ST_Contains(geom, feature_geom)
    LIMIT {target_false_count}
  ),
  combined AS (
    SELECT entity_a_id, entity_a_name, entity_b_id, entity_b_name,
           entity_c_id, entity_c_name, ground_truth
    FROM true_sample
    UNION ALL
    SELECT entity_a_id, entity_a_name, entity_b_id, entity_b_name,
           entity_c_id, entity_c_name, ground_truth
    FROM false_sample
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  entity_c_id,
  entity_c_name,
  ground_truth
FROM combined
LIMIT {total_sample};
