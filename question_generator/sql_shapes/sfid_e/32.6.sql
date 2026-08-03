-- TID 32.6 — Ranked full containment (least populated).
-- Mirror of 32.5 with ORDER BY rank_value ASC. See 56.7.sql for refactor
-- (per-country pool_c sampling + pool_b country prune BEFORE pop_tiles).
WITH pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
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
  anchor_countries AS (
    SELECT DISTINCT country_iso3 FROM anchors_c
  ),
  pool_b_pruned AS (
    SELECT /*+ REPARTITION(16) */
      b.id, b.name, b.country_iso3,
      ST_GeomFromWKT(b.geometry) AS geom,
      ST_Centroid(ST_GeomFromWKT(b.geometry)) AS centroid
    FROM {pool_b} b
    JOIN anchor_countries ac ON b.country_iso3 = ac.country_iso3
  ),
  -- See 56.7.sql for design notes — spatial pre-filter, then population.
  features_in_anchor AS (
    SELECT /*+ BROADCAST(c) */ DISTINCT
      b.id, b.name, b.country_iso3, b.geom, b.centroid,
      c.id   AS container_id,
      c.name AS container_name
    FROM pool_b_pruned b
    JOIN anchors_c c
      ON ST_Contains(c.geom, b.centroid)
    WHERE b.country_iso3 = c.country_iso3
  ),
  distinct_features AS (
    SELECT DISTINCT id, name, country_iso3, geom, centroid
    FROM features_in_anchor
  ),
  populations AS (
    SELECT
      df.id AS feature_id,
      CAST(
        COALESCE(SUM(RS_ZonalStats(t.rast, df.geom, 1, 'sum', true)), 0)
        AS BIGINT
      ) AS rank_value
    FROM distinct_features df
    CROSS JOIN pop_tiles t
    GROUP BY df.id
  ),
  ranked_features AS (
    SELECT /*+ BROADCAST(p) */
      fia.id           AS feature_id,
      fia.name         AS feature_name,
      fia.geom         AS feature_geom,
      fia.country_iso3 AS country_iso3,
      fia.container_id,
      fia.container_name,
      p.rank_value,
      ROW_NUMBER() OVER (
        PARTITION BY fia.container_id
        ORDER BY p.rank_value ASC
      ) AS rn
    FROM features_in_anchor fia
    JOIN populations p ON p.feature_id = fia.id
  ),
  rank1 AS (
    SELECT * FROM ranked_features WHERE rn = 1
  ),
  pool_a_pruned AS (
    SELECT /*+ REPARTITION(16) */ a.*
    FROM pool_a_geom a
    JOIN anchor_countries ac ON a.country_iso3 = ac.country_iso3
  ),
  true_sample AS (
    SELECT /*+ BROADCAST(rf) */
      a.id AS entity_a_id, a.name AS entity_a_name,
      rf.feature_id AS entity_b_id, rf.feature_name AS entity_b_name,
      rf.container_id AS entity_c_id, rf.container_name AS entity_c_name,
      'true' AS ground_truth
    FROM pool_a_pruned a
    JOIN rank1 rf
      ON ST_Contains(a.geom, rf.feature_geom)
    WHERE a.country_iso3 = rf.country_iso3
      AND a.id != rf.feature_id
      AND a.id != rf.container_id
    LIMIT {target_true_count}
  ),
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
      id AS entity_a_id, name AS entity_a_name,
      feature_id AS entity_b_id, feature_name AS entity_b_name,
      container_id AS entity_c_id, container_name AS entity_c_name,
      'false' AS ground_truth
    FROM false_candidates
    WHERE sample_rank <= 5
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
  entity_a_id, entity_a_name,
  entity_b_id, entity_b_name,
  entity_c_id, entity_c_name,
  ground_truth
FROM combined
LIMIT {total_sample};
