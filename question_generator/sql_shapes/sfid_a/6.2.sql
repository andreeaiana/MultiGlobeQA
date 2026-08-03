-- TID 6.2 — intra-region pairwise distance aggregate (AVG), per subcategory.
-- Mirror of 6.1 with MIN replaced by AVG. See 8.1.sql for design notes.
--
-- 2026-05 v6 fix: cap pool_b (regions) per country BEFORE the spatial
-- join. Earlier versions did the full pool_a × pool_b spatial join
-- (88K × 28K for USA+CAN) which made the downstream per-country
-- WindowExec hit a single-task collapse. Capping regions per country
-- bounds the spatial join to (88K × 100 sampled regions) and keeps
-- ALL POIs per chosen region for the exact aggregate. Mirrors TID
-- 56.x's per_country_cap pattern.
WITH pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_pair})
      ) AS region_rank
    FROM {pool_b} b
  ),
  pool_b_geom AS (
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM pool_b_capped
    WHERE region_rank <= {per_country_region_cap}
  ),
  anchor_countries AS (
    SELECT DISTINCT country_iso3 FROM pool_b_geom
  ),
  pool_a_geom AS (
    SELECT /*+ REPARTITION(16) */
      a.id, a.country_iso3, a.node_type AS subcategory,
      ST_GeomFromWKT(a.geometry) AS geom
    FROM {pool_a} a
    JOIN anchor_countries ac ON a.country_iso3 = ac.country_iso3
  ),
  -- BROADCAST sampled regions (small) onto the per-partition POI scan.
  poi_in_region AS (
    SELECT /*+ BROADCAST(r) */
      r.id AS region_id, r.name AS region_name, r.country_iso3,
      p.subcategory, p.id AS poi_id, p.geom AS poi_geom
    FROM pool_a_geom p
    JOIN pool_b_geom r
      ON ST_Contains(r.geom, p.geom)
    WHERE p.country_iso3 = r.country_iso3
  ),
  -- Per-(region, subcategory) count for degenerate filter.
  pair_summaries AS (
    SELECT
      region_id, region_name, country_iso3, subcategory,
      COUNT(*) AS pair_poi_count
    FROM poi_in_region
    GROUP BY region_id, region_name, country_iso3, subcategory
  ),
  filtered_pairs AS (
    SELECT region_id, region_name, country_iso3, subcategory
    FROM pair_summaries
    WHERE pair_poi_count >= {min_pois_per_pair}
  ),
  chosen_pois AS (
    SELECT /*+ REPARTITION(region_id, subcategory), BROADCAST(s) */
      pir.region_id, pir.region_name, pir.subcategory,
      pir.poi_id, pir.poi_geom
    FROM poi_in_region pir
    JOIN filtered_pairs s
      ON pir.region_id = s.region_id
     AND pir.subcategory = s.subcategory
  )
SELECT
  p1.subcategory AS entity_a_name,
  p1.region_id   AS entity_b_id,
  p1.region_name AS entity_b_name,
  CAST(
    AVG(ST_DistanceSphere(p1.poi_geom, p2.poi_geom) / 1000.0)
    AS STRING
  ) AS ground_truth
FROM chosen_pois p1
JOIN chosen_pois p2
  ON p1.region_id = p2.region_id
 AND p1.subcategory = p2.subcategory
WHERE p1.poi_id < p2.poi_id
GROUP BY p1.region_id, p1.region_name, p1.subcategory
LIMIT {total_sample};
