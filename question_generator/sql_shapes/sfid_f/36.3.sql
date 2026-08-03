-- TID 36.3 — Compound: radius around B AND EAST of C, per POI subcategory.
-- Question: "Which {{entity_A}} are within {{value}} km of {{entity_B}}
--            and east of {{entity_C}}?"
-- where {{entity_A}} renders the raw POI node_type (e.g. "school",
-- "Pharmacy") so the question is meaningful per subcategory.
-- Ground truth: SORTED COLLECT_LIST of pool_a entity names matching
-- both predicates per (b, c, subcategory) tuple — EXACT, no upper cap.
--
-- Grouping (mirror of TID 8.x): aggregating across ALL POI subcategories
-- for a given (b, c) pair yields degenerate questions ("which random
-- POIs are near X and north of Y"), so the grouping is by
-- (b, c, node_type) and each question targets a single meaningful POI
-- type. Cost control is at the ANCHOR PAIR level (per-country sampling
-- on pool_b and pool_c with divergent seeds); there is NO cap on the
-- per-pair COLLECT_LIST size, since the GT requires the EXACT
-- enumeration of every matching POI in the radius.
WITH pool_b_capped AS (
    SELECT
      b.*,
      ST_Centroid(ST_GeomFromWKT(b.geometry)) AS center,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM {pool_b} b
  ),
  pool_c_capped AS (
    SELECT
      c.*,
      ST_Centroid(ST_GeomFromWKT(c.geometry)) AS center,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_c})
      ) AS bucket_rank
    FROM {pool_c} c
  ),
  anchors_b AS (
    SELECT id, name, country_iso3, center, bucket_rank
    FROM pool_b_capped
    WHERE bucket_rank <= {per_country_anchor_cap}
  ),
  anchors_c AS (
    SELECT id, name, country_iso3, center, bucket_rank
    FROM pool_c_capped
    WHERE bucket_rank <= {per_country_anchor_cap}
  ),
  -- Pair anchors via rank-equality WITHIN country. Divergent seeds keep
  -- b and c rankings out of sync so the same entity rarely pairs with
  -- itself. Bounded at per_country_anchor_cap × N_countries pairs.
  anchor_pairs AS (
    SELECT
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      b.country_iso3,
      b.center AS b_center,
      c.id   AS entity_c_id,
      c.name AS entity_c_name,
      c.center AS c_center
    FROM anchors_b b
    JOIN anchors_c c
      ON b.country_iso3 = c.country_iso3
     AND b.bucket_rank = c.bucket_rank
     AND b.id != c.id
  ),
  -- Country pre-prune on pool_a so the radius join doesn't fan out
  -- across countries with no anchor pair. Carry node_type as the
  -- subcategory dimension for grouping.
  anchor_countries AS (
    SELECT DISTINCT country_iso3 FROM anchor_pairs
  ),
  pool_a_pruned AS (
    SELECT /*+ REPARTITION(16) */
      a.id, a.name, a.country_iso3,
      a.node_type AS subcategory,
      ST_Centroid(ST_GeomFromWKT(a.geometry)) AS center
    FROM {pool_a} a
    JOIN anchor_countries ac ON a.country_iso3 = ac.country_iso3
  ),
  -- Radius + directional match. Same-country WHERE keeps the join
  -- bounded; the directional ST_Y comparison is a post-filter.
  matches AS (
    SELECT /*+ BROADCAST(ap) */
      ap.entity_b_id, ap.entity_b_name,
      ap.entity_c_id, ap.entity_c_name,
      a.subcategory,
      a.name AS poi_name
    FROM pool_a_pruned a
    JOIN anchor_pairs ap
      ON a.country_iso3 = ap.country_iso3
    WHERE ST_DistanceSphere(a.center, ap.b_center) / 1000.0 <= {radius_km}
      AND ST_X(a.center) > ST_X(ap.c_center)
      AND a.id != ap.entity_b_id
      AND a.id != ap.entity_c_id
  )
SELECT
  subcategory     AS entity_a_name,
  entity_b_id, entity_b_name,
  entity_c_id, entity_c_name,
  CAST({radius_km} AS STRING) AS radius_km,
  SORT_ARRAY(COLLECT_LIST(poi_name)) AS ground_truth
FROM matches
GROUP BY entity_b_id, entity_b_name,
         entity_c_id, entity_c_name,
         subcategory
HAVING SIZE(COLLECT_LIST(poi_name)) > 0
LIMIT {total_sample};
