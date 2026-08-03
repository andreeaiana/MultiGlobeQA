-- TID 7.1 — 1-median facility location, per demand-point subcategory.
-- Question: "Which {{entity_A}} in {{entity_B}} minimises total distance
--            to all {{entity_C}}?"
--
-- Shape: 3-pool TID. For each candidate facility A inside a container
-- region B, compute the SUM of distances to every demand point C of
-- subcategory S also inside B. The winning facility per (region B,
-- subcategory S) is the one with the minimum total. {{entity_C}}
-- renders the C subcategory dynamically (e.g. "Pharmacy") so the
-- question is meaningful per demand-point type.
--
-- 2026-05 v7 changes vs v6:
--   - WHERE a.id != b.id excludes the container from being its own
--     candidate (fixes degenerate "Region X in Region X" rows for the
--     region_in_region variant where pool_a and pool_b overlap).
--   - GROUP BY adds c.subcategory (= c.node_type) so every (region,
--     c_subcategory) gets its own rank-1 facility, mirroring the 8.x
--     subcategory-aware redesign. entity_c_name renders as the raw
--     node_type instead of a literal "POIs".
WITH pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
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
      a.id, a.name, a.country_iso3,
      ST_GeomFromWKT(a.geometry) AS geom
    FROM {pool_a} a
    JOIN anchor_countries ac ON a.country_iso3 = ac.country_iso3
  ),
  pool_c_geom AS (
    SELECT /*+ REPARTITION(16) */
      c.id, c.country_iso3,
      c.node_type AS subcategory,
      ST_GeomFromWKT(c.geometry) AS geom
    FROM {pool_c} c
    JOIN anchor_countries ac ON c.country_iso3 = ac.country_iso3
  ),
  -- a IS contained in b (and is NOT b itself — fixes self-containment
  -- for region_in_region where pool_a and pool_b draw from the same
  -- region_category source and a region trivially "contains itself").
  a_in_b AS (
    SELECT /*+ BROADCAST(b) */
      a.id   AS a_id,
      a.name AS a_name,
      a.geom AS a_geom,
      b.id   AS b_id,
      b.name AS b_name,
      b.country_iso3
    FROM pool_a_geom a
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, a.geom)
    WHERE a.country_iso3 = b.country_iso3
      AND a.id != b.id
  ),
  c_in_b AS (
    SELECT /*+ BROADCAST(b) */
      c.id          AS c_id,
      c.subcategory AS c_subcategory,
      c.geom        AS c_geom,
      b.id          AS b_id
    FROM pool_c_geom c
    JOIN pool_b_geom b
      ON ST_Contains(b.geom, c.geom)
    WHERE c.country_iso3 = b.country_iso3
  ),
  candidate_totals AS (
    SELECT /*+ REPARTITION(entity_b_id) */
      a.a_id        AS entity_a_id,
      a.a_name      AS entity_a_name,
      a.b_id        AS entity_b_id,
      a.b_name      AS entity_b_name,
      c.c_subcategory,
      SUM(ST_DistanceSphere(a.a_geom, c.c_geom) / 1000.0) AS total_distance_km
    FROM a_in_b a
    JOIN c_in_b c
      ON a.b_id = c.b_id
     AND a.a_id != c.c_id
    GROUP BY a.a_id, a.a_name, a.b_id, a.b_name, c.c_subcategory
  ),
  ranked AS (
    SELECT
      entity_a_id, entity_a_name,
      entity_b_id, entity_b_name,
      c_subcategory,
      total_distance_km,
      ROW_NUMBER() OVER (
        PARTITION BY entity_b_id, c_subcategory
        ORDER BY total_distance_km ASC, entity_a_id ASC
      ) AS rank
    FROM candidate_totals
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  c_subcategory AS entity_c_name,
  CAST(total_distance_km AS STRING) AS ground_truth
FROM ranked
WHERE rank = 1
LIMIT {total_sample};
