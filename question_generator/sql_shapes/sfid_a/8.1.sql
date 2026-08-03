-- TID 8.1 — chained kNN + distance (k=1).
-- Question: "What is the distance between the nearest {{entity_A}} to
--            {{entity_B}} and {{entity_C}}?"
--
-- Shape: 3-pool TID — two-step distance computation.
--   Step 1: For each anchor in pool_b, rank pool_a candidates by distance
--           and keep rank = {k_rank} (the kNN-resolved entity_A).
--   Step 2: Compute distance from that resolved entity_A to every
--           candidate in pool_c. The output row pairs the (a, b) winner
--           from step 1 with each c, with ground_truth = distance(a, c).
--
-- Sampling: pool_b (the kNN anchor) gets per-country diversity capping.
-- pool_a is uncapped because the kNN must consider every candidate;
-- pool_c is uncapped so the second-step distance lands on every
-- candidate target. The ranked.entity_a is the chain link — c is paired
-- with it country-wise to keep the second distance compute in-country.
--
-- {k_rank}: 1 for 8.1, 2 for 8.2, 3 for 8.3 — SQL is shared; only
-- meta differs.
WITH pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM {pool_b} b
  ),
  anchors AS (
    SELECT *
    FROM pool_b_capped
    WHERE bucket_rank <= {anchor_cap}
  ),
  distances_to_b AS (
    SELECT
      a.id          AS entity_a_id,
      a.name        AS entity_a_name,
      a.geometry    AS entity_a_geometry,
      a.country_iso3 AS country_iso3,
      b.id          AS entity_b_id,
      b.name        AS entity_b_name,
      CASE
        WHEN ST_Intersects(ST_GeomFromWKT(a.geometry),
                           ST_GeomFromWKT(b.geometry))
          THEN 0.0
        ELSE ST_DistanceSphere(
               ST_ClosestPoint(ST_GeomFromWKT(a.geometry),
                               ST_GeomFromWKT(b.geometry)),
               ST_ClosestPoint(ST_GeomFromWKT(b.geometry),
                               ST_GeomFromWKT(a.geometry))
             ) / 1000.0
      END AS distance_to_b_km
    FROM {pool_a} a, anchors b
    WHERE a.country_iso3 = b.country_iso3
      AND a.id != b.id
  ),
  ranked AS (
    SELECT *, ROW_NUMBER() OVER (
      PARTITION BY entity_b_id
      ORDER BY distance_to_b_km ASC, entity_a_id ASC
    ) AS rank
    FROM distances_to_b
  ),
  resolved AS (
    SELECT * FROM ranked WHERE rank = {k_rank}
  )
SELECT
  r.entity_a_id AS entity_a_id,
  r.entity_a_name AS entity_a_name,
  r.entity_b_id AS entity_b_id,
  r.entity_b_name AS entity_b_name,
  c.id   AS entity_c_id,
  c.name AS entity_c_name,
  CAST(
    CASE
      WHEN ST_Intersects(ST_GeomFromWKT(r.entity_a_geometry),
                         ST_GeomFromWKT(c.geometry))
        THEN 0.0
      ELSE ST_DistanceSphere(
             ST_ClosestPoint(ST_GeomFromWKT(r.entity_a_geometry),
                             ST_GeomFromWKT(c.geometry)),
             ST_ClosestPoint(ST_GeomFromWKT(c.geometry),
                             ST_GeomFromWKT(r.entity_a_geometry))
           ) / 1000.0
    END AS STRING
  ) AS ground_truth
FROM resolved r, {pool_c} c
WHERE r.country_iso3 = c.country_iso3
  AND r.entity_a_id != c.id
  AND r.entity_b_id != c.id
LIMIT {total_sample};
