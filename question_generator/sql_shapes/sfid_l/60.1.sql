-- TID 60.1 — Qualitative comparative proximity (3-pool boolean).
-- Question: "Is {{entity_A}} more likely near {{entity_B}} than {{entity_C}}?"
-- Ground truth: 'true' if dist(A,B) < dist(A,C), 'false' otherwise.
--
-- Three-pool rank-equality structural-triple sampler (mirror of 4.1) with
-- distinct seeds per pool so the three rankings diverge (critical for
-- symmetric pools where pool_a == pool_b == pool_c).
--
-- Near-tie exclusion: pairs where |dist_AB - dist_AC| < 10% of the
-- smaller distance are dropped to avoid ambiguous GT (a relative
-- rather than absolute tolerance).
WITH pool_a_capped AS (
    SELECT a.*, ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_a})) AS unique_id
    FROM {pool_a} a
  ),
  pool_b_capped AS (
    SELECT b.*, ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_b})) AS unique_id
    FROM {pool_b} b
  ),
  pool_c_capped AS (
    SELECT c.*, ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_c})) AS unique_id
    FROM {pool_c} c
  ),
  dists AS (
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      c.id   AS entity_c_id,
      c.name AS entity_c_name,
      CASE
        WHEN a.geometry IS NULL OR b.geometry IS NULL THEN NULL
        WHEN ST_Intersects(ST_GeomFromWKT(a.geometry),
                           ST_GeomFromWKT(b.geometry))
          THEN 0.0
        ELSE ST_DistanceSphere(
               ST_ClosestPoint(ST_GeomFromWKT(a.geometry),
                               ST_GeomFromWKT(b.geometry)),
               ST_ClosestPoint(ST_GeomFromWKT(b.geometry),
                               ST_GeomFromWKT(a.geometry))
             ) / 1000.0
      END AS distance_ab_km,
      CASE
        WHEN a.geometry IS NULL OR c.geometry IS NULL THEN NULL
        WHEN ST_Intersects(ST_GeomFromWKT(a.geometry),
                           ST_GeomFromWKT(c.geometry))
          THEN 0.0
        ELSE ST_DistanceSphere(
               ST_ClosestPoint(ST_GeomFromWKT(a.geometry),
                               ST_GeomFromWKT(c.geometry)),
               ST_ClosestPoint(ST_GeomFromWKT(c.geometry),
                               ST_GeomFromWKT(a.geometry))
             ) / 1000.0
      END AS distance_ac_km
    FROM pool_a_capped a
    INNER JOIN pool_b_capped b ON a.unique_id = b.unique_id
    INNER JOIN pool_c_capped c ON a.unique_id = c.unique_id
    WHERE {join_condition}
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  entity_c_id,
  entity_c_name,
  CASE
    WHEN distance_ab_km IS NULL OR distance_ac_km IS NULL THEN NULL
    WHEN distance_ab_km < distance_ac_km THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM dists
WHERE distance_ab_km IS NULL
   OR distance_ac_km IS NULL
   OR ABS(distance_ab_km - distance_ac_km) > 0.1 * LEAST(distance_ab_km, distance_ac_km)
LIMIT {total_sample};
