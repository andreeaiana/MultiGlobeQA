-- TID 4.1 — comparative distance: which of A or B is CLOSER to C?
--
-- Shape: 3-pool rank-equality sampler, generalising 1.1's two-pool join
-- to three pools. Each pool gets a global ROW_NUMBER ordered by
-- RAND(seed_X) (three independent seeds so symmetric pool reuse —
-- POI × POI × POI — doesn't collapse to identical ranks). The triple
-- join pairs by unique_id equality and the country guard keeps the
-- triple in-country (a comparison across continents is meaningless).
--
-- Ground truth: name of the entity (a or b) closer to c — or 'both'
-- when ties or both intersect c.
WITH pool_a_capped AS (
    SELECT
      a.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_a})) AS unique_id
    FROM {pool_a} a
  ),
  pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_b})) AS unique_id
    FROM {pool_b} b
  ),
  pool_c_capped AS (
    SELECT
      c.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_c})) AS unique_id
    FROM {pool_c} c
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  c.id   AS entity_c_id,
  c.name AS entity_c_name,
  CAST(
    CASE
      WHEN ST_Intersects(ST_GeomFromWKT(a.geometry), ST_GeomFromWKT(c.geometry))
       AND NOT ST_Intersects(ST_GeomFromWKT(b.geometry), ST_GeomFromWKT(c.geometry))
        THEN a.name
      WHEN ST_Intersects(ST_GeomFromWKT(b.geometry), ST_GeomFromWKT(c.geometry))
       AND NOT ST_Intersects(ST_GeomFromWKT(a.geometry), ST_GeomFromWKT(c.geometry))
        THEN b.name
      WHEN ST_Intersects(ST_GeomFromWKT(a.geometry), ST_GeomFromWKT(c.geometry))
       AND ST_Intersects(ST_GeomFromWKT(b.geometry), ST_GeomFromWKT(c.geometry))
        THEN 'both'
      ELSE
        CASE
          WHEN ST_DistanceSphere(
                 ST_ClosestPoint(ST_GeomFromWKT(a.geometry), ST_GeomFromWKT(c.geometry)),
                 ST_ClosestPoint(ST_GeomFromWKT(c.geometry), ST_GeomFromWKT(a.geometry))
               ) < ST_DistanceSphere(
                 ST_ClosestPoint(ST_GeomFromWKT(b.geometry), ST_GeomFromWKT(c.geometry)),
                 ST_ClosestPoint(ST_GeomFromWKT(c.geometry), ST_GeomFromWKT(b.geometry))
               ) THEN a.name
          WHEN ST_DistanceSphere(
                 ST_ClosestPoint(ST_GeomFromWKT(b.geometry), ST_GeomFromWKT(c.geometry)),
                 ST_ClosestPoint(ST_GeomFromWKT(c.geometry), ST_GeomFromWKT(b.geometry))
               ) < ST_DistanceSphere(
                 ST_ClosestPoint(ST_GeomFromWKT(a.geometry), ST_GeomFromWKT(c.geometry)),
                 ST_ClosestPoint(ST_GeomFromWKT(c.geometry), ST_GeomFromWKT(a.geometry))
               ) THEN b.name
          ELSE 'both'
        END
    END AS STRING
  ) AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON a.country_iso3 = b.country_iso3
 AND a.unique_id = b.unique_id
INNER JOIN pool_c_capped c
  ON a.country_iso3 = c.country_iso3
 AND a.unique_id = c.unique_id
WHERE {join_condition}
LIMIT {total_sample};
