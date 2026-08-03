-- TID 2.4 — directional distance: west component.
-- Mirror of 2.3 (east) with the direction inverted. SELECT projects the
-- WESTWARD zonal component (same formula as 2.3 — magnitude is direction-
-- agnostic), with meta.join_condition adding ST_X(centroid(a)) <
-- ST_X(centroid(b)) so only A west-of-B pairs survive.
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
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CAST(
    CASE
      WHEN ST_Intersects(ST_GeomFromWKT(a.geometry),
                         ST_GeomFromWKT(b.geometry))
        THEN 0.0
      ELSE ST_DistanceSphere(
             ST_Point(
               ST_X(ST_Centroid(ST_GeomFromWKT(a.geometry))),
               ST_Y(ST_Centroid(ST_GeomFromWKT(b.geometry)))
             ),
             ST_Centroid(ST_GeomFromWKT(b.geometry))
           ) / 1000.0
    END AS STRING
  )                                     AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
