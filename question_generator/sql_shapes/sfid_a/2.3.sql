-- TID 2.3 — directional distance: east component.
-- Cardinal direction sibling of 2.1/2.2. SELECT projects the EASTWARD
-- (zonal) component: distance from a point at (A.lon, B.lat) to B's
-- centroid, computed when A is EAST of B (meta.join_condition adds
-- ST_X(centroid(a)) > ST_X(centroid(b))).
--
-- Note swap of axes vs 2.1: here A's longitude is the projected coord
-- and B's latitude is held constant — pure E-W distance.
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
      -- Project A's lon onto B's lat, measure to B's centroid.
      -- Pure E-W distance; ignores how far north/south A is from B.
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
