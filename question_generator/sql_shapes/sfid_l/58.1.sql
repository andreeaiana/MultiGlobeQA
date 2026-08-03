-- TID 58.1 — Qualitative near (boolean by distance threshold).
-- Question: "Is {{entity_A}} approximately near {{entity_B}}?"
-- Ground truth: 'true' if distance ≤ {near_threshold_km}, 'false' if
-- distance > {near_threshold_km} * 3 (ambiguous zone excluded).
--
-- Two-pool rank-equality structural-pair sampler (mirror of 1.1) with
-- the same-h3-cell constraint added (mirror of 49.1) so pairs are
-- co-located. Threshold is per-pairing (POI×POI: 1km, POI×natfeat: 5km,
-- natfeat×natfeat: 10km — pinned via meta.parameters.near_threshold_km).
--
-- Ambiguous-zone exclusion: pairs with distance in (threshold, 3*threshold]
-- are dropped via the outer WHERE so the GT is unambiguous.
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
  distances AS (
    SELECT
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
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
      END AS distance_km
    FROM pool_a_capped a
    INNER JOIN pool_b_capped b
      ON {join_condition}
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  CASE
    WHEN distance_km IS NULL THEN NULL
    WHEN distance_km <= {near_threshold_km} THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM distances
WHERE distance_km IS NULL
   OR distance_km <= {near_threshold_km}
   OR distance_km > {near_threshold_km} * 3.0
LIMIT {total_sample};
