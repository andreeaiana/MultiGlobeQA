-- TID 59.1 — Qualitative far (boolean by distance threshold).
-- Question: "Is {{entity_A}} far from {{entity_B}}?"
-- Ground truth: 'true' if distance ≥ {far_threshold_km}, 'false' if
-- distance ≤ {far_threshold_km} / 3 (ambiguous zone excluded).
--
-- Mirror of 58.1 with the threshold direction inverted. Per-pairing
-- threshold (POI×POI: 10km, POI×natfeat: 50km, POI×region: 200km, etc.
-- — pinned via meta.parameters.far_threshold_km).
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
    WHEN distance_km >= {far_threshold_km} THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM distances
WHERE distance_km IS NULL
   OR distance_km >= {far_threshold_km}
   OR distance_km <= {far_threshold_km} / 3.0
LIMIT {total_sample};
