-- TID 9.1 — entity to country boundary distance.
-- Question: "How far is {{entity_A}} from the boundary of {{entity_B}}?"
--
-- Shape: 2-pool TID. pool_b is countries; their multipolygon boundaries
-- are exploded via LATERAL VIEW EXPLODE(ST_Dump(...)) so each ring is
-- handled independently — needed because ST_ExteriorRing on a
-- MULTIPOLYGON would only return the first polygon's outer ring,
-- silently undercounting the boundary.
--
-- Sampling: pool_a anchors get per-country diversity sampling. pool_b
-- is uncapped — there are ~200 countries total so the cross-join with
-- the (capped) anchors is bounded.
WITH pool_a_capped AS (
    SELECT
      a.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS bucket_rank
    FROM {pool_a} a
  ),
  anchors AS (
    SELECT *
    FROM pool_a_capped
    WHERE bucket_rank <= {anchor_cap}
  ),
  exploded_boundaries AS (
    SELECT
      b.id    AS entity_b_id,
      b.name  AS entity_b_name,
      b.country_iso3,
      ST_ExteriorRing(exploded.geom) AS boundary_ring
    FROM {pool_b} b
    LATERAL VIEW EXPLODE(ST_Dump(ST_GeomFromWKT(b.geometry))) exploded AS geom
  ),
  boundary_distances AS (
    SELECT
      a.id    AS entity_a_id,
      a.name  AS entity_a_name,
      eb.entity_b_id,
      eb.entity_b_name,
      CASE
        WHEN ST_Intersects(ST_GeomFromWKT(a.geometry), eb.boundary_ring)
          THEN 0.0
        ELSE ST_DistanceSphere(
               ST_ClosestPoint(ST_GeomFromWKT(a.geometry), eb.boundary_ring),
               ST_ClosestPoint(eb.boundary_ring, ST_GeomFromWKT(a.geometry))
             ) / 1000.0
      END AS distance_km
    FROM anchors a, exploded_boundaries eb
    WHERE a.country_iso3 = eb.country_iso3
      AND a.id != eb.entity_b_id
  )
SELECT
  entity_a_id,
  entity_a_name,
  entity_b_id,
  entity_b_name,
  CAST(MIN(distance_km) AS STRING) AS ground_truth
FROM boundary_distances
GROUP BY entity_a_id, entity_a_name, entity_b_id, entity_b_name
LIMIT {total_sample};
