-- TID 61.1 — Qualitative vague nearest (top-k set enumeration).
-- Question: "Which {{entity_A}} are roughly closest to {{entity_B}}?"
-- Ground truth: JSON list of top {top_k} pool_a entity names nearest to
-- each pool_b anchor.
--
-- Mirror of 11.1's anchor + collect_set + ROW_NUMBER pattern with the
-- SELECT aggregating top-k by distance instead of all matches in radius.
-- Per-country anchor sampling on pool_b (mirror 10.1 / 12.1).
--
-- HAVING COUNT(entity_a_id) >= 3: anchors with < 3 candidates
-- in their h3 cell are dropped (the question becomes trivial / under-
-- constrained).
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
    SELECT id, name, country_iso3, h3_res3, geometry
    FROM pool_b_capped
    WHERE bucket_rank <= {per_country_cap}
  ),
  ranked AS (
    SELECT
      b.id   AS entity_b_id,
      b.name AS entity_b_name,
      a.id   AS entity_a_id,
      a.name AS entity_a_name,
      ROW_NUMBER() OVER (
        PARTITION BY b.id
        ORDER BY
          CASE
            WHEN a.geometry IS NULL OR b.geometry IS NULL THEN 1.0e18
            WHEN ST_Intersects(ST_GeomFromWKT(a.geometry),
                               ST_GeomFromWKT(b.geometry))
              THEN 0.0
            ELSE ST_DistanceSphere(
                   ST_ClosestPoint(ST_GeomFromWKT(a.geometry),
                                   ST_GeomFromWKT(b.geometry)),
                   ST_ClosestPoint(ST_GeomFromWKT(b.geometry),
                                   ST_GeomFromWKT(a.geometry))
                 ) / 1000.0
          END ASC,
          a.name ASC
      ) AS rn
    FROM anchors b
    LEFT JOIN {pool_a} a
      ON {join_condition}
  )
SELECT
  entity_b_id,
  entity_b_name,
  to_json(sort_array(collect_set(entity_a_name))) AS ground_truth
FROM ranked
WHERE rn <= {top_k} AND entity_a_id IS NOT NULL
GROUP BY entity_b_id, entity_b_name
HAVING COUNT(entity_a_id) >= 3
LIMIT {total_sample};
