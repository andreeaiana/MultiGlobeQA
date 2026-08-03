-- TID 43.1 — Event-to-entity distance, sampled via independent shuffles +
-- rank-equality join (mirror of 1.1).
--
-- Question: "How far away was the {{event_A}} from {{entity_B}} in {{time_period}}?"
-- Ground truth: distance in km between the event footprint and the entity.
--
-- pool_a = hazard event instances (active during [period_start, period_end]).
-- pool_b = POI / region instances (the "located" entity).
--
-- TEMPORAL FILTER lives in the pool_a CTE (NOT in join_condition) — we want
-- the event-active filter to narrow the shuffled pool BEFORE rank assignment,
-- so the rank-equality join still pairs evenly across the surviving events.
-- Putting the filter in join_condition would let inactive-period events
-- consume rank slots and silently shrink the answer set after the join.
--
-- Each pool gets a global ROW_NUMBER ordered by HASH(id, seed). pool_b uses
-- a different seed so rank k on each side points to different entities (per
-- 1.1's pattern). The {join_condition} carries the same-country and no-self
-- predicates plus the rank-equality structural join.
--
-- KG SCOPE: kwg only. Only kwg has hazard_event_category entities with
-- start_date / end_date populated; osmh3kg / worldkg either have no event
-- entities or no temporal columns on them. The variants block in this
-- shape's meta.yaml records the same per-KG scope.
WITH pool_a_capped AS (
    -- Filter to events active during the queried period BEFORE rank assignment.
    -- An event is "active" iff its [start_date, end_date] interval overlaps
    -- ['{period_start}', '{period_end}'].
    SELECT
      a.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_a})) AS unique_id
    FROM {pool_a} a
    WHERE a.start_date <= DATE('{period_end}')
      AND a.end_date   >= DATE('{period_start}')
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
  a.name AS event_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  -- Time-period label for {{time_period}} (literal range from parameters).
  CONCAT('{period_start}', ' to ', '{period_end}') AS time_period,
  CAST(
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
    END AS STRING
  )                                     AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
