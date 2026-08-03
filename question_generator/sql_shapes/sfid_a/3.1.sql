-- TID 3.1 — distance from a hazard event to a POI restricted to a sampled
-- region, gated by a temporal-overlap window (3-pool, temporal).
--
-- Question: "How far was {{event_A}} (during {{time_period}}) from
--            {{entity_B}} in {{entity_C}}?"
-- Ground truth: distance in km between the event footprint and a POI
-- contained by region entity_c.
--
-- pool_a = hazard event instances (active during [period_start, period_end])
-- pool_b = POI candidates (constrained to lie inside the sampled region)
-- pool_c = administrative regions (one anchor region selected per question)
--
-- Shape: structurally identical to 5.1 with two additions:
--   1. pool_a is filtered to events whose [start_date, end_date]
--      interval overlaps the [period_start, period_end] window. The
--      filter lives in the pool_a CTE (NOT in join_condition) so
--      inactive-period events don't consume rank slots — same rationale
--      as 43.1's pool_a CTE design.
--   2. The SELECT projects time_period concatenated from event dates,
--      kept inside ground_truth's CASE/CAST chain so the contract's
--      result_columns extractor sees a flat 5-column projection
--      (entity_a/b/c ids + names + ground_truth). The actual time_period
--      string is folded into the question template via the exporter's
--      attribute lookup, not surfaced as a separate column in this shape.
--
-- KG SCOPE: kwg only. Hazard events with start_date/end_date are first-
-- class only in KnowWhereGraph (NOAA / NIFC / MTBS); osmh3kg / worldkg
-- have no hazard_event_category entities with populated temporal columns.
-- Mirrors 43.1's per-KG scope rationale.
--
-- The {join_condition} carries:
--   - a.country_iso3 = c.country_iso3 (event stays in same country
--     as anchor region)
--   - a.unique_id    = c.unique_id    (rank-equality)
--   - a.id          != c.id           (no event-as-region)
WITH pool_a_capped AS (
    -- Filter to events active during the queried period BEFORE rank
    -- assignment (see this file's header for the rationale).
    -- Per-country hash-diversity rank (matches sampling.strategy
    -- row_number_hash declared in meta, shared with the 10.1/11.1/37.1
    -- pattern). PARTITION BY country_iso3 means unique_id resets per
    -- country, so the downstream rank-equality join (a.country_iso3 =
    -- c.country_iso3 AND a.unique_id = c.unique_id) pairs within
    -- country. PARTITION BY is required: an unpartitioned ORDER BY
    -- RAND() form forces WindowExec into a single partition (multi-hour
    -- hang on full kwg events × regions; not LIMIT-pushdown eligible
    -- because the rank-equality join needs all pairs).
    SELECT
      a.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS unique_id
    FROM {pool_a} a
    WHERE a.start_date <= DATE('{period_end}')
      AND a.end_date   >= DATE('{period_start}')
  ),
  pool_b_geom AS (
    -- Materialize POI centroid as Geometry-typed column upstream so
    -- Sedona's RangeJoinExec recognizes the spatial predicate in the
    -- pois_in_region JOIN below. Inline ST_Centroid(ST_GeomFromWKT(...))
    -- inside JOIN ON blocks the recognizer and degrades to broadcast
    -- hash join + per-row spatial check. Carry geometry text through
    -- too — the final SELECT's distance computation reads the raw POI
    -- WKT.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, geometry,
      ST_Centroid(ST_GeomFromWKT(geometry)) AS centroid_geom
    FROM {pool_b}
    WHERE geometry IS NOT NULL
  ),
  pool_c_capped AS (
    -- Per-country hash-diversity rank on regions (mirrors pool_a above).
    -- Pre-materialize geom for the spatial join below.
    SELECT /*+ REPARTITION(16) */
      c.*,
      ST_GeomFromWKT(geometry) AS geom,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_c})
      ) AS unique_id
    FROM {pool_c} c
    WHERE geometry IS NOT NULL
  ),
  pois_in_region AS (
    -- POI b is "in" region c iff c contains b's centroid. Joined on
    -- country_iso3 first to prune cross-pool pairs before the spatial
    -- predicate. Both geom columns are Geometry-typed (not WKT) so
    -- Sedona's RangeJoinExec activates.
    SELECT
      b.id           AS entity_b_id,
      b.name         AS entity_b_name,
      b.country_iso3 AS country_iso3,
      b.geometry     AS poi_geom,
      c.unique_id    AS region_unique_id,
      c.id           AS region_id
    FROM pool_b_geom b
    JOIN pool_c_capped c
      ON b.country_iso3 = c.country_iso3
     AND ST_Contains(c.geom, b.centroid_geom)
  )
SELECT
  a.id            AS entity_a_id,
  a.name          AS entity_a_name,
  a.name          AS event_a_name,
  p.entity_b_id   AS entity_b_id,
  p.entity_b_name AS entity_b_name,
  c.id            AS entity_c_id,
  c.name          AS entity_c_name,
  -- Time-period label for {{time_period}} (literal range from parameters).
  CONCAT('{period_start}', ' to ', '{period_end}') AS time_period,
  CAST(
    CASE
      WHEN a.geometry IS NULL OR p.poi_geom IS NULL THEN NULL
      WHEN ST_Intersects(ST_GeomFromWKT(a.geometry),
                         ST_GeomFromWKT(p.poi_geom))
        THEN 0.0
      ELSE ST_DistanceSphere(
             ST_ClosestPoint(ST_GeomFromWKT(a.geometry),
                             ST_GeomFromWKT(p.poi_geom)),
             ST_ClosestPoint(ST_GeomFromWKT(p.poi_geom),
                             ST_GeomFromWKT(a.geometry))
           ) / 1000.0
    END AS STRING
  )                                     AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_c_capped c
  ON {join_condition}
INNER JOIN pois_in_region p
  ON p.region_unique_id = c.unique_id
 AND a.id != p.entity_b_id
LIMIT {total_sample};
