-- TID 45.1 — Event hit date for one (event, intersecting region) pair.
-- Question: "When did {{event_A}} hit {{entity_A}}?"
-- Ground truth: e.start_date (the event's start date — when the event hit
-- the region the event geometry intersects).
--
-- pool_a = hazard event instances (the {{event_A}} slot)
-- pool_b = administrative regions (the {{entity_A}} slot)
--
-- COLUMN COLLISION AVOIDANCE.
-- The template uses BOTH {{event_A}} and {{entity_A}} in the same
-- row. If both mapped to entity_a_name, one slot would silently
-- overwrite the other in the rendered question
-- ("When did Greenwood County hit Greenwood County?"). The exporter
-- recognizes a separate event_a_* column family — this SQL projects
-- event_a_id / event_a_name (the event) AND entity_a_id / entity_a_name
-- (the region) so both placeholders resolve independently.
--
-- One row per (event, intersecting region) pair — no GROUP BY because
-- each region in the intersection set produces a distinct question
-- ("When did event X hit region R1?", "...region R2?", ...). The
-- ground truth is e.start_date in every row of a given event (the
-- event's hit date is a property of the event itself, not the pair),
-- but the questions are different.
--
-- KG SCOPE: kwg only. Hazard events with start_date are first-class
-- only in KnowWhereGraph (NOAA / NIFC / MTBS).
--
-- ANCHOR SAMPLING — per-country ROW_NUMBER hash diversity on the event
-- side. pool_b is NOT sampled — for each sampled event we want every
-- intersecting region pair (the answer set the question is asking about).
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
    SELECT id, name, geometry, country_iso3, start_date
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  e.id         AS event_a_id,
  e.name       AS event_a_name,
  r.id         AS entity_a_id,
  r.name       AS entity_a_name,
  CAST(e.start_date AS STRING) AS ground_truth
FROM anchors e
INNER JOIN {pool_b} r
  ON e.country_iso3 = r.country_iso3
 AND e.geometry IS NOT NULL
 AND r.geometry IS NOT NULL
 AND ST_Intersects(ST_GeomFromWKT(e.geometry), ST_GeomFromWKT(r.geometry))
WHERE e.start_date IS NOT NULL
LIMIT {total_sample};
