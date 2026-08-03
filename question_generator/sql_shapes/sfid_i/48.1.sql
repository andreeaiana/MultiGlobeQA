-- TID 48.1 — Event area in region (km²) per (event, intersecting region) pair.
-- Question: "What area of {{event_A}} fell in {{entity_A}}?"
-- Ground truth: ST_AreaSpheroid(ST_Intersection(event.geom, region.geom)) / 1e6
-- — area in km² of the intersection between the event geometry and the
-- region geometry, on the WGS84 spheroid.
--
-- pool_a = hazard event instances (the {{event_A}} slot)
-- pool_b = administrative regions (the {{entity_A}} slot)
--
-- COLUMN COLLISION RESOLUTION.
-- Like 45.1, the question template uses BOTH {{event_A}} and {{entity_A}}
-- in the same row. The exporter's event_*_name column family lets both
-- placeholders resolve independently — this SQL projects event_a_id /
-- event_a_name AND entity_a_id / entity_a_name. Without the column
-- family extension, one slot would silently overwrite the other.
--
-- One row per (event, intersecting region) pair — each pair has its own
-- intersection-area answer. No GROUP BY: questions like "what area of
-- event X fell in region R1" and "...region R2" are different questions
-- with different answers.
--
-- KG SCOPE: kwg only. Hazard events are first-class only in
-- KnowWhereGraph.
--
-- ANCHOR SAMPLING — per-country ROW_NUMBER hash diversity on the event
-- side. pool_b is NOT sampled — for each sampled event we want every
-- intersecting region pair (each pair is a distinct question).
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
    SELECT id, name, geometry, country_iso3
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  e.id   AS event_a_id,
  e.name AS event_a_name,
  r.id   AS entity_a_id,
  r.name AS entity_a_name,
  CAST(
    ST_AreaSpheroid(
      ST_Intersection(
        ST_GeomFromWKT(e.geometry),
        ST_GeomFromWKT(r.geometry)
      )
    ) / 1000000.0
    AS STRING
  ) AS ground_truth
FROM anchors e
INNER JOIN {pool_b} r
  ON e.country_iso3 = r.country_iso3
 AND e.geometry IS NOT NULL
 AND r.geometry IS NOT NULL
 AND ST_Intersects(ST_GeomFromWKT(e.geometry), ST_GeomFromWKT(r.geometry))
LIMIT {total_sample};
