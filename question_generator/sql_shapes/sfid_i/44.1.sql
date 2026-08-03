-- TID 44.1 — Event affected regions (set enumeration).
-- Question: "Which regions were affected by {{event_A}}?"
-- Ground truth: SORT_ARRAY(COLLECT_LIST(region_name)) of every region whose
-- geometry intersects the event's geometry, restricted to the same country.
--
-- pool_a = hazard event instances (the {{event_A}} slot)
-- pool_b = administrative regions
--
-- NOTE — column family for the event slot.
-- The exporter has an event_*_name column family that {{event_A}}
-- substitutes from. The per-ftid SQL must project
-- event_a_id / event_a_name DIRECTLY (not as entity_a_*) so the exporter
-- picks them up. 44.1 has no other entity slot in the question template,
-- so only the event_a_* columns are projected on the per-row side; the
-- aggregated region names are returned as ground_truth.
--
-- KG SCOPE: kwg only. Hazard events are first-class only in
-- KnowWhereGraph (NOAA / NIFC / MTBS). osmh3kg / worldkg have no
-- hazard_event_category entities.
--
-- ANCHOR SAMPLING — per-country ROW_NUMBER hash diversity on the event side
-- (pool_a). pool_b is NOT sampled: the COLLECT_LIST aggregate needs every
-- intersecting region, so capping pool_b would truncate the answer set.
-- HAVING SIZE >= 1 drops events with no intersecting region (otherwise
-- the question would render with an empty answer set and be unanswerable).
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
  -- Plural label for {{entity_A}} (the aggregated region set).
  '{entity_a_label}' AS entity_a_name,
  -- to_json keeps the ground_truth column STRING-typed, matching the
  -- canonical pattern used across sfid_m/sfid_i. Returning a raw
  -- ARRAY<STRING> diverges from sibling shapes whose ground_truth is
  -- STRING and breaks the parquet schema-merge at consolidation time
  -- (mismatch between this row's ARRAY<STRING> and the sibling shapes'
  -- STRING).
  to_json(SORT_ARRAY(COLLECT_LIST(r.name))) AS ground_truth
FROM anchors e
INNER JOIN {pool_b} r
  ON e.country_iso3 = r.country_iso3
 AND e.geometry IS NOT NULL
 AND r.geometry IS NOT NULL
 AND ST_Intersects(ST_GeomFromWKT(e.geometry), ST_GeomFromWKT(r.geometry))
GROUP BY e.id, e.name
HAVING SIZE(COLLECT_LIST(r.name)) >= 1
LIMIT {total_sample};
