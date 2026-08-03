-- Gold triples for TID 10.1: per-anchor (entity_b) PLUS each nearby pharmacy
-- (the entities the COUNT actually depends on).
--
-- BACKGROUND: Without this, gold only contained anchor B's triples
-- (name + geometry + type). The question "how many pharmacies are
-- within 5km of B?" cannot be answered from that alone — a model
-- retrieving the gold has no way to see the pharmacies it's supposed
-- to count. Gold-sufficiency validation across all 7 ported TIDs
-- flagged 10.1 as a TID with this gap; 11.1 inherits the same shape
-- and the same fix.
--
-- FIX: extend the URI set to fetch triples for to also include each
-- pharmacy within {radius_km} of the anchor, in the same country.
-- The radius+node_type filter is identical to the main SQL's join
-- predicate, so the gold lists exactly the entities the COUNT
-- aggregated. All those triples are attributed to the anchor's
-- local id (so the exporter joins them to the same question_id).
--
-- MAINTENANCE CONTRACT: Structurally identical to 20.1.gold.sql — if
-- one changes, the other should too. The two files exist to keep
-- per-TID gold concerns localized; their shape divergence (if any)
-- signals a real semantic difference, not just convention drift.
WITH picked_entities AS (
    SELECT entity_b_id AS id FROM _gt_result
  ),
  -- Anchor entities + their geometry/country (needed for the radius join below)
  entity_uris AS (
    SELECT p.id, e.entity_uri, e.geometry, e.country_iso3
    FROM picked_entities p
    JOIN {pool_b_all} e ON e.id = p.id
  ),
  -- URIs to fetch triples for: anchor URIs + nearby pharmacy URIs.
  -- All attributed to the anchor's `id` so they appear under the same
  -- question_id post-export.
  uris_to_fetch AS (
    -- Anchor entity URIs.
    SELECT id, entity_uri FROM entity_uris
    UNION
    -- Nearby pharmacies — same node_type filter and radius condition
    -- as the main SQL's join_condition.
    SELECT eu.id, t.entity_uri
    FROM entity_uris eu
    JOIN {pool_a_all} t
      ON eu.country_iso3 = t.country_iso3
     AND t.node_type = '{entity_a_category}'
     AND eu.id != t.id
     AND ST_DistanceSphere(
           ST_GeomFromWKT(t.geometry),
           ST_GeomFromWKT(eu.geometry)
         ) / 1000.0 <= {radius_km}
  ),
  direct_triples AS (
    SELECT /*+ BROADCAST(u) */
           u.id AS entity_id, t.subject, t.predicate, t.object
    FROM uris_to_fetch u
    JOIN {triples_view} t ON t.subject = u.entity_uri
    WHERE t.predicate IN ({predicate_iris})
  ),
  geometry_nodes AS (
    SELECT entity_id, object AS geo_node_uri
    FROM direct_triples
    WHERE predicate = '{pred_geometry_link}'
  ),
  geometry_triples AS (
    SELECT /*+ BROADCAST(gn) */
           gn.entity_id, t.subject, t.predicate, t.object
    FROM geometry_nodes gn
    JOIN {triples_view} t ON t.subject = gn.geo_node_uri
    WHERE t.predicate = '{pred_geometry_wkt}'
  )
SELECT entity_id, subject, predicate, object FROM direct_triples
UNION ALL
SELECT entity_id, subject, predicate, object FROM geometry_triples;
