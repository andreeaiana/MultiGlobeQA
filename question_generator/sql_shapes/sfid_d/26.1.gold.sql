-- Gold triples for TID 26.1: per-anchor (entity_b = road) PLUS each
-- POI matched by the proximity predicate (the entities the SET
-- enumeration aggregated).
--
-- BACKGROUND: Without this, gold only contained the road anchor's
-- triples (name + geometry + type). The question "Which cafes does
-- road X pass by?" cannot be answered from that alone — a model
-- retrieving the gold has no way to see the cafes it's supposed to
-- enumerate. Same gap class 10.1 / 11.1 hit; same fix shape applies.
--
-- FIX: extend the URI set to fetch triples for to also include each
-- POI within {proximity_buffer_meters} of the road, in the same
-- country, of the queried node_type. The proximity predicate is
-- identical to the main SQL's join — the gold lists exactly the
-- entities the collect_set aggregated. All triples are attributed
-- to the anchor's local id so the exporter joins them to the same
-- question_id (single-role pool_map per meta.yaml).
--
-- ANCHOR URI CAVEAT: the aggregated road view exposes a single
-- ``MIN(entity_uri)`` per merged road (the first OSM way's URI for
-- that name+country+type bucket). Gold for the road anchor attaches
-- to that single URI rather than every constituent way — acceptable
-- since the named road as a concept has no canonical OSM URI.
--
-- MAINTENANCE CONTRACT: Structurally identical to 43.1.gold.sql — if
-- one changes, the other should too. The two gold files exist to keep
-- per-TID gold concerns localized; their shape divergence (if any)
-- signals a real semantic difference, not just convention drift.
WITH picked_entities AS (
    SELECT entity_b_id AS id FROM _gt_result
  ),
  -- Anchor entities (roads) + their geometry/country (needed for the
  -- proximity join below).
  entity_uris AS (
    SELECT p.id, e.entity_uri, e.geometry, e.country_iso3
    FROM picked_entities p
    JOIN {pool_b_all} e ON e.id = p.id
  ),
  -- URIs to fetch triples for: anchor URIs + nearby POI URIs.
  -- All attributed to the anchor's `id` so they appear under the same
  -- question_id post-export.
  uris_to_fetch AS (
    -- Anchor entity URIs (road).
    SELECT id, entity_uri FROM entity_uris
    UNION
    -- Nearby POIs of the queried category — same proximity predicate
    -- and same node_type filter as the main SQL's join condition.
    SELECT eu.id, t.entity_uri
    FROM entity_uris eu
    JOIN {pool_a_all} t
      ON eu.country_iso3 = t.country_iso3
     AND t.node_type = '{entity_a_category}'
     AND eu.id != t.id
     AND ST_DWithin(
           ST_GeomFromWKT(eu.geometry),
           ST_GeomFromWKT(t.geometry),
           {proximity_buffer_meters} * 0.00001
         )
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
