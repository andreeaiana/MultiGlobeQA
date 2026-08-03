-- Gold triples for TID 17.1: per-anchor (entity_b) PLUS each pool_a
-- entity that intersects the anchor (the entities the COLLECT_LIST
-- enumeration actually returns).
--
-- Same shape as 26.1.gold.sql — the enumeration aggregate (17.1) and
-- the count aggregate (16.1) both depend on the same set of pool_a
-- entities matching the spatial predicate. A model retrieving the
-- gold should see every entity name that appears in the answer set;
-- without this extension, only the anchor's triples would be
-- available.
--
-- MAINTENANCE CONTRACT: Structurally identical to 26.1.gold.sql — if
-- one changes, the other should too. The two files exist to keep
-- per-TID gold concerns localized; their shape divergence (if any)
-- signals a real semantic difference, not just convention drift.
WITH picked_entities AS (
    SELECT entity_b_id AS id FROM _gt_result
  ),
  -- Anchor entities + their geometry/country (needed for the spatial
  -- join below to find every pool_a entity that intersected this anchor).
  entity_uris AS (
    SELECT p.id, e.entity_uri, e.geometry, e.country_iso3
    FROM picked_entities p
    JOIN {pool_b_all} e ON e.id = p.id
  ),
  -- URIs to fetch triples for: anchor URIs + intersecting pool_a URIs.
  uris_to_fetch AS (
    SELECT id, entity_uri FROM entity_uris
    UNION
    SELECT eu.id, t.entity_uri
    FROM entity_uris eu
    JOIN {pool_a_all} t
      ON eu.country_iso3 = t.country_iso3
     AND eu.id != t.id
     AND ST_Intersects(
           ST_GeomFromWKT(t.geometry),
           ST_GeomFromWKT(eu.geometry)
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
