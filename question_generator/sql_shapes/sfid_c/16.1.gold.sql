-- Gold triples for TID 16.1: per-anchor (entity_b) PLUS each pool_a
-- entity that intersects the anchor (the entities the COUNT actually
-- depends on).
--
-- BACKGROUND: Without this, gold only contains anchor B's triples
-- (name + geometry + type). The question "how many highways intersect
-- B?" cannot be answered from that alone — a model retrieving the gold
-- has no way to see the highways it's supposed to count. Same gap
-- 10.1 fixed for radius-containment counts.
--
-- FIX: extend the URI set to fetch triples for to also include each
-- pool_a entity that intersects the anchor, in the same country. The
-- spatial predicate is identical to the main SQL's join_condition so
-- the gold lists exactly the entities the COUNT aggregated. All those
-- triples are attributed to the anchor's local id (so the exporter
-- joins them to the same question_id).
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
  -- All attributed to the anchor's `id` so they appear under the same
  -- question_id post-export.
  uris_to_fetch AS (
    -- Anchor entity URIs.
    SELECT id, entity_uri FROM entity_uris
    UNION
    -- Intersecting pool_a entities — same country, no-self, and the
    -- ST_Intersects spatial predicate from the main SQL's
    -- join_condition. The pool_a_all view's filter already restricts
    -- to the queried node_type (e.g. highway), so we don't repeat it
    -- here.
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
