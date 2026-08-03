-- Gold triples for TID 12.1: per-anchor (entity_a) only.
--
-- Single-role provenance. The benchmark row is one-per-anchor and
-- ground_truth is a numeric population aggregate over the global
-- pop_tiles raster — population isn't an entity, so there are no
-- "matched neighbour" entities to pull triples for (unlike 10.1 /
-- 11.1, whose answer set IS a list of pool_a entities and whose
-- gold UNIONs anchor URIs with neighbour URIs). Same structural
-- pattern as 63.1.gold.sql: pull label / type / geometry triples
-- for the anchor itself, with the geometry-link → geometry-wkt
-- two-hop join so consumers see the full WKT (not just the geo-node
-- URI).
WITH picked_entities AS (
    SELECT entity_a_id AS id FROM _gt_result
  ),
  entity_uris AS (
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_a_all} e ON e.id = p.id
  ),
  direct_triples AS (
    SELECT /*+ BROADCAST(eu) */
           eu.id AS entity_id, t.subject, t.predicate, t.object
    FROM entity_uris eu
    JOIN {triples_view} t ON t.subject = eu.entity_uri
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
