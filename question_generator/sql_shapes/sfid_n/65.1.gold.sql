-- Gold triples for TID 65.1 — single-anchor (region) provenance.
-- Mirror sfid_p/112.1.gold.sql / sfid_o/111.1.gold.sql: pull label /
-- type / geometry triples for the answer entity.
--
-- The centroid IS the answer, derived from the entity's polygon
-- geometry triple chain. The standard two-hop ({pred_geometry_link} →
-- {pred_geometry_wkt}) gives the polygon WKT a model can centroid
-- itself.
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
