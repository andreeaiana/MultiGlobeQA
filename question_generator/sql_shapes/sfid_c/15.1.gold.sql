-- Gold triples for TID 15.1: per-pair (entity_a, entity_b) provenance.
--
-- The ground truth is an integer point count derived from
-- ST_Intersection(a, b) — it depends on the GEOMETRIES of the two anchor
-- entities, not on a set of further entities the way 16.1 / 17.1 / 18.1
-- aggregate. Per port instructions: "anchor-only gold (no extension)".
--
-- Mirrors 1.1.gold.sql structure: UNION the entity_a / entity_b ids,
-- resolve to URIs from each pool's unfiltered _all view, fetch label /
-- type / geometry_link triples, then chase the geometry_link to fetch
-- the geometry_wkt triple.
WITH picked_entities AS (
    SELECT entity_a_id AS id FROM _gt_result
    UNION
    SELECT entity_b_id AS id FROM _gt_result
  ),
  entity_uris AS (
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_a_all} e ON e.id = p.id
    UNION
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_b_all} e ON e.id = p.id
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
