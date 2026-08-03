-- 5.1 — 3-pool gold triples (provenance for entity_a, entity_b, entity_c).
--
-- Mirror of 13.1.gold.sql's 3-pool pattern — the "country bordering
-- the country containing entity_A" is computed from a Natural Earth
-- session view (NOT from KG triples), so it doesn't appear in the
-- gold provenance. A model retrieving the gold sees the three
-- entity URIs (anchor + 2 candidates) plus their label/type/geometry
-- triples, which is sufficient for the model to compute the
-- comparison itself if it knows which countries border which.
--
-- The country-adjacency lookup is benchmark-runtime infrastructure
-- (the executor's session view, populated from public data) — the
-- gold for the BENCHMARK QUESTION is just the entities involved.
WITH picked_entities AS (
    SELECT entity_a_id AS id FROM _gt_result
    UNION
    SELECT entity_b_id AS id FROM _gt_result
    UNION
    SELECT entity_c_id AS id FROM _gt_result
  ),
  entity_uris AS (
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_a_all} e ON e.id = p.id
    UNION
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_b_all} e ON e.id = p.id
    UNION
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_c_all} e ON e.id = p.id
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
