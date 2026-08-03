-- 45.1 — gold for "When did {{event_A}} hit {{entity_A}}?".
--
-- Two anchors per row: the event (pool_a, surfaced as event_a_id) and the
-- region (pool_b, surfaced as entity_a_id). Mirrors 71.1.gold.sql's
-- two-pool layout — single per-pair answer (the event start_date), no
-- aggregated neighbour set so no UNION over a radius like 19.1.
--
-- Plus the kwg temporal 3-hop chain on the event anchor so the start_date
-- (the answer) is retrievable in gold (mirrors 4.1.gold.sql):
--   event → hasTemporalScope → interval
--   interval → fireDiscoveryTime / fireOutTime → time_instant
--   time_instant → inXSDDate / inXSDDateTime → date_literal
--
-- The first hop flows naturally through direct_triples via {predicate_iris}
-- (temporal_scope role). Hops 2 and 3 require dedicated CTEs. Hardcoded
-- IRIs match 4.1.gold.sql since 45.1 is kwg-only.
WITH picked_entities AS (
    SELECT event_a_id AS id FROM _gt_result
    UNION
    SELECT entity_a_id AS id FROM _gt_result
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
  ),
  -- Temporal hop 1: pull interval node URIs from direct_triples
  -- (hasTemporalScope is in {predicate_iris} via temporal_scope role).
  temporal_scope_nodes AS (
    SELECT entity_id, object AS scope_node_uri
    FROM direct_triples
    WHERE predicate = 'http://stko-kwg.geog.ucsb.edu/lod/ontology/hasTemporalScope'
  ),
  -- Temporal hop 2: interval → time_instant (fireDiscoveryTime / fireOutTime).
  temporal_instant_triples AS (
    SELECT /*+ BROADCAST(tn) */
           tn.entity_id, t.subject, t.predicate, t.object
    FROM temporal_scope_nodes tn
    JOIN {triples_view} t ON t.subject = tn.scope_node_uri
    WHERE t.predicate IN (
      'http://stko-kwg.geog.ucsb.edu/lod/ontology/fireDiscoveryTime',
      'http://stko-kwg.geog.ucsb.edu/lod/ontology/fireOutTime'
    )
  ),
  temporal_instants AS (
    SELECT entity_id, object AS instant_uri
    FROM temporal_instant_triples
  ),
  -- Temporal hop 3: time_instant → date literal.
  temporal_date_triples AS (
    SELECT /*+ BROADCAST(ti) */
           ti.entity_id, t.subject, t.predicate, t.object
    FROM temporal_instants ti
    JOIN {triples_view} t ON t.subject = ti.instant_uri
    WHERE t.predicate IN (
      'http://www.w3.org/2006/time#inXSDDate',
      'http://www.w3.org/2006/time#inXSDDateTime'
    )
  )
SELECT entity_id, subject, predicate, object FROM direct_triples
UNION ALL
SELECT entity_id, subject, predicate, object FROM geometry_triples
UNION ALL
SELECT entity_id, subject, predicate, object FROM temporal_instant_triples
UNION ALL
SELECT entity_id, subject, predicate, object FROM temporal_date_triples;
