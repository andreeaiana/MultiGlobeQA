-- 3.1 — 3-pool gold triples with TEMPORAL provenance for the event anchor.
--
-- Mirrors 13.1.gold.sql's three-pool layout (entity_a / entity_b / entity_c)
-- plus the temporal CTE chain that kwg events require.
--
-- The kwg temporal model (per kg_prefixes.yaml::kwg.temporal_*) is a
-- 3-hop chain:
--   event → hasTemporalScope → interval        (captured in direct_triples
--                                               via {predicate_iris})
--   interval → fireDiscoveryTime / fireOutTime → time_instant
--   time_instant → inXSDDate / inXSDDateTime → date_literal
--
-- {predicate_iris} expands to ALL IRIs from gold_triples.predicates,
-- including the temporal roles. So direct_triples already pulls the
-- hasTemporalScope hop. The two downstream hops require dedicated
-- CTEs (analogous to geometry_nodes → geometry_triples):
--   - temporal_scope_nodes finds the interval URIs
--   - temporal_instant_triples joins to find the time_instant URIs
--   - temporal_instants extracts those URIs
--   - temporal_date_triples joins to find the date literals
--
-- The hasTemporalScope IRI is hardcoded as a literal here. 3.1 is
-- kwg-only (see 4.1.meta.yaml) so the predicate filter only needs
-- one KG's IRI — no per-KG token substitution needed. Same approach
-- as 9.1.gold.sql's {pred_geometry_link} hardcode for the geometry hop
-- (the executor substitutes that token; for temporal we substitute
-- inline since there is no analogous token defined).
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
  ),
  -- Temporal hop 1: pull the interval node URIs from direct_triples
  -- (the hasTemporalScope predicate is in {predicate_iris} via
  -- gold_triples.predicates: temporal_scope).
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
  -- Temporal hop 3: time_instant → date literal (inXSDDate / inXSDDateTime).
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
