-- 44.1 — aggregate gold for "Which regions were affected by {{event_A}}?".
--
-- 10.1-style extension: gold includes both the event anchor's triples AND
-- every region whose geometry intersects the event geometry (the regions
-- that show up in the main SQL's COLLECT_LIST). Without the region
-- extension, a model retrieving gold would see the event's triples but
-- no way to enumerate the affected regions in its answer.
--
-- Plus the kwg temporal 3-hop chain on the event anchor so start/end
-- dates of the event are retrievable in gold (mirrors 4.1.gold.sql):
--   event → hasTemporalScope → interval
--   interval → fireDiscoveryTime / fireOutTime → time_instant
--   time_instant → inXSDDate / inXSDDateTime → date_literal
--
-- The first hop (hasTemporalScope) flows naturally through direct_triples
-- via the {predicate_iris} expansion of gold_triples.predicates. The two
-- downstream hops require dedicated CTEs (analogous to geometry_nodes →
-- geometry_triples). Hardcoded IRIs match 4.1.gold.sql since 44.1 is
-- kwg-only.
WITH picked_entities AS (
    SELECT event_a_id AS id FROM _gt_result
  ),
  -- Event anchors with geometry + country (needed to expand to nearby regions).
  event_uris AS (
    SELECT p.id, e.entity_uri, e.geometry, e.country_iso3
    FROM picked_entities p
    JOIN {pool_a_all} e ON e.id = p.id
  ),
  -- URIs to fetch triples for: event URIs + every intersecting region URI.
  -- All region triples are attributed to the EVENT's id so gold rows show
  -- up under the same question_id post-export.
  uris_to_fetch AS (
    SELECT id, entity_uri FROM event_uris
    UNION
    SELECT eu.id, r.entity_uri
    FROM event_uris eu
    JOIN {pool_b_all} r
      ON eu.country_iso3 = r.country_iso3
     AND eu.geometry IS NOT NULL
     AND r.geometry IS NOT NULL
     AND ST_Intersects(ST_GeomFromWKT(eu.geometry),
                       ST_GeomFromWKT(r.geometry))
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
  ),
  -- Temporal hop 1: pull the interval node URIs from direct_triples
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
