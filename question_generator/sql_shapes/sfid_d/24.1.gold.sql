-- 24.1 gold triples — 10.1-style aggregate gold extension.
--
-- The COLLECT_LIST in 39.1.sql ranges over EVERY region of the variant's
-- category that ST_Intersects each anchor road. A model retrieving the
-- gold therefore needs to see triples for both:
--
--   1. the anchor road itself (its name + geometry chain), AND
--   2. every region the SORT_ARRAY actually contains (so the answer set
--      is reconstructible from triples alone).
--
-- Mirrors sfid_b/19.1.gold.sql's two-pool URI-extension shape:
-- uris_to_fetch UNIONs anchor pool_a URIs with pool_b URIs that match the
-- same ST_Intersects + same-country predicate the main SQL's join uses.
-- All triples are attributed to the anchor's local id so they appear
-- under the same question_id post-export.
--
-- CAVEAT (anchor URI representativeness): the aggregated road view's
-- entity_uri column carries the FIRST OSM way's URI per aggregated road
-- (MIN(n.entity_uri) inside road_setup.register_road_aggregation_views
-- GROUP BY). The aggregated road itself (one row per
-- (country_iso3, name, node_type)) is a synthesized aggregation with
-- NO triples in the KG — only its constituent OSM ways do. We use the
-- first-way URI as the anchor's representative for triple lookup; triples
-- for the OTHER OSM ways composing the same named road are not fetched.
-- This is intentional and matches the synthesized-anchor compromise
-- already encoded in the view: question text uses the road's `name`
-- (which is shared across all constituent ways), so a single
-- representative URI is sufficient for label / geometry provenance.
--
-- For the region URIs no caveat applies — region entities are real KG
-- entities with their own entity_uri values pulled directly from the
-- region pool.
WITH picked_entities AS (
    SELECT entity_b_id AS id FROM _gt_result
  ),
  -- Anchor entities + their geometry/country (needed for the
  -- ST_Intersects+country join below that fans out to the regions).
  entity_uris AS (
    SELECT p.id, e.entity_uri, e.geometry, e.country_iso3
    FROM picked_entities p
    JOIN {pool_a_all} e ON e.id = p.id
  ),
  -- URIs to fetch triples for: anchor road URIs + all regions in pool_b
  -- that (a) sit in the same country and (b) ST_Intersect the anchor
  -- road's geometry. Same predicate the main SQL uses to gate the
  -- COLLECT_LIST aggregate — gold lists exactly the region set whose
  -- names appear in the SORT_ARRAY ground_truth.
  uris_to_fetch AS (
    SELECT id, entity_uri FROM entity_uris
    UNION
    SELECT eu.id, reg.entity_uri
    FROM entity_uris eu
    JOIN {pool_b_all} reg
      ON eu.country_iso3 = reg.country_iso3
     AND reg.geometry_type IN ('POLYGON', 'MULTIPOLYGON')
     AND ST_Intersects(
           ST_GeomFromWKT(eu.geometry),
           ST_GeomFromWKT(reg.geometry)
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
