-- Gold triples for TID 36.6: per-pair (entity_b, entity_c) anchor POIs
-- PLUS every pool_a POI in the SORT_ARRAY(COLLECT_LIST(a.name)) ground
-- truth (i.e. matching the radius+directional predicates). 10.1-style
-- with two anchors — without these, the model retrieving the gold
-- cannot verify the COLLECTED list.
--
-- Each collected pool_a triple is attributed to BOTH anchor ids
-- (entity_b_id and entity_c_id) so the exporter joins them under the
-- same question_id post-export.
--
-- The radius+directional predicates here mirror the main SQL exactly,
-- so the gold lists EXACTLY the entities the COLLECT_LIST aggregated.
WITH picked_pairs AS (
    SELECT entity_b_id, entity_c_id FROM _gt_result
  ),
  -- Anchor B + geometry (geometry needed for the radius/dir join below).
  anchor_b AS (
    SELECT DISTINCT p.entity_b_id AS id, e.entity_uri, e.geometry
    FROM picked_pairs p
    JOIN {pool_b_all} e ON e.id = p.entity_b_id
  ),
  -- Anchor C + geometry.
  anchor_c AS (
    SELECT DISTINCT p.entity_c_id AS id, e.entity_uri, e.geometry
    FROM picked_pairs p
    JOIN {pool_c_all} e ON e.id = p.entity_c_id
  ),
  -- Pair-level (b, c) join — needed because the directional predicate
  -- compares against c's centroid while the radius compares against b.
  pair_geoms AS (
    SELECT
      p.entity_b_id, p.entity_c_id,
      b.geometry AS b_geom,
      c.geometry AS c_geom
    FROM picked_pairs p
    JOIN anchor_b b ON b.id = p.entity_b_id
    JOIN anchor_c c ON c.id = p.entity_c_id
  ),
  -- pool_a entities matching the main SQL's combined predicate, listed
  -- twice (once per anchor) so triples surface under both question
  -- attribution paths.
  pois_under_b AS (
    SELECT pg.entity_b_id AS id, t.entity_uri
    FROM pair_geoms pg
    JOIN {pool_a_all} t
      ON ST_DistanceSphere(
           ST_Centroid(ST_GeomFromWKT(t.geometry)),
           ST_Centroid(ST_GeomFromWKT(pg.b_geom))
         ) / 1000.0 <= {radius_km}
     AND ST_Y(ST_Centroid(ST_GeomFromWKT(t.geometry))) > ST_Y(ST_Centroid(ST_GeomFromWKT(pg.c_geom)))
         AND ST_X(ST_Centroid(ST_GeomFromWKT(t.geometry))) < ST_X(ST_Centroid(ST_GeomFromWKT(pg.c_geom)))
     AND t.id != pg.entity_b_id
     AND t.id != pg.entity_c_id
  ),
  pois_under_c AS (
    SELECT pg.entity_c_id AS id, t.entity_uri
    FROM pair_geoms pg
    JOIN {pool_a_all} t
      ON ST_DistanceSphere(
           ST_Centroid(ST_GeomFromWKT(t.geometry)),
           ST_Centroid(ST_GeomFromWKT(pg.b_geom))
         ) / 1000.0 <= {radius_km}
     AND ST_Y(ST_Centroid(ST_GeomFromWKT(t.geometry))) > ST_Y(ST_Centroid(ST_GeomFromWKT(pg.c_geom)))
         AND ST_X(ST_Centroid(ST_GeomFromWKT(t.geometry))) < ST_X(ST_Centroid(ST_GeomFromWKT(pg.c_geom)))
     AND t.id != pg.entity_b_id
     AND t.id != pg.entity_c_id
  ),
  uris_to_fetch AS (
    SELECT id, entity_uri FROM anchor_b
    UNION
    SELECT id, entity_uri FROM anchor_c
    UNION
    SELECT id, entity_uri FROM pois_under_b
    UNION
    SELECT id, entity_uri FROM pois_under_c
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
