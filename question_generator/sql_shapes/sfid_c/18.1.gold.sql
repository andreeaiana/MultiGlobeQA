-- Gold triples for TID 18.1: per-pair (entity_b, entity_c) anchor
-- provenance PLUS each pool_a entity that satisfies the set-difference
-- predicate (intersects B AND NOT intersects C).
--
-- BACKGROUND: The COLLECT_LIST aggregate enumerates pool_a entities
-- that intersect the B anchor but not the C anchor. Without aggregate
-- gold extension, only the B and C anchor triples would be available
-- and a model could not see the entities composing the answer set.
--
-- DESIGN CHOICE — exclusion role for C (per port instructions):
-- The C anchor's OWN triples are included (a model needs to know what
-- "C" means to evaluate "but not C"), but the URI extension does NOT
-- include entities that intersect C. Including C-intersectors would be
-- semantically wrong: those are precisely the entities the question
-- EXCLUDES from the answer set. Only B-intersecting (and not
-- C-intersecting) pool_a entities are added to the URI set.
WITH picked_entities AS (
    -- Both anchor roles. Each contributes its OWN triples (label / type /
    -- geometry) so the question text references resolve.
    SELECT entity_b_id AS id FROM _gt_result
    UNION
    SELECT entity_c_id AS id FROM _gt_result
  ),
  -- Anchor B entities + geometry/country (drives the predicate that
  -- selects the pool_a URI extension below).
  b_anchors AS (
    SELECT
      r.entity_b_id AS id,
      eb.entity_uri AS entity_uri,
      eb.geometry,
      eb.country_iso3
    FROM _gt_result r
    JOIN {pool_b_all} eb ON eb.id = r.entity_b_id
  ),
  -- Anchor C entities + geometry/country (used to filter OUT pool_a
  -- entities that intersect C — same semantics as the main SQL's
  -- NOT ST_Intersects clause).
  c_anchors AS (
    SELECT
      r.entity_b_id AS b_id,  -- key by B's id so the row aligns
      r.entity_c_id AS c_id,
      ec.geometry  AS c_geometry
    FROM _gt_result r
    JOIN {pool_c_all} ec ON ec.id = r.entity_c_id
  ),
  -- Anchor URIs for both B and C: provenance for the question text.
  -- Use picked_entities (already deduped via UNION) joined against the
  -- two pool_*_all views.
  anchor_uris AS (
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_b_all} e ON e.id = p.id
    UNION
    SELECT p.id, e.entity_uri
    FROM picked_entities p
    JOIN {pool_c_all} e ON e.id = p.id
  ),
  -- Pool_a entities matching the set-difference predicate. Same
  -- spatial conditions as the main SQL (intersects B, NOT intersects
  -- C, same country, no-self). Attribute to B's id so they appear
  -- under the same question_id post-export.
  --
  -- PERFORMANCE — mirror the main SQL's restructure:
  --   1. Materialize Geometry-typed columns upstream (so Sedona's
  --      RangeJoinExec can recognize the spatial predicate)
  --   2. Country semi-join pre-prune on pool_a (95%+ row reduction
  --      before the spatial scan)
  --   3. BROADCAST hint on the small (b, c) anchor pair set
  --   4. Pure-spatial JOIN ON (country filter in WHERE) so Spark
  --      doesn't pick SortMergeJoin on country_iso3 and serialize
  --      ST_Intersects as a post-merge filter on 1-2 partitions
  --   5. Split into intersects_b CTE (spatial join) + post-filter for
  --      NOT ST_Intersects (no spatial index possible for negation)
  -- The earlier inline ST_GeomFromWKT + LEFT-style combined predicate
  -- hung Job 77 (gold-triples generation) for hours on osmh3kg
  -- because Catalyst couldn't activate the Sedona optimizer.
  bc_pairs_geom AS (
    SELECT /*+ BROADCAST */
      ba.id            AS b_id,
      ba.country_iso3  AS country_iso3,
      ST_GeomFromWKT(ba.geometry)   AS b_geom,
      ca.c_id          AS c_id,
      ST_GeomFromWKT(ca.c_geometry) AS c_geom
    FROM b_anchors ba
    JOIN c_anchors ca ON ca.b_id = ba.id
  ),
  bc_anchor_countries AS (
    SELECT DISTINCT country_iso3 FROM bc_pairs_geom
  ),
  pool_a_pruned AS (
    SELECT /*+ REPARTITION(16) */
      t.id, t.entity_uri, t.country_iso3,
      ST_GeomFromWKT(t.geometry) AS geom
    FROM {pool_a_all} t
    JOIN bc_anchor_countries ac ON t.country_iso3 = ac.country_iso3
    WHERE t.geometry IS NOT NULL
  ),
  intersects_b AS (
    SELECT /*+ BROADCAST(p) */
      p.b_id, p.c_id, p.c_geom,
      a.id AS a_id, a.entity_uri AS a_entity_uri, a.geom AS a_geom
    FROM pool_a_pruned a
    INNER JOIN bc_pairs_geom p
      ON ST_Intersects(a.geom, p.b_geom)
    WHERE a.country_iso3 = p.country_iso3
      AND a.id != p.b_id
      AND a.id != p.c_id
  ),
  set_diff_pool_a AS (
    SELECT DISTINCT b_id AS id, a_entity_uri AS entity_uri
    FROM intersects_b
    WHERE NOT ST_Intersects(a_geom, c_geom)
  ),
  uris_to_fetch AS (
    SELECT id, entity_uri FROM anchor_uris
    UNION
    SELECT id, entity_uri FROM set_diff_pool_a
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
