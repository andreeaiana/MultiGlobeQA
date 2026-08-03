-- TID 26.1 — POI-near-road set enumeration ("path proximity").
-- Question: "Which {{entity_A}} does {{entity_B}} pass by?"
-- Ground truth: JSON-list string of pool_a (POI) entity NAMES whose
-- geometry lies within {proximity_buffer_meters} of each pool_b (road)
-- anchor's LINESTRING.
--
-- Same structural shape as TID 11.1 (LEFT JOIN + collect_set + to_json
-- + GROUP BY anchor + LIMIT total_sample) — what differs is the spatial
-- predicate (proximity buffer in metres, not radius in km) and the
-- anchor source (a per-session aggregated road view, not the entities
-- Delta). Pool_b is the OUTER anchor pool: total_sample bounds the
-- number of distinct road anchors returned. Anchors with ZERO matches
-- still produce a row whose ground_truth = '[]' (LEFT JOIN preserves
-- them) — see ZERO-MATCH SEMANTICS below for why this is a feature.
--
-- ZERO-MATCH SEMANTICS (verified):
--   With LEFT JOIN, an anchor road with no nearby POIs of the queried
--   category yields a single row whose `a.*` columns are NULL.
--   `collect_set(a.name)` over a single NULL row yields the empty
--   array []. `sort_array([])` is []. `to_json([])` is the literal
--   string '[]', which starts with '[' (passes validator's
--   set_enumeration / entity_list check) and is a syntactically valid
--   JSON list parseable to an empty Python list. So zero-match anchors
--   render the question "Which cafes does road X pass by?" with a valid
--   empty-set answer instead of being silently dropped.
--
-- DETERMINISTIC ORDERING:
--   sort_array() wraps collect_set() so re-runs of the same anchor
--   produce identical JSON-list strings even if Spark's collect_set
--   hash ordering shifts between runs. Reproducibility matters for
--   benchmark regeneration: a model evaluated against this benchmark
--   should see the same answer string on rerun, not a permutation.
--
-- ANCHOR SOURCE — osmh3kg_roads_aggregated.
-- pool_b reads from a per-session view registered by
-- ``road_setup.register_road_aggregation_views``. One row per
-- (country_iso3, name, node_type) named road with a merged geometry
-- (ST_LineMerge over ST_Union_Aggr of the OSM ways that share the
-- same name). The view's `id` is an MD5 of (country_iso3, name,
-- node_type) — NOT an OSM id, NOT present in the strata Delta — so
-- the executor SKIPS the strata join for this source and the
-- aggregated view's own country_iso3 column drives the per-country
-- ROW_NUMBER below. osmh3kg-only (kwg/worldkg lack an equivalent
-- view; see road_setup.py docstring).
--
-- ENTITY_URI CAVEAT: the aggregated view exposes
-- ``MIN(entity_uri)`` per merged road (the first OSM way's URI for
-- that name+country+type bucket). Gold triples for the road anchor
-- attach to that single URI rather than every constituent way's
-- URI. Acceptable for question-rendering provenance (the named road
-- as a concept has no canonical OSM URI) but worth knowing for
-- post-hoc gold inspection.
--
-- PROXIMITY PREDICATE — ST_DWithin in planar degrees.
--   ST_DWithin(b.geom, a.geom, {proximity_buffer_meters} * 0.00001)
-- The 0.00001 factor converts metres → degrees (roughly: at the
-- equator 1° ≈ 111 km, so 1m ≈ 9e-6° ≈ 0.00001°). A 100m buffer
-- becomes ~0.001°.
-- Sufficient for short proximity distances (<500m); near the poles
-- east-west distance per degree shrinks but the planar buffer over-
-- estimates so we keep all true positives. ST_DWithin is the bbox
-- prune; for short distances (<<1km) the spheroidal refine is not
-- worth the cost — the planar approximation rounds correctly.
--
-- PERFORMANCE — Sedona spatial join.
--   Same recipe as 10.1/11.1: pool_a_geom and pool_b_geom CTEs
--   pre-materialize Geometry-typed columns so Sedona's optimizer
--   recognizes ST_DWithin in JOIN ON as a spatial join and triggers
--   RangeJoinExec (R-tree). Inline ST_GeomFromWKT(...) inside JOIN
--   ON would block recognition and Spark would fall back to
--   CartesianProduct, which on a multi-million-row road × POI
--   product hangs.
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity.
-- A plain ``ORDER BY RAND()`` on USA-heavy osmh3kg would draw
-- anchors disproportionately from USA cells, masking question-
-- quality issues in long-tail countries. This hash-shuffles per-country
-- and caps each country's anchor count at {per_country_cap}, then
-- bounds the global total at {total_sample}. Geographically
-- balanced and deterministic across runs (hash, not RAND).
WITH pool_a_geom AS (
    -- POIs: id, name (collected into the enumeration set), country_iso3
    -- + node_type for the join, geom for the spatial predicate. Drop
    -- raw `geometry` WKT to keep the shuffled/joined frame narrow.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, node_type,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    -- Roads (aggregated): id (MD5 anchor key), name (rendered into
    -- {{entity_B}}), country_iso3 (PARTITION BY for diversity),
    -- node_type (carried for downstream debugging — not joined on),
    -- geom for the spatial predicate.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3, node_type,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
  pool_b_capped AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM pool_b_geom
  ),
  anchors AS (
    SELECT * FROM pool_b_capped WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  -- entity_a_name carries the PLURAL category label ('cafes') so the
  -- exporter's {{entity_A}} placeholder renders a grammatical question
  -- ("Which cafes does road X pass by?") rather than the singular
  -- ("Which cafe does road X pass by?") which uses the node_type value
  -- needed for the SQL filter.
  '{entity_a_category_label}' AS entity_a_name,
  -- entity_a_subcategory keeps the SINGULAR node_type for technical
  -- reference (matches what the SQL filter actually ran with).
  '{entity_a_category}'       AS entity_a_subcategory,
  b.id                        AS entity_b_id,
  b.name                      AS entity_b_name,
  -- ground_truth: JSON-list-of-name-strings. sort_array makes ordering
  -- deterministic across runs; collect_set drops duplicates and NULLs;
  -- to_json formats as a JSON literal so the validator's set_enumeration
  -- rule (startswith '[') accepts it. Zero-match anchors emit '[]'.
  to_json(sort_array(collect_set(a.name))) AS ground_truth
FROM anchors b
LEFT JOIN pool_a_geom a
  -- Same-country prune (the road and POI must share country_iso3 — the
  -- aggregated view's `country_iso3` originates from strata enrichment).
  -- a.id != b.id is technically unreachable (POI ids and road MD5 ids
  -- live in disjoint name-spaces) but kept for symmetry with 10.1/11.1
  -- and as a defensive guard if future merges allow id-collision.
  ON a.country_iso3 = b.country_iso3
  AND a.id != b.id
  AND a.node_type = '{entity_a_category}'
  AND ST_DWithin(b.geom, a.geom, {proximity_buffer_meters} * 0.00001)
GROUP BY b.id, b.name
-- Outer LIMIT: the per-country cap bounds anchors-per-country, but with
-- ~150 countries × per_country_cap the total can exceed total_sample.
-- This LIMIT enforces the global cap deterministically (Spark applies
-- LIMIT on the post-GROUP-BY result; ordering is stable given identical
-- inputs but not user-meaningful — sampling diversity comes from the
-- per-country PARTITION above).
LIMIT {total_sample};
