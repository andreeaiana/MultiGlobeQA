-- TID 10.1 — Radius containment count.
-- Question: "How many {{entity_A_category}} are within {{value}} km of {{entity_B}}?"
-- Ground truth: integer count of pool_a entities matching the parameter
-- node_type that are within {radius_km} of each pool_b anchor.
--
-- Structurally this is an aggregation. pool_b is the outer pool (each row =
-- one anchor entity). pool_a is the inner counted pool, narrowed by the
-- join_condition's node_type filter and radius distance. The LEFT JOIN +
-- COUNT/GROUP BY produces one row per anchor entity, with the count as
-- ground_truth — anchors with ZERO matches still produce a row whose
-- ground_truth = '0'. Do NOT switch this to INNER JOIN: that would
-- silently drop zero-match anchors and bias the benchmark toward
-- entities that happen to have at least one neighbour in radius.
--
-- entity_a_category is rendered into the join_condition (NOT the SQL
-- here) via the {join_condition} substitution. {radius_km} likewise
-- lives inside join_condition. This SQL just COUNTs and projects.
--
-- PERFORMANCE — Sedona spatial join.
-- The join_condition's spatial predicates run as a Sedona spatial join
-- (RangeJoinExec, R-tree on one side + probe with the other). This needs
-- two preconditions Sedona's optimizer is fussy about:
--   1. Both sides must expose Geometry-typed columns (NOT WKT strings).
--      Inline ST_GeomFromWKT(...) inside JOIN ON often blocks pattern
--      recognition and Spark falls back to CartesianProduct, which on
--      USA-dominated data hangs (single-task O(N×M) = 25M+ pairs).
--      So pool_a_geom and pool_b_geom CTEs materialize `geom` upstream.
--   2. The bbox-prune predicate (ST_DWithin) must be a SEPARATE conjunct
--      in JOIN ON — not nested inside another expression — for Sedona
--      to spot it. The spheroidal refinement (ST_DistanceSphere) is a
--      second JOIN-ON conjunct; non-spatial conjuncts in JOIN ON are
--      fine per Sedona's docs and let the rest of join_condition (the
--      country / node_type filter) participate naturally.
--
-- SPATIAL SEMANTICS — planar prune + spheroidal refine.
-- ST_DWithin(a.geom, b.geom, {radius_km}/100.0) operates in planar
-- degrees on lat/lon (no spheroid). 0.01° ≈ 1.11 km at the equator
-- (1° ≈ 111 km N-S everywhere; cos(lat)·111 km E-W elsewhere), so the
-- /100 factor buffers ~1.11 × radius_km at the equator and shrinks
-- east-west at higher latitudes. The factor is intentionally generous
-- so high-latitude longitudinal squashing can't drop true positives in
-- the prune. Correctness comes from the spheroidal ST_DistanceSphere
-- refine that follows; ST_DWithin only narrows the candidate set.
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity.
-- Purely random anchor selection (`ORDER BY RAND(seed)`) draws anchors
-- disproportionately from USA cells on USA-heavy KGs (osmh3kg, kwg),
-- masking question-quality issues in the long tail of countries. This
-- shape instead hash-shuffles per-country and caps each country's anchor
-- count at {per_country_cap}, then bounds the total at {total_sample}.
-- Geographically balanced sampling, deterministic across runs (hash,
-- not RAND).
WITH pool_a_geom AS (
    -- Materialize geometry as a Geometry-typed column (not a WKT string)
    -- so Sedona's optimizer can recognize the spatial join in the LEFT
    -- JOIN below and trigger its RangeJoin (R-tree) strategy. Without
    -- this CTE, inline ST_GeomFromWKT(...) inside JOIN ON often prevents
    -- recognition and Spark falls back to CartesianProduct.
    -- Project only the columns the downstream join + COUNT actually
    -- read (id for COUNT, country_iso3 + node_type for join_condition's
    -- same-country and category filters, geom for the spatial predicates).
    -- Drop the raw `geometry` WKT and `name` to keep the shuffled CTE narrow.
    SELECT /*+ REPARTITION(16) */
      id, country_iso3, node_type,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_a}
  ),
  pool_b_geom AS (
    -- Same idea on the anchor side: id + name go to SELECT/GROUP BY,
    -- country_iso3 drives PARTITION BY in pool_b_capped, geom feeds the
    -- spatial JOIN. node_type and the raw geometry WKT are unused
    -- downstream.
    SELECT /*+ REPARTITION(16) */
      id, name, country_iso3,
      ST_GeomFromWKT(geometry) AS geom
    FROM {pool_b}
  ),
  -- Hash-diversity anchor sampling. PARTITION BY country_iso3 means each
  -- country independently produces ranks 1..N from its own entities, so
  -- {per_country_cap} caps anchors per country (not per global RAND draw).
  -- ORDER BY hash(id, {rand_seed_b}) gives a deterministic per-(id, seed)
  -- shuffle within each country bucket; the seed defaults are stable so
  -- reruns produce identical anchors.
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
  -- entity_a_name carries the PLURAL label ('pharmacies') so the
  -- exporter's {{entity_A}} placeholder renders a grammatical question
  -- ("How many pharmacies are within …") rather than the singular
  -- ("How many pharmacy are within …") which uses the node_type value
  -- needed for the SQL filter.
  '{entity_a_category_label}' AS entity_a_name,
  -- entity_a_subcategory keeps the SINGULAR node_type for technical
  -- reference (matches what the SQL filter actually ran with).
  '{entity_a_category}' AS entity_a_subcategory,
  -- radius_km column drives the exporter's {{value}} substitution
  -- (it checks for a `radius_km` column before falling back to N/A).
  CAST({radius_km} AS STRING) AS radius_km,
  b.id                  AS entity_b_id,
  b.name                AS entity_b_name,
  CAST(COUNT(a.id) AS STRING) AS ground_truth
FROM anchors b
LEFT JOIN pool_a_geom a
  ON {join_condition}
GROUP BY b.id, b.name
-- Outer LIMIT: the per-country cap bounds anchors-per-country, but with
-- ~150 countries × per_country_cap the total can exceed total_sample.
-- This LIMIT enforces the global cap deterministically (Spark applies
-- LIMIT on the post-GROUP-BY result; ordering is stable given identical
-- inputs but not user-meaningful — the sampling diversity comes from the
-- per-country PARTITION above).
LIMIT {total_sample};
