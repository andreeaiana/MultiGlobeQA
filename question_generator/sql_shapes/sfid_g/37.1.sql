-- TID 37.1 — Centroid extraction (single-pool unary).
-- Question: "What is the most central point of {{entity_A}}?"
-- Ground truth: "lat,lon" string from ST_PointOnSurface (guaranteed
-- inside the polygon, vs ST_Centroid which can fall outside concave
-- shapes).
--
-- SHAPE — single pool, no spatial join.
-- Pure single-pool TID: no pool_b, no join, no {join_condition}. Each
-- anchor row from pool_a projects its own centroid coords as a
-- comma-joined string in the ground_truth slot. (Multi-column GT support
-- is on the backlog; until then, downstream consumers can split on ',' .)
-- Structurally this matches 12.1's single-pool shape (vs the two-pool
-- cross-join shapes 1.1 / 10.1 / 11.1 use), so the sampling pattern is
-- the per-country bucket form (see ANCHOR SAMPLING below) — NOT the
-- rank-equality / global-RAND form 1.1 uses for symmetric pair joins.
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity (mirror of 12.1).
-- A plain `ORDER BY RAND(seed)` gives purely random anchor selection,
-- which on USA-heavy KGs (osmh3kg, kwg) draws anchors disproportionately
-- from USA cells — the same failure mode the radius family (10.1 / 11.1
-- / 12.1) avoids. This pattern instead hash-shuffles per-country
-- and caps each country's anchor count at {per_country_cap}, then
-- bounds the total at {total_sample}. Geographically balanced,
-- deterministic across runs (hash, not RAND). See 37.1.meta.yaml for
-- the parameter defaults and per-country-cap reasoning.
WITH pool_a_capped AS (
    -- Per-country hash-diversity ranks. PARTITION BY country_iso3
    -- means each country independently produces ranks 1..N from its
    -- own entities, so {per_country_cap} caps anchors per country
    -- (not per global RAND draw). ORDER BY hash(id, {rand_seed_a})
    -- gives a deterministic per-(id, seed) shuffle within each
    -- country bucket; the seed defaults are stable so reruns produce
    -- identical anchors.
    SELECT
      a.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS bucket_rank
    FROM {pool_a} a
    -- Drop NULL + NaN-coord WKTs upstream. osmh3kg has occasional
    -- POINT Z (lat lon NaN) entries (e.g. lat=41.72, lon=23.44, Z=NaN)
    -- that crash JTS's ST_PointOnSurface with TopologyException.
    -- Regex on raw WKT survives Spark Catalyst pipelining (a Sedona-side
    -- filter would get fused into the same task as the failing
    -- ST_GeomFromWKT call).
    WHERE a.geometry IS NOT NULL
      AND NOT a.geometry RLIKE 'NaN'
  ),
  anchors AS (
    SELECT id, name, geometry
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  CASE
    -- NULL guard: surfaces missing geometry as NULL ground_truth so the
    -- validator drops the row, rather than emitting a broken coord string.
    WHEN a.geometry IS NULL THEN NULL
    ELSE CONCAT(
      CAST(ST_Y(ST_PointOnSurface(ST_GeomFromWKT(a.geometry))) AS STRING),
      ',',
      CAST(ST_X(ST_PointOnSurface(ST_GeomFromWKT(a.geometry))) AS STRING)
    )
  END    AS ground_truth
FROM anchors a
-- Outer LIMIT: the per-country cap bounds anchors-per-country, but with
-- ~150 countries × per_country_cap the total can exceed total_sample.
-- This LIMIT enforces the global cap deterministically (Spark applies
-- LIMIT on the post-CTE result; ordering is stable given identical
-- inputs but not user-meaningful — the sampling diversity comes from
-- the per-country PARTITION above, mirroring 12.1's pattern).
LIMIT {total_sample};
