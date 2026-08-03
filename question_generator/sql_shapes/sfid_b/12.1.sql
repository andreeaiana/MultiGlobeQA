-- TID 12.1 — Radius population buffer.
-- Question: "What is the total population of the area within {{value}} km of {{entity_A}}?"
-- Ground truth: integer count of people (raster zonal-stats SUM) inside the
-- buffered geometry around each anchor entity.
--
-- SHAPE — single pool, no spatial join.
-- Unlike 10.1 / 11.1 (the other two Wave-A radius TIDs) this is NOT a
-- two-pool cross-join. There is no pool_b; each anchor row produces a
-- buffered geometry, and population is computed by CROSS-JOIN against
-- the global `pop_tiles` raster view + RS_ZonalStats. So this file
-- carries NONE of 10.1 / 11.1's Sedona spatial-join scaffolding
-- (no pool_a_geom + pool_b_geom CTEs with REPARTITION hints, no
-- ST_DWithin / ST_DistanceSphere conjuncts, no LEFT JOIN). Those
-- primitives must stay ABSENT from this shape; copy-pasting from
-- 10.1 / 11.1 here would be a structural bug.
--
-- BUFFER GEOMETRY — planar degrees (`/111.32` divisor).
-- ST_Buffer(ST_GeomFromWKT(geometry), {radius_km} / 111.32) buffers in
-- planar degrees: 1° latitude ≈ 111.32 km, so dividing radius_km by
-- 111.32 yields the buffer distance in degrees. The buffer is therefore
-- planar (not spheroidal) and the longitudinal extent shrinks by
-- ~cos(lat) at higher latitudes; that's an intentional approximation
-- because the pop_tiles raster lives in the same lat/lon space and
-- RS_ZonalStats's `allTouched=true` cell selection (see below) absorbs
-- sub-cell boundary differences. The ground-truth VALUE is computed by
-- Sedona's RS_ZonalStats inside this SQL, over exactly the buffer
-- geometry built here; boundary cells are counted whole under
-- `allTouched=true`, so the total absorbs partially covered cells
-- rather than clipping them to the buffer outline. (See 21.1.meta.yaml
-- for spec range and per-KG radius tuning context.)
--
-- COALESCE(SUM(...), 0), not SUM(COALESCE(..., 0)) — the order matters.
-- Anchors over ocean / desert / unpopulated areas produce ZERO rows
-- after the CROSS JOIN expands and RS_ZonalStats returns no cells
-- with non-zero values; SUM over an empty input returns NULL (not 0).
-- COALESCE wraps the SUM so the empty-input NULL is coerced to numeric
-- 0. Inverting the order — SUM(COALESCE(RS_ZonalStats(...), 0)) — would
-- coerce per-row NULLs but still yield NULL when there are zero rows
-- to sum, defeating the point. Same zero-match principle 10.1 / 11.1
-- enforce via LEFT JOIN preservation.
--
-- RS_ZonalStats(t.rast, geom, 1, 'sum', true) — `allTouched=true` (5th arg)
-- counts every raster cell whose extent intersects the buffer boundary,
-- including cells only partially covered. Without it, small buffers
-- under-report by missing partially-covered boundary cells.
--
-- POP_TILES DEPENDENCY (fail-loud).
-- This SQL references `pop_tiles t` directly with no fallback; a missing
-- raster produces "Table or view not found: pop_tiles" at execute time.
-- That's intentional. See DEPLOYMENT NOTE in 21.1.meta.yaml for the
-- registration mechanism (the POPULATION_RASTER_PATH raster view)
-- and rationale for the fail-loud design.
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity (same as 10.1 / 11.1).
-- A plain `ORDER BY RAND(seed)` gives purely random anchor selection,
-- which on USA-heavy KGs (osmh3kg, kwg) draws anchors disproportionately
-- from USA cells. This pattern hash-shuffles per-country and caps
-- each country's anchor count at {per_country_cap}, then bounds the
-- total at {total_sample}. Geographically balanced, deterministic
-- across runs (hash, not RAND). See 21.1.meta.yaml for the parameter
-- defaults and per-country-cap reasoning.
WITH pool_a_capped AS (
    -- Per-country hash-diversity ranks. PARTITION BY country_iso3
    -- means each country independently produces ranks 1..N from its
    -- own entities, so {per_country_cap} caps anchors per country
    -- (not per global RAND draw). ORDER BY hash(id, {rand_seed_a})
    -- gives a deterministic per-(id, seed) shuffle within each
    -- country bucket; the seed defaults are stable so reruns produce
    -- identical anchors. No REPARTITION hint — there's no spatial
    -- join downstream, the raster CROSS JOIN's parallelism is set
    -- by the tile count.
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS bucket_rank
    FROM {pool_a}
  ),
  anchors AS (
    SELECT id, name, geometry
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  ),
  buffered AS (
    -- One buffered geometry per anchor. Buffering happens AFTER the
    -- per-country cap so we don't pay ST_Buffer cost on entities we
    -- end up dropping. ST_GeomFromWKT lives here (not in a pre-CTE
    -- like 10.1 / 11.1) because there is no spatial JOIN to optimize
    -- — the only consumer is RS_ZonalStats, which takes a Geometry
    -- argument directly without needing Sedona's RangeJoinExec.
    SELECT
      id   AS entity_a_id,
      name AS entity_a_name,
      ST_Buffer(ST_GeomFromWKT(geometry), {radius_km} / 111.32) AS buffer_geom
    FROM anchors
  )
SELECT
  -- All projections need explicit AS aliases (or bare identifiers) so the
  -- contract checker can extract result_columns from the outer SELECT.
  -- `b.entity_a_id` would fail extraction (qualified identifier, no AS);
  -- alias it back to its own name. Same for entity_a_name.
  b.entity_a_id   AS entity_a_id,
  b.entity_a_name AS entity_a_name,
  -- radius_km column drives the exporter's {{value}} substitution
  -- (it checks for a `radius_km` column before falling back to N/A).
  CAST({radius_km} AS STRING) AS radius_km,
  -- ground_truth: total population in buffer. CROSS JOIN with
  -- pop_tiles fans out per-tile; SUM aggregates the per-tile zonal
  -- stats into a single per-anchor population. See header for the
  -- COALESCE / allTouched rationale.
  CAST(
    -- Cast to BIGINT first: SUM(RS_ZonalStats(...)) returns a DOUBLE
    -- (population is fractionally split across boundary cells when
    -- allTouched=true), but ground_truth_type=count requires an integer
    -- — the validator's count rule rejects float strings like
    -- "3906.560591348246". Round-half-up via CAST(... + 0.5 AS BIGINT)
    -- would be more accurate, but plain truncation is what this shape
    -- applies, consistent with how raster zonal stats round elsewhere
    -- in the project.
    CAST(
      COALESCE(SUM(RS_ZonalStats(t.rast, b.buffer_geom, 1, 'sum', true)), 0)
      AS BIGINT
    ) AS STRING
  ) AS ground_truth
FROM buffered b
CROSS JOIN pop_tiles t
GROUP BY b.entity_a_id, b.entity_a_name
-- Outer LIMIT: the per-country cap bounds anchors-per-country, but with
-- ~150 countries × per_country_cap the total can exceed total_sample.
-- This LIMIT enforces the global cap deterministically (Spark applies
-- LIMIT on the post-GROUP-BY result; ordering is stable given identical
-- inputs but not user-meaningful — the sampling diversity comes from
-- the per-country PARTITION above).
LIMIT {total_sample};
