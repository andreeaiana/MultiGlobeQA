-- TID 39.1 — Centroid scale sensitivity (single-pool, two-resolution).
-- Question: "How does the centroid of {{entity_A}} change when computed
--            at spatial resolutions {{value_1}} km and {{value_2}} km?"
--   e.g.    "How does the centroid of France change when computed at
--            spatial resolutions 10 km and 100 km?"
-- Ground truth: numeric (float, km) — the geodesic distance between the
-- two centroids computed on the polygon simplified at the two resolutions.
--
-- SHAPE — single pool, no spatial join. Pure single-pool TID: no pool_b,
-- no join, no {join_condition}. Each anchor row from pool_a projects
-- TWO centroids (one per resolution) of its own polygon and emits the
-- inter-centroid distance as the ground_truth scalar. Structurally
-- identical to 37.1 (per-country anchor sampling, single-pool LIMIT)
-- except the SELECT computes a distance from two simplified polygons
-- instead of projecting one centroid pair.
--
-- RESOLUTION → SIMPLIFICATION TOLERANCE.
-- "Spatial resolution" maps to ST_SimplifyPreserveTopology's tolerance
-- argument. The conversion `tolerance_deg = value_km / 111.0` uses the
-- equatorial approximation 1° latitude ≈ 111 km. This is latitude-
-- dependent (1° longitude shrinks by cos(lat) at higher latitudes) but
-- standard for global-scale simplification. The /111.0 divisor is fixed
-- for this shape so the ground-truth distance stays comparable across
-- runs and KGs.
--
-- ST_SimplifyPreserveTopology (NOT ST_Simplify): preserves topology
-- so simplification can't introduce self-intersections / break the
-- polygon. ST_PointOnSurface (NOT ST_Centroid) for the centroid
-- projection — same rationale as 37.1 (must be inside the polygon for
-- concave shapes; ST_Centroid can fall outside).
--
-- ANCHOR SAMPLING — ROW_NUMBER hash diversity (mirror of 12.1 / 37.1).
-- Same per-country bucket pattern: PARTITION BY country_iso3, ORDER BY
-- hash(id, {rand_seed_a}), cap at {per_country_cap}. Region anchors are
-- USA-biased on osmh3kg / kwg — the per-country sampling is essential
-- for geographic balance (this TID's spec applies to regions only).
WITH pool_a_capped AS (
    -- Per-country hash-diversity ranks.
    SELECT
      a.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_a})
      ) AS bucket_rank
    FROM {pool_a} a
  ),
  anchors AS (
    SELECT id, name, geometry
    FROM pool_a_capped
    WHERE bucket_rank <= {per_country_cap}
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  -- {value_1} / {value_2} surfaced as columns so the exporter's
  -- {{value_1}} / {{value_2}} placeholders in the question template
  -- can substitute. Without these, the rendered question reads
  -- "… spatial resolutions N/A km and N/A km".
  CAST({value_1} AS STRING) AS value_1,
  CAST({value_2} AS STRING) AS value_2,
  -- ground_truth: geodesic distance (km) between the two simplified
  -- centroids. NULL guard: missing geometry → NULL ground_truth so the
  -- validator drops the row. ST_SimplifyPreserveTopology of NULL
  -- propagates NULL; the explicit IS NULL guard is for clarity and
  -- belt-and-suspenders against future Sedona behaviour changes.
  CAST(
    CASE
      WHEN a.geometry IS NULL THEN NULL
      ELSE ST_DistanceSphere(
        ST_PointOnSurface(ST_SimplifyPreserveTopology(
          ST_GeomFromWKT(a.geometry), {value_1} / 111.0
        )),
        ST_PointOnSurface(ST_SimplifyPreserveTopology(
          ST_GeomFromWKT(a.geometry), {value_2} / 111.0
        ))
      ) / 1000.0
    END AS STRING
  ) AS ground_truth
FROM anchors a
-- Outer LIMIT enforces the global cap deterministically; per-country
-- diversity comes from the PARTITION above.
LIMIT {total_sample};
