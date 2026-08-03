-- TID 29.1 — Component count (single-pool unary, ST_NumGeometries).
-- Question: "How many {{entity_A_label}} make up {{entity_B}}?"
-- Ground truth: integer count of sub-geometries inside a MULTI* / collection
-- geometry (e.g. islands in a country, line segments in a river system).
--
-- SHAPE — single pool, no spatial join.
-- entity_B is the composite geometry instance (the only sampled entity).
-- entity_A is a category LABEL only ("polygons" / "segments" / etc.) —
-- it fills a column slot but is not a sampled entity. We render the
-- label via {entity_a_label} at the SELECT.
--
-- ANCHOR SAMPLING - ROW_NUMBER hash diversity over the single pool.
-- Per-country bucket ranks. Geographically balanced and deterministic
-- across runs (hash, not RAND). See 29.1.meta.yaml for parameter
-- defaults and per-country-cap reasoning.
WITH pool_b_capped AS (
    -- Per-country hash-diversity ranks. PARTITION BY country_iso3 means
    -- each country independently produces ranks 1..N from its own
    -- entities, so {per_country_cap} caps anchors per country (not per
    -- global RAND draw). ORDER BY hash(id, {rand_seed_b}) gives a
    -- deterministic per-(id, seed) shuffle within each country bucket.
    SELECT
      b.*,
      ROW_NUMBER() OVER (
        PARTITION BY country_iso3
        ORDER BY hash(id, {rand_seed_b})
      ) AS bucket_rank
    FROM {pool_b} b
  ),
  anchors AS (
    SELECT id, name, geometry, geometry_type
    FROM pool_b_capped
    WHERE bucket_rank <= {per_country_cap}
      AND geometry_type IN (
        'MULTIPOLYGON', 'MULTILINESTRING', 'MULTIPOINT', 'GEOMETRYCOLLECTION'
      )
      AND ST_NumGeometries(ST_GeomFromWKT(geometry)) > 1
  )
SELECT
  -- entity_a_name carries the category label (e.g. "polygons", "segments")
  -- so the exporter's {{entity_A}} placeholder substitutes correctly.
  '{entity_a_label}'    AS entity_a_name,
  b.id                  AS entity_b_id,
  b.name                AS entity_b_name,
  CASE
    -- NULL guard: surfaces missing geometry as NULL ground_truth so the
    -- validator drops the row, rather than emitting a zero count.
    WHEN b.geometry IS NULL THEN NULL
    ELSE CAST(ST_NumGeometries(ST_GeomFromWKT(b.geometry)) AS STRING)
  END                   AS ground_truth
FROM anchors b
LIMIT {total_sample};
