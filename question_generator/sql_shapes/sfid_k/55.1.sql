-- TID 55.1 — Elongation comparison boolean (cross-country, polygons only).
-- Question: "Is {{entity_A}} more elongated than {{entity_B}}?"
-- Ground truth: 'true' iff A's elongation ratio > B's.
--
-- Elongation metric: max(width_km, height_km) / min(width_km, height_km)
-- where the envelope dims come from ST_Envelope, longitude width is
-- cosine-corrected to compensate for meridian convergence away from the
-- equator (so a 1° wide box near the pole shrinks to ~0 km, not ~111 km).
--
-- Inlined into a single CASE per row (vs the registry's nested CTEs)
-- because Spark's Catalyst handles the redundant ST_Envelope calls
-- well via subexpression elimination, and a flat shape mirrors the
-- 53.1 / 54.1 pattern for guard-rail consistency.
--
-- Cross-country join (no country_iso3 / h3_res3 in join_condition) —
-- elongation is a dimensionless shape ratio so global comparison is
-- meaningful.
WITH pool_a_capped AS (
    SELECT
      a.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_a})) AS unique_id
    FROM {pool_a} a
  ),
  pool_b_capped AS (
    SELECT
      b.*,
      ROW_NUMBER() OVER (ORDER BY RAND({rand_seed_b})) AS unique_id
    FROM {pool_b} b
  )
SELECT
  a.id   AS entity_a_id,
  a.name AS entity_a_name,
  b.id   AS entity_b_id,
  b.name AS entity_b_name,
  CASE
    WHEN a.geometry IS NULL OR b.geometry IS NULL THEN NULL
    -- Degenerate envelope guard: a single-vertex degenerate geometry
    -- has 0-width and 0-height envelope, which would divide by zero
    -- in min(). Surface as NULL so the validator drops the row.
    WHEN (ST_XMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
        - ST_XMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) <= 0
      OR (ST_YMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
        - ST_YMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) <= 0
      OR (ST_XMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
        - ST_XMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) <= 0
      OR (ST_YMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
        - ST_YMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) <= 0
    THEN NULL
    WHEN GREATEST(
           (ST_XMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
          - ST_XMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) * 111.0
            * COS(RADIANS((ST_YMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
                         + ST_YMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) / 2.0)),
           (ST_YMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
          - ST_YMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) * 111.0
         )
       / LEAST(
           (ST_XMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
          - ST_XMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) * 111.0
            * COS(RADIANS((ST_YMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
                         + ST_YMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) / 2.0)),
           (ST_YMax(ST_Envelope(ST_GeomFromWKT(a.geometry)))
          - ST_YMin(ST_Envelope(ST_GeomFromWKT(a.geometry)))) * 111.0
         )
       > GREATEST(
           (ST_XMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
          - ST_XMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) * 111.0
            * COS(RADIANS((ST_YMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
                         + ST_YMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) / 2.0)),
           (ST_YMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
          - ST_YMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) * 111.0
         )
       / LEAST(
           (ST_XMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
          - ST_XMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) * 111.0
            * COS(RADIANS((ST_YMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
                         + ST_YMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) / 2.0)),
           (ST_YMax(ST_Envelope(ST_GeomFromWKT(b.geometry)))
          - ST_YMin(ST_Envelope(ST_GeomFromWKT(b.geometry)))) * 111.0
         )
    THEN 'true'
    ELSE 'false'
  END AS ground_truth
FROM pool_a_capped a
INNER JOIN pool_b_capped b
  ON {join_condition}
LIMIT {total_sample};
