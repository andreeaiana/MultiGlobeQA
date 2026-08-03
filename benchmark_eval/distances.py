"""Per-answer-type ``d(â, a*)`` and ``τ(a*)`` functions.

Each answer type has two pure functions implemented here:

* ``distance(pred, gold, *, context) -> float`` — the underlying
  disagreement between prediction and gold in the natural units of
  that type. The scorer converts this to EM and NE.
* ``tolerance(gold, *, context) -> float`` — the type-appropriate
  ``τ`` that the scorer uses to compute
  ``NE = min(1, distance / tolerance)`` for most types.

This module is intentionally free of any parsing / canonicalization /
alias logic: those live in :mod:`benchmark_eval.canonicalize` and
:mod:`benchmark_eval.aliases`. The scorer orchestrates the three
layers.

Type notes:

* **Set enumeration** doesn't use ``tolerance`` — its NE is defined
  directly as ``1 − Jaccard`` (equivalently ``τ=1`` with
  ``d = 1 − Jaccard``, which gives the same NE).
* **Polygon geometry** also uses ``τ=1`` effectively; NE is
  ``1 − IoU`` with the IoU threshold only affecting EM per the plan.
* **Dates** use a separate path (components-match EM + day-distance
  NE) documented in :mod:`benchmark_eval.canonicalize`; this
  module doesn't implement a date distance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


# Default tolerance constants — can be overridden via ``ScorerConfig``.
DEFAULT_ALPHA: float = 0.05
DEFAULT_DISTANCE_DELTA_KM: float = 5.0
DEFAULT_AREA_DELTA_KM2: float = 25.0
DEFAULT_COUNT_DELTA: float = 1.0
# Absolute floor for the dimensionless ``ratio`` kind (TID 64.x shape
# indices). 0.0 → purely relative tolerance (``alpha*|y*|``); the natural
# scale-free criterion for a ratio spanning orders of magnitude.
DEFAULT_RATIO_DELTA: float = 0.0

# Hardcoded fallback for scale-from-magnitude breakpoints when the
# spec file is missing. Live values load from
# ``scoring_spec.json:numeric.scale_from_magnitude_buckets`` below;
# this dict is the fallback. Kept in sync with the spec by an
# assertion test. Area carries its own magnitude breakpoints
# (``upper_exclusive_area_km2``) so it is not bucketed on the linear
# distance breakpoints.
_FALLBACK_SCALE_BUCKETS: dict[str, dict[str, float | None]] = {
    "urban": {
        "upper_exclusive": 50.0,
        "upper_exclusive_area_km2": 2500.0,
        "delta_distance_km": 0.5,
        "delta_area_km2": 0.25,
    },
    "regional": {
        "upper_exclusive": 500.0,
        "upper_exclusive_area_km2": 250000.0,
        "delta_distance_km": 5.0,
        "delta_area_km2": 25.0,
    },
    "national": {
        "upper_exclusive": None,
        "upper_exclusive_area_km2": None,
        "delta_distance_km": 10.0,
        "delta_area_km2": 100.0,
    },
}

_BUCKET_NAMES = ("urban", "regional", "national")


def _load_scale_buckets() -> dict[str, dict[str, float | None]]:
    """Load scale-from-magnitude breakpoints from the spec; fall back
    to the hardcoded table when the spec is missing or malformed.

    Only the three named buckets are extracted, so documentation keys
    (e.g. ``breakpoints_description``) inside the spec block are ignored
    rather than treated as a malformed layout.
    """
    import json
    from pathlib import Path

    from benchmark_eval._paths import spec_path

    p = spec_path()
    if not p.exists():
        return {k: dict(v) for k, v in _FALLBACK_SCALE_BUCKETS.items()}
    raw = json.loads(Path(p).read_text())
    buckets = raw.get("numeric", {}).get("scale_from_magnitude_buckets")
    if not isinstance(buckets, dict) or not all(
        name in buckets for name in _BUCKET_NAMES
    ):
        return {k: dict(v) for k, v in _FALLBACK_SCALE_BUCKETS.items()}
    return {name: dict(buckets[name]) for name in _BUCKET_NAMES}


_SCALE_BUCKETS = _load_scale_buckets()

# Materialised constants for the hot path; loaded from the spec
# (or hardcoded fallback) once at import.
SCALE_BREAK_URBAN_REGIONAL: float = float(
    _SCALE_BUCKETS["urban"]["upper_exclusive"]  # type: ignore[arg-type]
)
SCALE_BREAK_REGIONAL_NATIONAL: float = float(
    _SCALE_BUCKETS["regional"]["upper_exclusive"]  # type: ignore[arg-type]
)
# Area magnitude breakpoints (km^2) — independent of the linear distance
# breakpoints above (#13).
SCALE_BREAK_URBAN_REGIONAL_AREA: float = float(
    _SCALE_BUCKETS["urban"]["upper_exclusive_area_km2"]  # type: ignore[arg-type]
)
SCALE_BREAK_REGIONAL_NATIONAL_AREA: float = float(
    _SCALE_BUCKETS["regional"]["upper_exclusive_area_km2"]  # type: ignore[arg-type]
)

URBAN_DELTA_KM: float = float(_SCALE_BUCKETS["urban"]["delta_distance_km"])  # type: ignore[arg-type]
REGIONAL_DELTA_KM: float = float(_SCALE_BUCKETS["regional"]["delta_distance_km"])  # type: ignore[arg-type]
NATIONAL_DELTA_KM: float = float(_SCALE_BUCKETS["national"]["delta_distance_km"])  # type: ignore[arg-type]

URBAN_DELTA_KM2: float = float(_SCALE_BUCKETS["urban"]["delta_area_km2"])  # type: ignore[arg-type]
REGIONAL_DELTA_KM2: float = float(_SCALE_BUCKETS["regional"]["delta_area_km2"])  # type: ignore[arg-type]
NATIONAL_DELTA_KM2: float = float(_SCALE_BUCKETS["national"]["delta_area_km2"])  # type: ignore[arg-type]


@dataclass
class NumericConfig:
    """Configuration for numeric tolerance computation.

    The scorer builds one of these from its :class:`ScorerConfig` and
    passes it into every numeric tolerance call. Keeping it small
    avoids threading six individual kwargs through the call chain.
    """

    alpha: float = DEFAULT_ALPHA
    default_delta_distance_km: float = DEFAULT_DISTANCE_DELTA_KM
    default_delta_area_km2: float = DEFAULT_AREA_DELTA_KM2
    default_delta_count: float = DEFAULT_COUNT_DELTA
    default_delta_ratio: float = DEFAULT_RATIO_DELTA
    use_scale_from_magnitude: bool = False


def numeric_distance(pred: float, gold: float) -> float:
    """``|ŷ − y*|``. Assumes both are already in the gold's unit."""
    return abs(float(pred) - float(gold))


def numeric_tolerance(
    gold: float, *, kind: str, config: NumericConfig
) -> float:
    """Return ``τ = max(δ, α·|y*|)`` for a numeric gold value.

    Args:
        gold: The gold value (in its canonical unit — km for distance,
            km² for area, a pure number for counts).
        kind: One of ``"distance"``, ``"area"``, ``"count"``. Selects
            the default δ when scale-from-magnitude is disabled.
        config: A :class:`NumericConfig` holding α and the defaults.

    Scale-from-magnitude (when enabled) chooses δ from the urban /
    regional / national breakpoints based on ``|gold|`` — that's the
    headline-sensitivity knob; disable it for the plan's default
    behavior.
    """
    g = abs(float(gold))
    if config.use_scale_from_magnitude and kind in ("distance", "area"):
        if kind == "distance":
            if g < SCALE_BREAK_URBAN_REGIONAL:
                delta = URBAN_DELTA_KM
            elif g < SCALE_BREAK_REGIONAL_NATIONAL:
                delta = REGIONAL_DELTA_KM
            else:
                delta = NATIONAL_DELTA_KM
        else:  # area — bucketed on the area breakpoints, not the
            # linear distance breakpoints (#13).
            if g < SCALE_BREAK_URBAN_REGIONAL_AREA:
                delta = URBAN_DELTA_KM2
            elif g < SCALE_BREAK_REGIONAL_NATIONAL_AREA:
                delta = REGIONAL_DELTA_KM2
            else:
                delta = NATIONAL_DELTA_KM2
    else:
        if kind == "distance":
            delta = config.default_delta_distance_km
        elif kind == "area":
            delta = config.default_delta_area_km2
        elif kind == "ratio":
            # Dimensionless shape indices (TID 64.x). Purely relative
            # tolerance by default (delta_ratio=0 → tau = alpha*|y*|),
            # scale-free across the value range (#5).
            delta = config.default_delta_ratio
        elif kind == "count":
            delta = config.default_delta_count
        else:
            raise ValueError(f"Unknown numeric kind: {kind!r}")
    return max(delta, config.alpha * g)


# ---------------------------------------------------------------------------
# Set enumeration — Jaccard
# ---------------------------------------------------------------------------


def jaccard(pred: Iterable[Any], gold: Iterable[Any]) -> float:
    """Return ``|Â ∩ A*| / |Â ∪ A*|``.

    Empty-both convention: ``Jaccard(∅, ∅) = 1`` (perfect match of
    empty sets). Anything else non-empty vs empty → 0.
    """
    pset = set(pred)
    gset = set(gold)
    if not pset and not gset:
        return 1.0
    union = pset | gset
    if not union:
        return 1.0  # defensive, unreachable given the above
    return len(pset & gset) / len(union)


def set_precision(pred: Iterable[Any], gold: Iterable[Any]) -> float:
    """``|Â ∩ A*| / |Â|`` — penalizes over-prediction.

    Convention: empty prediction → precision = 1 if gold is also
    empty, 0 otherwise. Mirrors the standard IR convention.
    """
    pset = set(pred)
    gset = set(gold)
    if not pset:
        return 1.0 if not gset else 0.0
    return len(pset & gset) / len(pset)


def set_recall(pred: Iterable[Any], gold: Iterable[Any]) -> float:
    """``|Â ∩ A*| / |A*|`` — penalizes missing gold items.

    Convention: empty gold → recall = 1 (no items to miss).
    """
    pset = set(pred)
    gset = set(gold)
    if not gset:
        return 1.0
    return len(pset & gset) / len(gset)


def set_f1(precision: float, recall: float) -> float:
    """F1 from precision and recall. ``NaN``-safe: returns 0 when
    precision + recall == 0."""
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


# ---------------------------------------------------------------------------
# Coordinates — geodesic
# ---------------------------------------------------------------------------


# Max possible geodesic distance on WGS84 is ~20,004 km (antipodal). Any
# real tolerance is in km, so this sentinel guarantees EM=0 / NE=1 for an
# unscoreable coordinate without putting an infinity into the metrics.
_MAX_GEODESIC_KM: float = 20100.0


def coordinate_distance_km(pred: tuple[float, float], gold: tuple[float, float]) -> float:
    """Geodesic distance (in km) between two ``(lat, lon)`` pairs.

    Uses :func:`geopy.distance.geodesic` which computes Vincenty on
    WGS84. Matches the model-section prompt's "geodesic along Earth's
    surface" spec. For same-point inputs, returns 0.

    Robustness: a *prediction* can carry an out-of-range coordinate
    (latitude beyond ±90, or lat/lon swapped) — geopy raises on those,
    which would otherwise crash the whole scoring cell. Gold is validated
    upstream, so a range violation is a prediction defect = wrong answer:
    return a maximal geodesic distance (> any tolerance ⇒ EM=0) instead of
    raising. NaN / non-numeric inputs are handled the same way.
    """
    from geopy.distance import geodesic

    def _in_range(pt: tuple[float, float]) -> bool:
        lat, lon = pt
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

    try:
        if not _in_range(pred) or not _in_range(gold):
            return _MAX_GEODESIC_KM
        return float(geodesic(pred, gold).km)
    except (ValueError, TypeError):
        return _MAX_GEODESIC_KM


# ---------------------------------------------------------------------------
# Geometry — polygons (IoU)
# ---------------------------------------------------------------------------


def polygon_iou(
    pred_coords: list[tuple[float, float]],
    gold_coords: list[tuple[float, float]],
) -> tuple[float, bool]:
    """Intersection-over-union of two polygons.

    Coordinates are ``(lat, lon)`` pairs forming an outer ring. The
    IoU is computed in EPSG:6933 (EASE-Grid 2.0 Cylindrical, equal-
    area) so high-latitude polygons are not artificially compressed
    along the longitude axis.

    Returns ``(iou, used_planar_fallback)``:

    * ``iou`` — the intersection-over-union ratio in ``[0, 1]``.
    * ``used_planar_fallback`` — ``True`` when EPSG:6933 reprojection
      failed and the IoU was computed on raw lon/lat coordinates as
      Cartesian X/Y (biased at high latitudes; only-defined-not-
      correct). Surfaces in ScoreRecord so downstream analysis can
      report what fraction of polygon rows used the degraded metric.

    Empty-both convention: both polygons degenerate (empty or <3
    points) → ``(1.0, False)``. One degenerate vs one real →
    ``(0.0, False)``.
    """
    from shapely.geometry import Polygon

    def _poly(coords: list[tuple[float, float]]) -> Polygon | None:
        if not coords or len(coords) < 3:
            return None
        # shapely expects (x, y) = (lon, lat)
        return Polygon([(lon, lat) for (lat, lon) in coords])

    p = _poly(pred_coords)
    g = _poly(gold_coords)
    if p is None and g is None:
        return 1.0, False
    if p is None or g is None:
        return 0.0, False

    used_planar_fallback = False
    try:
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform

        transformer = Transformer.from_crs(4326, 6933, always_xy=True)
        p = shp_transform(transformer.transform, p)
        g = shp_transform(transformer.transform, g)
    except Exception:
        # Reprojection failure (e.g. coords outside the projection
        # domain, or polygon self-intersects after transform). Fall
        # through to planar so the metric is still defined; surface
        # the degradation via the flag below.
        used_planar_fallback = True

    if not p.is_valid:
        p = p.buffer(0)
    if not g.is_valid:
        g = g.buffer(0)
    try:
        intersection = p.intersection(g).area
        union = p.union(g).area
    except Exception:
        return 0.0, used_planar_fallback
    if union <= 0:
        return (1.0 if intersection <= 0 else 0.0), used_planar_fallback
    return float(intersection / union), used_planar_fallback


# ---------------------------------------------------------------------------
# Geometry — lines (Hausdorff)
# ---------------------------------------------------------------------------


def line_hausdorff_km(
    pred_coords: list[tuple[float, float]],
    gold_coords: list[tuple[float, float]],
) -> float:
    """Approximate Hausdorff distance between two polylines in km.

    Uses shapely's Euclidean Hausdorff in lon/lat space, then
    multiplies by a mid-latitude-aware per-degree kilometer factor.
    The approximation is fine for sub-continental lines; for very
    long lines spanning multiple latitude bands the error grows.
    Consider this a reasonable default; a geodesic Hausdorff can
    replace this function without other changes.
    """
    from shapely.geometry import LineString

    # Shapely's LineString constructor requires at least two points;
    # callers that emit a single coordinate would otherwise crash here.
    # Treat both empty and single-coord sides as max-distance.
    if len(pred_coords) < 2 or len(gold_coords) < 2:
        return math.inf
    pline = LineString([(lon, lat) for (lat, lon) in pred_coords])
    gline = LineString([(lon, lat) for (lat, lon) in gold_coords])
    if pline.is_empty or gline.is_empty:
        return math.inf
    degrees = float(pline.hausdorff_distance(gline))
    mid_lat = (pred_coords[0][0] + gold_coords[0][0]) / 2.0
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians(mid_lat))
    # Conservative: use the smaller-per-degree factor as an upper bound.
    return degrees * max(km_per_deg_lat, km_per_deg_lon)


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_AREA_DELTA_KM2",
    "DEFAULT_COUNT_DELTA",
    "DEFAULT_DISTANCE_DELTA_KM",
    "NumericConfig",
    "coordinate_distance_km",
    "jaccard",
    "line_hausdorff_km",
    "numeric_distance",
    "numeric_tolerance",
    "polygon_iou",
    "set_f1",
    "set_precision",
    "set_recall",
]
