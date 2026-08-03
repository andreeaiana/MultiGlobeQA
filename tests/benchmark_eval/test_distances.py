"""Tests for :mod:`benchmark_eval.distances`.

Covers the per-type distance and tolerance functions: numeric
tolerance under default and scale-from-magnitude modes, Jaccard and
set-direction metrics, geodesic coordinate distance, polygon IoU.
"""

from __future__ import annotations

import math

import pytest

from benchmark_eval.distances import (
    NumericConfig,
    coordinate_distance_km,
    jaccard,
    line_hausdorff_km,
    numeric_distance,
    numeric_tolerance,
    polygon_iou,
    set_f1,
    set_precision,
    set_recall,
)

# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


class TestNumericTolerance:
    def test_default_distance_dominates_for_small_gold(self) -> None:
        """Small gold: δ=5 dominates α·|y*|=0.05·1=0.05."""
        cfg = NumericConfig()
        assert numeric_tolerance(1.0, kind="distance", config=cfg) == 5.0

    def test_relative_dominates_for_large_gold(self) -> None:
        """Large gold: α·|y*|=0.05·1000=50 dominates δ=5."""
        cfg = NumericConfig()
        assert numeric_tolerance(1000.0, kind="distance", config=cfg) == 50.0

    def test_area_default(self) -> None:
        cfg = NumericConfig()
        # Small area: δ=25 km² dominates.
        assert numeric_tolerance(0.5, kind="area", config=cfg) == 25.0
        # Large area: 0.05·10000 = 500 dominates δ=25.
        assert numeric_tolerance(10000.0, kind="area", config=cfg) == 500.0

    def test_count_default(self) -> None:
        cfg = NumericConfig()
        # Small count: δ=1 dominates.
        assert numeric_tolerance(2.0, kind="count", config=cfg) == 1.0
        # Large count: 0.05·100 = 5 dominates.
        assert numeric_tolerance(100.0, kind="count", config=cfg) == 5.0

    def test_negative_gold_uses_absolute(self) -> None:
        cfg = NumericConfig()
        assert numeric_tolerance(-100.0, kind="distance", config=cfg) == 5.0

    def test_zero_gold(self) -> None:
        """α·0=0, so δ dominates."""
        cfg = NumericConfig()
        assert numeric_tolerance(0.0, kind="distance", config=cfg) == 5.0

    def test_scale_from_magnitude_urban(self) -> None:
        cfg = NumericConfig(use_scale_from_magnitude=True)
        # 5 km → urban → δ=0.5.
        assert numeric_tolerance(5.0, kind="distance", config=cfg) == 0.5

    def test_scale_from_magnitude_regional(self) -> None:
        cfg = NumericConfig(use_scale_from_magnitude=True)
        # 100 km → regional → δ=5.
        assert numeric_tolerance(100.0, kind="distance", config=cfg) == 5.0

    def test_scale_from_magnitude_national(self) -> None:
        cfg = NumericConfig(use_scale_from_magnitude=True)
        # 5000 km → national → δ=10 (but α·|y*|=250 dominates).
        assert numeric_tolerance(5000.0, kind="distance", config=cfg) == 250.0

    def test_custom_alpha(self) -> None:
        cfg = NumericConfig(alpha=0.10)
        assert numeric_tolerance(1000.0, kind="distance", config=cfg) == 100.0

    def test_custom_delta(self) -> None:
        cfg = NumericConfig(default_delta_distance_km=2.0)
        assert numeric_tolerance(1.0, kind="distance", config=cfg) == 2.0

    def test_unknown_kind_raises(self) -> None:
        cfg = NumericConfig()
        with pytest.raises(ValueError):
            numeric_tolerance(1.0, kind="volume", config=cfg)


class TestNumericDistance:
    def test_basic(self) -> None:
        assert numeric_distance(3.0, 1.0) == 2.0

    def test_equal(self) -> None:
        assert numeric_distance(5.0, 5.0) == 0.0

    def test_negative(self) -> None:
        assert numeric_distance(-5.0, 5.0) == 10.0


# ---------------------------------------------------------------------------
# Set enumeration
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_equal(self) -> None:
        assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == 1.0

    def test_disjoint(self) -> None:
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial(self) -> None:
        # {a, b} vs {a, b, c} → 2/3
        assert jaccard({"a", "b"}, {"a", "b", "c"}) == pytest.approx(2 / 3)

    def test_both_empty(self) -> None:
        assert jaccard(set(), set()) == 1.0

    def test_pred_empty_gold_nonempty(self) -> None:
        assert jaccard(set(), {"a"}) == 0.0

    def test_accepts_list_input(self) -> None:
        assert jaccard(["a", "b"], ["a", "b", "c"]) == pytest.approx(2 / 3)


class TestSetPrecisionRecallF1:
    def test_precision_basic(self) -> None:
        # Predicted {a, b, extra}: 2 correct out of 3 → P=2/3
        assert set_precision({"a", "b", "extra"}, {"a", "b"}) == pytest.approx(2 / 3)

    def test_recall_basic(self) -> None:
        # Predicted {a, b}, gold {a, b, c, d}: 2 gold hit out of 4 → R=1/2
        assert set_recall({"a", "b"}, {"a", "b", "c", "d"}) == 0.5

    def test_precision_empty_pred_empty_gold(self) -> None:
        assert set_precision(set(), set()) == 1.0

    def test_precision_empty_pred_nonempty_gold(self) -> None:
        assert set_precision(set(), {"a"}) == 0.0

    def test_recall_empty_gold(self) -> None:
        assert set_recall({"a"}, set()) == 1.0

    def test_f1(self) -> None:
        # P=2/3, R=1/2 → F1 = 2·(2/3·1/2) / (2/3+1/2) = 2/3 / (7/6) = 4/7
        assert set_f1(2 / 3, 1 / 2) == pytest.approx(4 / 7)

    def test_f1_zero_zero(self) -> None:
        assert set_f1(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


class TestCoordinateDistance:
    def test_same_point(self) -> None:
        assert coordinate_distance_km((52.5, 13.4), (52.5, 13.4)) == 0.0

    def test_known_distance(self) -> None:
        # Berlin to Munich: ~504 km via geodesic.
        d = coordinate_distance_km((52.52, 13.405), (48.137, 11.576))
        assert 500 <= d <= 510


# ---------------------------------------------------------------------------
# Polygon IoU
# ---------------------------------------------------------------------------


class TestPolygonIoU:
    def test_identical_polygons(self) -> None:
        coords = [(52.0, 13.0), (52.0, 14.0), (53.0, 14.0), (53.0, 13.0)]
        iou, used_planar = polygon_iou(coords, coords)
        assert iou == pytest.approx(1.0)
        assert used_planar is False

    def test_disjoint_polygons(self) -> None:
        p1 = [(52.0, 13.0), (52.0, 14.0), (53.0, 14.0), (53.0, 13.0)]
        p2 = [(60.0, 20.0), (60.0, 21.0), (61.0, 21.0), (61.0, 20.0)]
        iou, used_planar = polygon_iou(p1, p2)
        assert iou == 0.0
        assert used_planar is False

    def test_partial_overlap(self) -> None:
        p1 = [(52.0, 13.0), (52.0, 14.0), (53.0, 14.0), (53.0, 13.0)]
        # Shifted one degree east, same size: IoU = 0/(2-0) = 0 (no overlap)
        # Shift 0.5 east: expected IoU about 1/3.
        p2 = [(52.0, 13.5), (52.0, 14.5), (53.0, 14.5), (53.0, 13.5)]
        iou, used_planar = polygon_iou(p1, p2)
        assert 0.3 <= iou <= 0.4
        assert used_planar is False

    def test_degenerate_inputs(self) -> None:
        iou, used_planar = polygon_iou([], [])
        assert iou == 1.0
        assert used_planar is False
        iou, used_planar = polygon_iou(
            [], [(52.0, 13.0), (52.0, 14.0), (53.0, 14.0)]
        )
        assert iou == 0.0
        assert used_planar is False

    def test_planar_fallback_flag_set_on_reprojection_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a reprojection failure (e.g. out-of-domain coords)
        by forcing the Transformer to raise. The IoU is still defined
        (planar fallback) but the flag must report the degradation.
        """
        import pyproj

        class _BrokenTransformer:
            @classmethod
            def from_crs(cls, *_a: object, **_kw: object) -> "_BrokenTransformer":
                raise RuntimeError("simulated reprojection failure")

        monkeypatch.setattr(pyproj, "Transformer", _BrokenTransformer)
        coords = [(52.0, 13.0), (52.0, 14.0), (53.0, 14.0), (53.0, 13.0)]
        iou, used_planar = polygon_iou(coords, coords)
        assert iou == pytest.approx(1.0)  # still defined (identical polys)
        assert used_planar is True


# ---------------------------------------------------------------------------
# Line Hausdorff
# ---------------------------------------------------------------------------


class TestLineHausdorff:
    def test_identical_lines_zero(self) -> None:
        line = [(52.0, 13.0), (53.0, 14.0), (54.0, 15.0)]
        assert line_hausdorff_km(line, line) == pytest.approx(0.0)

    def test_empty_returns_inf(self) -> None:
        assert math.isinf(line_hausdorff_km([], [(52.0, 13.0)]))

    def test_parallel_shifted_lines(self) -> None:
        line1 = [(52.0, 13.0), (52.0, 14.0)]
        line2 = [(52.1, 13.0), (52.1, 14.0)]
        d = line_hausdorff_km(line1, line2)
        # 0.1 deg lat ≈ 11 km
        assert 10 <= d <= 12


class TestCoordinateDistanceRobustness:
    """A model can emit an out-of-range coordinate (garbage bbox/point,
    or lat/lon swapped). Before the guard, geopy raised and crashed the
    ENTIRE scoring cell; now it scores a maximal miss."""

    def test_out_of_range_prediction_returns_max_not_raise(self) -> None:
        from benchmark_eval.distances import (
            _MAX_GEODESIC_KM,
            coordinate_distance_km,
        )
        # latitude > 90
        assert coordinate_distance_km((123.4, -66.0), (18.1, -66.0)) == _MAX_GEODESIC_KM
        # lat/lon swapped so "lat" is a longitude out of [-90, 90]
        assert coordinate_distance_km((18.1, -66.0), (-118.0, 34.0)) == _MAX_GEODESIC_KM
        # NaN coordinate
        assert coordinate_distance_km(
            (float("nan"), -66.0), (18.1, -66.0)
        ) == _MAX_GEODESIC_KM

    def test_valid_coordinates_still_compute_real_distance(self) -> None:
        from benchmark_eval.distances import coordinate_distance_km
        d = coordinate_distance_km((18.0, -66.0), (18.1, -66.0))
        assert 10.0 <= d <= 12.0  # ~0.1 deg lat ≈ 11 km
