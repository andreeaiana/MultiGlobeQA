"""Tests for :mod:`benchmark_eval.units`.

Covers: conversion-table loading from the spec, in-group conversion
correctness (round-trip + known-value checks), and the null-return
policy for cross-group / unknown / unitless inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_eval.units import (
    _FALLBACK_GROUPS,
    SCORING_SPEC_PATH,
    convert_to_gold_unit,
    dimensional_group,
)

# ---------------------------------------------------------------------------
# Spec file ↔ fallback parity
# ---------------------------------------------------------------------------


class TestSpecFileParity:
    def test_spec_file_exists(self) -> None:
        assert Path(SCORING_SPEC_PATH).exists(), (
            f"missing spec file: {SCORING_SPEC_PATH}"
        )

    def test_spec_groups_match_fallback(self) -> None:
        """Fallback table in code must agree with the spec — single
        source of truth when both are available."""
        raw = json.loads(Path(SCORING_SPEC_PATH).read_text())
        groups = raw["unit_conversion"]["groups"]
        for group_name, fallback_factors in _FALLBACK_GROUPS.items():
            spec_factors = groups[group_name]["factors_to_canonical"]
            for unit, factor in fallback_factors.items():
                assert unit in spec_factors, (
                    f"fallback group {group_name} has {unit!r} missing "
                    f"from spec"
                )
                assert spec_factors[unit] == factor, (
                    f"{group_name}/{unit}: fallback={factor} vs "
                    f"spec={spec_factors[unit]}"
                )

    def test_canonical_unit_has_factor_one(self) -> None:
        raw = json.loads(Path(SCORING_SPEC_PATH).read_text())
        groups = raw["unit_conversion"]["groups"]
        for group_name, group in groups.items():
            canonical = group["canonical"]
            factors = group["factors_to_canonical"]
            assert canonical in factors
            assert factors[canonical] == 1.0, (
                f"{group_name}: canonical {canonical} factor must be 1.0"
            )


# ---------------------------------------------------------------------------
# convert_to_gold_unit — in-group conversions
# ---------------------------------------------------------------------------


class TestConvertInGroupDistance:
    def test_same_unit_is_identity(self) -> None:
        assert convert_to_gold_unit(10.0, "km", "km") == 10.0

    def test_m_to_km(self) -> None:
        # 10000 m → 10 km — the motivating example for the whole feature.
        assert convert_to_gold_unit(10000.0, "m", "km") == pytest.approx(10.0)

    def test_km_to_m(self) -> None:
        assert convert_to_gold_unit(10.0, "km", "m") == pytest.approx(10000.0)

    def test_mi_to_km(self) -> None:
        # 1 mile = 1.609344 km exactly (international mile).
        assert convert_to_gold_unit(1.0, "mi", "km") == pytest.approx(1.609344)

    def test_km_to_mi(self) -> None:
        assert convert_to_gold_unit(1.609344, "km", "mi") == pytest.approx(1.0)

    def test_ft_to_km(self) -> None:
        # 1000 ft = 0.3048 km.
        assert convert_to_gold_unit(1000.0, "ft", "km") == pytest.approx(0.3048)

    def test_nmi_to_km(self) -> None:
        assert convert_to_gold_unit(1.0, "nmi", "km") == pytest.approx(1.852)

    def test_round_trip_mi_km_mi(self) -> None:
        # Conversion should be round-trip stable within float precision.
        out = convert_to_gold_unit(
            convert_to_gold_unit(42.0, "mi", "km"),  # type: ignore[arg-type]
            "km",
            "mi",
        )
        assert out == pytest.approx(42.0)


class TestConvertInGroupArea:
    def test_m2_to_km2(self) -> None:
        # 1_000_000 m² = 1 km².
        assert convert_to_gold_unit(1_000_000.0, "m2", "km2") == pytest.approx(1.0)

    def test_ha_to_km2(self) -> None:
        # 100 ha = 1 km².
        assert convert_to_gold_unit(100.0, "ha", "km2") == pytest.approx(1.0)

    def test_mi2_to_km2(self) -> None:
        assert convert_to_gold_unit(1.0, "mi2", "km2") == pytest.approx(
            2.589988110336
        )


# ---------------------------------------------------------------------------
# convert_to_gold_unit — null-return cases
# ---------------------------------------------------------------------------


class TestConvertReturnsNone:
    def test_pred_unit_none(self) -> None:
        assert convert_to_gold_unit(10.0, None, "km") is None

    def test_gold_unit_none(self) -> None:
        assert convert_to_gold_unit(10.0, "km", None) is None

    def test_both_none(self) -> None:
        assert convert_to_gold_unit(10.0, None, None) is None

    def test_unknown_pred_unit(self) -> None:
        assert convert_to_gold_unit(10.0, "parsec", "km") is None

    def test_unknown_gold_unit(self) -> None:
        assert convert_to_gold_unit(10.0, "km", "parsec") is None

    def test_cross_group_distance_to_area_blocked(self) -> None:
        # Distance → area must never be converted, even though both
        # units are known.
        assert convert_to_gold_unit(10.0, "km", "km2") is None

    def test_cross_group_area_to_distance_blocked(self) -> None:
        assert convert_to_gold_unit(10.0, "ha", "m") is None


# ---------------------------------------------------------------------------
# dimensional_group helper
# ---------------------------------------------------------------------------


class TestDimensionalGroup:
    def test_distance_strings(self) -> None:
        assert dimensional_group("km") == "distance"
        assert dimensional_group("kilometres") == "distance"
        assert dimensional_group("miles") == "distance"
        assert dimensional_group("ft") == "distance"

    def test_area_strings(self) -> None:
        assert dimensional_group("km²") == "area"
        assert dimensional_group("km^2") == "area"
        assert dimensional_group("hectares") == "area"

    def test_unitless_is_none(self) -> None:
        assert dimensional_group(None) is None
        assert dimensional_group("") is None
        assert dimensional_group("count") is None

    def test_unknown_is_none(self) -> None:
        assert dimensional_group("parsec") is None
