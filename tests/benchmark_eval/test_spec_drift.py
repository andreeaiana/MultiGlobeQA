"""D7: assertion tests that hardcoded fallback constants in the
``benchmark_eval`` package stay in sync with ``scoring_spec.json``.

The fallback dicts in ``scorer.py``, ``distances.py``,
``canonicalize.py``, ``units.py`` are only consulted when the JSON
spec is missing — in a healthy dev/prod environment they never
fire. But "missing spec" is the wrong moment to discover the
fallback has silently drifted from the canonical values. These
tests fail loudly the moment the two diverge.
"""

from __future__ import annotations

import json

import pytest

from benchmark_eval._paths import spec_path


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(spec_path().read_text())


def test_numeric_kind_by_unit_suffix_matches_spec(spec: dict) -> None:
    """D1: spec is the source of truth for unit→kind."""
    from benchmark_eval.scorer import (
        _FALLBACK_NUMERIC_KIND_BY_UNIT_SUFFIX,
        NUMERIC_KIND_BY_UNIT_SUFFIX,
    )

    spec_map = {
        k: v
        for k, v in spec["numeric"]["kind_by_unit_suffix"].items()
        if isinstance(v, str) and v in ("distance", "area", "count")
    }
    # The live map is loaded from spec → matches verbatim.
    assert spec_map == NUMERIC_KIND_BY_UNIT_SUFFIX
    # Hardcoded fallback is a subset of (or equal to) the spec — any
    # entry present in code must also be in the spec.
    for unit, kind in _FALLBACK_NUMERIC_KIND_BY_UNIT_SUFFIX.items():
        assert spec_map.get(unit) == kind, (
            f"Fallback says {unit}→{kind} but spec says {unit}→{spec_map.get(unit)}; "
            f"update one of them."
        )


def test_scale_buckets_match_spec(spec: dict) -> None:
    """D2: scale-from-magnitude breakpoints stay in sync."""
    from benchmark_eval.distances import (
        _FALLBACK_SCALE_BUCKETS,
        NATIONAL_DELTA_KM,
        NATIONAL_DELTA_KM2,
        REGIONAL_DELTA_KM,
        REGIONAL_DELTA_KM2,
        SCALE_BREAK_REGIONAL_NATIONAL,
        SCALE_BREAK_REGIONAL_NATIONAL_AREA,
        SCALE_BREAK_URBAN_REGIONAL,
        SCALE_BREAK_URBAN_REGIONAL_AREA,
        URBAN_DELTA_KM,
        URBAN_DELTA_KM2,
    )

    buckets = spec["numeric"]["scale_from_magnitude_buckets"]
    # Spec → live constants.
    assert buckets["urban"]["upper_exclusive"] == SCALE_BREAK_URBAN_REGIONAL
    assert buckets["regional"]["upper_exclusive"] == SCALE_BREAK_REGIONAL_NATIONAL
    # Area breakpoints are independent of the linear distance breakpoints (#13).
    assert (
        buckets["urban"]["upper_exclusive_area_km2"]
        == SCALE_BREAK_URBAN_REGIONAL_AREA
    )
    assert (
        buckets["regional"]["upper_exclusive_area_km2"]
        == SCALE_BREAK_REGIONAL_NATIONAL_AREA
    )
    assert buckets["urban"]["delta_distance_km"] == URBAN_DELTA_KM
    assert buckets["regional"]["delta_distance_km"] == REGIONAL_DELTA_KM
    assert buckets["national"]["delta_distance_km"] == NATIONAL_DELTA_KM
    assert buckets["urban"]["delta_area_km2"] == URBAN_DELTA_KM2
    assert buckets["regional"]["delta_area_km2"] == REGIONAL_DELTA_KM2
    assert buckets["national"]["delta_area_km2"] == NATIONAL_DELTA_KM2

    # Fallback dict matches spec exactly (used only when spec absent).
    for bucket_name in ("urban", "regional", "national"):
        for key in (
            "upper_exclusive",
            "upper_exclusive_area_km2",
            "delta_distance_km",
            "delta_area_km2",
        ):
            assert (
                _FALLBACK_SCALE_BUCKETS[bucket_name][key]
                == buckets[bucket_name][key]
            ), f"Fallback drift: {bucket_name}.{key}"


def test_default_delta_ratio_matches_spec(spec: dict) -> None:
    """#5: dimensionless-ratio floor stays in sync (spec ↔ fallbacks)."""
    from benchmark_eval.distances import DEFAULT_RATIO_DELTA
    from benchmark_eval.scorer import _FALLBACK_DEFAULTS, load_scoring_spec

    spec_val = spec["numeric"]["default_delta_ratio"]
    assert spec_val == DEFAULT_RATIO_DELTA
    assert _FALLBACK_DEFAULTS["default_delta_ratio"] == spec_val
    assert load_scoring_spec()["default_delta_ratio"] == spec_val


def test_polar_angle_tol_matches_spec(spec: dict) -> None:
    """Polar bearing tolerance stays in sync between spec and fallback."""
    from benchmark_eval.scorer import _FALLBACK_DEFAULTS, load_scoring_spec

    spec_val = spec["polar"]["angle_tol_degrees"]
    assert _FALLBACK_DEFAULTS["polar_angle_tol_degrees"] == spec_val
    # The loader surfaces it under the flat key the ScorerConfig reads.
    assert load_scoring_spec()["polar_angle_tol_degrees"] == spec_val


def test_date_precision_tolerances_match_spec(spec: dict) -> None:
    """D3: date-precision tolerances stay in sync."""
    from benchmark_eval.canonicalize import (
        _DATE_PRECISION_TOLERANCE,
        _FALLBACK_DATE_PRECISION_TOLERANCE,
    )

    spec_tols = spec["dates"]["precision_tolerance_days"]
    assert spec_tols == _DATE_PRECISION_TOLERANCE
    assert spec_tols == _FALLBACK_DATE_PRECISION_TOLERANCE


def test_coord_separator_chars_match_spec(spec: dict) -> None:
    """D4: coordinate-string separator regex stays in sync."""
    from benchmark_eval.scorer import (
        _FALLBACK_COORD_SEPARATOR_CHARS,
        _load_coord_separator_chars,
    )

    spec_chars = spec["coordinates"]["separator_chars"]
    assert _load_coord_separator_chars() == spec_chars
    assert spec_chars == _FALLBACK_COORD_SEPARATOR_CHARS


def test_unit_conversion_groups_match_spec(spec: dict) -> None:
    """D7: hardcoded fallback in units.py matches the spec exactly."""
    from benchmark_eval.units import _FALLBACK_GROUPS

    spec_groups = {
        group_name: dict(group["factors_to_canonical"])
        for group_name, group in spec["unit_conversion"]["groups"].items()
    }
    assert spec_groups == _FALLBACK_GROUPS


def test_spec_has_no_versioning_surface(spec: dict) -> None:
    """A1: spec is the canonical spec — no ``spec_version`` field
    should reappear (would suggest a backward-compat regression)."""
    assert "spec_version" not in spec


def test_cli_defaults_match_spec(spec: dict) -> None:
    """P1-4: ``benchmark_eval`` CLI defaults must read from the spec,
    not from hardcoded literals. Inspect the click options on the
    ``score`` command and assert each default == the spec value.

    If a future ``cli.py`` edit reverts to a hardcoded number, this
    test fires.
    """
    import click

    from benchmark_eval.cli import score

    options_by_name = {
        opt.name: opt for opt in score.params if isinstance(opt, click.Option)
    }
    assert options_by_name["alpha"].default == spec["numeric"]["alpha"]
    assert (
        options_by_name["default_delta_distance_km"].default
        == spec["numeric"]["default_delta_distance_km"]
    )
    assert (
        options_by_name["default_delta_area_km2"].default
        == spec["numeric"]["default_delta_area_km2"]
    )
    assert (
        options_by_name["polygon_iou_threshold"].default
        == spec["geometry"]["polygon_iou_threshold"]
    )
    assert (
        options_by_name["polar_angle_tol_degrees"].default
        == spec["polar"]["angle_tol_degrees"]
    )


def test_tid_override_map_single_source(spec: dict) -> None:  # noqa: ARG001
    """P0-2: the per-TID canonical-override map must be defined in
    one place. The runtime resolver in
    ``benchmark_models.schemas.type_registry`` is the source of truth;
    the build script imports it from there. This test asserts the
    build script's import binds the same dict object — drift becomes
    impossible by construction.

    If the build script ever falls back to defining its own copy
    inline, this test fails and forces a re-import.
    """
    from scripts.build_template_registry_from_benchmark import (
        _TID_TO_CANONICAL_OVERRIDE as BUILD_SCRIPT_COPY,
    )

    from benchmark_models.schemas.type_registry import (
        TID_TO_CANONICAL_OVERRIDE,
    )

    # The build script imports the runtime constant under its old
    # underscore name; we assert object identity, not value equality.
    # If someone re-defines it inline, ``is`` returns False.
    assert BUILD_SCRIPT_COPY is TID_TO_CANONICAL_OVERRIDE


def test_tid_override_map_contains_known_problems() -> None:
    """P0-2: pin the override map's contents so a future cleanup
    doesn't silently drop entries.

    All assertions use the released template ids. An override exists only
    where the released ``ground_truth_type`` tag is overloaded and cannot
    by itself select a canonical answer type; a tid whose tag already names
    its type (e.g. ``ground_truth_type=bounding_box``) needs no entry.
    """
    from benchmark_models.schemas.type_registry import (
        TID_TO_CANONICAL_OVERRIDE,
    )

    # `categorical` overloaded type — 4 distinct multiple-choice
    # families need 4 different canonicals.
    assert TID_TO_CANONICAL_OVERRIDE["50.1"] == "Predefined value (left, right)"
    assert TID_TO_CANONICAL_OVERRIDE["51.1"] == "Predefined value (upstream, downstream)"
    assert TID_TO_CANONICAL_OVERRIDE["52.1"] == "Cardinal direction (predefined)"
    assert TID_TO_CANONICAL_OVERRIDE["62.1"] == "Ternary (true/false/plausibly)"

    # `grid_cell_id` overloaded type — H3 / S2 / Geohash /
    # coordinate-string all share the same tag.
    assert TID_TO_CANONICAL_OVERRIDE["40.1"] == "H3 cell index"
    assert TID_TO_CANONICAL_OVERRIDE["41.1"] == "H3 cell index"
    assert TID_TO_CANONICAL_OVERRIDE["64.1"] == "H3 cell index"
    assert TID_TO_CANONICAL_OVERRIDE["64.2"] == "S2 cell index"
    assert TID_TO_CANONICAL_OVERRIDE["64.4"] == "Coordinates (lat/long pair)"

    # No override is needed for ``115.1``: the released
    # ``ground_truth_type`` names its type directly. Guard against
    # accidentally re-introducing an entry for it.
    assert "115.1" not in TID_TO_CANONICAL_OVERRIDE
