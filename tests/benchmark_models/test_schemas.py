"""Tests for ``benchmark_models.schemas``.

Two layers:

* Type-registry tests: every canonical Answer Type is registered, every
  ``AnswerTypeSpec`` has a sane shape, lookup helpers behave correctly,
  the dynamic-type allowlist is consistent with the possible-types map.

* Template-registry loader tests: synthetic JSON dicts exercise every
  validation path (well-formed entry, dynamic-type entry, unknown
  answer type, missing answer type, dynamic-type with answer_type set,
  non-dynamic with possible_answer_types set, dynamic-type allowlist
  drift, end-to-end load from a temp file).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark_models.schemas import (
    SPATIAL_FUNCTIONS,
    SPATIAL_SUBFUNCTIONS,
    TYPE_REGISTRY,
    AnswerTypeSpec,
    SpatialFunctionInfo,
    SpatialSubFunctionInfo,
    TemplateInfo,
    TemplateRegistry,
    get_spatial_function,
    get_spatial_subfunction,
    get_type_spec,
    is_known_sfid,
    is_known_subfunction,
    is_known_type,
    list_canonical_types,
    list_spatial_functions,
    list_spatial_subfunctions,
    load_template_registry,
)

# ============================================================================
# Type registry
# ============================================================================


# The full canonical vocabulary as observed in the Templates sheet on
# 2026-04-09. If a future template introduces a new answer type, this
# list should be updated alongside the registry entry. The test fails
# both ways: missing-from-registry and missing-from-this-list.
EXPECTED_CANONICAL_TYPES = frozenset(
    {
        # Pre-2026-04-23 registry.
        "Boolean (yes/no, true/false)",
        "Entity name (single)",
        "Set enumeration (list of entity names)",
        "Numeric (float)",
        "Numeric (integer)",
        "Numeric",
        "Coordinates (lat/long pair)",
        "Geometry (list of lat/long pairs)",
        "Entity hex string (single)",
        "Date",
        "Set enumeration (list of dates)",
        "Predefined value (left, right)",
        "Predefined value (upstream, downstream)",
        "Cardinal direction (predefined)",
        # Added for the 2026-04-23 benchmark (Wave 3a).
        "Ternary (true/false/plausibly)",
        "Geohash string",
        "Composite string (H3+S2)",
        "Bounding box (4-float)",
        "Polar coordinates (angle, distance)",
        # Added for the 2026-05-05 benchmark share (overloaded
        # ``grid_cell_id`` resolves to one of these per-TID).
        "H3 cell index",
        "S2 cell index",
    }
)


class TestTypeRegistry:
    def test_all_expected_types_are_registered(self) -> None:
        registered = set(TYPE_REGISTRY.keys())
        missing_from_registry = EXPECTED_CANONICAL_TYPES - registered
        extra_in_registry = registered - EXPECTED_CANONICAL_TYPES
        assert not missing_from_registry, (
            f"Type registry is missing canonical types: {missing_from_registry}"
        )
        assert not extra_in_registry, (
            f"Type registry has unexpected entries (update EXPECTED_CANONICAL_TYPES "
            f"if these are new canonical types): {extra_in_registry}"
        )

    def test_every_spec_has_canonical_name_matching_key(self) -> None:
        for key, spec in TYPE_REGISTRY.items():
            assert spec.canonical_name == key

    def test_every_spec_has_nl_description(self) -> None:
        for spec in TYPE_REGISTRY.values():
            assert spec.nl_description, f"empty nl_description for {spec.canonical_name}"

    def test_every_spec_json_schema_allows_null(self) -> None:
        # Either the type union includes "null", or the schema is a
        # oneOf that contains a null branch (used for Coordinates,
        # Geometry).
        for spec in TYPE_REGISTRY.values():
            schema = spec.json_schema
            if "type" in schema:
                type_field = schema["type"]
                if isinstance(type_field, list):
                    assert "null" in type_field, (
                        f"{spec.canonical_name}: type union does not include null"
                    )
                else:
                    pytest.fail(
                        f"{spec.canonical_name}: 'type' is a single value "
                        f"({type_field!r}), expected a list including 'null'"
                    )
            elif "oneOf" in schema:
                branches = schema["oneOf"]
                has_null = any(b.get("type") == "null" for b in branches)
                assert has_null, (
                    f"{spec.canonical_name}: oneOf does not include a null branch"
                )
            else:
                pytest.fail(
                    f"{spec.canonical_name}: schema has neither 'type' nor 'oneOf'"
                )

    def test_unit_field_only_on_appropriate_types(self) -> None:
        # wants_unit_field=True only for numeric types and the ambiguous
        # Numeric. Coordinates / geometry use degrees implicitly, no
        # response unit field.
        for spec in TYPE_REGISTRY.values():
            if spec.canonical_name in {"Numeric (float)", "Numeric"}:
                assert spec.wants_unit_field is True, (
                    f"{spec.canonical_name}: should have wants_unit_field=True"
                )
            elif "Numeric" not in spec.canonical_name and spec.canonical_name not in {
                "Coordinates (lat/long pair)",
                "Geometry (list of lat/long pairs)",
            }:
                # Non-numeric types should not have a unit field.
                assert spec.wants_unit_field is False, (
                    f"{spec.canonical_name}: unexpected wants_unit_field=True"
                )

    def test_get_type_spec_known(self) -> None:
        spec = get_type_spec("Boolean (yes/no, true/false)")
        assert isinstance(spec, AnswerTypeSpec)
        assert spec.nl_description == "yes or no"

    def test_get_type_spec_unknown_raises_with_helpful_message(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_type_spec("Definitely Not A Type")
        msg = str(excinfo.value)
        # Helpful error lists known types so the caller can suggest a fix.
        assert "Definitely Not A Type" in msg
        assert "Boolean (yes/no, true/false)" in msg

    def test_is_known_type(self) -> None:
        assert is_known_type("Boolean (yes/no, true/false)") is True
        assert is_known_type("nonexistent") is False

    def test_list_canonical_types_returns_all_entries(self) -> None:
        listed = list_canonical_types()
        assert set(listed) == set(TYPE_REGISTRY.keys())
        # No duplicates.
        assert len(listed) == len(set(listed))


# ============================================================================
# Spatial-function registry
# ============================================================================


# The spatial-function families the released benchmark uses. If a family is
# added, update both this set and the SPATIAL_FUNCTIONS registry; the test
# fails either way.
EXPECTED_SFIDS = frozenset(
    {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"}
)

# Total distinct (sfid, ssfid) pairs (with the two known gaps at
# A.VII and I.VI).
EXPECTED_SUBFUNCTION_COUNT = 55


class TestSpatialFunctions:
    def test_all_expected_sfids_are_registered(self) -> None:
        registered = set(SPATIAL_FUNCTIONS.keys())
        assert registered == EXPECTED_SFIDS

    def test_every_entry_has_sfid_matching_key(self) -> None:
        for key, info in SPATIAL_FUNCTIONS.items():
            assert info.sfid == key

    def test_every_entry_has_non_empty_name(self) -> None:
        for info in SPATIAL_FUNCTIONS.values():
            assert info.name

    def test_get_spatial_function_known(self) -> None:
        info = get_spatial_function("A")
        assert isinstance(info, SpatialFunctionInfo)
        # A's name is preserved verbatim from the Excel per design review.
        assert info.name == "Metric Distance & Proximity Computation / Reasoning"

    def test_get_spatial_function_unknown_raises_with_helpful_message(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_spatial_function("Z")
        msg = str(excinfo.value)
        assert "Z" in msg
        assert "Known SFIDs" in msg
        assert "spatial_functions.py" in msg

    def test_is_known_sfid(self) -> None:
        assert is_known_sfid("A") is True
        assert is_known_sfid("Z") is False

    def test_list_spatial_functions_in_registration_order(self) -> None:
        listed = list_spatial_functions()
        assert [s.sfid for s in listed] == list("ABCDEFGHIJKLMN")


class TestSpatialSubFunctions:
    def test_total_count_matches_excel_extraction(self) -> None:
        assert len(SPATIAL_SUBFUNCTIONS) == EXPECTED_SUBFUNCTION_COUNT

    def test_every_entry_keys_match_dataclass(self) -> None:
        for (sfid, ssfid), info in SPATIAL_SUBFUNCTIONS.items():
            assert info.sfid == sfid
            assert info.ssfid == ssfid

    def test_every_entry_has_non_empty_name(self) -> None:
        for info in SPATIAL_SUBFUNCTIONS.values():
            assert info.name

    def test_every_subfunction_sfid_is_in_function_registry(self) -> None:
        # No orphan sub-functions whose SFID isn't in SPATIAL_FUNCTIONS.
        for (sfid, _ssfid) in SPATIAL_SUBFUNCTIONS:
            assert sfid in SPATIAL_FUNCTIONS, (
                f"orphan sub-function under unknown SFID {sfid!r}"
            )

    def test_excel_gaps_are_absent(self) -> None:
        # A.VII and I.VI are the two gaps in the Excel; the registry
        # mirrors what's in the Excel and should not have these entries.
        assert ("A", "VII") not in SPATIAL_SUBFUNCTIONS
        assert ("I", "VI") not in SPATIAL_SUBFUNCTIONS

    def test_ssfids_are_roman_numerals(self) -> None:
        roman_chars = set("IVX")
        for (_sfid, ssfid) in SPATIAL_SUBFUNCTIONS:
            assert all(c in roman_chars for c in ssfid), (
                f"non-Roman ssfid: {ssfid!r}"
            )

    def test_get_spatial_subfunction_known(self) -> None:
        info = get_spatial_subfunction("A", "I")
        assert isinstance(info, SpatialSubFunctionInfo)
        # A.I had a double-space typo in Excel; the streamlined entry
        # collapses it.
        assert info.name == "Direct / constrained distance"
        assert "  " not in info.name

    def test_get_spatial_subfunction_unknown_raises(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_spatial_subfunction("A", "VII")
        msg = str(excinfo.value)
        assert "A.'VII'" in msg or "A.VII" in msg
        assert "spatial_functions.py" in msg

    def test_is_known_subfunction(self) -> None:
        assert is_known_subfunction("A", "I") is True
        assert is_known_subfunction("A", "VII") is False

    def test_list_spatial_subfunctions_filtered_by_sfid(self) -> None:
        a_subs = list_spatial_subfunctions(sfid="A")
        assert all(s.sfid == "A" for s in a_subs)
        # A has 7 sub-functions per the Excel (gap at VII).
        assert len(a_subs) == 7

    def test_list_spatial_subfunctions_unfiltered(self) -> None:
        all_subs = list_spatial_subfunctions()
        assert len(all_subs) == EXPECTED_SUBFUNCTION_COUNT

    def test_typo_fixes_applied(self) -> None:
        # I.II had "Occurence count" → fixed to "Occurrence count"
        assert get_spatial_subfunction("I", "II").name == "Occurrence count"
        # H.III had "Spatial Aggregation" → fixed to lowercase 'a'
        assert get_spatial_subfunction("H", "III").name == "Spatial aggregation"


# ============================================================================
# TemplateInfo — synthetic-dict validation
# ============================================================================


def _good_template_info() -> dict:
    return {
        "tid": "1.1",
        "sfid": "A",
        "ssfid": "I",
        "answer_type": "Numeric (float)",
        "answer_unit": "kilometres",
    }


class TestTemplateInfo:
    def test_well_formed_entry_validates(self) -> None:
        info = TemplateInfo.model_validate(_good_template_info())
        assert info.tid == "1.1"
        assert info.answer_type == "Numeric (float)"
        assert info.answer_unit == "kilometres"

    def test_missing_ssfid_is_ok(self) -> None:
        """ssfid is optional — benchmark-derived exports leave it unset."""
        row = _good_template_info()
        del row["ssfid"]
        info = TemplateInfo.model_validate(row)
        assert info.ssfid is None

    def test_unknown_answer_type_fails_loudly(self) -> None:
        row = _good_template_info()
        row["answer_type"] = "Numeric: 4"
        with pytest.raises(ValidationError) as excinfo:
            TemplateInfo.model_validate(row)
        assert "Numeric: 4" in str(excinfo.value)
        assert "Known types" in str(excinfo.value)

    def test_missing_answer_type_fails_loudly(self) -> None:
        row = _good_template_info()
        del row["answer_type"]
        with pytest.raises(ValidationError):
            TemplateInfo.model_validate(row)

    def test_extra_fields_are_forbidden(self) -> None:
        row = _good_template_info()
        row["future_field"] = "oops"
        with pytest.raises(ValidationError):
            TemplateInfo.model_validate(row)

    def test_unknown_sfid_fails_loudly(self) -> None:
        row = _good_template_info()
        row["sfid"] = "Z"
        with pytest.raises(ValidationError) as excinfo:
            TemplateInfo.model_validate(row)
        assert "unknown SFID 'Z'" in str(excinfo.value)
        assert "spatial_functions.py" in str(excinfo.value)

    def test_unknown_ssfid_fails_loudly(self) -> None:
        row = _good_template_info()
        row["ssfid"] = "XII"  # not a real sub-function under SFID A
        with pytest.raises(ValidationError) as excinfo:
            TemplateInfo.model_validate(row)
        msg = str(excinfo.value)
        assert "A.'XII'" in msg or "A.XII" in msg
        assert "spatial_functions.py" in msg

    def test_ssfid_must_belong_to_sfid_family(self) -> None:
        # Switching SFID without changing SSFID may produce a pair that
        # doesn't exist (e.g., A.VII is missing in the Excel).
        row = _good_template_info()
        row["sfid"] = "A"
        row["ssfid"] = "VII"  # A.VII is one of the gaps in the Excel
        with pytest.raises(ValidationError) as excinfo:
            TemplateInfo.model_validate(row)
        assert "A" in str(excinfo.value)

    def test_dropped_intent_fields_are_rejected(self) -> None:
        # Intent tags (multimodal / hop_complexity / reasoning axes /
        # etc.) are intentionally NOT in TemplateInfo. Per-question
        # metadata lives on the benchmark row itself; design-time
        # intent lives in the Templates sheet and is consumed by
        # paper-phase analysis, not the runtime. ``extra="forbid"``
        # rejects them on direct construction.
        for field in (
            "multimodal",
            "hop_complexity",
            "reasoning_navigational",
            "reasoning_relational",
            "reasoning_temporal",
            "cultural_grounding",
            "expert_usage",
        ):
            row = _good_template_info()
            row[field] = True
            with pytest.raises(ValidationError):
                TemplateInfo.model_validate(row)


# ============================================================================
# TemplateRegistry — full file
# ============================================================================


class TestTemplateRegistry:
    def test_minimal_registry_validates(self) -> None:
        reg = TemplateRegistry.model_validate(
            {
                "version": "1",
                "templates": {
                    "1.1": _good_template_info(),
                },
            }
        )
        assert len(reg) == 1
        assert "1.1" in reg
        assert reg.get("1.1").answer_type == "Numeric (float)"

    def test_provenance_fields_are_rejected(self) -> None:
        # Option A from the plan discussion: the registry carries no
        # provenance fields. exported_from_commit / exported_at /
        # exported_by must be rejected as unknown fields.
        with pytest.raises(ValidationError):
            TemplateRegistry.model_validate(
                {
                    "version": "1",
                    "exported_from_commit": "abc123",
                    "templates": {},
                }
            )

    def test_get_unknown_tid_raises(self) -> None:
        reg = TemplateRegistry.model_validate({"templates": {}})
        with pytest.raises(KeyError) as excinfo:
            reg.get("nope")
        assert "nope" in str(excinfo.value)


class TestLoadTemplateRegistry:
    def test_load_from_temp_file(self, tmp_path: Path) -> None:
        path = tmp_path / "template_registry.json"
        path.write_text(
            json.dumps(
                {
                    "version": "1",
                    "templates": {"1.1": _good_template_info()},
                }
            )
        )
        reg = load_template_registry(path)
        assert len(reg) == 1
        assert reg.get("1.1").answer_type == "Numeric (float)"

    def test_load_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_template_registry(tmp_path / "does_not_exist.json")

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_template_registry(path)

    def test_load_with_unknown_answer_type_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "template_registry.json"
        bad = _good_template_info()
        bad["answer_type"] = "WhoEvenKnows"
        path.write_text(json.dumps({"templates": {"1.1": bad}}))
        with pytest.raises(ValidationError) as excinfo:
            load_template_registry(path)
        assert "WhoEvenKnows" in str(excinfo.value)

    def test_bundled_registry_per_tid_overrides_for_overloaded_types(self) -> None:
        """The bundled registry must carry the per-TID canonical
        overrides for the overloaded benchmark vocab tags
        (``categorical``, ``grid_cell_id``). Without these, the model
        pipeline falls back to row-mode resolution and produces wrong
        canonical types for the overloaded TIDs.

        All TIDs are in the FINAL 1..72 taxonomy (post-drop renumber; TID
        24.1 dropped 2026-07-24 shifting parents 25..73 down to 24..72).
        The old bbox-mislabel override (115.1 → "Bounding box (4-float)")
        was removed because the renumbered share's tid 72.1 carries
        ``ground_truth_type='bounding_box'`` directly.
        """
        reg = load_template_registry()
        expected = {
            # categorical
            "50.1": "Predefined value (left, right)",           # was 55.1
            "51.1": "Predefined value (upstream, downstream)",  # was 56.1
            "52.1": "Cardinal direction (predefined)",          # was 57.1
            "62.1": "Ternary (true/false/plausibly)",           # was 68.1
            # grid_cell_id
            "40.1": "H3 cell index",                            # was 45.1
            "41.1": "H3 cell index",                            # was 46.1
            "64.1": "H3 cell index",                            # was 70.1
            "64.2": "S2 cell index",                            # was 70.2
            "64.3": "Geohash string",                           # was 70.3
            "64.4": "Coordinates (lat/long pair)",              # was 70.4
            # bbox — new tid 66.1 (was 72.1) derives from ground_truth_type.
            "66.1": "Bounding box (4-float)",                   # was 72.1
        }
        for tid, expected_canonical in expected.items():
            actual = reg.get(tid).answer_type
            assert actual == expected_canonical, (
                f"TID {tid}: expected {expected_canonical!r}, got {actual!r}"
            )

    def test_load_no_arg_uses_bundled_registry(self) -> None:
        # ``load_template_registry()`` with no args finds the bundled
        # registry via importlib.resources, regardless of cwd. This is
        # the form the runner uses at startup.
        reg = load_template_registry()
        assert isinstance(reg, TemplateRegistry)
        assert len(reg) > 0
        sfids = {t.sfid for t in reg.templates.values()}
        # SFID B is always present in the share (B is the smallest
        # always-shipped family); SFID A may be absent in shares where
        # SFID A is still in flight at the generator side.
        assert "B" in sfids


# ============================================================================
# Benchmark-type vocabulary loaded from JSON
# ============================================================================


class TestTypeVocabulary:
    def test_vocabulary_file_exists(self) -> None:
        from benchmark_models.schemas.type_registry import TYPE_VOCABULARY_PATH

        assert TYPE_VOCABULARY_PATH.exists(), (
            f"missing: {TYPE_VOCABULARY_PATH}"
        )

    def test_fallback_matches_json_file(self) -> None:
        """Hardcoded fallbacks and JSON must agree — one source of
        truth per snake_case key."""
        import json

        from benchmark_models.schemas.type_registry import (
            _FALLBACK_BENCHMARK_TO_CANONICAL,
            _FALLBACK_DEFAULT_UNIT,
            TYPE_VOCABULARY_PATH,
        )

        raw = json.loads(TYPE_VOCABULARY_PATH.read_text())
        for bt, entry in raw["types"].items():
            assert (
                _FALLBACK_BENCHMARK_TO_CANONICAL.get(bt) == entry["canonical_name"]
            ), f"fallback diverges from JSON for {bt}"
            if entry.get("default_unit") is not None:
                assert _FALLBACK_DEFAULT_UNIT.get(bt) == entry["default_unit"]

    def test_every_canonical_is_registered(self) -> None:
        from benchmark_models.schemas.type_registry import (
            BENCHMARK_TYPE_TO_CANONICAL,
            TYPE_REGISTRY,
        )

        for canonical in BENCHMARK_TYPE_TO_CANONICAL.values():
            assert canonical in TYPE_REGISTRY, (
                f"vocabulary points at unregistered type {canonical!r}"
            )

    def test_set_vocab_added_for_2026_05_07_share(self) -> None:
        """The 2026-05-07 share introduces the bare ``set`` ground_truth_type
        on what were TIDs 27.1 / 28.1 / 29.1 pre-renumbering (22.1 / 23.1 /
        24.1 after the first renumber; the land-use 24.1 was dropped
        2026-07-24, so 22.1 / 23.1 remain). Without this vocab mapping,
        those rows silently format-error at eval time."""
        from benchmark_models.schemas.type_registry import (
            BENCHMARK_TYPE_TO_CANONICAL,
            canonicalize_answer_type,
        )

        assert BENCHMARK_TYPE_TO_CANONICAL.get("set") == (
            "Set enumeration (list of entity names)"
        )
        assert canonicalize_answer_type("set") == (
            "Set enumeration (list of entity names)"
        )


# ============================================================================
# Bundled registry freshness vs the new share
# ============================================================================


class TestBundledRegistryFreshness:
    """Pin the shape of the regenerated registry so an accidentally-stale
    regeneration doesn't silently shrink coverage.
    """

    def test_covers_all_14_sfids_in_the_share(self) -> None:
        reg = load_template_registry()
        sfids = {t.sfid for t in reg.templates.values()}
        # The benchmark covers fourteen spatial-function families, A-N.
        expected = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"}
        assert sfids == expected, (
            f"unexpected SFID coverage: missing={expected - sfids} "
            f"unexpected={sfids - expected}"
        )

    def test_registry_has_at_least_129_tids(self) -> None:
        # Lower bound to catch a future regeneration that drops sub-templates.
        # The released registry holds 129 across 65 parent templates.
        reg = load_template_registry()
        assert len(reg) >= 129

    def test_set_typed_tids_present(self) -> None:
        """The set-enumeration tids in the renumbered share (was
        27.1/28.1/29.1 pre-renumbering; 22.1/23.1/24.1 after the first
        renumber). The land-use 24.1 was dropped 2026-07-24, so 22.1/23.1
        remain and must resolve to the set-enumeration canonical — depends
        on the ``set`` vocab entry being present."""
        reg = load_template_registry()
        for tid in ("17.1", "18.1"):  # were 22.1/23.1 pre-2026-07-26 renumber
            info = reg.get(tid)
            assert info.answer_type == "Set enumeration (list of entity names)"

    def test_tid_52_1_now_area(self) -> None:
        """The area-km2 TID (was 80.1 pre-2026-05-09, then 57.1, 53.1 in
        the 1..73 renumber, now 52.1 after the 2026-07-24 TID-24.1 drop)
        is typed as area_km2 → Numeric (float). Catches a future
        regeneration against un-fixed JSON that would re-introduce the
        distance mislabel that motivated this fixture originally."""
        reg = load_template_registry()
        info = reg.get("47.1")  # was 52.1 pre-2026-07-26 renumber
        assert info.answer_type == "Numeric (float)"
        # Default unit pulled from the canonical, which for
        # ``area_km2`` is ``km^2``.
        assert info.answer_unit == "km^2"

    def test_no_orphan_tids_outside_known_sfids(self) -> None:
        """Every registry TID has its sfid in SPATIAL_FUNCTIONS."""
        from benchmark_models.schemas import SPATIAL_FUNCTIONS

        reg = load_template_registry()
        for tid, info in reg.templates.items():
            assert info.sfid in SPATIAL_FUNCTIONS, (
                f"TID {tid} references unknown SFID {info.sfid}"
            )
