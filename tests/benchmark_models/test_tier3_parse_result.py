"""Tests for :meth:`SmolagentsModel._parse_result` and its helpers.

Covers the post-validation + coercion rewrite that prevents raw Python
source or meta-statements from leaking into the ``answer`` field of
tier-3 predictions. The smoke run surfaced two concrete failure modes
that these tests pin:

* qid ``a9d9b79f`` (TID 19.1): answer leaked as
  ``"from geopy.geocoders import Nominatim..."`` — a Python source
  tail. Must be detected as malformed for a Numeric template.
* qid ``5698fe07`` (TID 67.1): answer leaked as
  ``"Please provide a task for me to solve."`` — a meta-statement.
  Must be detected as malformed for an Entity-hex template.

All smolagents imports are mocked so the tests run without the real
framework installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from benchmark_models.tier3 import (
    _coerce_boolean,
    _coerce_coordinates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _TInfo:
    """Minimal TemplateInfo stand-in for _parse_result."""

    answer_type: str | None = None
    answer_unit: str | None = None


def _make_model() -> Any:
    """Construct a SmolagentsModel without running __init__.

    ``_parse_result`` is a pure method — it only reads ``self`` for
    the ``_H3_RE`` class attribute. Sidestepping __init__ avoids
    booting the real smolagents / Spark stack.
    """
    from benchmark_models.tier3 import SmolagentsModel

    return SmolagentsModel.__new__(SmolagentsModel)


# ---------------------------------------------------------------------------
# _coerce_boolean
# ---------------------------------------------------------------------------


class TestCoerceBoolean:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("yes", True),
            ("YES", True),
            ("Yes", True),
            ("y", True),
            ("true", True),
            ("True", True),
            ("t", True),
            ("1", True),
            ("no", False),
            ("No", False),
            ("n", False),
            ("false", False),
            ("False", False),
            ("f", False),
            ("0", False),
            ("  yes  ", True),  # whitespace tolerant
        ],
    )
    def test_truthy_and_falsy(self, value: Any, expected: bool) -> None:
        assert _coerce_boolean(value) is expected

    @pytest.mark.parametrize(
        "value",
        ["maybe", "likely", "partially", "2", "", None, [], {}],
    )
    def test_rejects_non_boolean(self, value: Any) -> None:
        assert _coerce_boolean(value) is None

    def test_bare_numeric_rejected(self) -> None:
        """0/1 as bare ints are too easy to hit accidentally — the
        coercer returns None so the caller treats it as malformed."""
        assert _coerce_boolean(1) is None
        assert _coerce_boolean(0) is None
        assert _coerce_boolean(1.0) is None


# ---------------------------------------------------------------------------
# _coerce_coordinates
# ---------------------------------------------------------------------------


class TestCoerceCoordinates:
    def test_string_with_comma(self) -> None:
        assert _coerce_coordinates("52.517, 13.390") == "52.517, 13.39"

    def test_string_with_semicolon(self) -> None:
        assert _coerce_coordinates("52.517; 13.390") == "52.517, 13.39"

    def test_string_with_space(self) -> None:
        assert _coerce_coordinates("52.517 13.390") == "52.517, 13.39"

    def test_list(self) -> None:
        assert _coerce_coordinates([52.517, 13.390]) == "52.517, 13.39"

    def test_dict_lat_lon(self) -> None:
        assert (
            _coerce_coordinates({"lat": 52.517, "lon": 13.390})
            == "52.517, 13.39"
        )

    def test_dict_latitude_longitude(self) -> None:
        assert (
            _coerce_coordinates({"latitude": 52.517, "longitude": 13.390})
            == "52.517, 13.39"
        )

    @pytest.mark.parametrize(
        "value",
        [
            "not coords",
            "Please provide a task for me to solve.",
            "from geopy.geocoders import Nominatim",
            "",
            None,
            [],
            [1, 2, 3],  # wrong arity
            {"lat": 52.517},  # missing lon
        ],
    )
    def test_rejects_garbage(self, value: Any) -> None:
        assert _coerce_coordinates(value) is None

    def test_rejects_out_of_range(self) -> None:
        assert _coerce_coordinates("95.0, 13.39") is None   # lat > 90
        assert _coerce_coordinates("52.5, 200.0") is None   # lon > 180
        assert _coerce_coordinates("-95.0, 13.39") is None  # lat < -90


# ---------------------------------------------------------------------------
# _parse_result + _validate_against_type
# ---------------------------------------------------------------------------


class TestParseResultBoolean:
    def test_yes_no_strings_coerced_to_bool(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Boolean (yes/no, true/false)")
        answer, unit, refusal, malformed = m._parse_result("yes", info)
        assert answer is True and not refusal and not malformed
        answer, unit, refusal, malformed = m._parse_result("no", info)
        assert answer is False and not refusal and not malformed

    def test_real_bool_passes_through(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Boolean (yes/no, true/false)")
        answer, _, refusal, malformed = m._parse_result(
            {"answer": False}, info
        )
        assert answer is False and not refusal and not malformed

    def test_garbage_for_boolean_is_malformed(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Boolean (yes/no, true/false)")
        # Any string that's not a known yes/no variant → no_final_answer.
        answer, _, refusal, malformed = m._parse_result(
            "please provide more context", info
        )
        assert answer is None and refusal and malformed


class TestParseResultNumeric:
    def test_float_string_parsed(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Numeric (float)", answer_unit="kilometres")
        answer, unit, refusal, malformed = m._parse_result("4.24", info)
        assert answer == 4.24 and unit == "kilometres" and not refusal

    def test_dict_payload_with_unit(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Numeric (float)", answer_unit="kilometres")
        answer, unit, refusal, malformed = m._parse_result(
            {"answer": 4.24, "unit": "km"}, info
        )
        assert answer == 4.24 and unit == "km" and not refusal and not malformed

    def test_python_source_leak_is_malformed(self) -> None:
        """Smoke-run regression: qid a9d9b79f (TID 19.1) leaked Python
        source into answer because the agent step was flagged
        is_final_answer=True but the payload was garbage."""
        m = _make_model()
        info = _TInfo(answer_type="Numeric (integer)")
        answer, _, refusal, malformed = m._parse_result(
            "from geopy.geocoders import Nominatim\nfrom geopy.distance import geodesic",
            info,
        )
        assert answer is None and refusal and malformed

    def test_bool_rejected_as_numeric(self) -> None:
        """``bool`` is an ``int`` subclass — reject to avoid spurious
        1.0/0.0 numeric answers from what was really a True/False."""
        m = _make_model()
        info = _TInfo(answer_type="Numeric (float)")
        answer, _, refusal, malformed = m._parse_result({"answer": True}, info)
        assert answer is None and refusal and malformed


class TestParseResultEntityHex:
    def test_valid_h3_accepted(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Entity hex string (single)")
        answer, _, refusal, malformed = m._parse_result(
            "891f1d48227ffff", info
        )
        assert answer == "891f1d48227ffff"
        assert not refusal and not malformed

    def test_valid_h3_uppercase_lowered(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Entity hex string (single)")
        answer, _, _, malformed = m._parse_result("891F1D48227FFFF", info)
        assert answer == "891f1d48227ffff" and not malformed

    def test_meta_statement_is_malformed(self) -> None:
        """Smoke-run regression: qid 5698fe07 (TID 67.1) leaked
        'Please provide a task for me to solve.' as the answer."""
        m = _make_model()
        info = _TInfo(answer_type="Entity hex string (single)")
        answer, _, refusal, malformed = m._parse_result(
            "Please provide a task for me to solve.", info
        )
        assert answer is None and refusal and malformed

    def test_wrong_length_h3_is_malformed(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Entity hex string (single)")
        # Too short
        _, _, _, malformed = m._parse_result("891f1d48", info)
        assert malformed
        # Non-hex
        _, _, _, malformed = m._parse_result("zzz1d48227ffffff", info)
        assert malformed


class TestParseResultCoordinates:
    def test_dict_accepted(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Coordinates (lat/long pair)")
        answer, _, refusal, malformed = m._parse_result(
            {"answer": {"lat": 52.517, "lon": 13.390}}, info
        )
        assert answer == "52.517, 13.39" and not refusal and not malformed

    def test_string_accepted(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Coordinates (lat/long pair)")
        answer, _, _, malformed = m._parse_result("52.517, 13.390", info)
        assert answer == "52.517, 13.39" and not malformed

    def test_out_of_range_malformed(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Coordinates (lat/long pair)")
        _, _, _, malformed = m._parse_result("95.0, 200.0", info)
        assert malformed


class TestParseResultExplicitNull:
    """Explicit ``final_answer(None)`` / ``{"answer": None}`` is an
    abstention — *not* malformed. Distinguishing matters: abstained
    signals the model deliberately didn't answer; no_final_answer
    signals the model failed to produce a valid answer."""

    def test_none_result_is_abstention_not_malformed(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Boolean (yes/no, true/false)")
        answer, _, refusal, malformed = m._parse_result(None, info)
        assert answer is None and refusal and not malformed

    def test_dict_with_answer_none(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Numeric (float)")
        answer, _, refusal, malformed = m._parse_result(
            {"answer": None}, info
        )
        assert answer is None and refusal and not malformed


class TestEnvelopeFreeRecovery:
    """Smoke regression (GPT-5.4-tier3c): the model called
    ``final_answer({"lat": X, "lon": Y})`` on 5 questions — an
    envelope without the canonical ``"answer"`` key.

    Before the fix: all 5 classified as ``"abstained"`` because
    ``dict.get("answer")`` returned None, conflating schema-miss with
    explicit abstention. Worst case: one row was an *exactly correct*
    Coordinates answer that we threw away.

    After the fix: type-aware recovery accepts ``{"lat", "lon"}`` for
    Coordinate questions and flags it malformed for Boolean /
    Entity-hex questions (where a coord dict is obviously wrong).
    """

    def test_coord_dict_accepted_for_coordinate_template(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Coordinates (lat/long pair)")
        # The exact GPT-5.4 payload from qid ebafcd53 (TID 112.1).
        payload = {
            "lat": 52.523130800000004,
            "lon": 13.440441100000001,
        }
        answer, _, refusal, malformed = m._parse_result(payload, info)
        assert not refusal and not malformed
        assert answer.startswith("52.523")
        assert "13.44" in answer

    def test_coord_dict_with_latitude_longitude_keys(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Coordinates (lat/long pair)")
        answer, _, refusal, malformed = m._parse_result(
            {"latitude": 52.517, "longitude": 13.39}, info
        )
        assert not refusal and not malformed
        assert answer == "52.517, 13.39"

    def test_coord_dict_rejected_for_boolean_template(self) -> None:
        """TID 90.1 / 101.1 case: GPT returned coordinates for a
        Boolean question. Pre-fix: abstained. Post-fix: malformed."""
        m = _make_model()
        info = _TInfo(answer_type="Boolean (yes/no, true/false)")
        answer, _, refusal, malformed = m._parse_result(
            {"lat": 52.517, "lon": 13.39}, info
        )
        assert answer is None and refusal and malformed

    def test_coord_dict_rejected_for_entity_hex_template(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Entity hex string (single)")
        _, _, refusal, malformed = m._parse_result(
            {"lat": 52.517, "lon": 13.39}, info
        )
        assert refusal and malformed

    def test_numeric_single_field_dict_accepted(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Numeric (float)", answer_unit="km")
        answer, unit, refusal, malformed = m._parse_result(
            {"value": 4.24}, info
        )
        assert answer == 4.24 and unit == "km"
        assert not refusal and not malformed

    def test_numeric_named_field_dict_accepted(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Numeric (integer)")
        answer, _, refusal, malformed = m._parse_result(
            {"count": 99}, info
        )
        assert answer == 99.0 and not refusal and not malformed

    def test_numeric_multi_field_dict_rejected_when_no_preferred_key(
        self,
    ) -> None:
        """Multiple numeric fields with no answer-bearing key — we
        can't tell which is the answer. Conservative: malformed."""
        m = _make_model()
        info = _TInfo(answer_type="Numeric (float)")
        _, _, refusal, malformed = m._parse_result(
            {"foo": 4.24, "bar": 99}, info
        )
        assert refusal and malformed

    def test_numeric_multi_field_dict_recovers_via_preferred_key(
        self,
    ) -> None:
        """When a known answer-bearing key (``distance_km`` /
        ``count`` / ``answer`` / ``value`` / ``area_km2`` /
        ``result``) is present, the recovery picks it even when
        other numeric fields exist (e.g. Claude appending a
        ``confidence`` value)."""
        m = _make_model()
        info = _TInfo(answer_type="Numeric (float)")
        answer, _, refusal, malformed = m._parse_result(
            {"distance_km": 4.24, "confidence": 0.95}, info
        )
        assert answer == 4.24
        assert not refusal and not malformed

    def test_dict_with_answer_none_is_abstention_not_malformed(self) -> None:
        """``{"answer": None}`` is an *explicit* abstention — the
        ``answer`` key is present. Must NOT be classified as malformed
        even though the value is None (that's a separate path)."""
        m = _make_model()
        info = _TInfo(answer_type="Boolean (yes/no, true/false)")
        answer, _, refusal, malformed = m._parse_result(
            {"answer": None, "unit": None}, info
        )
        assert answer is None and refusal and not malformed

    def test_json_string_without_answer_key_triggers_recovery(self) -> None:
        """JSON-string envelope missing the ``answer`` key follows the
        same recovery path as a raw dict — no special casing needed."""
        m = _make_model()
        info = _TInfo(answer_type="Coordinates (lat/long pair)")
        answer, _, refusal, malformed = m._parse_result(
            '{"lat": 52.517, "lon": 13.39}', info
        )
        assert not refusal and not malformed
        assert answer == "52.517, 13.39"


class TestParseResultUnknownType:
    """Answer types we don't explicitly validate pass through — so
    future type additions don't silently fail as malformed."""

    def test_free_text_passes_through(self) -> None:
        m = _make_model()
        info = _TInfo(answer_type="Entity name (single)")
        answer, _, refusal, malformed = m._parse_result(
            "Alexanderplatz", info
        )
        assert answer == "Alexanderplatz" and not refusal and not malformed
