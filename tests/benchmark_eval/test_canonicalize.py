"""Tests for :mod:`benchmark_eval.canonicalize`.

Covers the pure-function helpers:

* string canonicalization — NFC / casefold / punctuation / whitespace
* unit canonicalization — table + pluralization fallback
* numeric parsing — native types, strings, trailing unit tokens
* date parsing — ISO strings, dicts, datetime.date
* date midpoint and components-match rules (Option C)
* day-distance and precision tolerance
"""

from __future__ import annotations

from datetime import date as date_cls

import pytest

from benchmark_eval.canonicalize import (
    DateComponents,
    NumericParse,
    canonicalize_string,
    canonicalize_unit,
    date_components_match,
    date_day_distance,
    date_precision_tolerance_days,
    date_to_midpoint,
    parse_date,
    parse_numeric,
)

# ---------------------------------------------------------------------------
# canonicalize_string
# ---------------------------------------------------------------------------


class TestCanonicalizeString:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Berlin", "berlin"),
            ("Berlin, Germany", "berlin germany"),
            ("  Berlin  ", "berlin"),
            ("BERLIN", "berlin"),
            ("Berlín", "berlín"),  # preserves accented chars
            ("St. Michael", "st michael"),
            ("Cafe-Kanel", "cafe kanel"),
            ("", ""),
            ("  ", ""),
        ],
    )
    def test_basic(self, raw: str, expected: str) -> None:
        assert canonicalize_string(raw) == expected

    def test_none_returns_empty(self) -> None:
        assert canonicalize_string(None) == ""  # type: ignore[arg-type]

    def test_unicode_nfc(self) -> None:
        """``é`` as composed + decomposed codepoints collapse to same form."""
        composed = "caf\u00e9"            # U+00E9
        decomposed = "cafe\u0301"         # e + combining acute
        assert canonicalize_string(composed) == canonicalize_string(decomposed)


# ---------------------------------------------------------------------------
# canonicalize_unit
# ---------------------------------------------------------------------------


class TestCanonicalizeUnit:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("km", "km"),
            ("KM", "km"),
            ("kilometres", "km"),
            ("kilometers", "km"),
            ("Kilometer", "km"),
            ("km²", "km2"),
            ("km^2", "km2"),
            ("square kilometres", "km2"),
            ("kilometres squared", "km2"),
            ("m", "m"),
            ("metres", "m"),
            ("meters", "m"),
            ("ha", "ha"),
            ("hectares", "ha"),
            (None, None),
            ("", None),
            ("  ", None),
        ],
    )
    def test_common_units(self, raw: str | None, expected: str | None) -> None:
        assert canonicalize_unit(raw) == expected

    def test_unknown_unit_returns_canonicalized(self) -> None:
        """Unknown units pass through canonicalized (so the caller can
        compare them, but the caller retains authority over whether
        they match)."""
        assert canonicalize_unit("parsec") == "parsec"
        assert canonicalize_unit("PARSEC") == "parsec"

    def test_plural_fallback(self) -> None:
        """Unknown plural that matches a known singular in the table."""
        # ``"feet"`` is known; ``"foots"`` is not, but "foot" is.
        assert canonicalize_unit("foots") == "ft"


# ---------------------------------------------------------------------------
# parse_numeric
# ---------------------------------------------------------------------------


class TestParseNumeric:
    def test_float(self) -> None:
        p = parse_numeric(13.5)
        assert p.ok and p.value == 13.5 and p.declared_unit is None

    def test_int(self) -> None:
        p = parse_numeric(42)
        assert p.ok and p.value == 42.0 and p.declared_unit is None

    def test_bool_rejected(self) -> None:
        """Booleans are not numbers; reject loudly."""
        assert parse_numeric(True).ok is False
        assert parse_numeric(False).ok is False

    def test_string_plain(self) -> None:
        p = parse_numeric("13.5")
        assert p.ok and p.value == 13.5 and p.declared_unit is None

    def test_string_with_unit(self) -> None:
        p = parse_numeric("13.5 km")
        assert p.ok and p.value == 13.5 and p.declared_unit == "km"

    def test_string_with_unit_no_space(self) -> None:
        p = parse_numeric("13.5km")
        assert p.ok and p.value == 13.5 and p.declared_unit == "km"

    def test_string_with_comma_thousands(self) -> None:
        p = parse_numeric("1,234.5 km")
        assert p.ok and p.value == 1234.5 and p.declared_unit == "km"

    def test_scientific(self) -> None:
        p = parse_numeric("1.5e3")
        assert p.ok and p.value == 1500.0

    def test_negative(self) -> None:
        p = parse_numeric("-12.5")
        assert p.ok and p.value == -12.5

    def test_none_rejected(self) -> None:
        assert parse_numeric(None).ok is False

    def test_empty_string_rejected(self) -> None:
        assert parse_numeric("").ok is False
        assert parse_numeric("   ").ok is False

    def test_garbage_rejected(self) -> None:
        assert parse_numeric("not a number").ok is False

    def test_list_rejected(self) -> None:
        assert parse_numeric([13.5]).ok is False  # type: ignore[arg-type]

    # --- Locale-aware separator handling (multilingual safety) ---

    def test_us_thousands_unambiguous(self) -> None:
        """``"1,234"`` and ``"12,345,678"`` are unambiguously thousands
        in US convention — every comma group is exactly three digits."""
        assert parse_numeric("1,234").value == 1234.0
        assert parse_numeric("12,345,678").value == 12345678.0
        assert parse_numeric("-1,000 km").value == -1000.0

    def test_us_with_decimal_point(self) -> None:
        """``"1,234.56"`` — comma thousands, dot decimal."""
        assert parse_numeric("1,234.56").value == 1234.56
        assert parse_numeric("12,345,678.9").value == 12345678.9

    def test_european_thousands_with_comma_decimal(self) -> None:
        """``"1.234,56"`` (German) — dot thousands, comma decimal.
        Disambiguated by the dot appearing before the comma."""
        p = parse_numeric("1.234,56")
        assert p.ok and p.value == 1234.56

    def test_european_decimal_only_is_ambiguous_and_rejected(self) -> None:
        """``"12,5"`` could be German 12.5 OR a US-thousands typo for
        125 — silent assumption either way introduces 10× scale errors.
        Rejected as format_error so the caller can flag it loudly."""
        assert parse_numeric("12,5").ok is False
        assert parse_numeric("12,34 km").ok is False

    def test_european_thousands_only_is_ambiguous_and_rejected(
        self,
    ) -> None:
        """``"1.234.567"`` could be European thousands (1234567) or a
        date-like artefact. Reject."""
        assert parse_numeric("1.234.567").ok is False

    def test_single_dot_is_decimal(self) -> None:
        """Single-dot strings are unambiguously US decimal."""
        assert parse_numeric("1234.5").value == 1234.5
        assert parse_numeric("0.5").value == 0.5

    def test_underscore_separator(self) -> None:
        """Python literal-style underscores are stripped."""
        assert parse_numeric("1_234.5").value == 1234.5
        assert parse_numeric("1_000_000").value == 1000000.0


class TestParseDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2019-06-15", DateComponents(2019, 6, 15)),
            ("2019-06", DateComponents(2019, 6, None)),
            ("2019", DateComponents(2019, None, None)),
            ({"year": 2019, "month": 6, "day": 15}, DateComponents(2019, 6, 15)),
            ({"y": 2019, "m": 6, "d": 15}, DateComponents(2019, 6, 15)),
            ({"year": 2019}, DateComponents(2019, None, None)),
            (date_cls(2019, 6, 15), DateComponents(2019, 6, 15)),
            (None, None),
            ("not a date", None),
            ("invalid-date", None),
        ],
    )
    def test_variants(self, raw, expected) -> None:
        assert parse_date(raw) == expected

    def test_precision_property(self) -> None:
        assert DateComponents(2019, 6, 15).precision == "day"
        assert DateComponents(2019, 6, None).precision == "month"
        assert DateComponents(2019, None, None).precision == "year"


# ---------------------------------------------------------------------------
# date midpoint
# ---------------------------------------------------------------------------


class TestDateToMidpoint:
    def test_day_precision_identity(self) -> None:
        assert date_to_midpoint(DateComponents(2019, 6, 15)) == date_cls(2019, 6, 15)

    def test_month_precision_day_15(self) -> None:
        assert date_to_midpoint(DateComponents(2019, 6, None)) == date_cls(2019, 6, 15)

    def test_year_precision_july_2(self) -> None:
        assert date_to_midpoint(DateComponents(2019, None, None)) == date_cls(2019, 7, 2)

    def test_invalid_day_clamped(self) -> None:
        """Feb 30 → Feb 28 (or 29 in leap years)."""
        result = date_to_midpoint(DateComponents(2019, 2, 30))
        assert result.month == 2
        assert result.day in (28, 29)


# ---------------------------------------------------------------------------
# date_components_match (EM rule)
# ---------------------------------------------------------------------------


class TestDateComponentsMatch:
    def test_exact_match_day_level(self) -> None:
        g = DateComponents(2019, 6, 15)
        assert date_components_match(g, g) is True

    def test_wrong_year(self) -> None:
        assert date_components_match(
            DateComponents(2020, 6, 15), DateComponents(2019, 6, 15)
        ) is False

    def test_wrong_day(self) -> None:
        assert date_components_match(
            DateComponents(2019, 6, 14), DateComponents(2019, 6, 15)
        ) is False

    def test_gold_year_only_ignores_prediction_detail(self) -> None:
        """Year-only gold accepts any same-year prediction."""
        assert date_components_match(
            DateComponents(2019, 6, 15), DateComponents(2019, None, None)
        ) is True

    def test_gold_month_level_requires_month_match(self) -> None:
        """Month-level gold still requires month match but ignores day."""
        assert date_components_match(
            DateComponents(2019, 6, 15), DateComponents(2019, 6, None)
        ) is True
        assert date_components_match(
            DateComponents(2019, 7, 15), DateComponents(2019, 6, None)
        ) is False

    def test_day_gold_with_month_prediction_fails(self) -> None:
        """Gold specifies day; prediction only specifies month → fail."""
        assert date_components_match(
            DateComponents(2019, 6, None), DateComponents(2019, 6, 15)
        ) is False


# ---------------------------------------------------------------------------
# Day distance + tolerance
# ---------------------------------------------------------------------------


class TestDateDistanceAndTolerance:
    def test_day_distance_same_day(self) -> None:
        g = DateComponents(2019, 6, 15)
        assert date_day_distance(g, g) == 0

    def test_day_distance_one_day(self) -> None:
        assert date_day_distance(
            DateComponents(2019, 6, 16), DateComponents(2019, 6, 15)
        ) == 1

    def test_day_distance_cross_year(self) -> None:
        assert date_day_distance(
            DateComponents(2020, 1, 1), DateComponents(2019, 12, 31)
        ) == 1

    def test_tolerance_day_level(self) -> None:
        g = DateComponents(2019, 6, 15)
        assert date_precision_tolerance_days(g, g) == 1

    def test_tolerance_month_level(self) -> None:
        assert date_precision_tolerance_days(
            DateComponents(2019, 6, None), DateComponents(2019, 6, None)
        ) == 15

    def test_tolerance_year_level(self) -> None:
        assert date_precision_tolerance_days(
            DateComponents(2019, None, None), DateComponents(2019, None, None)
        ) == 365

    def test_tolerance_uses_coarser(self) -> None:
        """Year-only gold + day-level prediction → year tolerance
        (the coarser of the two)."""
        assert date_precision_tolerance_days(
            DateComponents(2019, 6, 15), DateComponents(2019, None, None)
        ) == 365


# ---------------------------------------------------------------------------
# NumericParse dataclass smoke
# ---------------------------------------------------------------------------


class TestNumericParseDataclass:
    def test_fields(self) -> None:
        p = NumericParse(value=1.0, declared_unit="km", ok=True)
        assert p.value == 1.0
        assert p.declared_unit == "km"
        assert p.ok is True
