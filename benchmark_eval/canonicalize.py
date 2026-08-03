"""String, unit, numeric and date canonicalization helpers.

Used by the scorer to normalize predictions and gold values before
comparison. Canonicalization is deliberately separate from the
alias layer (:mod:`benchmark_eval.aliases`): canonicalization
handles spelling / whitespace / Unicode / unit-word variants that
don't need a lookup table, while alias resolution handles
translation and genuinely distinct names that share a canonical
entity.

Core helpers:

* :func:`canonicalize_string` — NFC normalization, casefolding,
  whitespace + punctuation stripping.
* :func:`canonicalize_unit` — table-driven unit normalization
  (e.g. ``"km"``, ``"kilometres"``, ``"kilometer"``, ``"Kilometer"``
  all map to ``"km"``).
* :func:`parse_numeric` — best-effort numeric parse that strips a
  trailing unit token and returns ``(value, declared_unit,
  format_ok)``.
* :func:`parse_date` — parse an ISO string or ``(y, m, d)`` dict into
  :class:`DateComponents`. Partial dates (year-only, month-only) are
  preserved with ``None`` for unspecified components.
* :func:`date_to_midpoint` — collapse a partial date to a
  representative ``date`` object for day-distance arithmetic
  (year-only → July 2; month-only → day 15; full date → itself).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import timedelta
from typing import Any

# ---------------------------------------------------------------------------
# String canonicalization
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def canonicalize_string(s: str) -> str:
    """Return a canonical form suitable for string-equality lookup.

    Applies (in order): Unicode NFC normalization, casefold
    (lowercase for comparison purposes that handles German ß etc.),
    punctuation stripping, whitespace collapsing, and edge trimming.

    Two strings with the same canonical form are treated as
    equivalent for alias lookup and entity resolution. This does NOT
    perform translation — use :class:`benchmark_eval.aliases.AliasStore`
    for cross-language equivalence.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Unit canonicalization
# ---------------------------------------------------------------------------


# Maps any accepted unit spelling (in its already-casefolded form) to
# a canonical short code. Used by both :func:`canonicalize_unit` and
# by the scorer's numeric drill-down to decide whether a prediction's
# declared unit matches the gold unit.
UNIT_CANONICAL: dict[str, str | None] = {
    # Distance
    "km": "km",
    "kms": "km",
    "kilometre": "km",
    "kilometres": "km",
    "kilometer": "km",
    "kilometers": "km",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "mi": "mi",
    "mile": "mi",
    "miles": "mi",
    "ft": "ft",
    "feet": "ft",
    "foot": "ft",
    "nm": "nmi",
    "nautical mile": "nmi",
    "nautical miles": "nmi",
    # Area — accept both squared and abbreviated forms
    "km2": "km2",
    "km²": "km2",
    "km 2": "km2",
    "sqkm": "km2",
    "sq km": "km2",
    "square km": "km2",
    "square kilometer": "km2",
    "square kilometers": "km2",
    "square kilometre": "km2",
    "square kilometres": "km2",
    "kilometres squared": "km2",
    "kilometers squared": "km2",
    "m2": "m2",
    "m²": "m2",
    "square meter": "m2",
    "square meters": "m2",
    "square metre": "m2",
    "square metres": "m2",
    "ha": "ha",
    "hectare": "ha",
    "hectares": "ha",
    "mi2": "mi2",
    "mi²": "mi2",
    "square mile": "mi2",
    "square miles": "mi2",
    # Counts
    "": None,  # empty string → no unit
    "count": None,
    "counts": None,
    # Time
    "day": "day",
    "days": "day",
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
}


def canonicalize_unit(s: str | None) -> str | None:
    """Return a canonical short code for a unit string, or ``None``.

    Canonicalizes the input (casefold + strip + NFC + replace ``²``
    with ``"2"``) then looks it up in :data:`UNIT_CANONICAL`.
    Unknown units return the canonicalized input unchanged — the
    caller can decide whether to treat that as a mismatch.
    Empty / whitespace-only input returns ``None``.
    """
    if s is None:
        return None
    s = unicodedata.normalize("NFC", str(s)).strip().casefold()
    s = s.replace("²", "2").replace("^2", "2")
    s = _WS_RE.sub(" ", s)
    if not s:
        return None
    # Direct lookup.
    if s in UNIT_CANONICAL:
        return UNIT_CANONICAL[s]
    # Try stripping trailing ``s`` as a last resort (handles the long
    # tail of pluralization we didn't enumerate).
    if s.endswith("s") and s[:-1] in UNIT_CANONICAL:
        return UNIT_CANONICAL[s[:-1]]
    return s  # unknown — return canonicalized form unchanged


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------


@dataclass
class NumericParse:
    """Result of :func:`parse_numeric`."""

    value: float | None
    declared_unit: str | None
    ok: bool


_TRAILING_UNIT_RE = re.compile(
    r"^\s*(?P<num>[-+]?[\d.,_]+(?:[eE][-+]?\d+)?)\s*(?P<unit>[^\d\s].*?)?\s*$"
)

# Comma-as-thousands pattern: digits, then one or more groups of
# exactly three digits each preceded by a comma. Optional sign.
# Matches ``"1,234"``, ``"12,345,678"``, ``"-9,999"``. Does NOT match
# ``"12,5"`` (German decimal — ambiguous; rejected).
_THOUSANDS_ONLY_RE = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})+$")

# Dot-as-thousands pattern: same shape but with dots. Matches
# ``"1.234"``, ``"12.345.678"``. ``"1.234"`` *could* also be a US
# decimal — see :func:`_normalize_numeric_string` for tie-breaking.
_DOT_THOUSANDS_RE = re.compile(r"^[-+]?\d{1,3}(?:\.\d{3})+$")


def _normalize_numeric_string(num_str: str) -> str | None:
    """Coerce a numeric token to a Python-float-parseable string.

    Multilingual-safe rules:

    * Strip Python literal-style underscores.
    * If both ``,`` and ``.`` are present: the **last** separator
      wins as the decimal point. ``"1,234.56"`` (US) → ``"1234.56"``;
      ``"1.234,56"`` (German) → ``"1234.56"``.
    * If only ``,`` is present:
        - Pure thousands shape (``"1,234"``, ``"12,345,678"``) → strip
          commas. The pattern is unambiguous: every comma-separated
          group after the first must be exactly three digits.
        - Anything else (``"12,5"``, ``"12,34"``) → ambiguous between
          European decimal and a typo of US thousands; return
          ``None`` so the caller flags the prediction as a format
          error rather than silently introducing a 10×–1000× scale
          bug. The benchmark spec calls for dot-decimal output;
          locale-specific coercion is the model / pipeline's
          responsibility upstream.
    * If only ``.`` is present:
        - Single dot → US decimal.
        - Multi-dot pure-thousands shape (``"1.234.567"``) → ambiguous
          (could be European thousands or a date-like artefact);
          return ``None``.
    * No separators → return as-is.
    """
    s = num_str.replace("_", "")
    has_comma = "," in s
    has_dot = "." in s

    if has_comma and has_dot:
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")
        if last_dot > last_comma:
            # US convention: dot decimal, comma thousands.
            return s.replace(",", "")
        else:
            # European convention: comma decimal, dot thousands.
            return s.replace(".", "").replace(",", ".")

    if has_comma:
        if _THOUSANDS_ONLY_RE.match(s):
            return s.replace(",", "")
        return None  # ambiguous — let caller flag as format error

    if has_dot and _DOT_THOUSANDS_RE.match(s) and s.count(".") > 1:
        return None  # ambiguous European thousands

    return s


def parse_numeric(value: Any) -> NumericParse:
    """Best-effort numeric parse.

    Handles:

    * Native ``int`` / ``float`` — returned as-is with no declared unit.
    * ``bool`` — rejected (``ok=False``); booleans are not numbers
      even though Python treats them as ints.
    * ``str`` with an optional trailing unit token (``"13.5 km"`` →
      ``value=13.5, declared_unit="km"``).
    * ``str`` without a unit (``"13.5"``) — ``declared_unit=None``.
    * ``None`` and everything else — ``ok=False``.

    Numeric-string normalization is locale-aware in a paper-safe way:
    unambiguous US (``"1,234.56"``) and European (``"1.234,56"``)
    forms are accepted; ambiguous shapes (``"12,5"`` — German decimal
    or US-thousands typo?) are rejected as format errors rather than
    silently scaled. See :func:`_normalize_numeric_string` for the
    full ruleset.

    Returns a :class:`NumericParse`. When ``ok=False``, the caller
    should treat the prediction as a format error: the scorer sets
    ``em=0``, ``ne=1``, and ``format_error=True`` in that case.
    """
    if isinstance(value, bool):
        return NumericParse(value=None, declared_unit=None, ok=False)
    if isinstance(value, (int, float)):
        return NumericParse(value=float(value), declared_unit=None, ok=True)
    if not isinstance(value, str):
        return NumericParse(value=None, declared_unit=None, ok=False)

    s = value.strip()
    if not s:
        return NumericParse(value=None, declared_unit=None, ok=False)

    match = _TRAILING_UNIT_RE.match(s)
    if not match:
        return NumericParse(value=None, declared_unit=None, ok=False)
    normalized = _normalize_numeric_string(match.group("num"))
    if normalized is None:
        return NumericParse(value=None, declared_unit=None, ok=False)
    try:
        num = float(normalized)
    except ValueError:
        return NumericParse(value=None, declared_unit=None, ok=False)
    unit_str = match.group("unit")
    return NumericParse(
        value=num,
        declared_unit=unit_str.strip() if unit_str else None,
        ok=True,
    )


# ---------------------------------------------------------------------------
# Date canonicalization
# ---------------------------------------------------------------------------


@dataclass
class DateComponents:
    """A partial or full date.

    ``year`` is required; ``month`` and ``day`` may be ``None`` for
    year-level or month-level dates. :func:`date_to_midpoint`
    collapses a :class:`DateComponents` to a representative full
    ``date`` object for day-distance arithmetic.
    """

    year: int
    month: int | None = None
    day: int | None = None

    @property
    def precision(self) -> str:
        """One of ``"day"``, ``"month"``, ``"year"``."""
        if self.day is not None:
            return "day"
        if self.month is not None:
            return "month"
        return "year"


_ISO_FULL = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{1,2})$")
_ISO_YEAR = re.compile(r"^(\d{4})$")


def parse_date(value: Any) -> DateComponents | None:
    """Parse an ISO-like string or a ``(y, m, d)`` dict into components.

    Accepted forms:

    * ``"2019-06-15"`` → day-precision.
    * ``"2019-06"`` → month-precision (``day=None``).
    * ``"2019"`` → year-precision (``month=None``, ``day=None``).
    * ``{"year": 2019, "month": 6, "day": 15}`` — any subset of keys.
    * ``{"y": ..., "m": ..., "d": ...}`` — short form.
    * :class:`datetime.date` — converted directly.

    Returns ``None`` if the input couldn't be parsed. The scorer
    treats that as a format error.
    """
    if value is None:
        return None
    if isinstance(value, date_cls):
        return DateComponents(year=value.year, month=value.month, day=value.day)
    if isinstance(value, dict):
        y = value.get("year") or value.get("y")
        m = value.get("month") or value.get("m")
        d = value.get("day") or value.get("d")
        if y is None:
            return None
        try:
            return DateComponents(
                year=int(y),
                month=int(m) if m is not None else None,
                day=int(d) if d is not None else None,
            )
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip()
        for pattern, builder in (
            (_ISO_FULL, lambda m: DateComponents(int(m[1]), int(m[2]), int(m[3]))),
            (_ISO_MONTH, lambda m: DateComponents(int(m[1]), int(m[2]), None)),
            (_ISO_YEAR, lambda m: DateComponents(int(m[1]), None, None)),
        ):
            match = pattern.match(s)
            if match:
                try:
                    return builder(match)
                except (TypeError, ValueError):
                    return None
    return None


def date_to_midpoint(dc: DateComponents) -> date_cls:
    """Collapse a partial date to a representative day.

    Midpoint convention (for NE day-distance arithmetic):

    * Day-precision: the date itself.
    * Month-precision: day 15 of the month.
    * Year-precision: July 2 (midpoint of the year).

    Clamps invalid days (e.g., Feb 30) to the last valid day of the
    month so arithmetic never fails.
    """
    year = dc.year
    month = dc.month if dc.month is not None else 7
    day = dc.day if dc.day is not None else (2 if dc.month is None else 15)
    try:
        return date_cls(year, month, day)
    except ValueError:
        # Clamp to last valid day of that month.
        for candidate in range(day, 0, -1):
            try:
                return date_cls(year, month, candidate)
            except ValueError:
                continue
        # Should be unreachable — fall back to first of year.
        return date_cls(year, 1, 1)


def date_components_match(pred: DateComponents, gold: DateComponents) -> bool:
    """Return True if every gold-specified component matches the
    prediction.

    EM rule for dates: compare only the fields
    that the gold specifies; prediction-only extra precision is
    ignored. A year-only gold of 2019 matches a prediction of
    2019-06-15 (both specify year 2019); a day-level gold of
    2019-06-15 and a year-only prediction of 2019 does NOT match
    (gold specifies day but prediction doesn't).
    """
    if pred.year != gold.year:
        return False
    if gold.month is not None and (pred.month is None or pred.month != gold.month):
        return False
    # Early-return form reads more clearly than a negated one-liner here.
    return not (
        gold.day is not None and (pred.day is None or pred.day != gold.day)
    )


# Hardcoded fallback for date-precision tolerances. Live values load
# from ``scoring_spec.json:dates.precision_tolerance_days`` below;
# this dict is the fallback when the spec is missing. Kept in sync
# by an assertion test.
_FALLBACK_DATE_PRECISION_TOLERANCE: dict[str, int] = {
    "year": 365,
    "month": 15,
    "day": 1,
}


def _load_date_precision_tolerance() -> dict[str, int]:
    """Load date-precision tolerances from the spec; fall back to the
    hardcoded table when the spec is missing or malformed."""
    import json

    from benchmark_eval._paths import spec_path

    p = spec_path()
    if not p.exists():
        return dict(_FALLBACK_DATE_PRECISION_TOLERANCE)
    raw = json.loads(p.read_text())
    block = raw.get("dates", {}).get("precision_tolerance_days", {})
    out: dict[str, int] = {}
    for key in ("year", "month", "day"):
        val = block.get(key)
        if isinstance(val, int):
            out[key] = val
        else:
            out[key] = _FALLBACK_DATE_PRECISION_TOLERANCE[key]
    return out


_DATE_PRECISION_TOLERANCE = _load_date_precision_tolerance()


def date_precision_tolerance_days(pred: DateComponents, gold: DateComponents) -> int:
    """Return the day-tolerance ``τ`` for NE given the coarser of
    prediction and gold precision.

    Tolerances are loaded from
    ``scoring_spec.json:dates.precision_tolerance_days`` at import time;
    the default values are 365 / 15 / 1 for year / month / day.

    The rule takes the coarser precision to avoid over-penalizing
    the model when the gold itself only specifies year.
    """
    precisions = (pred.precision, gold.precision)
    if "year" in precisions:
        return _DATE_PRECISION_TOLERANCE["year"]
    if "month" in precisions:
        return _DATE_PRECISION_TOLERANCE["month"]
    return _DATE_PRECISION_TOLERANCE["day"]


def date_day_distance(pred: DateComponents, gold: DateComponents) -> int:
    """Return the absolute day distance between two partial dates,
    using the midpoint convention for the coarser side."""
    return abs((date_to_midpoint(pred) - date_to_midpoint(gold)).days)


# Re-export for convenience.
__all__ = [
    "DateComponents",
    "NumericParse",
    "UNIT_CANONICAL",
    "canonicalize_string",
    "canonicalize_unit",
    "date_components_match",
    "date_day_distance",
    "date_precision_tolerance_days",
    "date_to_midpoint",
    "parse_date",
    "parse_numeric",
]


# Silence unused-import warnings for types surfaced in type hints only.
_ = timedelta
