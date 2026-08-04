"""Type registry — canonical Answer Type → :class:`AnswerTypeSpec`.

This module is the single source of truth for answer-type semantics:
JSON schema fragment, natural-language description rendered in the
prompt, and unit-handling policy. The canonical display names keyed
here are what every downstream consumer (backends, scorer, prompt
renderer) expects. The benchmark's snake_case ``ground_truth_type``
values are mapped into this vocabulary via
:data:`BENCHMARK_TYPE_TO_CANONICAL`.

``AnswerTypeSpec`` fields:

* ``json_schema`` — JSON schema fragment for the ``answer`` field.
  Always includes ``null`` in the type union so the abstention
  sentinel (``answer = None``) is structurally valid.
* ``nl_description`` — short phrase rendered into the prompt's
  ``{answer_type_description}`` slot.
* ``default_unit`` — canonical unit string when one applies
  (``"km"``, ``"km^2"``, ...); ``None`` for non-unit types.
* ``wants_unit_field`` — whether the response schema includes a
  ``unit`` field so the model declares what unit it produced.
  ``True`` for numeric / geometric types, ``False`` elsewhere.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from loguru import logger

TYPE_VOCABULARY_PATH = Path(__file__).parent / "type_vocabulary.json"
"""Canonical benchmark-type vocabulary. Loaded at import time to
populate :data:`BENCHMARK_TYPE_TO_CANONICAL` and
:data:`DEFAULT_UNIT_BY_BENCHMARK_TYPE`. Hardcoded fallbacks below are
used (with a warning) when the file is missing so dev workflows keep
working.
"""

# ----------------------------------------------------------------------------
# AnswerTypeSpec
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnswerTypeSpec:
    """One canonical Answer Type's schema, description, and unit handling.

    Frozen so the global registry is effectively immutable; treat
    ``TYPE_REGISTRY`` as read-only data.
    """

    canonical_name: str
    """Canonical Answer Type name (e.g. ``"Numeric (float)"``). The
    key under which this spec is registered in :data:`TYPE_REGISTRY`."""

    json_schema: dict[str, Any]
    """JSON schema fragment for the ``answer`` field. Always includes
    ``null`` in the type union so abstention is structurally valid."""

    nl_description: str
    """Natural-language phrase rendered into the prompt's
    ``{answer_type_description}`` slot."""

    default_unit: str | None = None
    """Canonical unit string for numeric/geometric types when the
    template does not specify one (e.g., ``"km"`` for distances by
    default). ``None`` for non-numeric types."""

    wants_unit_field: bool = False
    """Whether the response schema for this type includes a ``unit``
    field. ``True`` for numeric / coordinate / geometry types so the
    model declares what unit it produced. ``False`` for everything
    else."""

    description: str = ""
    """Internal documentation describing what this answer type is and
    when it applies. Not rendered to the model — the prompt uses
    ``nl_description`` for that."""


# ----------------------------------------------------------------------------
# Schema fragment helpers
# ----------------------------------------------------------------------------

# A nullable JSON schema for the abstention sentinel. Most entries
# build on these.

_NULLABLE_BOOL: dict[str, Any] = {"type": ["boolean", "null"]}
_NULLABLE_NUMBER: dict[str, Any] = {"type": ["number", "null"]}
_NULLABLE_INTEGER: dict[str, Any] = {"type": ["integer", "null"]}
_NULLABLE_STRING: dict[str, Any] = {"type": ["string", "null"]}
_NULLABLE_STRING_LIST: dict[str, Any] = {
    "type": ["array", "null"],
    "items": {"type": "string"},
}

_COORDINATE_PAIR: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
            },
            "required": ["lat", "lon"],
            "additionalProperties": False,
        },
        {"type": "null"},
    ],
}

_GEOMETRY_LIST: dict[str, Any] = {
    "oneOf": [
        {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number"},
                    "lon": {"type": "number"},
                },
                "required": ["lat", "lon"],
                "additionalProperties": False,
            },
        },
        {"type": "null"},
    ],
}


# ----------------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------------

# One entry per canonical Answer Type. Order is for source-file
# readability only; lookups go through :data:`TYPE_REGISTRY` by name.
_ENTRIES: tuple[AnswerTypeSpec, ...] = (
    # ------------------- Boolean -------------------
    AnswerTypeSpec(
        canonical_name="Boolean (yes/no, true/false)",
        json_schema=_NULLABLE_BOOL,
        nl_description="yes or no",
        description=(
            "Most common answer type (~30 templates). The schema uses a real "
            "JSON boolean; the prompt asks the model in natural language to "
            "answer 'yes' or 'no'. Structured-output handles the conversion."
        ),
    ),
    # ------------------- Entity name -------------------
    AnswerTypeSpec(
        canonical_name="Entity name (single)",
        json_schema=_NULLABLE_STRING,
        nl_description="a single entity name",
        description=(
            "Plain string holding an entity name. The eval pipeline applies "
            "alias-aware matching against the source KG when scoring."
        ),
    ),
    # ------------------- Set enumeration of entity names -------------------
    AnswerTypeSpec(
        canonical_name="Set enumeration (list of entity names)",
        json_schema=_NULLABLE_STRING_LIST,
        nl_description="a list of entity names",
        description=(
            "Order-insensitive set of entity names. Eval reports both exact-set "
            "match and F1 over the set."
        ),
    ),
    # ------------------- Numeric (float) -------------------
    AnswerTypeSpec(
        canonical_name="Numeric (float)",
        json_schema=_NULLABLE_NUMBER,
        nl_description="a numeric value",
        default_unit="km",
        wants_unit_field=True,
        description=(
            "Real-valued numeric answer (distances, areas, durations). "
            "Default unit is kilometres; ``km^2`` applies for area-flavoured "
            "TIDs. Response schema includes a free ``unit`` field so the "
            "model declares what unit it actually produced."
        ),
    ),
    # ------------------- Numeric (integer) -------------------
    AnswerTypeSpec(
        canonical_name="Numeric (integer)",
        json_schema=_NULLABLE_INTEGER,
        nl_description="an integer count",
        default_unit=None,
        wants_unit_field=False,
        description=(
            "Discrete-valued numeric answer (counts, occurrences). No unit "
            "since counts are unitless. Strict integer equality at eval time."
        ),
    ),
    # ------------------- Numeric (ambiguous) -------------------
    AnswerTypeSpec(
        canonical_name="Numeric",
        json_schema=_NULLABLE_NUMBER,
        nl_description="a numeric value",
        default_unit="km",
        wants_unit_field=True,
        description=(
            "Generic numeric type. Treated as a synonym of "
            "``Numeric (float)`` — templates that fall here should ideally "
            "be split into ``Numeric (float)`` or ``Numeric (integer)`` "
            "at the source."
        ),
    ),
    # ------------------- Coordinates -------------------
    AnswerTypeSpec(
        canonical_name="Coordinates (lat/long pair)",
        json_schema=_COORDINATE_PAIR,
        nl_description="a latitude/longitude pair as {\"lat\": <num>, \"lon\": <num>}",
        default_unit="degrees",
        wants_unit_field=False,
        description=(
            "Single coordinate point. The schema is a {lat, lon} object so "
            "the model can never confuse the order. Eval applies a "
            "distance-tolerance metric, not exact match."
        ),
    ),
    # ------------------- Geometry -------------------
    AnswerTypeSpec(
        canonical_name="Geometry (list of lat/long pairs)",
        json_schema=_GEOMETRY_LIST,
        nl_description="a list of latitude/longitude pairs forming a polyline or polygon",
        default_unit="degrees",
        wants_unit_field=False,
        description=(
            "Polyline or polygon as a list of {lat, lon} objects. Eval "
            "applies an IoU / Hausdorff metric, not exact match."
        ),
    ),
    # ------------------- H3 hex string -------------------
    AnswerTypeSpec(
        canonical_name="Entity hex string (single)",
        json_schema=_NULLABLE_STRING,
        nl_description=(
            "an H3 cell identifier, either as a 15-character hexadecimal "
            "string (e.g. '85283473fffffff') or a decimal integer"
        ),
        description=(
            "H3 spatial-index cell ID. Gold can appear as a 14–16 char "
            "hex string, a decimal integer, or a negative signed-int64 "
            "value (res-15 cells overflow). The scorer normalises all "
            "forms to lowercase unsigned hex before comparison."
        ),
    ),
    # ------------------- Geohash -------------------
    AnswerTypeSpec(
        canonical_name="Geohash string",
        json_schema=_NULLABLE_STRING,
        nl_description=(
            "a geohash string (base-32 encoded, e.g. 'u39wju')"
        ),
        description=(
            "Alternative spatial index (vs H3). Lower-case base-32 string. "
            "Strict equality by default, with optional prefix-match "
            "tolerance in the scorer for partial credit."
        ),
    ),
    # ------------------- Composite string (H3+S2) -------------------
    AnswerTypeSpec(
        canonical_name="Composite string (H3+S2)",
        json_schema=_NULLABLE_STRING,
        nl_description=(
            "a composite cell identifier in the exact form "
            "'H3(res=15): <int> | S2(lvl=20): <int>'"
        ),
        description=(
            "TID 48.1 only. Gold is a pipe-separated pair of H3 and S2 "
            "cell IDs at fixed resolutions. Strict equality of both halves "
            "at eval time."
        ),
    ),
    # ------------------- H3 cell index -------------------
    AnswerTypeSpec(
        canonical_name="H3 cell index",
        json_schema=_NULLABLE_STRING,
        nl_description=(
            "an H3 cell index as a decimal integer string "
            "(e.g. '609336537410174975')"
        ),
        description=(
            "Single H3 spatial-index cell ID, gold as decimal-encoded "
            "uint64 string. Distinct from `Entity hex string (single)` "
            "which prompts for hex; this canonical prompts for the "
            "decimal form that the share emits."
        ),
    ),
    # ------------------- S2 cell index -------------------
    AnswerTypeSpec(
        canonical_name="S2 cell index",
        json_schema=_NULLABLE_STRING,
        nl_description=(
            "an S2 cell index as a (signed) decimal integer string "
            "(e.g. '-8350462053613305856')"
        ),
        description=(
            "Single S2 spatial-index cell ID, gold as decimal-encoded "
            "signed int64 string. Negative values are normal — fine S2 "
            "cells overflow the int64 sign bit."
        ),
    ),
    # ------------------- Bounding box -------------------
    AnswerTypeSpec(
        canonical_name="Bounding box (4-float)",
        json_schema={
            "oneOf": [
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                {"type": "null"},
            ],
        },
        nl_description=(
            "a bounding box as four numbers in the order "
            "[lat_min, lon_min, lat_max, lon_max]"
        ),
        default_unit="degrees",
        wants_unit_field=False,
        description=(
            "Axis-aligned bounding box in geographic coordinates. Gold "
            "format is a comma-separated string of four floats; the "
            "scorer parses both into a (min, max) tuple pair and scores "
            "via IoU."
        ),
    ),
    # ------------------- Polar coordinates -------------------
    AnswerTypeSpec(
        canonical_name="Polar coordinates (angle, distance)",
        json_schema={
            "oneOf": [
                {
                    "type": "object",
                    "properties": {
                        "angle_degrees": {"type": "number"},
                        "distance_km": {"type": "number"},
                    },
                    "required": ["angle_degrees", "distance_km"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ],
        },
        nl_description=(
            "polar coordinates as 'θ degrees, r km' "
            "(angle first, then distance)"
        ),
        default_unit="km",
        wants_unit_field=False,
        description=(
            "Angle + radial-distance pair. Scorer compares angle modulo "
            "360 and distance under numeric tolerance separately; both "
            "must match for EM."
        ),
    ),
    # ------------------- Ternary -------------------
    AnswerTypeSpec(
        canonical_name="Ternary (true/false/plausibly)",
        json_schema={
            "type": ["string", "null"],
            "enum": ["true", "false", "plausibly", None],
        },
        nl_description=(
            "one of 'true', 'false', or 'plausibly' "
            "(confirmed true, confirmed false, or plausible-but-uncertain)"
        ),
        description=(
            "Three-valued logic for questions whose answer may be "
            "epistemically uncertain. Strict string equality after "
            "case-normalisation. Upstream gold uses 'Plausibly' capitalised; "
            "both scorer and backend lowercase at comparison time."
        ),
    ),
    # ------------------- Date -------------------
    AnswerTypeSpec(
        canonical_name="Date",
        json_schema=_NULLABLE_STRING,
        nl_description="a date in ISO 8601 format (YYYY-MM-DD)",
        description=(
            "Single date as an ISO 8601 string. Eval normalizes parses and "
            "compares as canonical date."
        ),
    ),
    # ------------------- Set of dates -------------------
    AnswerTypeSpec(
        canonical_name="Set enumeration (list of dates)",
        json_schema=_NULLABLE_STRING_LIST,
        nl_description="a list of dates in ISO 8601 format (YYYY-MM-DD)",
        description=(
            "Order-insensitive set of dates as ISO 8601 strings. Eval "
            "normalizes parses then applies set comparison."
        ),
    ),
    # ------------------- Predefined value (left, right) -------------------
    AnswerTypeSpec(
        canonical_name="Predefined value (left, right)",
        json_schema={"type": ["string", "null"], "enum": ["left", "right", None]},
        nl_description="either 'left' or 'right'",
        description=(
            "Closed-vocab string answer. The schema enforces the allowed "
            "values via JSON enum (with null included for abstention)."
        ),
    ),
    # ------------------- Predefined value (upstream, downstream) -------------------
    AnswerTypeSpec(
        canonical_name="Predefined value (upstream, downstream)",
        json_schema={
            "type": ["string", "null"],
            "enum": ["upstream", "downstream", None],
        },
        nl_description="either 'upstream' or 'downstream'",
        description=(
            "Closed-vocab string answer for river-flow direction questions."
        ),
    ),
    # ------------------- Cardinal direction (predefined) -------------------
    AnswerTypeSpec(
        canonical_name="Cardinal direction (predefined)",
        json_schema={
            "type": ["string", "null"],
            "enum": [
                "north",
                "south",
                "east",
                "west",
                "northeast",
                "northwest",
                "southeast",
                "southwest",
                None,
            ],
        },
        nl_description=(
            "one of the cardinal or intercardinal directions: north, south, "
            "east, west, northeast, northwest, southeast, southwest"
        ),
        description=(
            "Closed-vocab string answer for directional questions. The "
            "natural-language prompt asks the model to pick from the listed "
            "directions; the schema enforces the choice."
        ),
    ),
)


TYPE_REGISTRY: Mapping[str, AnswerTypeSpec] = MappingProxyType(
    {entry.canonical_name: entry for entry in _ENTRIES}
)
"""Read-only registry of canonical Answer Type → ``AnswerTypeSpec``.

Use :func:`get_type_spec` for lookups; mutating ``TYPE_REGISTRY`` is
not supported (it's wrapped in :class:`types.MappingProxyType`).
"""


# ----------------------------------------------------------------------------
# Benchmark-vocabulary bridge
# ----------------------------------------------------------------------------

# The benchmark's ``ground_truth_type`` field uses snake_case values
# (``"boolean"``, ``"numeric_integer"``, ``"bounding_box"``, …). The
# internal vocabulary (keys of :data:`TYPE_REGISTRY`) uses canonical
# display names. This mapping lives alongside :data:`TYPE_REGISTRY`
# so both the eval pipeline (scoring) and the model pipeline
# (response-schema construction) share one source of truth.

# Hardcoded fallbacks — kept in sync with ``type_vocabulary.json``.
# Used only when the vocabulary file is missing.
_FALLBACK_BENCHMARK_TO_CANONICAL: dict[str, str] = {
    "boolean": "Boolean (yes/no, true/false)",
    "ternary": "Ternary (true/false/plausibly)",
    "numeric": "Numeric (float)",
    "numeric_integer": "Numeric (integer)",
    "numeric_float": "Numeric (float)",
    "numeric_km": "Numeric (float)",
    "distance_kilometers": "Numeric (float)",
    "area_km2": "Numeric (float)",
    "coordinates": "Coordinates (lat/long pair)",
    "polar_coordinates": "Polar coordinates (angle, distance)",
    "bounding_box": "Bounding box (4-float)",
    "hex_string": "Entity hex string (single)",
    "geohash": "Geohash string",
    "composite_string": "Composite string (H3+S2)",
    "entity_single_name": "Entity name (single)",
    "set_enumeration": "Set enumeration (list of entity names)",
    "date_set": "Set enumeration (list of dates)",
    "directional_value": "Predefined value (left, right)",
    "cardinal_direction": "Cardinal direction (predefined)",
    "count": "Numeric (integer)",
    "entity_name": "Entity name (single)",
    "entity_list": "Set enumeration (list of entity names)",
    "temporal": "Date",
    "qualitative": "Composite string (H3+S2)",
    # TIDs 27.1, 28.1, 29.1 use the bare ``set`` tag for
    # set-of-entity-names answers.
    "set": "Set enumeration (list of entity names)",
}

_FALLBACK_DEFAULT_UNIT: dict[str, str] = {
    "distance_kilometers": "km",
    "area_km2": "km^2",
    "numeric_km": "km",
    "polar_coordinates": "km",
}


def _load_vocabulary() -> tuple[dict[str, str], dict[str, str]]:
    """Load the benchmark-type vocabulary JSON. Falls back to the
    hardcoded dicts (with a warning) when the file is missing.
    """
    if not TYPE_VOCABULARY_PATH.exists():
        logger.warning(
            "Type vocabulary not found at {}; using hardcoded fallback "
            "values. Regenerate with the committed JSON.",
            TYPE_VOCABULARY_PATH,
        )
        return dict(_FALLBACK_BENCHMARK_TO_CANONICAL), dict(_FALLBACK_DEFAULT_UNIT)
    raw = json.loads(TYPE_VOCABULARY_PATH.read_text())
    types = raw.get("types", {})
    canonical = {
        bt: entry["canonical_name"]
        for bt, entry in types.items()
    }
    units = {
        bt: entry["default_unit"]
        for bt, entry in types.items()
        if entry.get("default_unit") is not None
    }
    return canonical, units


_canonical_view, _unit_view = _load_vocabulary()

BENCHMARK_TYPE_TO_CANONICAL: Mapping[str, str] = MappingProxyType(_canonical_view)

DEFAULT_UNIT_BY_BENCHMARK_TYPE: Mapping[str, str] = MappingProxyType(_unit_view)


def canonicalize_answer_type(benchmark_type: str | None) -> str | None:
    """Map a benchmark ``ground_truth_type`` to the canonical display name.

    Returns ``None`` when the input is ``None``. For unknown snake_case
    names, returns the input verbatim so downstream dispatch falls
    through to the unknown-type branch (``format_error=True`` in the
    scorer) rather than crashing.
    """
    if benchmark_type is None:
        return None
    return BENCHMARK_TYPE_TO_CANONICAL.get(benchmark_type, benchmark_type)


def default_unit_for_benchmark_type(benchmark_type: str | None) -> str | None:
    """Conventional unit for a benchmark answer type, or ``None``."""
    if benchmark_type is None:
        return None
    return DEFAULT_UNIT_BY_BENCHMARK_TYPE.get(benchmark_type)


# Per-TID canonical overrides for benchmark vocab tags that are
# *overloaded* — a single ``ground_truth_type`` value covers multiple
# distinct answer shapes that resolve to different canonical types per
# TID. The flat vocabulary (``type_vocabulary.json``) intentionally
# does not map these tags; the registry exporter and the runtime
# resolver both apply the override below so each TID lands on the
# right canonical.
#
# The two overloaded tags:
#   - ``categorical`` (Predefined L/R, Predefined U/D, Cardinal, Ternary)
#   - ``grid_cell_id`` (H3 / S2 / Geohash / MGRS-like / Coords-string)
#
# Single source of truth — both the build script
# (``scripts/build_template_registry_from_benchmark.py``) and the
# runtime ``resolve_canonical_answer_type`` consume this dict.
# ``test_spec_drift`` asserts they stay in lock-step via direct equality.
TID_TO_CANONICAL_OVERRIDE: dict[str, str] = {
    # Keys are released sub-template ids.

    # `categorical` — single benchmark type covers several distinct
    # multiple-choice families (left/right, upstream/downstream,
    # cardinal direction, yes/no/plausibly). One canonical type per
    # TID, not per benchmark-type tag.
    "50.1": "Predefined value (left, right)",
    "51.1": "Predefined value (upstream, downstream)",
    "52.1": "Cardinal direction (predefined)",
    "62.1": "Ternary (true/false/plausibly)",

    # `grid_cell_id` — same tag for multiple distinct spatial grid
    # encodings (H3, S2, Geohash, coordinate-string). Per-TID override
    # picks the right canonical.
    "40.1": "H3 cell index",
    "41.1": "H3 cell index",
    "64.1": "H3 cell index",
    "64.2": "S2 cell index",
    "64.3": "Geohash string",
    "64.4": "Coordinates (lat/long pair)",

    # NOTE: bounding-box questions need no override. TID 66.1 carries
    # ``ground_truth_type='bounding_box'`` directly, and
    # ``canonicalize_answer_type`` maps that without help.
}


def resolve_canonical_answer_type(
    ground_truth_type: str | None,
    fallback: str | None = None,
    *,
    tid: str | None = None,
) -> str:
    """Pick the canonical answer type for one question.

    Resolution order:

    1. If ``tid`` is in :data:`TID_TO_CANONICAL_OVERRIDE`, return the
       override. Highest precedence — handles benchmark mislabels and
       overloaded vocab tags that ``ground_truth_type`` alone can't
       disambiguate. The eval-side ``benchmark_eval/loaders.py``
       applies the same resolution.
    2. Per-row ``Example.ground_truth_type`` (authoritative per
       question when the TID has no override).
    3. Per-TID ``TemplateInfo.answer_type`` fallback (used when the
       row lacks its own type label).

    Args:
        ground_truth_type: The benchmark's snake_case value from
            ``Example.ground_truth_type``.
        fallback: Canonical display name from
            ``TemplateInfo.answer_type``.
        tid: Per-question template id (``Example.tid``). When set
            and in the override map, takes precedence. Pass it
            explicitly from every backend's schema-build site so
            model-side prompting agrees with eval-side scoring on
            mislabeled or overloaded TIDs.

    Returns:
        A canonical display name from :data:`TYPE_REGISTRY`.

    Raises:
        KeyError: if neither source resolves to a registered type.
    """
    if tid is not None and tid in TID_TO_CANONICAL_OVERRIDE:
        return TID_TO_CANONICAL_OVERRIDE[tid]
    if ground_truth_type is not None:
        canonical = canonicalize_answer_type(ground_truth_type)
        if canonical is not None and canonical in TYPE_REGISTRY:
            return canonical
    if fallback is not None and fallback in TYPE_REGISTRY:
        return fallback
    known = ", ".join(sorted(TYPE_REGISTRY.keys()))
    raise KeyError(
        f"Could not resolve canonical answer type from "
        f"tid={tid!r}, ground_truth_type={ground_truth_type!r}, or "
        f"fallback={fallback!r}. Known canonical types: [{known}]"
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def get_type_spec(name: str) -> AnswerTypeSpec:
    """Look up an :class:`AnswerTypeSpec` by canonical Answer Type name.

    Args:
        name: A canonical Answer Type string from :data:`TYPE_REGISTRY`.

    Returns:
        The matching :class:`AnswerTypeSpec`.

    Raises:
        KeyError: if ``name`` is not in the registry. The error message
            lists every known type so the caller can suggest a fix.
    """
    try:
        return TYPE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(TYPE_REGISTRY.keys()))
        raise KeyError(
            f"Unknown answer type {name!r}. Known types: [{known}]"
        ) from exc


def is_known_type(name: str) -> bool:
    """Return ``True`` if ``name`` is a canonical Answer Type."""
    return name in TYPE_REGISTRY


def list_canonical_types() -> list[str]:
    """Return all canonical Answer Type names in registration order."""
    return [entry.canonical_name for entry in _ENTRIES]


