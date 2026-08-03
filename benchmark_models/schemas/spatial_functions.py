"""Canonical spatial-function and sub-function mapping.

Each template belongs to a spatial-function family identified by an
SFID (e.g. ``"A"``) and — when finer granularity matters — a
sub-function SSFID (e.g. ``"I"``). This module pins the streamlined
display names that the rest of the codebase uses.

Two read-only mappings:

* :data:`SPATIAL_FUNCTIONS`: ``sfid`` → :class:`SpatialFunctionInfo`.
  16 entries, A through P.
* :data:`SPATIAL_SUBFUNCTIONS`: ``(sfid, ssfid)`` →
  :class:`SpatialSubFunctionInfo`. 60 entries.

SFIDs and SSFIDs are authoritative: if a new spatial-function family
appears in the benchmark, add it here. ``TemplateInfo`` validates
against these mappings at registry-load time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

# ----------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpatialFunctionInfo:
    """Streamlined name for one spatial function (SFID)."""

    sfid: str
    """The single-letter SFID (``"A"`` through ``"P"``)."""

    name: str
    """The streamlined human-readable family name (e.g.,
    ``"Topological Relationships"``)."""


@dataclass(frozen=True, slots=True)
class SpatialSubFunctionInfo:
    """Streamlined name for one sub-function (SFID + SSFID)."""

    sfid: str
    """The single-letter SFID (``"A"`` through ``"P"``)."""

    ssfid: str
    """Roman-numeral sub-function id within the SFID family
    (``"I"``, ``"II"``, …)."""

    name: str
    """The streamlined human-readable sub-function name."""


# ----------------------------------------------------------------------------
# SPATIAL_FUNCTIONS — 14 entries (A through N)
# ----------------------------------------------------------------------------


_SF_ENTRIES: tuple[SpatialFunctionInfo, ...] = (
    # A, B, E, F: Excel names preserved verbatim per the design review.
    SpatialFunctionInfo("A", "Metric Distance & Proximity Computation / Reasoning"),
    SpatialFunctionInfo("B", "Metric Containment"),
    SpatialFunctionInfo("C", "Topological Relationships"),
    SpatialFunctionInfo("D", "Network & Path-Based Reasoning"),
    SpatialFunctionInfo("E", "Containment & Administrative Hierarchy"),
    SpatialFunctionInfo("F", "Spatial Comparison & Ordering"),
    SpatialFunctionInfo("G", "Centrality & Medial Geometry"),
    SpatialFunctionInfo("H", "Grid & Spatial Indexing"),
    SpatialFunctionInfo("I", "Spatio-Temporal Events"),
    SpatialFunctionInfo("J", "Directional & Orientation"),
    SpatialFunctionInfo("K", "Shape & Geometry Properties"),
    SpatialFunctionInfo("L", "Uncertainty & Fuzzy Reasoning"),
    SpatialFunctionInfo("M", "Spatial Transformation & Reference Change"),
    SpatialFunctionInfo("N", "Spatial Reference & Geometry"),
)

SPATIAL_FUNCTIONS: Mapping[str, SpatialFunctionInfo] = MappingProxyType(
    {entry.sfid: entry for entry in _SF_ENTRIES}
)
"""Read-only mapping ``sfid`` → :class:`SpatialFunctionInfo`."""


# ----------------------------------------------------------------------------
# SPATIAL_SUBFUNCTIONS — 55 entries
# ----------------------------------------------------------------------------


_SSF_ENTRIES: tuple[SpatialSubFunctionInfo, ...] = (
    # ---- A: Metric Distance & Proximity (7 sub-functions, gap at VII) ----
    SpatialSubFunctionInfo("A", "I", "Direct / constrained distance"),  # ws fix
    SpatialSubFunctionInfo("A", "II", "Event-to-location"),
    SpatialSubFunctionInfo("A", "III", "Comparative"),
    SpatialSubFunctionInfo("A", "IV", "Aggregation"),
    SpatialSubFunctionInfo("A", "V", "Optimization"),
    SpatialSubFunctionInfo("A", "VI", "Distance-based selection"),
    SpatialSubFunctionInfo("A", "VIII", "Boundary distance"),
    # ---- B: Metric Containment ----
    SpatialSubFunctionInfo("B", "I", "Counting"),
    SpatialSubFunctionInfo("B", "II", "Enumeration"),
    SpatialSubFunctionInfo("B", "III", "Spatial join"),
    # ---- C: Topological Relationships ----
    SpatialSubFunctionInfo("C", "I", "Intersection (boolean)"),
    SpatialSubFunctionInfo("C", "II", "Intersection (count)"),
    SpatialSubFunctionInfo("C", "III", "Intersection (enumeration)"),
    SpatialSubFunctionInfo("C", "IV", "Overlap"),
    SpatialSubFunctionInfo("C", "V", "Adjacency"),
    SpatialSubFunctionInfo("C", "VI", "Disjointness"),
    SpatialSubFunctionInfo("C", "VII", "Enclosure"),
    # ---- D: Network & Path-Based Reasoning ----
    SpatialSubFunctionInfo("D", "I", "Connectivity"),
    SpatialSubFunctionInfo("D", "II", "Traversal"),
    SpatialSubFunctionInfo("D", "III", "Path proximity"),
    SpatialSubFunctionInfo("D", "IV", "Path ordering"),
    SpatialSubFunctionInfo("D", "V", "Required crossings"),
    SpatialSubFunctionInfo("D", "VI", "Optimal routing"),
    # ---- E: Containment & Administrative Hierarchy ----
    SpatialSubFunctionInfo("E", "I", "Membership"),
    SpatialSubFunctionInfo("E", "II", "Counting"),
    SpatialSubFunctionInfo("E", "III", "Full containment"),
    # ---- F: Spatial Comparison & Ordering ----
    SpatialSubFunctionInfo("F", "I", "Attribute comparison"),
    SpatialSubFunctionInfo("F", "II", "Extremes"),
    SpatialSubFunctionInfo("F", "III", "Ranking"),
    SpatialSubFunctionInfo("F", "IV", "Compound comparison"),
    # ---- G: Centrality & Medial Geometry ----
    SpatialSubFunctionInfo("G", "I", "Centroid / medoid"),
    SpatialSubFunctionInfo("G", "II", "Entity centrality"),
    SpatialSubFunctionInfo("G", "III", "Scale sensitivity"),
    # ---- H: Grid & Spatial Indexing ----
    SpatialSubFunctionInfo("H", "I", "Spatial unit containment"),
    SpatialSubFunctionInfo("H", "II", "Multi-indexing"),
    SpatialSubFunctionInfo("H", "III", "Spatial aggregation"),  # case fix
    # ---- I: Spatio-Temporal Events (gap at VI) ----
    SpatialSubFunctionInfo("I", "I", "Event distance"),
    SpatialSubFunctionInfo("I", "II", "Occurrence count"),  # typo fix
    SpatialSubFunctionInfo("I", "III", "Temporal extent"),
    SpatialSubFunctionInfo("I", "IV", "Temporal overlap"),
    SpatialSubFunctionInfo("I", "V", "Spatial overlap"),
    SpatialSubFunctionInfo("I", "VII", "Spatial impact"),
    # ---- J: Directional & Orientation ----
    SpatialSubFunctionInfo("J", "I", "Cardinal direction"),  # streamlined
    SpatialSubFunctionInfo("J", "II", "Relative to path / location"),
    SpatialSubFunctionInfo("J", "III", "Angular orientation"),
    # ---- K: Shape & Geometry Properties ----
    SpatialSubFunctionInfo("K", "I", "Area comparison"),
    SpatialSubFunctionInfo("K", "II", "Perimeter"),
    SpatialSubFunctionInfo("K", "III", "Shape reasoning"),  # streamlined
    # ---- L: Uncertainty & Fuzzy Reasoning ----
    SpatialSubFunctionInfo("L", "I", "Approximate distance"),
    SpatialSubFunctionInfo("L", "II", "Vague spatial reference"),  # streamlined
    SpatialSubFunctionInfo("L", "III", "Probabilistic containment"),
    # ---- M: Spatial Transformation & Reference Change ----
    SpatialSubFunctionInfo("M", "I", "Relative position"),
    SpatialSubFunctionInfo("M", "II", "Reference system conversion"),  # streamlined
    # ---- N: Spatial Reference & Geometry ----
    SpatialSubFunctionInfo("N", "I", "Absolute coordinate retrieval"),
    SpatialSubFunctionInfo("N", "II", "Spatial extent representation"),
)

SPATIAL_SUBFUNCTIONS: Mapping[tuple[str, str], SpatialSubFunctionInfo] = MappingProxyType(
    {(entry.sfid, entry.ssfid): entry for entry in _SSF_ENTRIES}
)
"""Read-only mapping ``(sfid, ssfid)`` → :class:`SpatialSubFunctionInfo`.

SSFIDs are Roman numerals (``"I"``, ``"II"``, …). The numbering is
not fully dense — ``A.VII`` and ``I.VI`` are unused — because some
spatial sub-functions don't exist in that family's taxonomy.
"""


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def get_spatial_function(sfid: str) -> SpatialFunctionInfo:
    """Look up a :class:`SpatialFunctionInfo` by SFID.

    Args:
        sfid: A single-letter SFID (``"A"`` through ``"P"``).

    Returns:
        The matching :class:`SpatialFunctionInfo`.

    Raises:
        KeyError: if ``sfid`` is not in the registry.
    """
    try:
        return SPATIAL_FUNCTIONS[sfid]
    except KeyError as exc:
        known = ", ".join(sorted(SPATIAL_FUNCTIONS.keys()))
        raise KeyError(
            f"Unknown SFID {sfid!r}. Known SFIDs: [{known}]. "
            f"Add the entry to benchmark_models/schemas/spatial_functions.py "
            f"if this is a new spatial function family."
        ) from exc


def get_spatial_subfunction(sfid: str, ssfid: str) -> SpatialSubFunctionInfo:
    """Look up a :class:`SpatialSubFunctionInfo` by ``(sfid, ssfid)``.

    Args:
        sfid: A single-letter SFID (``"A"`` through ``"P"``).
        ssfid: A Roman-numeral SSFID (``"I"``, ``"II"``, ...).

    Returns:
        The matching :class:`SpatialSubFunctionInfo`.

    Raises:
        KeyError: if ``(sfid, ssfid)`` is not in the registry.
    """
    key = (sfid, ssfid)
    try:
        return SPATIAL_SUBFUNCTIONS[key]
    except KeyError as exc:
        known = ", ".join(
            f"{s}.{ss}" for (s, ss) in sorted(SPATIAL_SUBFUNCTIONS.keys())
        )
        raise KeyError(
            f"Unknown sub-function {sfid}.{ssfid!r}. Known sub-functions: "
            f"[{known}]. Add the entry to benchmark_models/schemas/"
            f"spatial_functions.py if this is a new sub-function."
        ) from exc


def list_spatial_functions() -> list[SpatialFunctionInfo]:
    """Return all SFID entries in registration order (A through P)."""
    return list(_SF_ENTRIES)


def list_spatial_subfunctions(sfid: str | None = None) -> list[SpatialSubFunctionInfo]:
    """Return sub-function entries.

    Args:
        sfid: If given, only return sub-functions under this SFID.
            Otherwise return all 60 entries in registration order.
    """
    if sfid is None:
        return list(_SSF_ENTRIES)
    return [entry for entry in _SSF_ENTRIES if entry.sfid == sfid]


def is_known_sfid(sfid: str) -> bool:
    """Return ``True`` if ``sfid`` is in the registry."""
    return sfid in SPATIAL_FUNCTIONS


def is_known_subfunction(sfid: str, ssfid: str) -> bool:
    """Return ``True`` if ``(sfid, ssfid)`` is in the registry."""
    return (sfid, ssfid) in SPATIAL_SUBFUNCTIONS
