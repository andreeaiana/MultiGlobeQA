"""Geometry simplification for oracle gold contexts.

The kwg/osmh3kg/worldkg gold triples carry verbatim WKT geometries —
ZCTA / county / country / lake polygons run from ~100 chars to
>1 M chars. At any reasonable deployment context cap (8 K – 1 M
tokens), a fraction of these overflow when injected into the
oracle prompt. Format choice (raw / structured / verbalized) saves
nothing — the WKT *bytes themselves* are the bottleneck (measured
on kwg/eng/sfid_a: the three formats differ by <0.5 % per row).

Douglas-Peucker simplification (via shapely's
:meth:`shapely.geometry.base.BaseGeometry.simplify`) bounds the
rendered context length. Applied **uniformly across all oracle
runs regardless of backend** so the closed-source-vs-open-weights
oracle comparison stays apples-to-apples — same gold-context
fidelity per backend, simplified or not.

Tolerance is in coordinate units. On lat/lon WKT (the format
across all three KGs), ε = 0.001° ≈ 111 m on the equator. The
chosen default was picked empirically to:

* Reduce a typical ZCTA polygon from ~50 K chars to ~5 K chars
  (~10× shrinkage on the kwg/sfid_a development slice).
* Keep simplified polygon-area within ~0.05 % of the original.
* Stay well below the spatial resolution at which a typical
  benchmark distance / containment answer would differ from the
  unsimplified ground truth.

The methodology footnote in the paper quotes the actual reduction
ratios measured on the eval slice — see
:class:`SimplifyStats` and the audit log emitted by
``OracleContext.__post_init__`` once simplification is active.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from benchmark_models.oracle import Triple, _is_aswkt, _strip_typed_literal


@dataclass(frozen=True)
class SimplifyStats:
    """Per-triple summary recorded when a WKT object is simplified.

    The :class:`benchmark_models.oracle.OracleContext` keeps a
    ``simplify_stats`` mapping (question_id → list[SimplifyStats]) so
    downstream eval / methodology footnotes can quote actual
    reduction ratios on the headline slice rather than estimates.
    """
    subject: str
    original_chars: int
    simplified_chars: int
    geometry_type: str
    n_coords_before: int
    n_coords_after: int


_TYPED_LITERAL_SUFFIX = re.compile(r"\^\^<[^>]+>$")
"""Matches the trailing ``^^<http://...wktLiteral>`` annotation that
worldkg appends to its WKT objects. We strip it before parsing,
then re-attach so downstream formatters see the original wrapper
shape."""


def _split_wrapping(obj: str) -> tuple[str, str, str]:
    """Split a WKT object into ``(left_wrap, body, right_wrap)``.

    Three wrapper shapes appear across the gold KGs:

    * osmh3kg: bare ``POLYGON ((...))`` → ``('', 'POLYGON ((...))', '')``
    * kwg: quote-wrapped ``"POLYGON ((...))"`` →
      ``('"', 'POLYGON ((...))', '"')``
    * worldkg: typed literal ``"POLYGON ((...))"^^<...wktLiteral>`` →
      ``('"', 'POLYGON ((...))', '"^^<...wktLiteral>')``

    Wrappers are preserved through the round-trip so the simplified
    triple looks shape-identical to the original to downstream
    code (formatters, eval). Only the WKT body changes.
    """
    s = obj
    typed_suffix = ""
    m = _TYPED_LITERAL_SUFFIX.search(s)
    if m:
        typed_suffix = m.group(0)
        s = s[: m.start()]
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        return ('"', s[1:-1], '"' + typed_suffix)
    return ("", s, typed_suffix)


def _wkt_count_coords(wkt: str) -> int:
    """Approximate coordinate count via comma counting.

    Off by a constant factor (commas separate coords AND
    sub-geometries) but consistent before/after, which is what we
    need for the reduction-ratio telemetry. The shapely-based
    accurate count is available but not worth the import cost on
    every triple — this is audit data, not eval data.
    """
    return wkt.count(",") + 1


def simplify_wkt(wkt_body: str, tolerance: float) -> tuple[str, dict[str, Any]]:
    """Simplify a bare WKT string (no wrapper) via shapely.

    Returns ``(new_wkt, metadata)`` where ``metadata`` carries
    ``geometry_type``, ``n_coords_before``, ``n_coords_after``, or
    an ``error`` key if shapely couldn't parse the input. On parse
    failure the original WKT is returned untouched — oracle
    rendering must remain robust to malformed gold rows.

    ``tolerance=0.0`` is an identity pass (shapely returns the
    original geometry). Callers should normally short-circuit
    earlier; this branch exists for defensive use.
    """
    from shapely import wkt as shp_wkt

    try:
        geom = shp_wkt.loads(wkt_body)
    except Exception as exc:
        return wkt_body, {"error": f"parse-failed: {exc}"}

    n_before = _wkt_count_coords(wkt_body)
    try:
        simplified = geom.simplify(tolerance, preserve_topology=False)
    except Exception as exc:
        return wkt_body, {
            "error": f"simplify-failed: {exc}",
            "geometry_type": geom.geom_type,
        }

    new_wkt = simplified.wkt
    n_after = _wkt_count_coords(new_wkt)
    return new_wkt, {
        "geometry_type": geom.geom_type,
        "n_coords_before": n_before,
        "n_coords_after": n_after,
    }


def simplify_triples(
    triples: list[Triple],
    tolerance: float,
) -> tuple[list[Triple], list[SimplifyStats]]:
    """Simplify every WKT object in ``triples`` and return the new list
    plus per-triple stats.

    Identifies WKT triples via :func:`benchmark_models.oracle._is_aswkt`
    (predicate ends in ``#asWKT`` / ``/asWKT``). Non-WKT objects (type
    URIs, labels, geometry-chain links) are returned untouched in the
    same order.

    Wrapper shapes are preserved: a quote-wrapped kwg WKT comes back
    quote-wrapped, a typed-literal worldkg WKT comes back with the
    annotation re-attached, a bare osmh3kg WKT stays bare.

    ``tolerance=0`` or ``tolerance=None`` bypasses simplification
    entirely and returns ``(list(triples), [])``. Tolerance is in
    coordinate units (decimal degrees on lat/lon WKT).
    """
    if not tolerance:
        return list(triples), []

    out: list[Triple] = []
    stats: list[SimplifyStats] = []
    for t in triples:
        if not _is_aswkt(t["predicate"]):
            out.append(t)
            continue

        left, body, right = _split_wrapping(t["object"])
        # Belt-and-suspenders: ``_strip_typed_literal`` handles edge
        # cases _split_wrapping may miss (rare typed-literal nesting).
        body = _strip_typed_literal(body)

        new_body, meta = simplify_wkt(body, tolerance)

        if "error" in meta:
            logger.warning(
                "simplify_triples: skipping {} on subject={!r} pred={!r} "
                "({})",
                meta["error"], t["subject"], t["predicate"],
                "geometry kept verbatim",
            )
            out.append(t)
            continue

        new_t: Triple = {
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": f"{left}{new_body}{right}",
        }
        if "entity_role" in t:
            new_t["entity_role"] = t["entity_role"]
        out.append(new_t)
        stats.append(SimplifyStats(
            subject=t["subject"],
            original_chars=len(body),
            simplified_chars=len(new_body),
            geometry_type=meta["geometry_type"],
            n_coords_before=meta["n_coords_before"],
            n_coords_after=meta["n_coords_after"],
        ))

    return out, stats


__all__ = ["SimplifyStats", "simplify_triples", "simplify_wkt"]
