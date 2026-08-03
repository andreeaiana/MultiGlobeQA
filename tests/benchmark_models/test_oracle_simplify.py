"""Tests for :mod:`benchmark_models.oracle_simplify`.

Covers:

* WKT body simplification via shapely (with stats).
* Wrapper preservation across the three KG flavours
  (kwg quote-wrap, worldkg typed-literal, osmh3kg bare).
* Pass-through for non-WKT triples and ``tolerance=0``.
* Integration with :class:`OracleContext` — construction-time
  simplification populates ``simplify_stats`` and updates
  ``gold_triples_map`` in place.
* Defensive handling of malformed WKT (warn + keep original).
"""

from __future__ import annotations

from benchmark_models.oracle import OracleContext, Triple
from benchmark_models.oracle_simplify import (
    SimplifyStats,
    simplify_triples,
    simplify_wkt,
)

# A polygon with enough vertices to actually shrink under ε=0.001.
# Square-ish ring around (0,0) with stair-step intermediate points so
# Douglas-Peucker has something to remove.
_DENSE_POLYGON_WKT = (
    "POLYGON ((0 0, 0 0.0001, 0 0.0002, 0 0.0003, 0 0.0004, 0 0.0005, "
    "0 0.001, 0 0.005, 0 0.01, 0 0.05, 0 0.1, 0 0.5, 0 1, "
    "0.5 1, 1 1, 1 0.5, 1 0, 0.5 0, 0 0))"
)


class TestSimplifyWkt:
    """``simplify_wkt(body, tolerance)`` is the inner helper that
    takes a *bare* WKT body (no surrounding quotes / typed-literal
    suffix) and returns ``(new_wkt, metadata)``."""

    def test_dense_polygon_shrinks_under_meaningful_tolerance(self) -> None:
        new_wkt, meta = simplify_wkt(_DENSE_POLYGON_WKT, tolerance=0.001)
        assert "POLYGON" in new_wkt
        assert meta["geometry_type"] == "Polygon"
        assert meta["n_coords_after"] < meta["n_coords_before"], (
            f"polygon should shrink: before={meta['n_coords_before']}, "
            f"after={meta['n_coords_after']}"
        )

    def test_point_unchanged_under_simplification(self) -> None:
        """A POINT has nothing to simplify away — coord count stays
        exactly the same regardless of tolerance."""
        new_wkt, meta = simplify_wkt("POINT (13.32 52.50)", tolerance=0.001)
        assert "POINT" in new_wkt
        # Approximate coord-count helper rounds to 1 for a single point.
        assert meta["n_coords_after"] == meta["n_coords_before"]

    def test_invalid_wkt_returns_original_with_error_metadata(self) -> None:
        """The renderer must remain robust to malformed gold rows —
        we don't raise, we report the error and return verbatim."""
        new_wkt, meta = simplify_wkt("not actually wkt!", tolerance=0.001)
        assert new_wkt == "not actually wkt!"
        assert "error" in meta

    def test_zero_tolerance_is_identity_on_geom_type(self) -> None:
        """``tolerance=0`` is shapely's identity simplify — geometry
        survives, but `n_coords_after` may differ slightly from the
        approximate-counter perspective; we just assert geometry
        type is preserved."""
        new_wkt, meta = simplify_wkt(_DENSE_POLYGON_WKT, tolerance=0.0)
        assert meta["geometry_type"] == "Polygon"


class TestSimplifyTriples:
    """Public API: ``simplify_triples(triples, tolerance)`` walks the
    triple list, simplifies WKT objects, preserves wrapper shape, and
    returns per-triple stats."""

    def test_passthrough_for_non_wkt_triples(self) -> None:
        """``rdf:type`` and label triples are returned untouched."""
        triples: list[Triple] = [
            {"subject": "X", "predicate": "http://...#type", "object": "cafe"},
            {"subject": "X", "predicate": "http://...#label", "object": "Mochi"},
        ]
        out, stats = simplify_triples(triples, tolerance=0.001)
        assert out == triples
        assert stats == []

    def test_simplifies_polygon_at_tolerance(self) -> None:
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        out, stats = simplify_triples(triples, tolerance=0.001)
        assert len(stats) == 1
        assert stats[0].subject == "X"
        assert stats[0].original_chars == len(_DENSE_POLYGON_WKT)
        assert stats[0].simplified_chars < stats[0].original_chars
        assert stats[0].geometry_type == "Polygon"
        # The output triple must still carry the simplified WKT.
        assert "POLYGON" in out[0]["object"]

    def test_zero_tolerance_short_circuits_identity(self) -> None:
        """``tolerance=0`` (or ``None``) returns the list unchanged
        and emits zero stats — same as no-simplification behaviour
        before this module existed."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        out, stats = simplify_triples(triples, tolerance=0.0)
        assert out == triples
        assert stats == []
        out_none, stats_none = simplify_triples(triples, tolerance=None)  # type: ignore[arg-type]
        assert out_none == triples
        assert stats_none == []

    def test_preserves_kwg_quote_wrapping(self) -> None:
        """kwg gold rows have the WKT inside straight double-quotes —
        ``"POLYGON ((...))"``. The output must come back wrapped the
        same way, or downstream formatters mis-render it."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": f'"{_DENSE_POLYGON_WKT}"',
            },
        ]
        out, _ = simplify_triples(triples, tolerance=0.001)
        body = out[0]["object"]
        assert body.startswith('"') and body.endswith('"')
        assert "POLYGON" in body
        # The inner WKT must have been simplified (shorter than the input).
        assert len(body) < len(triples[0]["object"])

    def test_preserves_worldkg_typed_literal_wrapping(self) -> None:
        """worldkg gold rows have the typed-literal annotation
        ``"...WKT..."^^<...wktLiteral>``. Re-attach on output."""
        wktl_suffix = '^^<http://www.opengis.net/ont/geosparql#wktLiteral>'
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": f'"{_DENSE_POLYGON_WKT}"{wktl_suffix}',
            },
        ]
        out, _ = simplify_triples(triples, tolerance=0.001)
        body = out[0]["object"]
        assert body.endswith(wktl_suffix), (
            f"typed-literal suffix lost: {body[-80:]!r}"
        )
        assert '"' in body  # quote-wrap still present

    def test_passes_through_bare_osmh3kg_style(self) -> None:
        """osmh3kg gold rows are bare WKT (no quotes, no typed
        literal). Output must also be bare."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        out, _ = simplify_triples(triples, tolerance=0.001)
        body = out[0]["object"]
        assert not body.startswith('"')
        assert "^^<" not in body
        assert "POLYGON" in body

    def test_invalid_wkt_keeps_original_and_emits_no_stats(self) -> None:
        """Malformed WKT shouldn't fail the run — keep the triple
        verbatim, skip the stats entry, log a warning."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "not actually wkt",
            },
        ]
        out, stats = simplify_triples(triples, tolerance=0.001)
        assert out == triples
        assert stats == []

    def test_preserves_entity_role(self) -> None:
        """``Triple`` carries an optional ``entity_role`` field on the
        new flat-format share — must round-trip through simplification."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
                "entity_role": "a",
            },
        ]
        out, _ = simplify_triples(triples, tolerance=0.001)
        assert out[0].get("entity_role") == "a"


class TestOracleContextSimplification:
    """End-to-end integration: ``OracleContext`` runs the simplifier
    in ``__post_init__`` when a positive tolerance is supplied."""

    def test_construction_with_explicit_none_passthrough(self) -> None:
        """Explicit ``wkt_simplify_tolerance=None`` (or 0) disables
        the simplifier — useful for ablation runs that want raw gold
        WKT. The dataclass default is now 0.001 (matches
        EntityService); callers must opt OUT explicitly to bypass."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        ctx = OracleContext(
            mode="structured",
            gold_triples_map={"qid1": triples},
            wkt_simplify_tolerance=None,  # explicit opt-out
        )
        # No simplification: triples unchanged, stats dict empty.
        assert ctx.gold_triples_map["qid1"][0]["object"] == _DENSE_POLYGON_WKT
        assert ctx.simplify_stats == {}

    def test_default_construction_simplifies(self) -> None:
        """The dataclass default ``wkt_simplify_tolerance=0.001`` fires
        the simplifier without explicit configuration. Symmetric with
        ``EntityService.wkt_simplify_tolerance=0.001`` so oracle and
        tier-3a/3c use-kg modes produce comparable simplified WKT out
        of the box. Regression guard against accidentally re-flipping
        the default to None and silently re-introducing the
        WKT-bloat overflow that motivated the simplifier (commit
        19c9dfa) in the first place."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        ctx = OracleContext(
            mode="structured",
            gold_triples_map={"qid1": triples},
            # no wkt_simplify_tolerance: defaults to 0.001
        )
        assert ctx.wkt_simplify_tolerance == 0.001
        assert ctx.gold_triples_map["qid1"][0]["object"] != _DENSE_POLYGON_WKT
        assert "qid1" in ctx.simplify_stats
        assert len(ctx.simplify_stats["qid1"]) == 1

    def test_construction_with_tolerance_runs_simplifier(self) -> None:
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        ctx = OracleContext(
            mode="structured",
            gold_triples_map={"qid1": triples},
            wkt_simplify_tolerance=0.001,
        )
        # Simplification ran: WKT is shorter, stats populated.
        new_obj = ctx.gold_triples_map["qid1"][0]["object"]
        assert len(new_obj) < len(_DENSE_POLYGON_WKT)
        assert "qid1" in ctx.simplify_stats
        assert len(ctx.simplify_stats["qid1"]) == 1
        s = ctx.simplify_stats["qid1"][0]
        assert isinstance(s, SimplifyStats)
        assert s.simplified_chars < s.original_chars

    def test_explicit_zero_tolerance_passthrough(self) -> None:
        """``wkt_simplify_tolerance=0.0`` is functionally identical to
        passing ``None`` — bypass entirely."""
        triples: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        ctx = OracleContext(
            mode="structured",
            gold_triples_map={"qid1": triples},
            wkt_simplify_tolerance=0.0,
        )
        assert ctx.gold_triples_map["qid1"][0]["object"] == _DENSE_POLYGON_WKT
        assert ctx.simplify_stats == {}

    def test_multiple_questions_each_get_own_stats(self) -> None:
        """``simplify_stats`` is keyed by question_id; questions
        without WKT triples don't show up in the map."""
        triples_with_wkt: list[Triple] = [
            {
                "subject": "X",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": _DENSE_POLYGON_WKT,
            },
        ]
        triples_no_wkt: list[Triple] = [
            {"subject": "Y", "predicate": "http://...#type", "object": "cafe"},
        ]
        ctx = OracleContext(
            mode="structured",
            gold_triples_map={
                "qid_with_wkt": triples_with_wkt,
                "qid_no_wkt": triples_no_wkt,
            },
            wkt_simplify_tolerance=0.001,
        )
        assert "qid_with_wkt" in ctx.simplify_stats
        assert "qid_no_wkt" not in ctx.simplify_stats
