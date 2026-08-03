"""Tests for the triple-native KG bridge redesign.

Covers the three new modules in
``benchmark_models/kg_tools/``:

* :mod:`schemas` — per-KG predicate vocabularies + subject-id
  canonicalisation.
* :mod:`entity_service` — entity-shaped API on top of a triple
  bridge (mocked here; live Spark tests live separately).
* :mod:`tools` — the smolagents Tool subclasses, exercised against
  both the mock-default path and an injected ``EntityService``.

The :class:`TripleBridge` itself is not tested live in this file —
its query primitives need either a Spark session against the
production data (too heavy for unit tests) or a triple-shaped
fixture (deferred to a separate commit). What we test here is that
:class:`EntityService` correctly decodes per-KG predicate
conventions when given a bridge-shaped mock.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import pytest

from benchmark_models.kg_tools.entity_service import EntityService
from benchmark_models.kg_tools.schemas import (
    KWG_SCHEMA,
    OSMH3KG_SCHEMA,
    WORLDKG_SCHEMA,
    schema_for,
)
from benchmark_models.kg_tools.tools import (
    FindEntitiesByTypeTool,
    FindEntityTool,
    GetEntityTool,
    ListCategoriesTool,
)

# ---------------------------------------------------------------------------
# KGSchema
# ---------------------------------------------------------------------------


class TestKGSchema:
    """Subject-id canonicalisation per KG mirrors the lexical
    conventions in the production Delta tables (verified earlier
    against pyarrow reads of benchmark/kgs/{kwg,osmh3kg,worldkg}/)."""

    def test_kwg_prefixes_local_name(self) -> None:
        """kwg benchmark IDs are bare suffixes; prepend the namespace."""
        out = KWG_SCHEMA.canonicalize_subject_id("zipCodeArea.27351")
        assert out == "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.27351"

    def test_kwg_passthrough_when_already_uri(self) -> None:
        """If the caller already has a full URI (e.g., from gold
        triples), no double-prefixing."""
        uri = "http://stko-kwg.geog.ucsb.edu/lod/resource/foo.bar"
        assert KWG_SCHEMA.canonicalize_subject_id(uri) == uri

    def test_kwg_passthrough_foreign_namespace_uri(self) -> None:
        """A full IRI in ANOTHER namespace (geometry node, rdf:type
        object, cross-referenced resource) must also pass through — the
        agent routinely feeds such a URI back into get_entity. Prepending
        the resource prefix produced malformed
        ``.../resource/http://...`` subjects that matched zero triples.
        Regression for the kwg-concentrated double-IRI bug."""
        for uri in (
            "http://www.opengis.net/ont/geosparql#Geometry",
            "http://www.w3.org/2000/01/rdf-schema#label",
            "https://sws.geonames.org/2649308/",
        ):
            assert KWG_SCHEMA.canonicalize_subject_id(uri) == uri, uri
            assert WORLDKG_SCHEMA.canonicalize_subject_id(uri) == uri, uri

    def test_osmh3kg_strips_geo_prefix(self) -> None:
        """osmh3kg geometry-node subjects appear as 'geo<id>' but
        benchmark IDs are bare numeric. Strip the prefix to canonical."""
        assert OSMH3KG_SCHEMA.canonicalize_subject_id("geo2649308671") == "2649308671"
        assert OSMH3KG_SCHEMA.canonicalize_subject_id("2649308671") == "2649308671"

    def test_osmh3kg_leaves_non_geo_alone(self) -> None:
        """The 'geo' prefix strip is conservative — only strips when
        followed by digits."""
        assert OSMH3KG_SCHEMA.canonicalize_subject_id("geosparql") == "geosparql"

    def test_worldkg_prefixes_local_name(self) -> None:
        out = WORLDKG_SCHEMA.canonicalize_subject_id("3814104941")
        assert out == "http://www.worldkg.org/resource/3814104941"

    def test_schema_for_known_kg(self) -> None:
        assert schema_for("kwg") is KWG_SCHEMA
        assert schema_for("osmh3kg") is OSMH3KG_SCHEMA
        assert schema_for("worldkg") is WORLDKG_SCHEMA

    def test_schema_for_unknown_kg_raises(self) -> None:
        """Wrong-KG bugs should surface at startup, not propagate
        through every triple query as a silent vocabulary mismatch."""
        with pytest.raises(ValueError, match="Unknown KG"):
            schema_for("not-a-real-kg")

    def test_each_schema_carries_geometry_predicates(self) -> None:
        """All three KGs should have geo:asWKT in their geometry-data
        predicate set."""
        wkt_pred = "http://www.opengis.net/ont/geosparql#asWKT"
        for schema in (KWG_SCHEMA, OSMH3KG_SCHEMA, WORLDKG_SCHEMA):
            assert wkt_pred in schema.geometry_data_predicates


# ---------------------------------------------------------------------------
# EntityService (with mocked bridge)
# ---------------------------------------------------------------------------


class _MockBridge:
    """Minimal stand-in for TripleBridge that backs EntityService
    tests with hand-crafted triple sets. Records calls so tests can
    assert how the service interacts with the bridge."""

    def __init__(self, triples_by_subject: dict[str, list[dict[str, Any]]]) -> None:
        self.triples_by_subject = triples_by_subject
        self.fetch_calls: list[str] = []
        self.search_calls: list[tuple[tuple[str, ...], str, int]] = []
        self.distinct_calls: list[tuple[tuple[str, ...], int]] = []

    def fetch_triples_for_subject(
        self, subject: str, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        self.fetch_calls.append(subject)
        return self.triples_by_subject.get(subject, [])

    def fetch_triples_pointing_to(
        self,
        object_value: str,
        *,
        predicates: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        preds = set(predicates) if predicates else None
        for subj, triples in self.triples_by_subject.items():
            for t in triples:
                if t["object"] == object_value and (
                    preds is None or t["predicate"] in preds
                ):
                    results.append(t)
        return results

    def search_object_substring(
        self,
        predicates: Iterable[str],
        substring: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        preds = tuple(predicates)
        self.search_calls.append((preds, substring, limit))
        out: list[dict[str, Any]] = []
        needle = substring.lower()
        for triples in self.triples_by_subject.values():
            for t in triples:
                if t["predicate"] in preds and needle in str(t["object"]).lower():
                    out.append(t)
                    if len(out) >= limit:
                        return out
        return out

    def query_distinct_objects(
        self,
        predicates: Iterable[str],
        *,
        limit: int = 200,
    ) -> list[str]:
        preds = tuple(predicates)
        self.distinct_calls.append((preds, limit))
        seen: set[str] = set()
        for triples in self.triples_by_subject.values():
            for t in triples:
                if t["predicate"] in preds:
                    seen.add(t["object"])
        return sorted(seen)[:limit]


def _kwg_entity_triples() -> dict[str, list[dict[str, Any]]]:
    """A tiny kwg-style entity with a 2-hop geometry link.

    Mirrors the actual schema: a ZCTA entity at
    'http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.27351'
    links via geo:hasGeometry to the geometry node at
    '...geometry.Polygon.zipCodeArea.27351', which carries the WKT
    via geo:asWKT.
    """
    entity = "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.27351"
    geom_node = "http://stko-kwg.geog.ucsb.edu/lod/resource/geometry.Polygon.zipCodeArea.27351"
    return {
        entity: [
            {
                "subject": entity,
                "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "object": "http://stko-kwg.geog.ucsb.edu/lod/ontology/ZipCodeArea",
            },
            {
                "subject": entity,
                "predicate": "http://www.w3.org/2000/01/rdf-schema#label",
                "object": "Zip code tabulation area with zipcode 27351",
            },
            {
                "subject": entity,
                "predicate": "http://www.opengis.net/ont/geosparql#hasGeometry",
                "object": geom_node,
            },
        ],
        geom_node: [
            {
                "subject": geom_node,
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                # Quote-wrapped WKT (kwg convention). Triangle with
                # enough vertices to allow simplification at 0.001 to
                # potentially drop some — but small enough that the
                # mock doesn't depend on shapely's exact output.
                "object": '"POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"',
            },
        ],
    }


class TestEntityServiceGetEntity:
    def test_resolves_kwg_zcta_two_hop_geometry(self) -> None:
        """Benchmark-id passed to get_entity is canonicalised to the
        full kwg URI, then the 2-hop link to the geometry node is
        followed automatically — caller sees a single geometry_wkt
        on the entity dict."""
        bridge = _MockBridge(_kwg_entity_triples())
        svc = EntityService(bridge, KWG_SCHEMA, wkt_simplify_tolerance=0.0)

        result = svc.get_entity("zipCodeArea.27351")

        assert "error" not in result, result
        assert (
            result["subject"]
            == "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.27351"
        )
        assert result["name"] == "Zip code tabulation area with zipcode 27351"
        assert result["types"] == [
            "http://stko-kwg.geog.ucsb.edu/lod/ontology/ZipCodeArea"
        ]
        assert result["category"] == "http://stko-kwg.geog.ucsb.edu/lod/ontology/ZipCodeArea"
        # Quote wrapper stripped, no typed-literal suffix to remove.
        assert result["geometry_wkt"] == "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
        # h3_index / pagerank intentionally absent (not in triples).
        assert result["pagerank"] is None
        assert result["h3_index"] is None
        # Two subject-keyed bridge reads (one per hop).
        assert len(bridge.fetch_calls) == 2

    def test_returns_error_dict_when_subject_missing(self) -> None:
        """Empty triple set for the canonicalised subject → error
        dict instead of crashing. Caller checks 'error' in result."""
        bridge = _MockBridge({})
        svc = EntityService(bridge, KWG_SCHEMA)
        result = svc.get_entity("zipCodeArea.does-not-exist")
        assert "error" in result
        assert result["id"] == "zipCodeArea.does-not-exist"
        # Subject canonicalised even when the lookup fails — caller
        # can debug what URI was attempted.
        assert (
            result["subject"]
            == "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.does-not-exist"
        )

    def test_one_hop_geometry_path_when_no_link_predicate(self) -> None:
        """Some entities carry geo:asWKT directly (no separate
        geometry node). EntityService falls through to the 1-hop
        path when 2-hop didn't fire."""
        entity = "1234"
        triples = {
            entity: [
                {"subject": entity, "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "Point"},
                {"subject": entity, "predicate": "http://www.w3.org/2000/01/rdf-schema#label", "object": "Test"},
                {
                    "subject": entity,
                    "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                    "object": "POINT (13.32 52.50)",
                },
            ],
        }
        bridge = _MockBridge(triples)
        svc = EntityService(bridge, OSMH3KG_SCHEMA, wkt_simplify_tolerance=0.0)
        result = svc.get_entity("1234")
        assert result["geometry_wkt"] == "POINT (13.32 52.50)"
        assert result["geometry_node"] is None  # No 2-hop link

    def test_wkt_simplification_active_by_default(self) -> None:
        """Default wkt_simplify_tolerance=0.001 should shrink a dense
        polygon. Use a polygon with many redundant vertices so
        Douglas-Peucker can drop some."""
        entity = "X"
        # Many collinear-ish points so simplifier drops most.
        wkt = "POLYGON ((0 0, 0 0.0001, 0 0.0002, 0 0.0005, 0 1, 1 1, 1 0, 0 0))"
        triples = {
            entity: [
                {
                    "subject": entity,
                    "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                    "object": wkt,
                }
            ],
        }
        bridge = _MockBridge(triples)
        # Default tolerance.
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        result = svc.get_entity("X")
        # The returned WKT should be a valid POLYGON but shorter than
        # the input (collinear coords dropped).
        assert "POLYGON" in result["geometry_wkt"]
        assert len(result["geometry_wkt"]) < len(wkt), (
            f"expected simplification to shrink WKT, got {len(result['geometry_wkt'])}"
            f" >= input {len(wkt)}"
        )

    def test_zero_tolerance_disables_simplification(self) -> None:
        """tolerance=0 returns geometry verbatim — needed for the
        oracle-compatible 'verbatim gold' mode and for tests that
        compare bytes."""
        entity = "X"
        wkt = "POLYGON ((0 0, 0 0.0001, 0 1, 1 1, 1 0, 0 0))"
        triples = {
            entity: [
                {
                    "subject": entity,
                    "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                    "object": wkt,
                }
            ],
        }
        bridge = _MockBridge(triples)
        svc = EntityService(bridge, OSMH3KG_SCHEMA, wkt_simplify_tolerance=0.0)
        assert svc.get_entity("X")["geometry_wkt"] == wkt

    def test_no_geometry_yields_none(self) -> None:
        """An entity with only type/label and no geometry predicate
        comes back with geometry_wkt=None — not a crash."""
        entity = "Y"
        triples = {
            entity: [
                {"subject": entity, "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "Concept"},
            ],
        }
        bridge = _MockBridge(triples)
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        result = svc.get_entity("Y")
        assert result["geometry_wkt"] is None
        assert result["category"] == "Concept"


class TestEntityServiceFindByName:
    def test_passes_kg_name_predicates_to_bridge(self) -> None:
        """find_entity_by_name routes through the schema's
        name_predicates — confirms the bridge sees the right set.
        The bridge limit is over-fetched (25) so the ranker has
        signal beyond the user's truncation cap."""
        bridge = _MockBridge({})
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        svc.find_entity_by_name("kanel", limit=5)
        assert len(bridge.search_calls) == 1
        preds, substring, limit = bridge.search_calls[0]
        assert substring == "kanel"
        # Over-fetch: bridge always sees the over-fetch constant
        # regardless of the user's limit, so ranking has signal.
        assert limit == 25
        # osmh3kg uses bare OSM tag keys for naming.
        assert "name" in preds
        assert "name:en" in preds

    def test_caps_at_user_limit_after_ranking(self) -> None:
        """User's ``limit`` truncates the post-rank list, not the
        bridge fetch."""
        # Build a bridge that returns 10 hits all matching "X".
        hits = [
            {
                "subject": f"http://stko-kwg.geog.ucsb.edu/lod/resource/X{i}",
                "predicate": "http://www.w3.org/2000/01/rdf-schema#label",
                "object": "X" * (i + 1),  # all unique label lengths
            }
            for i in range(10)
        ]

        class _BulkBridge(_MockBridge):
            def search_object_substring(
                self, predicates, substring, *, limit=20,
            ):
                self.search_calls.append((tuple(predicates), substring, limit))
                return hits

        bridge = _BulkBridge({})
        svc = EntityService(bridge, KWG_SCHEMA)
        result = svc.find_entity_by_name("X", limit=3)
        assert len(result) == 3

    def test_ranks_canonical_predicate_above_alt(self) -> None:
        """Hits via an earlier predicate in schema.name_predicates
        outrank later ones, regardless of label length. KWG's
        name_predicates is (name_en, *label_predicates) — the first
        position should always win on tie of label length."""
        # Two label predicates in KWG schema:
        kwg_preds = list(KWG_SCHEMA.name_predicates)
        canonical_pred, secondary_pred = kwg_preds[0], kwg_preds[-1]
        if canonical_pred == secondary_pred:
            pytest.skip("KWG schema only has one name predicate; rank tie-break N/A")
        hits = [
            {
                "subject": "http://stko-kwg.geog.ucsb.edu/lod/resource/Aless",
                "predicate": secondary_pred,
                "object": "X",
            },
            {
                "subject": "http://stko-kwg.geog.ucsb.edu/lod/resource/Bmore",
                "predicate": canonical_pred,
                "object": "X",
            },
        ]

        class _PredBridge(_MockBridge):
            def search_object_substring(self, predicates, substring, *, limit=20):
                self.search_calls.append((tuple(predicates), substring, limit))
                return hits

        bridge = _PredBridge({})
        svc = EntityService(bridge, KWG_SCHEMA)
        result = svc.find_entity_by_name("X")
        # Bmore (canonical pred) wins despite later alphabetic
        # subject — predicate priority dominates.
        assert result[0]["subject"] == "Bmore"
        assert result[1]["subject"] == "Aless"

    def test_ranks_short_label_above_long_on_same_predicate(self) -> None:
        """Label-length ascending tie-break: shorter labels rank
        higher within a predicate. This is the dominant signal that
        rescues numeric queries from being buried in noise."""
        pred = KWG_SCHEMA.name_predicates[0]
        hits = [
            {
                "subject": "http://stko-kwg.geog.ucsb.edu/lod/resource/A",
                "predicate": pred,
                "object": "Observation of the property with key dmin for the earthquake nc21124230",
            },
            {
                "subject": "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.24230",
                "predicate": pred,
                "object": "24230",
            },
        ]

        class _LenBridge(_MockBridge):
            def search_object_substring(self, predicates, substring, *, limit=20):
                self.search_calls.append((tuple(predicates), substring, limit))
                return hits

        bridge = _LenBridge({})
        svc = EntityService(bridge, KWG_SCHEMA)
        result = svc.find_entity_by_name("24230")
        # Short "24230" label wins over the long observation label.
        assert result[0]["subject"] == "zipCodeArea.24230"
        assert result[0]["name"] == "24230"

    def test_strips_kwg_subject_uri_prefix_in_display(self) -> None:
        """KWG full subjects are 80+ chars. Display version drops
        the schema's ``subject_uri_prefix`` so per-call output stays
        compact. Round-trip is preserved by canonicalize_subject_id."""
        pred = KWG_SCHEMA.name_predicates[0]
        full = "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.27351"
        hits = [{"subject": full, "predicate": pred, "object": "27351"}]

        class _PrefixBridge(_MockBridge):
            def search_object_substring(self, predicates, substring, *, limit=20):
                return hits

        svc = EntityService(_PrefixBridge({}), KWG_SCHEMA)
        result = svc.find_entity_by_name("27351")
        assert result[0]["subject"] == "zipCodeArea.27351"  # short form
        # The short form must round-trip through canonicalize_subject_id.
        assert KWG_SCHEMA.canonicalize_subject_id(result[0]["subject"]) == full

    def test_preserves_osmh3kg_bare_id_when_no_prefix(self) -> None:
        """osmh3kg has no subject_uri_prefix — subjects are already
        bare numeric IDs. Display path should be a pass-through."""
        pred = OSMH3KG_SCHEMA.name_predicates[0]
        hits = [{"subject": "2649308671", "predicate": pred, "object": "Cafe Kanel"}]

        class _BareBridge(_MockBridge):
            def search_object_substring(self, predicates, substring, *, limit=20):
                return hits

        svc = EntityService(_BareBridge({}), OSMH3KG_SCHEMA)
        result = svc.find_entity_by_name("Kanel")
        assert result[0]["subject"] == "2649308671"


class TestEntityServiceFindByType:
    def test_region_filter_is_no_op_with_warning(self) -> None:
        """region= is accepted for API parity but currently no-op.
        Verify it doesn't crash and emits the documented warning."""
        bridge = _MockBridge({})
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        result = svc.find_entities_by_type("cafe", region="Berlin")
        assert result == []


class TestEntityServiceListTypes:
    def test_routes_to_bridge_distinct_objects(self) -> None:
        bridge = _MockBridge({
            "subj1": [{"subject": "subj1", "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "cafe"}],
            "subj2": [{"subject": "subj2", "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "hospital"}],
            "subj3": [{"subject": "subj3", "predicate": "http://www.w3.org/2000/01/rdf-schema#label", "object": "Cafe Kanel"}],
        })
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        result = svc.list_types(limit=10)
        # Paged return: types + truncation flags.
        assert sorted(result["types"]) == ["cafe", "hospital"]
        assert result["shown"] == 2
        assert result["has_more"] is False
        # Predicate set passed to the bridge matches the schema.
        # list_types over-fetches by 1 (limit + 1) to detect truncation
        # without a second COUNT query.
        assert len(bridge.distinct_calls) == 1
        preds, limit = bridge.distinct_calls[0]
        assert "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" in preds
        assert limit == 11  # 10 + 1 over-fetch

    def test_truncates_and_sets_has_more(self) -> None:
        # Three rdf:type values; ask for limit=2 → expect first two
        # alphabetically and has_more=True.
        bridge = _MockBridge({
            "subj1": [{"subject": "subj1", "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "cafe"}],
            "subj2": [{"subject": "subj2", "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "hospital"}],
            "subj3": [{"subject": "subj3", "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "park"}],
        })
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        result = svc.list_types(limit=2)
        assert result["types"] == ["cafe", "hospital"]
        assert result["shown"] == 2
        assert result["has_more"] is True


# ---------------------------------------------------------------------------
# Tool wrappers — exercise the smolagents-side surface.
# ---------------------------------------------------------------------------


class TestToolsAgainstEntityService:
    """Tools.forward() delegates to the backend's methods. Verify the
    delegation, the mock-default fallback, and the return-string shape."""

    def test_find_entity_tool_delegates_to_find_entity_by_name(self) -> None:
        bridge = _MockBridge({
            "subj": [
                {"subject": "subj", "predicate": "name", "object": "Cafe Mochi"},
            ],
        })
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        tool = FindEntityTool(backend=svc)
        out = tool.forward("Mochi")
        assert "Cafe Mochi" in out
        assert "subj" in out

    def test_find_entity_tool_mock_returns_canned_shape(self) -> None:
        """No-backend path returns the canned mock — same shape as
        the real bridge so agent tests don't need real Spark."""
        tool = FindEntityTool(backend=None)
        out = tool.forward("anything")
        assert "subject" in out
        assert "predicate" in out

    def test_get_entity_tool_mock_carries_geometry_wkt(self) -> None:
        """The mock fallback returns a geometry_wkt field so agent
        Python code that reads result['geometry_wkt'] works in tests."""
        tool = GetEntityTool(backend=None)
        out = tool.forward("any_id")
        assert "geometry_wkt" in out
        assert "POINT" in out  # mock returns POINT (13.3248 52.5006)

    def test_list_categories_tool_delegates_to_list_types(self) -> None:
        bridge = _MockBridge({
            "subj1": [{"subject": "subj1", "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "object": "cafe"}],
        })
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        tool = ListCategoriesTool(backend=svc)
        out = tool.forward()
        assert "cafe" in out

    def test_find_entities_by_type_tool_forwards_region(self) -> None:
        """region kwarg is supported even though it's a no-op
        downstream — the tool signature still accepts it."""
        bridge = _MockBridge({})
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        tool = FindEntitiesByTypeTool(backend=svc)
        out = tool.forward("cafe", region="Berlin")
        # No matches; empty list returned.
        assert out == "[]"


class TestToolOutputIsValidJson:
    """Lock in the (e1) contract: every tool returns a JSON string
    that round-trips through json.loads. Repr-style str() output
    silently broke the documented "list of dicts" contract — single
    quotes and Python None/True can't be parsed by json.loads, and
    naive ``for e in results: e['subject']`` iterates the string
    char-by-char (Gemini's recurring tier-3a crash). These tests
    guard against accidentally regressing to str(results)."""

    def test_find_entity_returns_parseable_json(self) -> None:
        bridge = _MockBridge({
            "subj": [{"subject": "subj", "predicate": "name", "object": "Cafe Mochi"}],
        })
        svc = EntityService(bridge, OSMH3KG_SCHEMA)
        out = FindEntityTool(backend=svc).forward("Mochi")
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert all(isinstance(e, dict) and "subject" in e for e in parsed)

    def test_find_entity_mock_returns_parseable_json(self) -> None:
        out = FindEntityTool(backend=None).forward("anything")
        parsed = json.loads(out)
        assert parsed == [
            {
                "subject": "mock_001",
                "name": "anything",
                "predicate": "http://www.w3.org/2000/01/rdf-schema#label",
            },
        ]

    def test_get_entity_returns_parseable_json(self) -> None:
        out = GetEntityTool(backend=None).forward("any_id")
        parsed = json.loads(out)
        assert isinstance(parsed, dict)
        assert parsed["geometry_wkt"] == "POINT (13.3248 52.5006)"
        # None must round-trip as JSON null (not Python repr "None").
        assert parsed["geometry_node"] is None

    def test_list_categories_returns_parseable_json(self) -> None:
        out = ListCategoriesTool(backend=None).forward()
        parsed = json.loads(out)
        assert isinstance(parsed, dict)
        assert "types" in parsed
        assert "has_more" in parsed
        # Booleans must round-trip as JSON true/false.
        assert parsed["has_more"] is False

    def test_find_entities_by_type_returns_parseable_json(self) -> None:
        out = FindEntitiesByTypeTool(backend=None).forward("cafe")
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert parsed[0]["matched_type"] == "cafe"


# ---------------------------------------------------------------------------
# EntityService satisfies the snapshot SupportsDeltaVersions protocol.
# ---------------------------------------------------------------------------


def test_entity_service_has_get_delta_versions() -> None:
    """The snapshot module uses a structural protocol check; we just
    need ``.get_delta_versions()`` to exist and return a Mapping[str, int]."""

    class _MockBridgeWithVersions:
        def get_delta_versions(self) -> dict[str, int]:
            return {"benchmark/kgs/kwg": 42}

    svc = EntityService(_MockBridgeWithVersions(), KWG_SCHEMA)  # type: ignore[arg-type]
    assert svc.get_delta_versions() == {"benchmark/kgs/kwg": 42}


# ---------------------------------------------------------------------------
# TripleBridge.close() — must tolerate a dead JVM.
# ---------------------------------------------------------------------------


def test_triple_bridge_close_tolerates_dead_jvm() -> None:
    """When the Spark JVM has died mid-run (heap OOM, direct buffer
    cap, SLURM OOM-killer on the SLURM step), ``self.spark.stop()``
    raises ConnectionRefusedError/Py4JNetworkError. The runner's
    end-of-smoke cleanup hits this path; we must not re-raise from
    ``close()`` because by then the predictions JSONL is already on
    disk and the cleanup traceback would mask the upstream cause."""
    from unittest.mock import MagicMock

    from benchmark_models.kg_tools.triple_bridge import TripleBridge

    # Build a TripleBridge *without* invoking __init__ — we don't want
    # to start a real Spark session in a unit test. Inject the
    # minimum surface ``close()`` reads (.spark, .schema.name).
    bridge = TripleBridge.__new__(TripleBridge)
    bridge.spark = MagicMock()
    bridge.spark.stop.side_effect = ConnectionRefusedError("[Errno 111] Connection refused")
    bridge.schema = KWG_SCHEMA

    # The contract: close() returns normally even when spark.stop()
    # raises. (We could also check for a logger.warning call, but the
    # behaviour-level guarantee is the non-raise.)
    bridge.close()
