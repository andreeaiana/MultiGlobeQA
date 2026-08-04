"""Tests for :mod:`benchmark_models.oracle` — loaders and formatters.

Covers:

* ``load_gold_triples`` against the committed sample file and a
  handcrafted fixture with known contents.
* ``format_structured`` / ``format_raw`` / ``format_verbalized`` on a
  small fixture where the expected output can be asserted exactly.
* ``format_verbalized`` fallback for predicates outside the standard
  ``rdf:type`` / ``geo:hasGeometry`` / ``geo:asWKT`` set.
* ``assert_coverage`` raises when question ids are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_models.oracle import (
    OracleContext,
    Triple,
    assert_coverage,
    format_raw,
    format_structured,
    format_triples,
    format_verbalized,
    load_gold_triples,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_triples() -> list[Triple]:
    """One question's worth of triples, matching the shape in the
    committed sample file (one cafe + one hospital, both with
    hasGeometry + asWKT)."""
    return [
        {
            "subject": "12147663074",
            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "object": "cafe",
        },
        {
            "subject": "12147663074",
            "predicate": "http://www.opengis.net/ont/geosparql#hasGeometry",
            "object": "geo12147663074",
        },
        {
            "subject": "geo12147663074",
            "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
            "object": "POINT (13.32 52.50)",
        },
        {
            "subject": "4148219",
            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "object": "hospital",
        },
        {
            "subject": "4148219",
            "predicate": "http://www.opengis.net/ont/geosparql#hasGeometry",
            "object": "geo4148219",
        },
        {
            "subject": "geo4148219",
            "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
            "object": "POLYGON ((13.34 52.46, 13.35 52.46, 13.35 52.47, 13.34 52.47))",
        },
    ]


# ---------------------------------------------------------------------------
# load_gold_triples
# ---------------------------------------------------------------------------


class TestLoadGoldTriples:
    """Loader against both real and synthetic inputs.

    The gold-triples format is flat one-row-per-triple with explicit
    ``entity_role`` provenance, not a double-encoded JSON-string per
    question. The loader groups rows by ``question_id`` into
    ``list[Triple]``.
    """

    def test_loads_a_real_share_shard(self) -> None:
        """The committed gold-triples shards under ``benchmark/raw_eng/`` load
        cleanly. Structural assertions only — row counts shift with
        every benchmark release."""
        # Pick a small but non-empty shard to keep the test fast.
        # Some shards (e.g. kwg/sfid_j) are empty during in-flight
        # generation; skip them so the test exercises a populated shard.
        candidates = list(
            (Path(__file__).parents[2] / "benchmark/raw_eng").glob(
                "*/sfid_*/gold_triples.jsonl"
            )
        )
        non_empty = [
            p for p in candidates
            if sum(f.stat().st_size for f in p.rglob("part-*") if f.is_file()) > 0
        ]
        if not non_empty:
            pytest.skip("no non-empty gold_triples.jsonl shard dir found")
        # Smallest non-empty by total bytes so the test is cheap.
        shard_dir = min(
            non_empty,
            key=lambda p: sum(f.stat().st_size for f in p.rglob("part-*") if f.is_file()),
        )
        m = load_gold_triples(shard_dir)
        assert len(m) > 0
        # Every value is a list of triples with the canonical keys.
        for qid in list(m)[:20]:
            triples = m[qid]
            assert isinstance(qid, str)
            assert isinstance(triples, list)
            assert len(triples) >= 1
            for t in triples:
                # subject / predicate / object are required; entity_role
                # is optional. The loader always populates entity_role
                # for the new flat format, so it should be present.
                assert "subject" in t
                assert "predicate" in t
                assert "object" in t
                assert "entity_role" in t

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError, not a silent empty map."""
        with pytest.raises(FileNotFoundError):
            load_gold_triples(tmp_path / "nope.json")

    def test_handcrafted_flat_fixture(self, tmp_path: Path) -> None:
        """Round-trip a tiny flat-format JSONL through the loader.

        The new share emits one row per triple with explicit
        ``entity_role``; the loader groups rows sharing
        ``question_id`` into a list preserving file order.
        """
        rows = [
            {"question_id": "q1", "entity_role": "a", "subject": "s1", "predicate": "p1", "object": "o1"},
            {"question_id": "q1", "entity_role": "b", "subject": "s2", "predicate": "p2", "object": "o2"},
            {"question_id": "q2", "entity_role": "a", "subject": "s3", "predicate": "p3", "object": "o3"},
        ]
        path = tmp_path / "tiny.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        m = load_gold_triples(path)
        assert set(m) == {"q1", "q2"}
        assert len(m["q1"]) == 2
        assert len(m["q2"]) == 1
        # entity_role propagates to the Triple dict.
        assert m["q1"][0]["entity_role"] == "a"
        assert m["q1"][1]["entity_role"] == "b"

    def test_geo_subject_normalization(self, tmp_path: Path) -> None:
        """``geo<digits>`` osmh3kg subjects are stripped to bare digits.

        Without normalisation the same entity appears under two
        distinct ids (`geo12345` from asWKT triples vs `12345` from
        image_url triples), which breaks downstream entity grouping
        in the verbalised formatter.
        """
        rows = [
            {"question_id": "q1", "entity_role": "a", "subject": "geo12345", "predicate": "http://x/asWKT", "object": "POINT (1 2)"},
            {"question_id": "q1", "entity_role": "a", "subject": "12345", "predicate": "multimodal:image_url", "object": "http://example/img.jpg"},
        ]
        path = tmp_path / "geo.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        m = load_gold_triples(path)
        # Both rows now share the same canonical subject.
        assert {t["subject"] for t in m["q1"]} == {"12345"}

    def test_geosparql_uri_not_stripped(self, tmp_path: Path) -> None:
        """``geosparql:...`` URIs (and any non-digit-only ``geo`` prefix)
        survive subject normalisation untouched."""
        rows = [
            {"question_id": "q1", "entity_role": "a", "subject": "geosparql:wktLiteral",
             "predicate": "p", "object": "o"},
        ]
        path = tmp_path / "uri.jsonl"
        path.write_text(json.dumps(rows[0]) + "\n")
        m = load_gold_triples(path)
        assert m["q1"][0]["subject"] == "geosparql:wktLiteral"

    def test_loads_directory_of_shards(self, tmp_path: Path) -> None:
        """A leaf shard directory of ``part-*.json`` files loads as
        a single grouped map (matches the Spark output convention)."""
        shard_dir = tmp_path / "gold_triples.jsonl"
        shard_dir.mkdir()
        rows_a = [
            {"question_id": "q1", "entity_role": "a", "subject": "s", "predicate": "p", "object": "o"}
        ]
        rows_b = [
            {"question_id": "q2", "entity_role": "a", "subject": "s2", "predicate": "p", "object": "o"}
        ]
        (shard_dir / "part-00000.json").write_text(
            "\n".join(json.dumps(r) for r in rows_a) + "\n"
        )
        (shard_dir / "part-00001.json").write_text(
            "\n".join(json.dumps(r) for r in rows_b) + "\n"
        )
        m = load_gold_triples(shard_dir)
        assert set(m) == {"q1", "q2"}

    def test_dedupes_duplicate_triples_per_qid(self, tmp_path: Path) -> None:
        """Identical (role, subject, predicate, object) rows within the
        same ``question_id`` collapse to a single triple. Upstream
        materialization sometimes emits duplicates (KG triples asserted
        from multiple ingest paths and unioned without dedup); they
        bloat oracle prompts without adding semantic content.
        """
        rows = [
            # q1: triple appears 3× — should collapse to 1.
            {"question_id": "q1", "entity_role": "a",
             "subject": "s1", "predicate": "p1", "object": "POLYGON ((...))"},
            {"question_id": "q1", "entity_role": "a",
             "subject": "s1", "predicate": "p1", "object": "POLYGON ((...))"},
            {"question_id": "q1", "entity_role": "a",
             "subject": "s1", "predicate": "p1", "object": "POLYGON ((...))"},
            # q1: a distinct second triple — kept.
            {"question_id": "q1", "entity_role": "b",
             "subject": "s2", "predicate": "p1", "object": "POLYGON ((...))"},
            # q2: independent question — its own dedup.
            {"question_id": "q2", "entity_role": "a",
             "subject": "s3", "predicate": "p1", "object": "POINT (1 2)"},
            {"question_id": "q2", "entity_role": "a",
             "subject": "s3", "predicate": "p1", "object": "POINT (1 2)"},
        ]
        path = tmp_path / "dups.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        m = load_gold_triples(path)
        # q1: 4 raw rows → 2 unique (role=a + role=b).
        assert len(m["q1"]) == 2
        # q2: 2 raw rows → 1 unique.
        assert len(m["q2"]) == 1

    def test_same_triple_under_different_role_kept(self, tmp_path: Path) -> None:
        """The same (subject, predicate, object) under role=a vs role=b
        is preserved — different role makes the triple semantically
        distinct from the oracle prompt's perspective."""
        rows = [
            {"question_id": "q1", "entity_role": "a",
             "subject": "shared_s", "predicate": "p", "object": "o"},
            {"question_id": "q1", "entity_role": "b",
             "subject": "shared_s", "predicate": "p", "object": "o"},
        ]
        path = tmp_path / "roles.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        m = load_gold_triples(path)
        # Both kept — role distinguishes them.
        assert len(m["q1"]) == 2
        roles = {t.get("entity_role") for t in m["q1"]}
        assert roles == {"a", "b"}


# ---------------------------------------------------------------------------
# format_structured
# ---------------------------------------------------------------------------


class TestFormatStructured:
    def test_contains_prefixes_header(self, sample_triples: list[Triple]) -> None:
        """The structured form includes a PREFIXES block and short predicates."""
        out = format_structured(sample_triples)
        assert "PREFIXES:" in out
        assert "rdf: http://www.w3.org/1999/02/22-rdf-syntax-ns#" in out
        assert "geo: http://www.opengis.net/ont/geosparql#" in out
        assert "TRIPLES:" in out

    def test_short_predicates_used(self, sample_triples: list[Triple]) -> None:
        """Long predicate URIs are replaced with short prefixed forms."""
        out = format_structured(sample_triples)
        # Full URIs should not appear inside the TRIPLES body (they're
        # only in the PREFIXES header).
        body = out.split("TRIPLES:", 1)[1]
        assert "http://www.w3.org/1999/02" not in body
        assert "rdf:type" in body
        assert "geo:hasGeometry" in body
        assert "geo:asWKT" in body

    def test_output_is_valid_json_block(self, sample_triples: list[Triple]) -> None:
        """The TRIPLES block parses back as a JSON array."""
        out = format_structured(sample_triples)
        body = out.split("TRIPLES:\n", 1)[1]
        parsed = json.loads(body)
        assert isinstance(parsed, list)
        assert len(parsed) == len(sample_triples)


# ---------------------------------------------------------------------------
# format_raw
# ---------------------------------------------------------------------------


class TestFormatRaw:
    def test_full_uris_and_dot_terminators(
        self, sample_triples: list[Triple]
    ) -> None:
        """Raw form uses full URIs in angle brackets and dot-terminated lines."""
        out = format_raw(sample_triples)
        lines = out.splitlines()
        assert len(lines) == len(sample_triples)
        for line in lines:
            assert line.endswith(" .")
        assert "<http://www.opengis.net/ont/geosparql#asWKT>" in out
        assert "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>" in out

    def test_literals_quoted(self, sample_triples: list[Triple]) -> None:
        '''Literal objects ("cafe", WKT strings) are JSON-quoted.'''
        out = format_raw(sample_triples)
        assert '"cafe"' in out
        assert '"hospital"' in out
        assert '"POINT (13.32 52.50)"' in out

    def test_entity_refs_in_brackets(self, sample_triples: list[Triple]) -> None:
        """Object values that are entity IDs (appear as subjects elsewhere)
        get angle brackets, not JSON quotes."""
        out = format_raw(sample_triples)
        assert "<geo12147663074>" in out
        assert "<geo4148219>" in out
        # "cafe" should NOT be wrapped in angle brackets.
        assert "<cafe>" not in out


# ---------------------------------------------------------------------------
# format_verbalized
# ---------------------------------------------------------------------------


class TestFormatVerbalized:
    def test_entity_aggregated_output(self, sample_triples: list[Triple]) -> None:
        """Option B: one sentence per entity, combining type + geometry."""
        out = format_verbalized(sample_triples)
        assert (
            "Entity 12147663074 is a cafe with geometry POINT (13.32 52.50)."
            in out
        )
        assert "Entity 4148219 is a hospital with geometry POLYGON" in out
        # The geometry *subjects* (geoNNN) should not appear as standalone
        # sentences — they're folded into the owning entity.
        assert "Entity geo12147663074" not in out

    def test_fallback_for_unknown_predicates(self) -> None:
        """Unknown predicates render via the predicate-level fallback."""
        triples: list[Triple] = [
            {"subject": "e1", "predicate": "http://example.org/custom#note", "object": "hi"},
        ]
        out = format_verbalized(triples)
        assert "Entity e1 has note hi." in out

    def test_type_only_entity(self) -> None:
        """An entity with only rdf:type renders without geometry."""
        triples: list[Triple] = [
            {
                "subject": "e1",
                "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "object": "park",
            },
        ]
        out = format_verbalized(triples)
        assert out == "Entity e1 is a park."

    def test_geometry_only_entity(self) -> None:
        """An entity with only hasGeometry (but no type) renders without type."""
        triples: list[Triple] = [
            {
                "subject": "e1",
                "predicate": "http://www.opengis.net/ont/geosparql#hasGeometry",
                "object": "geo-e1",
            },
            {
                "subject": "geo-e1",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POINT (1 2)",
            },
        ]
        out = format_verbalized(triples)
        assert "Entity e1 has geometry POINT (1 2)." in out

    def test_spatial_object_chain_includes_geometry(self) -> None:
        """Regression: WorldKG uses ``spatialObject`` instead of
        ``hasGeometry`` for the geometry chain. The previous
        implementation hardcoded ``hasGeometry`` and silently dropped
        the WKT for WorldKG entities, biasing tier-1
        ``oracle_verbalized`` baselines downward. The fix detects the
        chain dynamically (any predicate whose object is a
        ``geoXYZ`` subject of an ``asWKT`` triple)."""
        triples: list[Triple] = [
            {
                "subject": "e1",
                "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "object": "Museum",
            },
            {
                "subject": "e1",
                "predicate": "http://www.worldkg.org/schema/spatialObject",
                "object": "geo-e1",
            },
            {
                "subject": "geo-e1",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POINT (14.50 46.05)",
            },
        ]
        out = format_verbalized(triples)
        # Geometry must appear, folded into the entity's sentence.
        assert "Entity e1 is a Museum with geometry POINT (14.50 46.05)." in out
        # Geometry subject must NOT appear as a standalone sentence.
        assert "Entity geo-e1" not in out
        # The chain predicate (``has spatialObject ...``) must NOT
        # appear — the chain is folded into "with geometry ...".
        assert "spatialObject" not in out

    def test_arbitrary_chain_predicate_includes_geometry(self) -> None:
        """Generic case: chain detection is by *object*, not predicate
        name. A made-up predicate like ``customLink`` works as a chain
        if its object happens to be a WKT-bearing subject."""
        triples: list[Triple] = [
            {
                "subject": "e1",
                "predicate": "http://example.org/schema#customLink",
                "object": "geo-e1",
            },
            {
                "subject": "geo-e1",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POINT (1 2)",
            },
        ]
        out = format_verbalized(triples)
        assert "Entity e1 has geometry POINT (1 2)." in out
        assert "Entity geo-e1" not in out

    def test_entity_with_direct_aswkt_inlines_geometry(self) -> None:
        """An entity with no chain — ``asWKT`` is its own predicate —
        still gets a geometry sentence rather than dropping the WKT."""
        triples: list[Triple] = [
            {
                "subject": "e1",
                "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                "object": "park",
            },
            {
                "subject": "e1",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POINT (3 4)",
            },
        ]
        out = format_verbalized(triples)
        assert "Entity e1 is a park with geometry POINT (3 4)." in out


# ---------------------------------------------------------------------------
# format_triples dispatch
# ---------------------------------------------------------------------------


class TestFormatTriplesDispatch:
    def test_t3_uses_structured(self, sample_triples: list[Triple]) -> None:
        """``mode='t3'`` produces the same output as ``mode='structured'``."""
        assert format_triples(sample_triples, "t3") == format_triples(
            sample_triples, "structured"
        )

    def test_raw_mode(self, sample_triples: list[Triple]) -> None:
        assert format_triples(sample_triples, "raw") == format_raw(sample_triples)

    def test_verbalized_mode(self, sample_triples: list[Triple]) -> None:
        assert format_triples(sample_triples, "verbalized") == format_verbalized(
            sample_triples
        )

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            format_triples([], "bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# assert_coverage
# ---------------------------------------------------------------------------


class TestAssertCoverage:
    def test_full_coverage_passes(self) -> None:
        m: dict[str, list[Triple]] = {"q1": [], "q2": []}
        assert_coverage(m, ["q1", "q2"])  # no exception

    def test_missing_raises_with_list(self) -> None:
        m: dict[str, list[Triple]] = {"q1": []}
        with pytest.raises(LookupError) as excinfo:
            assert_coverage(m, ["q1", "q2", "q3"])
        # Error includes at least a sample of missing ids.
        assert "q2" in str(excinfo.value)
        assert "q3" in str(excinfo.value)


# ---------------------------------------------------------------------------
# OracleContext
# ---------------------------------------------------------------------------


class TestOracleContext:
    def test_constructs_and_exposes_fields(self) -> None:
        ctx = OracleContext(
            mode="raw",
            gold_triples_map={"q1": [{"subject": "s", "predicate": "p", "object": "o"}]},
        )
        assert ctx.mode == "raw"
        assert "q1" in ctx.gold_triples_map


# ---------------------------------------------------------------------------
# WKT object-format heterogeneity (3 styles)
# ---------------------------------------------------------------------------


class TestWktStyleHandling:
    """Three distinct WKT object formats land in our gold triples:

    * osmh3kg: bare ``POLYGON ((...))``
    * kwg: literal-quoted ``"POLYGON ((...))"``
    * worldkg: RDF typed-literal ``"Point(...)"^^<...wktLiteral>``

    All three must classify as WKT (and therefore stay literal-typed,
    not get angle-bracket-wrapped as entity refs) in :func:`format_raw`.
    """

    def test_strip_typed_literal_bare(self) -> None:
        from benchmark_models.oracle import _strip_typed_literal

        assert _strip_typed_literal("POLYGON ((1 2, 3 4))") == "POLYGON ((1 2, 3 4))"

    def test_strip_typed_literal_quoted(self) -> None:
        from benchmark_models.oracle import _strip_typed_literal

        assert _strip_typed_literal('"POLYGON ((1 2, 3 4))"') == "POLYGON ((1 2, 3 4))"

    def test_strip_typed_literal_rdf_typed(self) -> None:
        from benchmark_models.oracle import _strip_typed_literal

        s = '"Point(7.034 61.256)"^^<http://www.opengis.net/ont/geosparql#wktLiteral>'
        assert _strip_typed_literal(s) == "Point(7.034 61.256)"

    def test_strip_typed_literal_no_op_on_plain_string(self) -> None:
        from benchmark_models.oracle import _strip_typed_literal

        assert _strip_typed_literal("cafe") == "cafe"
        assert _strip_typed_literal("") == ""

    def test_looks_like_wkt_three_styles(self) -> None:
        from benchmark_models.oracle import _looks_like_wkt

        assert _looks_like_wkt("POLYGON ((1 2, 3 4))")
        assert _looks_like_wkt('"POLYGON ((1 2, 3 4))"')
        assert _looks_like_wkt(
            '"Point(7.034 61.256)"^^<http://www.opengis.net/ont/geosparql#wktLiteral>'
        )
        assert _looks_like_wkt("MULTIPOINT ((1 2))")
        assert _looks_like_wkt("GEOMETRYCOLLECTION (POINT (1 2))")
        assert not _looks_like_wkt("cafe")
        assert not _looks_like_wkt("")
        assert not _looks_like_wkt("12345")

    def test_format_raw_wkt_quoted_kwg_style(self) -> None:
        """kwg-style quoted WKT must render as a literal, not an entity ref."""
        triples: list[Triple] = [
            {
                "subject": "kwg-entity",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": '"POLYGON ((1 2, 3 4))"',
            }
        ]
        out = format_raw(triples)
        # Object should be JSON-quoted (literal), not angle-bracket-wrapped.
        assert '<"POLYGON' not in out
        # The literal output is JSON-encoded; the inner quotes get escaped.
        assert "POLYGON ((1 2, 3 4))" in out

    def test_format_raw_wkt_typed_literal_worldkg_style(self) -> None:
        """worldkg-style typed-literal WKT must render as a literal."""
        wkt_literal = '"Point(7.034 61.256)"^^<http://www.opengis.net/ont/geosparql#wktLiteral>'
        triples: list[Triple] = [
            {
                "subject": "wkg-entity",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": wkt_literal,
            }
        ]
        out = format_raw(triples)
        # Should land as a JSON-quoted literal, not angle-bracket-wrapped.
        assert "<\"Point" not in out
        # The full typed-literal string survives in the JSON-encoded literal.
        assert "Point(7.034 61.256)" in out

    def test_format_raw_wkt_bare_osmh3kg_style(self) -> None:
        triples: list[Triple] = [
            {
                "subject": "osmh3kg-entity",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "POLYGON ((130.985 30.62, 130.986 30.63))",
            }
        ]
        out = format_raw(triples)
        assert "<POLYGON" not in out
        assert "POLYGON ((130.985 30.62, 130.986 30.63))" in out

    def test_format_raw_geometrycollection_recognised(self) -> None:
        """``GEOMETRYCOLLECTION`` is in the WKT prefix list."""
        triples: list[Triple] = [
            {
                "subject": "e",
                "predicate": "http://www.opengis.net/ont/geosparql#asWKT",
                "object": "GEOMETRYCOLLECTION (POINT (1 2))",
            }
        ]
        out = format_raw(triples)
        assert "<GEOMETRYCOLLECTION" not in out
