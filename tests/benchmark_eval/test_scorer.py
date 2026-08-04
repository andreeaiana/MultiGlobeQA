"""End-to-end scorer tests.

One test class per answer type, exercising:

* Happy path (EM=1).
* Near-miss within tolerance (EM=1 for continuous types).
* Far-off (EM=0, NE=1).
* Refusal (``answer=None``).
* Format error (wrong Python type for the expected shape).

The fixtures build ``Prediction`` and ``GoldRow`` by hand — no
backend / runner machinery involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmark_eval.aliases import AliasEntry, AliasStore
from benchmark_eval.base import GoldRow
from benchmark_eval.scorer import Scorer, ScorerConfig
from benchmark_models.base import Prediction, ToolCall, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(answer: Any, *, unit: str | None = None, refusal: bool = False) -> Prediction:
    return Prediction(
        question_id="q1",
        tid="1.1",
        sfid="A",
        answer_type="_placeholder_",
        model_id="m",
        tier="tier1",
        language="en",
        prompt_template_id="tier1.canonical",
        seed=42,
        snapshot_id="x",
        answer=answer,
        unit=unit,
        refusal=refusal,
        usage=Usage(),
    )


def _gold(
    answer_type: str,
    ground_truth: Any,
    *,
    answer_unit: str | None = None,
) -> GoldRow:
    return GoldRow(
        question_id="q1",
        tid="1.1",
        sfid="A",
        answer_type=answer_type,
        answer_unit=answer_unit,
        ground_truth=ground_truth,
        language="en",
    )


@pytest.fixture
def alias_store() -> AliasStore:
    return AliasStore.from_entries([
        AliasEntry(
            canonical_id="osm:1",
            display_name="Berlin",
            aliases={"Berlin", "Berlín", "柏林"},
        ),
        AliasEntry(
            canonical_id="osm:2",
            display_name="Munich",
            aliases={"Munich", "München"},
        ),
        AliasEntry(
            canonical_id="cardinal:N",
            display_name="N",
            aliases={"N", "North", "Nord"},
        ),
        AliasEntry(
            canonical_id="predef:left",
            display_name="left",
            aliases={"left", "gauche"},
        ),
    ])


@pytest.fixture
def scorer(alias_store: AliasStore) -> Scorer:
    return Scorer(alias_store=alias_store)


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------


class TestScoreBoolean:
    def test_correct(self, scorer: Scorer) -> None:
        r = scorer.score(_pred(True), _gold("Boolean (yes/no, true/false)", True))
        assert r.em == 1 and r.ne == 0 and r.covered

    def test_wrong(self, scorer: Scorer) -> None:
        r = scorer.score(_pred(False), _gold("Boolean (yes/no, true/false)", True))
        assert r.em == 0 and r.ne == 1 and r.covered

    def test_string_yes(self, scorer: Scorer) -> None:
        r = scorer.score(_pred("yes"), _gold("Boolean (yes/no, true/false)", True))
        assert r.em == 1

    def test_refusal(self, scorer: Scorer) -> None:
        r = scorer.score(_pred(None), _gold("Boolean (yes/no, true/false)", True))
        assert r.em == 0 and r.ne == 1 and not r.covered

    def test_format_error(self, scorer: Scorer) -> None:
        r = scorer.score(_pred([1, 2]), _gold("Boolean (yes/no, true/false)", True))
        assert r.em == 0 and r.format_error


# ---------------------------------------------------------------------------
# Entity name (single)
# ---------------------------------------------------------------------------


class TestScoreEntityName:
    def test_exact(self, scorer: Scorer) -> None:
        r = scorer.score(_pred("Berlin"), _gold("Entity name (single)", "osm:1"))
        assert r.em == 1 and not r.resolve_failure

    def test_multilingual(self, scorer: Scorer) -> None:
        r = scorer.score(_pred("Berlín"), _gold("Entity name (single)", "osm:1"))
        assert r.em == 1

    def test_wrong_entity(self, scorer: Scorer) -> None:
        r = scorer.score(_pred("Munich"), _gold("Entity name (single)", "osm:1"))
        assert r.em == 0 and not r.resolve_failure  # resolved, just wrong

    def test_unresolvable(self, scorer: Scorer) -> None:
        r = scorer.score(_pred("Atlantis"), _gold("Entity name (single)", "osm:1"))
        assert r.em == 0 and r.resolve_failure

    def test_bare_numeric_gold_prefixed(self, scorer: Scorer) -> None:
        """Gold stored as ``"1"`` is normalized to ``"osm:1"`` for lookup."""
        r = scorer.score(_pred("Berlin"), _gold("Entity name (single)", "1"))
        assert r.em == 1

    def test_empty_pred_does_not_match_empty_gold(self, scorer: Scorer) -> None:
        """An empty / refusal-shaped prediction must not silently match
        an empty gold string. This protects against EM-inflation on
        rows whose gold was lost upstream and on refusal predictions
        the model emitted as ``""`` instead of ``null``."""
        r = scorer.score(
            _pred(""),
            _gold("Entity name (single)", ""),
        )
        assert r.em == 0


class TestNormalizeGoldCanonical:
    """Phase 3 / 2c: gold-side URL → canonical-id mapping.

    Gold triples carry full URIs as subject / object values
    (``http://www.worldkg.org/resource/...`` for worldkg,
    ``http://stko-kwg.geog.ucsb.edu/lod/resource/...`` for kwg). The
    scorer must normalise these to the per-language alias-store key
    form (``wkg:<id>`` / ``kwg:<id>``) before lookup.
    """

    def test_already_prefixed_passthrough(self) -> None:
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical("osm:12345") == "osm:12345"
        assert Scorer._normalize_gold_canonical("wkg:67890") == "wkg:67890"
        assert Scorer._normalize_gold_canonical("kwg:zipCodeArea.12054") == "kwg:zipCodeArea.12054"

    def test_bare_numeric_becomes_osm_prefix(self) -> None:
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical("12345") == "osm:12345"

    def test_worldkg_uri_becomes_wkg_prefix(self) -> None:
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical(
            "http://www.worldkg.org/resource/12345"
        ) == "wkg:12345"

    def test_worldkg_geo_prefix_stripped(self) -> None:
        """Synthetic ``geo<digits>`` geometry nodes share the entity's
        canonical id after the ``geo`` prefix is stripped — same
        normalisation as ``oracle.load_gold_triples`` does on the
        gold-triples loader path."""
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical(
            "http://www.worldkg.org/resource/geo12345"
        ) == "wkg:12345"

    def test_worldkg_geosparql_uri_not_geo_stripped(self) -> None:
        """Only ``geo<digits>`` is stripped; non-numeric ``geo*`` tails
        survive (e.g. ``geometry`` would not be a numeric id)."""
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical(
            "http://www.worldkg.org/resource/geometry"
        ) == "wkg:geometry"

    def test_kwg_uri_becomes_kwg_prefix(self) -> None:
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical(
            "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.12054"
        ) == "kwg:zipCodeArea.12054"

    def test_bare_name_passthrough(self) -> None:
        """Plain entity names are returned unchanged for the alias store
        to handle via ``AliasStore.resolve``."""
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical("Berlin") == "Berlin"
        assert Scorer._normalize_gold_canonical("Cafe Kanel") == "Cafe Kanel"

    def test_none_returns_none(self) -> None:
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical(None) is None

    def test_strips_whitespace(self) -> None:
        from benchmark_eval.scorer import Scorer

        assert Scorer._normalize_gold_canonical(
            "  http://www.worldkg.org/resource/12345  "
        ) == "wkg:12345"


# ---------------------------------------------------------------------------
# Entity hex string (strict)
# ---------------------------------------------------------------------------


class TestScoreEntityHex:
    """H3 cell id matching.

    Wave 2c: the scorer normalises decimal integers,
    negative signed int64 overflow forms, and hex to a canonical
    15-character lower-case hex string before comparison. The test
    cases below use realistic H3 res-7 values rather than placeholder
    ``"0xdeadbeef"`` shapes that never occur in the benchmark."""

    H3_A = "87c2f2a51ffffff"  # decimal 611415508226408447
    H3_B = "87c2e3480ffffff"  # decimal 611414452452982783

    def test_exact(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.H3_A),
            _gold("Entity hex string (single)", self.H3_A),
        )
        assert r.em == 1

    def test_case_insensitive(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.H3_A.upper()),
            _gold("Entity hex string (single)", self.H3_A),
        )
        assert r.em == 1

    def test_wrong(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.H3_A),
            _gold("Entity hex string (single)", self.H3_B),
        )
        assert r.em == 0

    def test_decimal_gold_hex_prediction(self, scorer: Scorer) -> None:
        """Benchmark gold is decimal int; model emits hex. Must match."""
        r = scorer.score(
            _pred(self.H3_A),
            _gold("Entity hex string (single)", "611415508226408447"),
        )
        assert r.em == 1

    def test_negative_int64_gold(self, scorer: Scorer) -> None:
        """H3 res-15 cells overflow signed int64 to negative form.
        Must unsigned-cast and still match the canonical hex. Uses a
        real gold value observed in the benchmark."""
        signed = -8190665156108222464
        canonical_hex = format(signed & 0xFFFFFFFFFFFFFFFF, "x")
        r = scorer.score(
            _pred(canonical_hex),
            _gold("Entity hex string (single)", str(signed)),
        )
        assert r.em == 1


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------


class TestScoreNumeric:
    def test_exact_match(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(12.247, unit="km"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 1 and r.ne == pytest.approx(0.0, abs=1e-9)
        assert r.unit_match is True
        assert r.numeric_failure_category is None

    def test_within_tolerance(self, scorer: Scorer) -> None:
        # Default ScorerConfig has use_scale_from_magnitude=True;
        # gold=12.247 km is "urban" (< 50) → δ=0.5, but
        # τ = max(δ, α·|y*|) = max(0.5, 0.05·12.247) = 0.612.
        # Pick a prediction within that tighter tolerance.
        r = scorer.score(
            _pred(12.5, unit="km"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 1
        # d = 0.253, τ ≈ 0.612 → NE ≈ 0.413
        assert r.ne == pytest.approx(0.253 / 0.6123, rel=1e-2)

    def test_beyond_tolerance(self, scorer: Scorer) -> None:
        # d=100, τ=5 → NE=1, EM=0, both_wrong? no, unit matches.
        r = scorer.score(
            _pred(112.247, unit="km"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 0 and r.ne == 1.0
        assert r.numeric_failure_category == "number_wrong_unit_correct"

    def test_unit_mismatch_number_within(self, scorer: Scorer) -> None:
        # Gold = 12.247 km. Pred = 12.3 in "metres" — unit mismatch.
        r = scorer.score(
            _pred(12.3, unit="metres"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 0
        assert r.unit_match is False
        assert r.numeric_failure_category == "number_correct_unit_wrong"

    def test_unit_mismatch_number_wrong_too(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(1000, unit="metres"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 0
        assert r.numeric_failure_category == "both_wrong"

    def test_canonicalized_unit_match(self, scorer: Scorer) -> None:
        """``km`` and ``kilometres`` canonicalize to the same code."""
        r = scorer.score(
            _pred(12.247, unit="km"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.unit_match is True

    def test_no_gold_unit_auto_match(self, scorer: Scorer) -> None:
        """Count questions have no gold unit; unit_match defaults True."""
        r = scorer.score(
            _pred(7),
            _gold("Numeric (integer)", 7, answer_unit=None),
        )
        assert r.em == 1 and r.unit_match is True

    def test_string_numeric_parses(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("13.5 km"),
            _gold("Numeric (float)", 13.5, answer_unit="kilometres"),
        )
        assert r.em == 1

    def test_prediction_unit_field_overrides_inline_unit(
        self, scorer: Scorer
    ) -> None:
        """Md10 regression: when ``prediction.unit`` and the trailing
        unit on the answer string disagree, the explicit unit field
        wins. The model-side prompt asks for the unit in the
        structured-output ``unit`` field; that's authoritative even
        if the answer-string parse pulls a different token off the
        end."""
        # answer_string says "5 km" (parse_numeric → unit "km").
        # prediction.unit declares "m". Gold unit is "km". The
        # explicit field wins → unit_match=False.
        r = scorer.score(
            _pred("5 km", unit="m"),
            _gold("Numeric (float)", 5.0, answer_unit="kilometres"),
        )
        assert r.unit_match is False
        # number value is 5.0 — within tolerance of gold 5.0 — but
        # unit mismatch makes it EM=0 with the appropriate failure
        # category. ne_unit_normalized would convert 5 m → 0.005 km,
        # hugely off; that's expected.
        assert r.em == 0
        assert r.numeric_failure_category == "number_correct_unit_wrong"

    def test_format_error(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("not a number", unit="km"),
            _gold("Numeric (float)", 13.5, answer_unit="kilometres"),
        )
        assert r.em == 0 and r.format_error

    def test_refusal(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(None),
            _gold("Numeric (float)", 13.5, answer_unit="kilometres"),
        )
        assert r.em == 0 and not r.covered


# ---------------------------------------------------------------------------
# Set enumeration
# ---------------------------------------------------------------------------


class TestScoreSetEnumeration:
    def test_exact_set(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(["Berlin", "Munich"]),
            _gold("Set enumeration (list of entity names)", ["osm:1", "osm:2"]),
        )
        assert r.em == 1 and r.ne == 0
        assert r.set_precision == 1.0 and r.set_recall == 1.0 and r.set_f1 == 1.0

    def test_partial_jaccard(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(["Berlin"]),
            _gold("Set enumeration (list of entity names)", ["osm:1", "osm:2"]),
        )
        # Jaccard = 1/2 → NE = 0.5. EM=0 (set not equal).
        assert r.em == 0 and r.ne == pytest.approx(0.5)
        assert r.set_precision == 1.0 and r.set_recall == 0.5

    def test_set_of_dates(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(["2019-01-01", "2020-01-01"]),
            _gold("Set enumeration (list of dates)", ["2019-01-01", "2020-01-01"]),
        )
        assert r.em == 1 and r.ne == 0.0

    def test_empty_sets(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred([]),
            _gold("Set enumeration (list of entity names)", []),
        )
        assert r.em == 1 and r.ne == 0.0

    def test_format_error(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("Berlin, Munich"),  # string, not list
            _gold("Set enumeration (list of entity names)", ["osm:1"]),
        )
        assert r.em == 0 and r.format_error

    def test_json_encoded_string_gold(self, scorer: Scorer) -> None:
        """The benchmark share stores set-enumeration gold as
        JSON-encoded list strings (Spark's default array
        serialization). The scorer's ``_coerce_list`` parses these
        before comparison so the share's 2,653 set-enum rows score
        correctly without preprocessing."""
        gold_as_json_string = '["Berlin", "Munich"]'
        r = scorer.score(
            _pred(["Berlin", "Munich"]),
            _gold("Set enumeration (list of entity names)", gold_as_json_string),
        )
        assert r.em == 1 and r.ne == 0
        assert not r.format_error

    def test_quote_wrapped_gold_elements_match_clean_pred(
        self, scorer: Scorer
    ) -> None:
        """Phase 3 / 2c: set-enumeration gold elements can arrive
        quote-wrapped (e.g. ``'"Berlin"'``); D2 cleaned the
        committed JSON, but the scorer also strips a single wrapping
        ``"…"`` pair on read so a future regression that re-introduces
        the pathology doesn't depress recall. Uses entities already
        in the test alias store to verify alias resolution survives
        the strip.
        """
        r = scorer.score(
            _pred(["Berlin", "Munich"]),
            _gold(
                "Set enumeration (list of entity names)",
                ['"Berlin"', '"Munich"'],
            ),
        )
        assert r.em == 1 and r.ne == 0

    def test_quote_wrapped_pred_elements_match_clean_gold(
        self, scorer: Scorer
    ) -> None:
        """Symmetric: pred-side quote-wrapping also cleaned on read."""
        r = scorer.score(
            _pred(['"Berlin"', '"Munich"']),
            _gold(
                "Set enumeration (list of entity names)",
                ["osm:1", "osm:2"],
            ),
        )
        assert r.em == 1 and r.ne == 0

    def test_bare_names_absent_from_alias_store_match(
        self, scorer: Scorer
    ) -> None:
        """Regression for the gold/prediction canonicalization asymmetry.

        When set members are NOT in the alias store, the gold side must
        canonicalize identically to the prediction side. The old
        ``resolved or normalized or canonicalize_string(...)`` fallback
        kept the gold's RAW string (``_normalize_gold_canonical`` returns
        bare names unchanged) while the prediction was
        casefolded/punctuation-stripped, so the two sets could never
        intersect — a byte-identical perfect answer scored EM=0. Since
        100% of released set-gold members are bare names absent from the
        store, this zeroed EM on ~65% of set rows. These members are
        deliberately outside the fixture alias store.
        """
        members = [
            "Road segment with OBJECT ID 184598",
            "Road segment with OBJECT ID 184599",
        ]
        r = scorer.score(
            _pred(list(members)),
            _gold("Set enumeration (list of entity names)", list(members)),
        )
        assert r.em == 1 and r.ne == 0.0
        assert r.set_precision == 1.0
        assert r.set_recall == 1.0
        assert r.set_f1 == 1.0

    def test_bare_name_casefold_absent_from_store(self, scorer: Scorer) -> None:
        """Absent-from-store members differing only by case still match:
        both sides run through ``canonicalize_string`` (casefold)."""
        r = scorer.score(
            _pred(["ROAD SEGMENT WITH OBJECT ID 42"]),
            _gold(
                "Set enumeration (list of entity names)",
                ["Road segment with OBJECT ID 42"],
            ),
        )
        assert r.em == 1 and r.ne == 0.0


class TestSetEntityAliasCollisionDeterminism:
    """A2: set-entity scoring must be deterministic across alias-store
    regenerations. Two AliasStores built from JSON files with different
    key orders (but identical alias→canonical mappings) must produce
    byte-identical ScoreRecord output."""

    def _build_store(self, entries_order: list[str]) -> AliasStore:
        """Build a store with entries inserted in a caller-specified
        order. Each entry shares one alias with the next ("Capital")
        so the collision-resolution path is exercised."""
        defs = {
            "osm:1": ("Berlin", {"Berlin", "Capital"}),
            "osm:2": ("Paris", {"Paris", "Capital"}),
            "osm:3": ("Madrid", {"Madrid", "Capital"}),
        }
        return AliasStore.from_entries([
            AliasEntry(
                canonical_id=cid,
                display_name=defs[cid][0],
                aliases=defs[cid][1],
            )
            for cid in entries_order
        ])

    def test_identical_score_across_two_key_orders(self) -> None:
        store_a = self._build_store(["osm:1", "osm:2", "osm:3"])
        store_b = self._build_store(["osm:3", "osm:1", "osm:2"])
        scorer_a = Scorer(alias_store=store_a)
        scorer_b = Scorer(alias_store=store_b)

        # All three predictions use the ambiguous "Capital" alias —
        # the collision-resolution path is the one exercised.
        pred = _pred(["Capital", "Madrid", "Berlin"])
        gold = _gold(
            "Set enumeration (list of entity names)",
            ["osm:1", "osm:2", "osm:3"],
        )

        rec_a = scorer_a.score(pred, gold)
        rec_b = scorer_b.score(pred, gold)
        assert rec_a.em == rec_b.em
        assert rec_a.ne == rec_b.ne
        assert rec_a.set_precision == rec_b.set_precision
        assert rec_a.set_recall == rec_b.set_recall
        assert rec_a.set_f1 == rec_b.set_f1

    def test_no_gold_hint_picks_alphabetically_first_canonical(self) -> None:
        """When no gold canonical matches the prediction-alias collision,
        the fallback ``candidates[0]`` must be the alphabetically-first
        canonical id (stable across alias-store regenerations)."""
        store = self._build_store(["osm:3", "osm:2", "osm:1"])
        # The alias "Capital" maps to all three osm:1/2/3. With no
        # gold hint, candidates[0] is "osm:1" (alphabetical first).
        # Build a gold that doesn't intersect the prediction so the
        # gold-hint branch can't fire.
        gold = _gold(
            "Set enumeration (list of entity names)",
            ["osm:nonexistent"],
        )
        rec = Scorer(alias_store=store).score(_pred(["Capital"]), gold)
        # Pred resolves to osm:1; gold resolves to its fallback string.
        # Sets are disjoint → em=0.
        assert rec.em == 0


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


class TestScoreCoordinates:
    def test_exact(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred({"lat": 52.52, "lon": 13.405}),
            _gold("Coordinates (lat/long pair)", {"lat": 52.52, "lon": 13.405}),
        )
        assert r.em == 1

    def test_within_tolerance(self, scorer: Scorer) -> None:
        # ~0.01 deg shift ≈ 1.1 km, well within default τ=10km.
        r = scorer.score(
            _pred({"lat": 52.53, "lon": 13.415}),
            _gold("Coordinates (lat/long pair)", {"lat": 52.52, "lon": 13.405}),
        )
        assert r.em == 1

    def test_beyond_tolerance(self, scorer: Scorer) -> None:
        # Berlin to Munich: ~504 km, far beyond τ.
        r = scorer.score(
            _pred({"lat": 48.137, "lon": 11.576}),
            _gold("Coordinates (lat/long pair)", {"lat": 52.52, "lon": 13.405}),
        )
        assert r.em == 0 and r.ne == 1.0

    def test_list_form(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred([52.52, 13.405]),
            _gold("Coordinates (lat/long pair)", [52.52, 13.405]),
        )
        assert r.em == 1

    def test_format_error(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("not coords"),
            _gold("Coordinates (lat/long pair)", [52.52, 13.405]),
        )
        assert r.em == 0 and r.format_error

    def test_underscore_joined_string(self, scorer: Scorer) -> None:
        """Some coordinate gold values are emitted as
        ``'<lat>_<lon>'`` underscore-joined strings. The regex must
        accept this separator alongside comma / semicolon /
        whitespace."""
        r = scorer.score(
            _pred("52.52_13.405"),
            _gold("Coordinates (lat/long pair)", "52.52_13.405"),
        )
        assert r.em == 1

    def test_underscore_joined_negative(self, scorer: Scorer) -> None:
        # Real shape from the share: ``'42.33_-83.02'``.
        r = scorer.score(
            _pred("42.33_-83.02"),
            _gold("Coordinates (lat/long pair)", "42.33_-83.02"),
        )
        assert r.em == 1


class TestCoordinateGridQuantizationTID70_4:
    """TID 64.4 ("Convert the location to a lat/long grid at {res} degrees")
    tightens the coordinate tolerance to a half-cell diagonal derived from the
    gold's decimal precision, so only a prediction in the *correct* grid cell
    passes. The scorer selects this route on an exact tid match, so these tests
    pin the behaviour against the route silently going dormant."""

    def _grid_gold(self, value: str) -> GoldRow:
        return GoldRow(
            question_id="q1", tid="64.4", sfid="N",
            answer_type="Coordinates (lat/long pair)", answer_unit=None,
            ground_truth=value, language="en",
        )

    def test_in_cell_passes(self, scorer: Scorer) -> None:
        # 2-decimal gold => 0.01 deg cell (~0.78 km half-diagonal tol).
        r = scorer.score(_pred("52.50_13.40"), self._grid_gold("52.50_13.40"))
        assert r.em == 1

    def test_wrong_cell_fails(self, scorer: Scorer) -> None:
        # ~1.3 km away is a different 0.01 deg cell; the tightened tolerance
        # rejects it (the generic ~10 km coordinate tolerance would wrongly
        # pass — see the contrast test below).
        r = scorer.score(_pred("52.51_13.41"), self._grid_gold("52.50_13.40"))
        assert r.em == 0 and r.ne == 1.0

    def test_generic_coord_tid_would_pass_same_prediction(
        self, scorer: Scorer
    ) -> None:
        # The identical ~1.3 km-off prediction against a NON-grid coordinate
        # tid uses the generic tolerance and passes — proving the 64.4 route
        # is what tightens scoring (guards against re-dormancy).
        generic = GoldRow(
            question_id="q1", tid="1.1", sfid="A",
            answer_type="Coordinates (lat/long pair)", answer_unit=None,
            ground_truth="52.50_13.40", language="en",
        )
        assert scorer.score(_pred("52.51_13.41"), generic).em == 1


# ---------------------------------------------------------------------------
# H3 / S2 cell index — share the entity-hex matcher
# ---------------------------------------------------------------------------


class TestScoreH3CellIndex:
    """``H3 cell index`` routes to the existing ``_score_entity_hex``
    matcher. The benchmark share emits H3 cells as decimal-encoded
    uint64 strings (e.g. ``'609336537410174975'``)."""

    DECIMAL = "609336537410174975"
    HEX = format(int(DECIMAL), "x")  # canonical lowercase hex

    def test_exact_decimal(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.DECIMAL),
            _gold("H3 cell index", self.DECIMAL),
        )
        assert r.em == 1

    def test_decimal_gold_hex_prediction(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.HEX),
            _gold("H3 cell index", self.DECIMAL),
        )
        assert r.em == 1

    def test_wrong_decimal(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("609336537410174999"),
            _gold("H3 cell index", self.DECIMAL),
        )
        assert r.em == 0


class TestScoreS2CellIndex:
    """``S2 cell index`` shares the entity-hex matcher. S2 cell IDs
    round-trip through the same hex/decimal/signed-int64
    normalization. Real gold shape from the share is a negative
    signed-int64 decimal string."""

    SIGNED = "-8350462053613305856"
    HEX = format(int(SIGNED) & 0xFFFFFFFFFFFFFFFF, "x")

    def test_exact_signed(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.SIGNED),
            _gold("S2 cell index", self.SIGNED),
        )
        assert r.em == 1

    def test_signed_gold_hex_prediction(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.HEX),
            _gold("S2 cell index", self.SIGNED),
        )
        assert r.em == 1

    def test_wrong_signed(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("-8350462053613305000"),
            _gold("S2 cell index", self.SIGNED),
        )
        assert r.em == 0


class TestScoreH3SpatialCompanion:
    """Resolution-agnostic spatial companion for standalone H3 cells.

    Strict ``em`` still requires the exact cell, but ``em_spatial``
    credits a prediction that localizes the target at a different
    resolution (the common "right area, wrong resolution" failure).
    S2 tokens don't decode as H3, so they receive no companion."""

    def _gold_hex(self) -> str:
        import h3

        return h3.latlng_to_cell(37.0, -122.0, 9)  # res-9 gold cell

    def test_coarser_prediction_same_location(self, scorer: Scorer) -> None:
        import h3

        gold = self._gold_hex()
        parent = h3.cell_to_parent(gold, 6)  # coarser cell that contains gold
        r = scorer.score(
            _pred(parent),
            _gold("H3 cell index", str(int(gold, 16))),  # decimal gold
        )
        assert r.em == 0  # wrong resolution -> strict miss
        assert r.em_spatial == 1  # correct location -> spatial credit
        assert r.ne_spatial == 0.0

    def test_finer_prediction_same_location(self, scorer: Scorer) -> None:
        import h3

        gold = self._gold_hex()
        child = h3.cell_to_children(gold, 12)[0]  # finer cell inside gold
        r = scorer.score(_pred(child), _gold("H3 cell index", gold))
        assert r.em == 0
        assert r.em_spatial == 1

    def test_far_prediction_no_spatial_credit(self, scorer: Scorer) -> None:
        import h3

        far = h3.latlng_to_cell(-40.0, 170.0, 9)  # opposite hemisphere
        r = scorer.score(_pred(far), _gold("H3 cell index", self._gold_hex()))
        assert r.em == 0
        assert r.em_spatial == 0

    def test_exact_match_scores_both(self, scorer: Scorer) -> None:
        gold = self._gold_hex()
        r = scorer.score(_pred(gold), _gold("H3 cell index", gold))
        assert r.em == 1
        assert r.em_spatial == 1

    def test_s2_gold_has_no_h3_companion(self, scorer: Scorer) -> None:
        s2 = "-8350462053613305856"
        r = scorer.score(_pred(s2), _gold("S2 cell index", s2))
        assert r.em == 1
        assert r.em_spatial is None


class TestScoreCompositeString:
    """``Composite string (H3+S2)``. Strict equality of both
    pipe-separated halves, plus an H3-half spatial companion scored by
    H3 hierarchy containment. The S2 half is not spatially scored."""

    GOLD = "H3(res=3): 590804843560435711 | S2(lvl=6): 6924002952105426944"

    @staticmethod
    def _fine_cell_in_gold(res: int = 15) -> str:
        import h3

        gold_h3_hex = format(590804843560435711 & 0xFFFFFFFFFFFFFFFF, "x")
        lat, lon = h3.cell_to_latlng(gold_h3_hex)
        return h3.latlng_to_cell(lat, lon, res)

    def test_exact_match(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.GOLD),
            _gold("Composite string (H3+S2)", self.GOLD),
        )
        assert r.em == 1
        assert r.em_spatial == 1

    def test_h3_containment_credits_spatial(self, scorer: Scorer) -> None:
        # Model emits the H3 half at res 15 (per the schema) inside the
        # gold res-3 cell, with a different resolution/encoding → strict
        # 0, spatial 1 via containment.
        fine = self._fine_cell_in_gold(15)
        pred = f"H3(res=15): {fine} | S2(lvl=20): 10000000000000000"
        r = scorer.score(
            _pred(pred),
            _gold("Composite string (H3+S2)", self.GOLD),
        )
        assert r.em == 0
        assert r.em_spatial == 1
        assert r.ne_spatial == pytest.approx(0.0)

    def test_h3_far_away(self, scorer: Scorer) -> None:
        import h3

        far = h3.latlng_to_cell(0.0, 0.0, 15)  # Gulf of Guinea
        pred = f"H3(res=15): {far} | S2(lvl=20): 10000000000000000"
        r = scorer.score(
            _pred(pred),
            _gold("Composite string (H3+S2)", self.GOLD),
        )
        assert r.em == 0
        assert r.em_spatial == 0

    def test_non_h3_prediction(self, scorer: Scorer) -> None:
        # Model answered but the H3 half is garbage → em_spatial=0.
        pred = "H3(res=15): 1836 | S2(lvl=20): 100"
        r = scorer.score(
            _pred(pred),
            _gold("Composite string (H3+S2)", self.GOLD),
        )
        assert r.em == 0
        assert r.em_spatial == 0


class TestS2NoSpatialCompanion:
    """S2 cell index is intentionally scored by strict-ID only — no
    spatial companion (Decision: no S2 decoder dependency)."""

    SIGNED = "-8350462053613305856"

    def test_exact_has_no_spatial(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(self.SIGNED),
            _gold("S2 cell index", self.SIGNED),
        )
        assert r.em == 1
        assert r.em_spatial is None
        assert r.ne_spatial is None

    def test_wrong_has_no_spatial(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("-8357862421850750976"),
            _gold("S2 cell index", self.SIGNED),
        )
        assert r.em == 0
        assert r.em_spatial is None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestScoreGeometry:
    def test_polygon_identical(self, scorer: Scorer) -> None:
        ring = [(52.0, 13.0), (52.0, 14.0), (53.0, 14.0), (53.0, 13.0), (52.0, 13.0)]
        r = scorer.score(
            _pred(ring),
            _gold("Geometry (list of lat/long pairs)", ring),
        )
        assert r.em == 1 and r.ne == pytest.approx(0.0, abs=1e-6)

    def test_polygon_disjoint(self, scorer: Scorer) -> None:
        ring_a = [(52.0, 13.0), (52.0, 14.0), (53.0, 14.0), (53.0, 13.0), (52.0, 13.0)]
        ring_b = [(60.0, 20.0), (60.0, 21.0), (61.0, 21.0), (61.0, 20.0), (60.0, 20.0)]
        r = scorer.score(
            _pred(ring_a),
            _gold("Geometry (list of lat/long pairs)", ring_b),
        )
        assert r.em == 0 and r.ne == pytest.approx(1.0, abs=1e-6)

    def test_line_identical(self, scorer: Scorer) -> None:
        # Non-closed ring → line.
        line = [(52.0, 13.0), (52.1, 13.1), (52.2, 13.2)]
        r = scorer.score(
            _pred(line),
            _gold("Geometry (list of lat/long pairs)", line),
        )
        assert r.em == 1


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------


class TestScoreDate:
    def test_exact(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("2019-06-15"), _gold("Date", "2019-06-15")
        )
        assert r.em == 1 and r.ne == 0.0

    def test_one_day_off(self, scorer: Scorer) -> None:
        """Components-match EM=0; NE=1 since tolerance is 1 day."""
        r = scorer.score(
            _pred("2019-06-16"), _gold("Date", "2019-06-15")
        )
        assert r.em == 0 and r.ne == 1.0

    def test_year_only_gold_matches_any_day_in_year(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("2019-11-30"), _gold("Date", "2019")
        )
        assert r.em == 1

    def test_day_gold_with_month_prediction(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("2019-06"), _gold("Date", "2019-06-15")
        )
        assert r.em == 0

    def test_format_error(self, scorer: Scorer) -> None:
        r = scorer.score(_pred("never"), _gold("Date", "2019-06-15"))
        assert r.em == 0 and r.format_error


# ---------------------------------------------------------------------------
# Cardinal + Predefined
# ---------------------------------------------------------------------------


class TestScoreCardinalAndPredefined:
    def test_cardinal_exact(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("North"),
            _gold("Cardinal direction (predefined)", "N"),
        )
        assert r.em == 1

    def test_cardinal_wrong(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("South"),
            _gold("Cardinal direction (predefined)", "N"),
        )
        assert r.em == 0 and r.resolve_failure

    def test_predefined_multilingual(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("gauche"),
            _gold("Predefined value (left, right)", "left"),
        )
        assert r.em == 1


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def test_basic(self, scorer: Scorer) -> None:
        preds = [_pred(True), _pred(False)]
        # Make qids unique
        preds[0] = preds[0].model_copy(update={"question_id": "qa"})
        preds[1] = preds[1].model_copy(update={"question_id": "qb"})
        golds = {
            "qa": GoldRow(
                question_id="qa", tid="1.1", sfid="A",
                answer_type="Boolean (yes/no, true/false)",
                ground_truth=True, language="en",
            ),
            "qb": GoldRow(
                question_id="qb", tid="1.1", sfid="A",
                answer_type="Boolean (yes/no, true/false)",
                ground_truth=True, language="en",
            ),
        }
        recs = scorer.score_batch(preds, golds)
        assert len(recs) == 2
        assert recs[0].em == 1 and recs[1].em == 0

    def test_missing_gold_skipped(self, scorer: Scorer) -> None:
        preds = [_pred(True).model_copy(update={"question_id": "qa"})]
        recs = scorer.score_batch(preds, {})  # no gold for qa
        assert recs == []


# ---------------------------------------------------------------------------
# ScorerConfig knobs
# ---------------------------------------------------------------------------


class TestScorerConfig:
    def test_alpha_sensitivity(self, alias_store: AliasStore) -> None:
        """Changing α changes tolerance for large-gold numeric."""
        # gold=1000 km, pred=1060 km → d=60
        gold = _gold("Numeric (float)", 1000.0, answer_unit="kilometres")
        # α=0.05 → τ=max(5, 50) = 50 → d=60 > 50 → EM=0
        r1 = Scorer(
            alias_store=alias_store, config=ScorerConfig(alpha=0.05),
        ).score(_pred(1060, unit="km"), gold)
        assert r1.em == 0
        # α=0.10 → τ=max(5, 100) = 100 → d=60 < 100 → EM=1
        r2 = Scorer(
            alias_store=alias_store, config=ScorerConfig(alpha=0.10),
        ).score(_pred(1060, unit="km"), gold)
        assert r2.em == 1

    def test_scale_from_magnitude_toggle(self, alias_store: AliasStore) -> None:
        gold = _gold("Numeric (float)", 5.0, answer_unit="kilometres")
        # Default has scale-from-magnitude=True: urban (gold<50)
        # δ=0.5, τ=max(0.5, 0.05·5)=0.5 → d=0.8 > 0.5 → EM=0.
        r_default = Scorer(alias_store=alias_store).score(
            _pred(5.8, unit="km"), gold
        )
        assert r_default.em == 0
        # Explicit opt-out flips to flat global δ=5: τ=5 → d=0.8 < 5 → EM=1.
        r_flat = Scorer(
            alias_store=alias_store,
            config=ScorerConfig(use_scale_from_magnitude=False),
        ).score(_pred(5.8, unit="km"), gold)
        assert r_flat.em == 1


# ============================================================================
# ScorerConfig spec loading + deviations
# ============================================================================


class TestScorerConfigSpec:
    def test_defaults_match_spec_file(self) -> None:
        """Hardcoded fallbacks in scorer.py must match
        ``scoring_spec.json`` so there is exactly one source of
        truth."""
        import json
        from pathlib import Path

        from benchmark_eval.scorer import SCORING_SPEC_PATH, ScorerConfig

        spec_path = Path(SCORING_SPEC_PATH)
        assert spec_path.exists(), f"missing: {spec_path}"
        raw = json.loads(spec_path.read_text())

        cfg = ScorerConfig()
        # Numeric block.
        assert cfg.alpha == raw["numeric"]["alpha"]
        assert (
            cfg.default_delta_distance_km
            == raw["numeric"]["default_delta_distance_km"]
        )
        assert (
            cfg.default_delta_area_km2
            == raw["numeric"]["default_delta_area_km2"]
        )
        assert cfg.use_scale_from_magnitude == raw["numeric"][
            "use_scale_from_magnitude"
        ]
        # Geometry block.
        assert (
            cfg.polygon_iou_threshold
            == raw["geometry"]["polygon_iou_threshold"]
        )

    def test_deviations_empty_for_spec_defaults(self) -> None:
        from benchmark_eval.scorer import ScorerConfig

        assert ScorerConfig().deviations_from_spec() == {}

    def test_deviations_reports_overrides(self) -> None:
        from benchmark_eval.scorer import ScorerConfig

        cfg = ScorerConfig(alpha=0.1, polygon_iou_threshold=0.9)
        dev = cfg.deviations_from_spec()
        assert dev == {"alpha": 0.1, "polygon_iou_threshold": 0.9}


# ============================================================================
# Numeric — unit-normalized companions (em_unit_normalized / ne_unit_normalized)
# ============================================================================


class TestNumericUnitNormalized:
    """Companion metrics that convert dimensionally-compatible
    prediction units into the gold unit before computing EM / NE.
    Exercises the 'correct answer in the wrong unit' case the metric
    is designed to catch."""

    def test_m_vs_km_strict_fails_normalized_passes(
        self, scorer: Scorer
    ) -> None:
        # Gold 12.247 km; model says 12247 m (semantically correct).
        r = scorer.score(
            _pred(12247.0, unit="m"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        # Strict headline: value+unit mismatch → EM=0, NE clipped to 1.
        assert r.em == 0
        assert r.ne == pytest.approx(1.0)
        # Normalized companion: convert 12247 m → 12.247 km → EM=1, NE=0.
        assert r.em_unit_normalized == 1
        assert r.ne_unit_normalized == pytest.approx(0.0)
        assert r.unit_match is False

    def test_mi_vs_km_in_same_group_converts(self, scorer: Scorer) -> None:
        # 7.610 mi ≈ 12.247 km within 0.01% → passes normalized EM.
        r = scorer.score(
            _pred(7.610, unit="mi"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 0  # strict: unit mismatch
        assert r.em_unit_normalized == 1  # normalized: within tolerance
        assert r.ne_unit_normalized is not None
        assert r.ne_unit_normalized < 0.1

    def test_matching_units_normalized_equals_strict(
        self, scorer: Scorer
    ) -> None:
        r = scorer.score(
            _pred(12.247, unit="km"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 1
        assert r.em_unit_normalized == 1
        assert r.ne == r.ne_unit_normalized

    def test_cross_group_unit_falls_back_to_strict(
        self, scorer: Scorer
    ) -> None:
        # Gold is a distance (km); model says an area unit (km²).
        # Conversion impossible → normalized falls back to strict.
        r = scorer.score(
            _pred(12.0, unit="km2"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 0
        assert r.em_unit_normalized == r.em
        assert r.ne_unit_normalized == r.ne

    def test_unknown_unit_falls_back_to_strict(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(12.0, unit="parsec"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em_unit_normalized == r.em
        assert r.ne_unit_normalized == r.ne

    def test_unitless_gold_falls_back_to_strict(self, scorer: Scorer) -> None:
        # Numeric (integer) count has no gold unit — no conversion
        # happens; normalized == strict.
        r = scorer.score(
            _pred(7, unit=None),
            _gold("Numeric (integer)", 7, answer_unit=None),
        )
        assert r.em == 1
        assert r.em_unit_normalized == 1
        assert r.ne == r.ne_unit_normalized

    def test_area_m2_to_km2(self, scorer: Scorer) -> None:
        # Gold 5 km²; model says 5_000_000 m² (= 5 km²). Strict fails,
        # normalized passes.
        r = scorer.score(
            _pred(5_000_000.0, unit="m²"),
            _gold("Numeric (float)", 5.0, answer_unit="km^2"),
        )
        assert r.em == 0
        assert r.em_unit_normalized == 1

    def test_wrong_value_right_unit_stays_wrong(self, scorer: Scorer) -> None:
        # Model has the unit right but the value is far off.
        # Normalized cannot help — no conversion changes this.
        r = scorer.score(
            _pred(500.0, unit="km"),
            _gold("Numeric (float)", 12.247, answer_unit="kilometres"),
        )
        assert r.em == 0
        assert r.em_unit_normalized == 0
        assert r.ne == r.ne_unit_normalized


# ============================================================================
# Denormalised counters: tool calls, SC agreement, false-premise
# ============================================================================


def _pred_with(
    answer: Any,
    *,
    tool_calls: list[ToolCall] | None = None,
    sc_samples: list[Any] | None = None,
    final_state: str | None = None,
) -> Prediction:
    """Prediction helper that exposes the tier-3 / tier-2 SC fields."""
    return Prediction(
        question_id="q1",
        tid="1.1",
        sfid="A",
        answer_type="_placeholder_",
        model_id="m",
        tier="tier3" if tool_calls is not None else "tier2",
        language="en",
        prompt_template_id="t.canonical",
        seed=42,
        snapshot_id="x",
        answer=answer,
        tool_calls=tool_calls,
        sc_samples=sc_samples,
        final_state=final_state,  # type: ignore[arg-type]
        usage=Usage(),
    )


class TestScoreRecordToolCallCounters:
    def test_none_for_tier1(self, scorer: Scorer) -> None:
        r = scorer.score(_pred(True), _gold("Boolean (yes/no, true/false)", True))
        assert r.tool_call_count is None
        assert r.tool_error_count is None

    def test_counts_tool_calls_and_errors(self, scorer: Scorer) -> None:
        calls = [
            ToolCall(name="find_entity", args={"name": "Berlin"}, result={"id": 1}),
            ToolCall(name="get_entity", args={"id": 1}, error="boom"),
            ToolCall(name="get_entity", args={"id": 2}, result={"area": 891}),
        ]
        r = scorer.score(
            _pred_with(True, tool_calls=calls, final_state="answered"),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.tool_call_count == 3
        assert r.tool_error_count == 1

    def test_zero_calls_when_list_empty(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred_with(True, tool_calls=[], final_state="answered"),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.tool_call_count == 0
        assert r.tool_error_count == 0


class TestScoreRecordSCAgreement:
    def test_none_when_sc_samples_absent(self, scorer: Scorer) -> None:
        r = scorer.score(_pred(True), _gold("Boolean (yes/no, true/false)", True))
        assert r.sc_agreement is None

    def test_none_for_n_below_two(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred_with(True, sc_samples=[True]),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.sc_agreement is None

    def test_unanimous_three_of_three(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred_with(True, sc_samples=[True, True, True]),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.sc_agreement == pytest.approx(1.0)

    def test_majority_two_of_three(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred_with(True, sc_samples=[True, True, False]),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.sc_agreement == pytest.approx(2 / 3)

    def test_dict_samples_compare_via_json(self, scorer: Scorer) -> None:
        # Same dict serialises identically regardless of insertion order
        # (sort_keys=True). Two of three samples are the same.
        a = {"x": 1, "y": 2}
        a_reordered = {"y": 2, "x": 1}
        r = scorer.score(
            _pred_with([a, a_reordered, {"x": 9}], sc_samples=[a, a_reordered, {"x": 9}]),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.sc_agreement == pytest.approx(2 / 3)


class TestScoreRecordFalsePremiseFlag:
    def test_carries_through_from_gold(self, scorer: Scorer) -> None:
        gold = GoldRow(
            question_id="q1",
            tid="1.1",
            sfid="A",
            answer_type="Boolean (yes/no, true/false)",
            ground_truth=True,
            language="en",
            is_true_premise=True,
        )
        r = scorer.score(_pred(True), gold)
        assert r.is_true_premise is True

    def test_false_premise_correct_abstention(self, scorer: Scorer) -> None:
        gold = GoldRow(
            question_id="q1",
            tid="1.1",
            sfid="A",
            answer_type="Entity name (single)",
            ground_truth=None,
            language="en",
            is_true_premise=False,
        )
        # Model abstained → EM=1, covered=True (per existing scoring rule).
        # is_true_premise must propagate to the record so the aggregator
        # can compute negative_rejection_rate.
        r = scorer.score(_pred(None, refusal=True), gold)
        assert r.em == 1
        assert r.is_true_premise is False

    def test_none_when_benchmark_omits_field(self, scorer: Scorer) -> None:
        r = scorer.score(_pred(True), _gold("Boolean (yes/no, true/false)", True))
        assert r.is_true_premise is None


# ---------------------------------------------------------------------------
# Ternary (K7 from the round-2 plan)
# ---------------------------------------------------------------------------


class TestScoreTernary:
    """K7: ternary (true / false / plausibly) value coverage."""

    def test_true_matches_true(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("true"),
            _gold("Ternary (true/false/plausibly)", "true"),
        )
        assert r.em == 1

    def test_false_matches_false(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("false"),
            _gold("Ternary (true/false/plausibly)", "false"),
        )
        assert r.em == 1

    def test_plausibly_matches_plausibly(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("plausibly"),
            _gold("Ternary (true/false/plausibly)", "plausibly"),
        )
        assert r.em == 1

    def test_capitalisation_normalised(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("Plausibly"),
            _gold("Ternary (true/false/plausibly)", "plausibly"),
        )
        assert r.em == 1

    def test_true_vs_false_mismatch(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("true"),
            _gold("Ternary (true/false/plausibly)", "false"),
        )
        assert r.em == 0

    def test_bool_true_coerced(self, scorer: Scorer) -> None:
        """A bool prediction is coerced to its string form."""
        r = scorer.score(
            _pred(True),
            _gold("Ternary (true/false/plausibly)", "true"),
        )
        assert r.em == 1

    def test_bool_false_coerced(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(False),
            _gold("Ternary (true/false/plausibly)", "false"),
        )
        assert r.em == 1

    def test_invalid_value_format_error(self, scorer: Scorer) -> None:
        """Values outside {true, false, plausibly} are format errors."""
        r = scorer.score(
            _pred("maybe"),
            _gold("Ternary (true/false/plausibly)", "true"),
        )
        assert r.em == 0
        assert r.format_error is True


# ---------------------------------------------------------------------------
# Harness-error handling (infra / code_gen must not be scored as refusals)
# ---------------------------------------------------------------------------


def _error_pred(
    error_class: str, *, answer: Any = None, refusal: bool = False
) -> Prediction:
    """A prediction that terminated with ``final_state='error'``."""
    return Prediction(
        question_id="q1",
        tid="1.1",
        sfid="A",
        answer_type="_placeholder_",
        model_id="m",
        tier="tier1",
        language="en",
        prompt_template_id="tier1.canonical",
        seed=42,
        snapshot_id="x",
        answer=answer,
        refusal=refusal,
        final_state="error",
        error_class=error_class,  # type: ignore[arg-type]
        usage=Usage(),
    )


def _gold_premise(
    answer_type: str, ground_truth: Any, *, is_true_premise: bool
) -> GoldRow:
    return GoldRow(
        question_id="q1",
        tid="1.1",
        sfid="A",
        answer_type=answer_type,
        ground_truth=ground_truth,
        language="en",
        is_true_premise=is_true_premise,
    )


class TestHarnessErrorNotRefusal:
    def test_infra_error_on_false_premise_not_credited(
        self, scorer: Scorer
    ) -> None:
        """An infra crash on a false-premise row must NOT be credited
        EM=1 as a 'correct rejection' — the substrate failed, the model
        did not choose to abstain."""
        r = scorer.score(
            _error_pred("infra"),
            _gold_premise(
                "Boolean (yes/no, true/false)", None, is_true_premise=False
            ),
        )
        assert r.em == 0
        assert r.covered is False
        assert r.refusal is False

    def test_code_gen_error_not_refusal(self, scorer: Scorer) -> None:
        r = scorer.score(
            _error_pred("code_gen"),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.em == 0 and r.covered is False and r.refusal is False

    def test_non_harness_error_not_counted_as_refusal(
        self, scorer: Scorer
    ) -> None:
        """A ``config`` / ``schema_violation`` error row is a substrate
        failure, not an abstention: the broadened ``final_state==error``
        guard scores it EM=0 / covered=False / refusal=False (it does NOT
        route through the refusal branch, so it can't inflate the refusal
        rate). Its non-harness ``error_class`` still keeps it in the
        ``em_adjusted`` denominator — that split lives in aggregate.py."""
        r = scorer.score(
            _error_pred("config", answer=None),
            _gold("Boolean (yes/no, true/false)", True),
        )
        assert r.em == 0 and r.covered is False and r.refusal is False

    def test_genuine_refusal_on_false_premise_still_credited(
        self, scorer: Scorer
    ) -> None:
        """Guard is scoped to errors: a real abstention (no error state)
        on a false-premise row is still the correct answer (EM=1)."""
        r = scorer.score(
            _pred(None),
            _gold_premise(
                "Boolean (yes/no, true/false)", None, is_true_premise=False
            ),
        )
        assert r.em == 1 and r.covered is True and r.refusal is True

    def test_false_premise_commission_is_covered_wrong_not_format_error(
        self, scorer: Scorer
    ) -> None:
        """A model that ANSWERS a false-premise question (gold None,
        is_true_premise=False) instead of abstaining produced a parseable
        typed answer: it counts as covered=True but wrong (EM=0). It must
        NOT be mislabeled a format_error (which would deflate coverage and
        inflate the format-error rate)."""
        r = scorer.score(
            _pred(True),  # a non-refusal boolean answer (commission)
            _gold_premise(
                "Boolean (yes/no, true/false)", None, is_true_premise=False
            ),
        )
        assert r.em == 0
        assert r.covered is True
        assert r.format_error is False


# ---------------------------------------------------------------------------
# Bounding box — degenerate (zero-area point) gold
# ---------------------------------------------------------------------------


class TestScoreBoundingBoxDegenerate:
    def test_zero_area_gold_scored_by_proximity(self, scorer: Scorer) -> None:
        """A point 'bounding box' (zero area) is scored by geodesic
        proximity, not exact float equality: a prediction ~0.9 km away
        (well inside the 5 km coordinate tolerance) scores EM=1."""
        gold_pt = "40.00000, -75.00000, 40.00000, -75.00000"
        near = "40.00800, -75.00000, 40.00800, -75.00000"  # ~0.89 km north
        r = scorer.score(
            _pred(near), _gold("Bounding box (4-float)", gold_pt)
        )
        assert r.em == 1 and r.covered is True

    def test_zero_area_gold_far_prediction_fails(self, scorer: Scorer) -> None:
        gold_pt = "40.0, -75.0, 40.0, -75.0"
        far = "41.0, -75.0, 41.0, -75.0"  # ~111 km north
        r = scorer.score(
            _pred(far), _gold("Bounding box (4-float)", gold_pt)
        )
        assert r.em == 0

    def test_nondegenerate_gold_still_uses_iou(self, scorer: Scorer) -> None:
        """A real (positive-area) gold box still scores by IoU."""
        gold_box = "40.0, -75.0, 41.0, -74.0"
        r = scorer.score(
            _pred(gold_box), _gold("Bounding box (4-float)", gold_box)
        )
        assert r.em == 1  # identical boxes → IoU 1.0


# ---------------------------------------------------------------------------
# #17: previously-untested scorers — geohash + bbox/polar coercion
# ---------------------------------------------------------------------------


class TestScoreGeohash:
    def test_exact_match(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("dpsby3"), _gold("Geohash string", "dpsby3")
        )
        assert r.em == 1 and r.ne == 0.0 and r.covered

    def test_case_insensitive(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("DPSBY3"), _gold("Geohash string", "dpsby3")
        )
        assert r.em == 1

    def test_wrong_cell(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred("dpsby2"), _gold("Geohash string", "dpsby3")
        )
        assert r.em == 0 and r.ne == 1.0 and r.covered

    def test_non_base32_pred_format_error(self, scorer: Scorer) -> None:
        # 'a', 'i', 'l', 'o' are outside the geohash base-32 alphabet.
        r = scorer.score(
            _pred("ailo!!"), _gold("Geohash string", "dpsby3")
        )
        assert r.em == 0 and r.format_error is True

    def test_refusal(self, scorer: Scorer) -> None:
        r = scorer.score(
            _pred(None), _gold("Geohash string", "dpsby3")
        )
        assert r.em == 0 and not r.covered and r.refusal


class TestCoerceBboxAndPolar:
    def test_coerce_bbox_string_list_dict(self) -> None:
        from benchmark_eval.scorer import _coerce_bbox

        want = (45.187926, -67.279667, 45.187926, -67.279667)
        assert _coerce_bbox(
            "45.187926,-67.279667,45.187926,-67.279667"
        ) == want
        assert _coerce_bbox(
            [45.187926, -67.279667, 45.187926, -67.279667]
        ) == want
        assert _coerce_bbox(
            {
                "lat_min": 45.187926, "lon_min": -67.279667,
                "lat_max": 45.187926, "lon_max": -67.279667,
            }
        ) == want

    def test_coerce_bbox_swapped_corners_normalized(self) -> None:
        from benchmark_eval.scorer import _coerce_bbox

        # Upstream occasionally emits max,min swapped — normalize.
        assert _coerce_bbox("41.0,-74.0,40.0,-75.0") == (40.0, -75.0, 41.0, -74.0)

    def test_coerce_bbox_malformed(self) -> None:
        from benchmark_eval.scorer import _coerce_bbox

        assert _coerce_bbox("not,a,box") is None
        assert _coerce_bbox([1, 2, 3]) is None  # wrong arity
        assert _coerce_bbox({"lat_min": 1}) is None  # missing keys
        assert _coerce_bbox(None) is None

    def test_coerce_polar_forms(self) -> None:
        from benchmark_eval.scorer import _coerce_polar

        assert _coerce_polar("150.2365, 33.5497") == (150.2365, 33.5497)
        assert _coerce_polar([150.2365, 33.5497]) == (150.2365, 33.5497)
        assert _coerce_polar(
            {"angle_degrees": 150.2365, "distance_km": 33.5497}
        ) == (150.2365, 33.5497)
        # Zero radius / angle must parse (membership, not truthiness).
        assert _coerce_polar({"angle_degrees": 0.0, "distance_km": 0.0}) == (
            0.0, 0.0,
        )

    def test_coerce_polar_malformed(self) -> None:
        from benchmark_eval.scorer import _coerce_polar

        assert _coerce_polar("nonsense") is None
        assert _coerce_polar([1]) is None
        assert _coerce_polar({"angle_degrees": 1.0}) is None  # no distance
        assert _coerce_polar(None) is None


# ---------------------------------------------------------------------------
# #5: TID 57.x dimensionless shape indices use a relative (ratio) tolerance
# ---------------------------------------------------------------------------


def _ratio_gold(tid: str, value: float) -> GoldRow:
    return GoldRow(
        question_id="q1", tid=tid, sfid="K", answer_type="Numeric (float)",
        answer_unit=None, ground_truth=value, language="en",
    )


class TestRatioToleranceTID63:
    def test_within_5pct_passes(self, scorer: Scorer) -> None:
        # gold 4.5e-4; a prediction within 5% passes.
        g = _ratio_gold("57.1", 4.5e-4)
        assert scorer.score(_pred(4.5e-4 * 1.02), g).em == 1

    def test_off_by_50pct_fails(self, scorer: Scorer) -> None:
        g = _ratio_gold("57.1", 4.5e-4)
        assert scorer.score(_pred(4.5e-4 * 1.5), g).em == 0

    def test_constant_guess_no_longer_wins(self, scorer: Scorer) -> None:
        # Under the old count δ=1.0, ANY prediction within ±1.0 of a
        # ~4.5e-4 gold scored EM=1 (a constant 0.0 gamed it). With the
        # relative ratio tolerance it must actually be close.
        g = _ratio_gold("57.1", 4.5e-4)
        assert scorer.score(_pred(0.0), g).em == 0

    def test_larger_scale_sub_tid(self, scorer: Scorer) -> None:
        # 57.5 golds reach ~75; 5% relative still applies.
        g = _ratio_gold("57.5", 40.0)
        assert scorer.score(_pred(41.0), g).em == 1   # 2.5% off
        assert scorer.score(_pred(60.0), g).em == 0   # 50% off
