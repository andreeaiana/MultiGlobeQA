"""Tests for ``benchmark_models.base``.

Two layers:

* Unit tests with synthetic dicts — fast, isolated, no I/O. Cover the
  important loader paths (entity slot extraction A through D, parameters
  parsing, gold-field stripping, strata_cell promotion, language
  override, raw escape hatch, partially-populated entity slot warning,
  malformed parameters JSON warning).

* Integration test that loads the real ``output/sample_benchmark.jsonl``
  file end-to-end and verifies every row maps cleanly through
  ``Example.from_benchmark_row``. Catches schema-drift bugs the unit
  tests would miss.

The Prediction / Usage / ToolCall / Model layer is exercised by a small
``DummyModel`` round-trip so we know the abstractions compose without
import-time issues.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_models.base import (
    EntityRef,
    Example,
    Model,
    Prediction,
    ToolCall,
    Usage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "sample_benchmark.jsonl"
)


# ============================================================================
# parse_bool_or_none — gold-field boolean helper
# ============================================================================


class TestParseBoolOrNone:
    """Cover the gold-field boolean coercion helper.

    The upstream benchmark writes ``is_true_premise`` as the *string*
    ``"false"`` / ``"true"`` rather than a JSON boolean. Without this
    helper, ``if row.get('is_true_premise')`` treats the string
    ``"false"`` as truthy and silently inverts premise-aware scoring.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("t", True),
            ("f", False),
            ("yes", True),
            ("no", False),
            ("1", True),
            ("0", False),
            ("  true  ", True),
            (None, None),
            ("", None),
        ],
    )
    def test_valid_values(self, value, expected) -> None:
        from benchmark_models.base import parse_bool_or_none

        assert parse_bool_or_none(value) is expected

    @pytest.mark.parametrize("value", ["maybe", "partly", "2", "TrueFalse"])
    def test_unparseable_raises(self, value: str) -> None:
        """Fail loud on unknown string values so data-entry regressions
        surface immediately rather than masquerading as False."""
        from benchmark_models.base import parse_bool_or_none

        with pytest.raises(ValueError, match="cannot interpret"):
            parse_bool_or_none(value)

    def test_unparseable_type_raises(self) -> None:
        from benchmark_models.base import parse_bool_or_none

        with pytest.raises(ValueError):
            parse_bool_or_none([True])

    def test_regression_benchmark_uses_string_false(self) -> None:
        """Pin the observed upstream format so a cast change to real
        bools upstream (future) will flag here first."""
        from benchmark_models.base import parse_bool_or_none

        # This is the exact literal in output/sample_benchmark_oracle_mini.json.
        assert parse_bool_or_none("false") is False


# ============================================================================
# EntityRef
# ============================================================================


class TestEntityRef:
    def test_minimal_construction(self) -> None:
        ref = EntityRef(id="42", name="Eiffel Tower")
        assert ref.id == "42"
        assert ref.name == "Eiffel Tower"
        assert ref.category is None

    def test_with_category(self) -> None:
        ref = EntityRef(id="42", name="Eiffel Tower", category="points of interest")
        assert ref.category == "points of interest"

    def test_with_subcategory(self) -> None:
        ref = EntityRef(id="42", name="Pharmacy Central", subcategory="Pharmacy")
        assert ref.subcategory == "Pharmacy"
        assert ref.category is None

    def test_with_category_and_subcategory(self) -> None:
        ref = EntityRef(
            id="42", name="Pharmacy Central",
            category="points of interest", subcategory="Pharmacy",
        )
        assert ref.category == "points of interest"
        assert ref.subcategory == "Pharmacy"

    def test_category_only_slot(self) -> None:
        # A slot that represents a *type* ("any Pharmacy") rather than
        # an instance: empty id and name are explicit and legal.
        ref = EntityRef(id="", name="", subcategory="Pharmacy")
        assert ref.id == ""
        assert ref.name == ""
        assert ref.subcategory == "Pharmacy"

    def test_id_and_name_required(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
            EntityRef(name="Eiffel Tower")  # type: ignore[call-arg]
        with pytest.raises(Exception):  # noqa: B017
            EntityRef(id="42")  # type: ignore[call-arg]


# ============================================================================
# Example.from_benchmark_row — synthetic-dict unit tests
# ============================================================================


def _row_two_entities() -> dict:
    """Mirrors a row from ``output/sample_benchmark.jsonl``."""
    return {
        "question_id": "1285b5ef-7e01-4bfe-ae5a-525940982294",
        "tid": "1.1",
        "sfid": "A",
        "question": "How far apart are Cafe Kanel and Seniorenzentrum St. Michael?",
        "base_template": "What is the distance between {{entity_A}} and {{entity_B}}?",
        "ground_truth": "12.247331472514373",
        "ground_truth_type": "distance_kilometers",
        "entity_a_id": "2476384356",
        "entity_a_name": "Cafe Kanel",
        "entity_b_id": "2068224",
        "entity_b_name": "Seniorenzentrum St. Michael",
        "kg_source": "osmh3kg",
        "strata_cell": "HIC_high",
        "is_valid": True,
    }


class TestExampleFromBenchmarkRow:
    def test_two_entity_row_loads_cleanly(self) -> None:
        ex = Example.from_benchmark_row(_row_two_entities())
        assert ex.question_id == "1285b5ef-7e01-4bfe-ae5a-525940982294"
        assert ex.tid == "1.1"
        assert ex.sfid == "A"
        assert ex.kg_source == "osmh3kg"
        assert ex.strata_cell == "HIC_high"
        assert ex.language == "eng"
        assert ex.image_path is None

    def test_two_entity_slots_populated(self) -> None:
        ex = Example.from_benchmark_row(_row_two_entities())
        assert set(ex.entities) == {"A", "B"}
        assert ex.entity_a is not None
        assert ex.entity_a.id == "2476384356"
        assert ex.entity_a.name == "Cafe Kanel"
        assert ex.entity_a.category is None
        assert ex.entity_b is not None
        assert ex.entity_b.name == "Seniorenzentrum St. Michael"
        assert ex.entity_c is None
        assert ex.entity_d is None

    def test_entity_iteration_helpers(self) -> None:
        ex = Example.from_benchmark_row(_row_two_entities())
        assert ex.all_entity_ids() == ["2476384356", "2068224"]
        assert ex.all_entity_names() == ["Cafe Kanel", "Seniorenzentrum St. Michael"]
        assert ex.has_entity("A") is True
        assert ex.has_entity("a") is True  # case-insensitive
        assert ex.has_entity("C") is False

    def test_gold_fields_are_stripped(self) -> None:
        row = _row_two_entities()
        # Seed two more gold-like fields so the stripping is actually
        # exercised (the shared fixture only carries the core three).
        row["validation_errors"] = ["some earlier check"]
        row["is_true_premise"] = "false"
        ex = Example.from_benchmark_row(row)
        # Gold answer-value fields are stripped. ``ground_truth_type``
        # is intentionally NOT stripped: it's the schema label for the
        # expected answer, not the answer itself, and backends use it
        # to build per-question JSON schemas.
        for field in (
            "ground_truth", "is_valid",
            "validation_errors", "error", "is_true_premise",
        ):
            assert field not in ex.raw, f"{field} leaked into Example.raw"
            assert not hasattr(ex, field), f"{field} leaked onto Example attrs"

    def test_is_true_premise_does_not_trigger_unknown_field_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # is_true_premise is gold metadata — it must be silently
        # stripped by from_benchmark_row, not emitted as an
        # "unrecognized field" warning.
        import logging
        row = _row_two_entities()
        row["is_true_premise"] = "false"
        with caplog.at_level(logging.WARNING, logger="benchmark_models.base"):
            ex = Example.from_benchmark_row(row)
        assert "is_true_premise" not in ex.raw
        for rec in caplog.records:
            assert "is_true_premise" not in rec.getMessage(), (
                f"unexpected warning: {rec.getMessage()!r}"
            )

    def test_four_entity_row(self) -> None:
        row = {
            "question_id": "qid-4ent",
            "tid": "24.1",
            "sfid": "C",
            "question": (
                "Does the country on which Eiffel Tower is located intersect "
                "the river on which Seine bridge is located?"
            ),
            "base_template": (
                "Does the {{entity_A}} on which {{entity_B}} is located "
                "intersect the {{entity_C}} on which {{entity_D}} is located?"
            ),
            "entity_a_id": "1",
            "entity_a_name": "France",
            "entity_b_id": "2",
            "entity_b_name": "Eiffel Tower",
            "entity_c_id": "3",
            "entity_c_name": "Seine",
            "entity_d_id": "4",
            "entity_d_name": "Pont Neuf",
            "kg_source": "worldkg",
        }
        ex = Example.from_benchmark_row(row)
        assert set(ex.entities) == {"A", "B", "C", "D"}
        assert ex.entity_d is not None
        assert ex.entity_d.name == "Pont Neuf"
        assert ex.all_entity_ids() == ["1", "2", "3", "4"]

    def test_parameters_parsed_from_json_string(self) -> None:
        row = {
            "question_id": "qid-params",
            "tid": "2.1",
            "sfid": "A",
            "question": "How far north is X from Y?",
            "base_template": "How far {{cardinal_direction}} is {{entity_A}} from {{entity_B}}?",
            "entity_a_id": "1",
            "entity_a_name": "X",
            "entity_b_id": "2",
            "entity_b_name": "Y",
            "kg_source": "osmh3kg",
            "parameters": json.dumps(
                {
                    "cardinal_direction": "North",
                    "entity_A_category": "natural features",
                }
            ),
        }
        ex = Example.from_benchmark_row(row)
        assert ex.parameters == {
            "cardinal_direction": "North",
            "entity_A_category": "natural features",
        }
        # Category from parameters propagates onto the EntityRef.
        assert ex.entity_a is not None
        assert ex.entity_a.category == "natural features"
        # B has no category constraint, stays None.
        assert ex.entity_b is not None
        assert ex.entity_b.category is None

    def test_parameters_already_dict(self) -> None:
        # Some loaders may pre-parse parameters; the loader should accept
        # both string and dict forms.
        row = _row_two_entities()
        row["parameters"] = {"cardinal_direction": "South"}
        ex = Example.from_benchmark_row(row)
        assert ex.parameters == {"cardinal_direction": "South"}

    def test_malformed_parameters_json_falls_back_to_empty(self) -> None:
        row = _row_two_entities()
        row["parameters"] = "{not valid json"
        ex = Example.from_benchmark_row(row)
        assert ex.parameters == {}

    def test_missing_parameters_field_defaults_to_empty(self) -> None:
        # The current sample_benchmark.jsonl has no `parameters` field.
        row = _row_two_entities()
        assert "parameters" not in row
        ex = Example.from_benchmark_row(row)
        assert ex.parameters == {}

    def test_partial_entity_slot_is_skipped_with_warning(self) -> None:
        row = _row_two_entities()
        # Drop entity_b_name; loader should skip slot B rather than guess.
        del row["entity_b_name"]
        ex = Example.from_benchmark_row(row)
        assert "A" in ex.entities
        assert "B" not in ex.entities

    def test_subcategory_on_instance_slot(self) -> None:
        # An instance slot (id + name) can additionally carry a
        # top-level entity_<letter>_subcategory; the loader plumbs it
        # into EntityRef.subcategory.
        row = _row_two_entities()
        row["entity_a_subcategory"] = "Pharmacy"
        ex = Example.from_benchmark_row(row)
        assert ex.entity_a is not None
        assert ex.entity_a.name == "Cafe Kanel"  # name still populated
        assert ex.entity_a.subcategory == "Pharmacy"
        # entity_b has no subcategory, stays None.
        assert ex.entity_b is not None
        assert ex.entity_b.subcategory is None

    def test_category_only_slot_from_subcategory(self) -> None:
        # Category-based question: entity_a_id / entity_a_name are
        # absent; entity_a_subcategory carries the slot's identity
        # ("Pharmacy" in the real benchmark). Loader creates an
        # EntityRef with empty id/name so downstream iteration keeps
        # the slot.
        row = _row_two_entities()
        del row["entity_a_id"]
        del row["entity_a_name"]
        row["entity_a_subcategory"] = "Pharmacy"
        ex = Example.from_benchmark_row(row)
        assert ex.entity_a is not None
        assert ex.entity_a.id == ""
        assert ex.entity_a.name == ""
        assert ex.entity_a.subcategory == "Pharmacy"
        # entity_b — the instance — still present.
        assert ex.entity_b is not None
        assert ex.entity_b.name == "Seniorenzentrum St. Michael"

    def test_subcategory_does_not_trigger_unknown_field_warning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # entity_<letter>_subcategory is consumed by the entity-slot
        # extractor, so it must not reach the "unrecognized field"
        # warning path and must not land in Example.raw.
        import logging
        row = _row_two_entities()
        row["entity_a_subcategory"] = "Pharmacy"
        with caplog.at_level(logging.WARNING, logger="benchmark_models.base"):
            ex = Example.from_benchmark_row(row)
        assert "entity_a_subcategory" not in ex.raw
        # No warning message mentioning subcategory.
        for rec in caplog.records:
            assert "subcategory" not in rec.getMessage().lower(), (
                f"unexpected warning: {rec.getMessage()!r}"
            )

    def test_empty_slot_without_subcategory_still_skipped(self) -> None:
        # Removing both id and name with NO subcategory should yield
        # no EntityRef for that letter — same as before this change.
        row = _row_two_entities()
        del row["entity_a_id"]
        del row["entity_a_name"]
        ex = Example.from_benchmark_row(row)
        assert "A" not in ex.entities
        assert "B" in ex.entities

    def test_language_override(self) -> None:
        row = _row_two_entities()
        ex = Example.from_benchmark_row(row, language="deu")
        assert ex.language == "deu"

    def test_language_from_row_field(self) -> None:
        row = _row_two_entities()
        row["language"] = "fra"
        ex = Example.from_benchmark_row(row)
        assert ex.language == "fra"

    def test_language_override_beats_row(self) -> None:
        row = _row_two_entities()
        row["language"] = "fra"
        ex = Example.from_benchmark_row(row, language="zho_CN")
        assert ex.language == "zho_CN"

    def test_language_default_when_absent(self) -> None:
        row = _row_two_entities()
        ex = Example.from_benchmark_row(row)
        assert ex.language == "eng"

    def test_unknown_field_lands_in_raw_with_warning(self) -> None:
        row = _row_two_entities()
        row["future_field"] = {"some": "data"}
        ex = Example.from_benchmark_row(row)
        assert ex.raw == {"future_field": {"some": "data"}}

    def test_strata_cell_optional(self) -> None:
        row = _row_two_entities()
        del row["strata_cell"]
        ex = Example.from_benchmark_row(row)
        assert ex.strata_cell is None

    def test_extra_fields_on_example_are_forbidden(self) -> None:
        # Construction directly (not via from_benchmark_row) should
        # reject unknown fields, so the typed surface stays clean.
        with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
            Example(
                question_id="x",
                tid="x",
                sfid="x",
                question="x",
                base_template="x",
                kg_source="x",
                bogus_field="oops",  # type: ignore[call-arg]
            )


class TestExampleEvents:
    """Coverage for the ``event_<letter>_*`` slot extraction used by
    SFID I event-driven templates."""

    def _row_with_event_a(self) -> dict:
        """SFID I row with both an entity slot and an event slot."""
        return {
            "question_id": "evt-1",
            "tid": "77.1",
            "sfid": "I",
            "question": (
                'When did "Earthquake from USGS Earthquake Catalog with ID '
                'pr71432043" hit the Federal Judicial District?'
            ),
            "base_template": "When did {{event_A}} hit {{entity_A}}?",
            "ground_truth": "2023-11-24",
            "ground_truth_type": "temporal",
            "entity_a_id": "federalJudicialDistrict.87",
            "entity_a_name": "Federal Judicial District: District of Puerto Rico",
            "event_a_id": "earthquake.pr71432043",
            "event_a_name": (
                '"Earthquake from USGS Earthquake Catalog with ID pr71432043"'
            ),
            "kg_source": "kwg",
            "strata_cell": "HIC_high",
            "is_valid": True,
        }

    def test_event_slot_populated(self) -> None:
        ex = Example.from_benchmark_row(self._row_with_event_a())
        assert set(ex.events) == {"A"}
        assert ex.events["A"].id == "earthquake.pr71432043"
        assert ex.events["A"].name.startswith('"Earthquake')

    def test_entity_and_event_coexist(self) -> None:
        ex = Example.from_benchmark_row(self._row_with_event_a())
        # Entity slot still populated; event lives in the sibling dict.
        assert set(ex.entities) == {"A"}
        assert ex.entity_a is not None
        assert ex.entity_a.id == "federalJudicialDistrict.87"
        assert ex.events["A"].id == "earthquake.pr71432043"

    def test_event_only_row(self) -> None:
        # SFID I TIDs 78.1/79.1/80.1 emit ``event_a_name`` only
        # (no event_a_id — known design / open question with the
        # benchmark owner). Loader emits a partial-pair warning and
        # skips the slot rather than guessing.
        row = {
            "question_id": "evt-name-only",
            "tid": "78.1",
            "sfid": "I",
            "question": "x",
            "base_template": "x",
            "ground_truth": "true",
            "ground_truth_type": "boolean",
            "kg_source": "kwg",
            "is_valid": True,
            "entity_a_id": "x",
            "entity_a_name": "x",
            "event_a_name": '"hurricane.x"',
        }
        ex = Example.from_benchmark_row(row)
        # Partial event slot is dropped; the warning was logged.
        assert "A" not in ex.events
        # event_a_name should still be marked consumed (no
        # unrecognized-field warning, no leak into raw).
        assert "event_a_name" not in ex.raw

    def test_no_event_fields_yields_empty_dict(self) -> None:
        ex = Example.from_benchmark_row(_row_two_entities())
        assert ex.events == {}

    def test_event_b_name_does_not_leak_to_raw(self) -> None:
        # SFID I TIDs 78.1/79.1 carry event_a_name + event_b_name without
        # corresponding ids — both partial pairs get skipped, neither
        # field should leak into ``raw``.
        row = {
            "question_id": "evt-both-name-only",
            "tid": "78.1",
            "sfid": "I",
            "question": "x",
            "base_template": "x",
            "ground_truth": "true",
            "ground_truth_type": "boolean",
            "kg_source": "kwg",
            "is_valid": True,
            "entity_a_id": "x",
            "entity_a_name": "x",
            "event_a_name": '"hurricane.x"',
            "event_b_name": '"earthquake.y"',
        }
        ex = Example.from_benchmark_row(row)
        assert "event_a_name" not in ex.raw
        assert "event_b_name" not in ex.raw


class TestExampleImages:
    """Coverage for the multimodal ``images`` field and the related
    ``is_multimodal`` flag; the cleanup script strips data URIs out of
    the question text."""

    def _row_with_images(self, image_count: int = 1) -> dict:
        images = [
            {
                "slot": chr(ord("A") + i),
                "path": f"benchmark/raw_eng/multimodal/x/images/q1_{chr(ord('A') + i)}.png",
                "mime_type": "image/png",
            }
            for i in range(image_count)
        ]
        return {
            "question_id": "q1",
            "tid": "1.1",
            "sfid": "A",
            "question": " and ".join(
                [f"[IMAGE_{img['slot']}]" for img in images]
            ),
            "base_template": "...",
            "ground_truth_type": "numeric",
            "kg_source": "osmh3kg",
            "is_multimodal": True,
            "images": images,
            "entity_a_id": "ent-1",
            "entity_a_name": "Mt Test",
        }

    def test_images_populate_into_imagecontent_list(self) -> None:
        ex = Example.from_benchmark_row(self._row_with_images(image_count=2))
        assert len(ex.images) == 2
        assert ex.images[0].slot == "A"
        assert ex.images[1].slot == "B"
        assert ex.images[0].mime_type == "image/png"

    def test_is_multimodal_propagates(self) -> None:
        ex = Example.from_benchmark_row(self._row_with_images())
        assert ex.is_multimodal is True

    def test_images_default_empty_list(self) -> None:
        ex = Example.from_benchmark_row(_row_two_entities())
        assert ex.images == []
        assert ex.is_multimodal is False

    def test_malformed_images_entry_is_skipped(self) -> None:
        row = self._row_with_images()
        # A non-dict entry — should be skipped with a warning, not crash.
        row["images"].append("not-a-dict")  # type: ignore[arg-type]
        ex = Example.from_benchmark_row(row)
        # The good entry survived; the bad one was dropped.
        assert len(ex.images) == 1

    def test_images_path_unchanged_without_source_path(self) -> None:
        # When source_path is not provided, the stored path is preserved
        # verbatim — matches prior behaviour for synthetic tests.
        ex = Example.from_benchmark_row(self._row_with_images())
        assert ex.images[0].path.startswith("benchmark/raw_eng/")

    def test_images_path_resolved_with_source_path(self, tmp_path) -> None:
        # When source_path is provided and lives under a multimodal/
        # tree, ImageContent.path is rewritten to the absolute resolved
        # location regardless of what the JSONL row stored.
        from pathlib import Path

        shard = (
            tmp_path / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        )
        shard.mkdir(parents=True)
        row = self._row_with_images()
        # Row's stored path is the legacy stub; resolver should
        # re-root it under tmp_path using the row's kg_source.
        ex = Example.from_benchmark_row(row, source_path=shard)
        assert Path(ex.images[0].path).is_absolute()
        assert ex.images[0].path == str(
            tmp_path / "multimodal" / "osmh3kg" / "images" / "q1_A.png"
        )

    def test_images_path_resolved_clean_form(self, tmp_path) -> None:
        from pathlib import Path

        shard = (
            tmp_path / "multimodal" / "worldkg" / "eng" / "benchmark.jsonl"
        )
        shard.mkdir(parents=True)
        row = self._row_with_images()
        row["kg_source"] = "worldkg"
        # Override with the published convention path shape.
        row["images"][0]["path"] = "images/q1_A.png"
        ex = Example.from_benchmark_row(row, source_path=shard)
        assert Path(ex.images[0].path).is_absolute()
        assert ex.images[0].path == str(
            tmp_path / "multimodal" / "worldkg" / "images" / "q1_A.png"
        )

    def test_image_attribution_fields_land_in_raw_silently(self) -> None:
        # Source / URL attribution fields preserved by the cleanup
        # script land in ``raw`` so analysis code can surface them,
        # but without producing schema-drift warnings.
        from benchmark_models.base import (
            _WARNED_UNKNOWN_FIELDS,
            reset_schema_drift_warnings,
        )

        reset_schema_drift_warnings()
        row = self._row_with_images()
        row["entity_a_image_source"] = "https://example.com/photo"
        row["entity_a_image_url"] = "https://example.com/x.png"
        ex = Example.from_benchmark_row(row)
        # Preserved on raw for analysis access.
        assert ex.raw["entity_a_image_source"] == "https://example.com/photo"
        assert ex.raw["entity_a_image_url"] == "https://example.com/x.png"
        # And the warn-once registry didn't pick them up.
        assert "entity_a_image_source" not in _WARNED_UNKNOWN_FIELDS
        assert "entity_a_image_url" not in _WARNED_UNKNOWN_FIELDS


class TestExampleDoesNotMutateInputRow:
    """``Example.from_benchmark_row`` must not mutate the caller's
    dict. Regression guard: the previous implementation ``row.pop``-ed
    gold fields in place, which leaked to callers that reused the
    same dict across calls (e.g., class-scoped test fixtures).
    """

    def test_gold_fields_remain_in_caller_dict(self) -> None:
        row = {
            "question_id": "q-no-mutate",
            "tid": "24.1",
            "sfid": "C",
            "question": "q?",
            "base_template": "t",
            "kg_source": "osm",
            "ground_truth": True,
            "ground_truth_type": "boolean",
            "is_true_premise": True,
            "is_valid": True,
            "validation_errors": [],
        }
        original_keys = set(row.keys())
        original_gt = row["ground_truth"]

        Example.from_benchmark_row(row)

        assert set(row.keys()) == original_keys
        assert row["ground_truth"] == original_gt
        assert row["ground_truth_type"] == "boolean"
        assert row["is_true_premise"] is True

    def test_repeated_conversion_yields_same_example(self) -> None:
        """Calling twice on the same dict must produce equal Examples
        (would have regressed under in-place mutation if a second call
        saw a stripped dict)."""
        row = {
            "question_id": "q-rep",
            "tid": "24.1",
            "sfid": "C",
            "question": "q?",
            "base_template": "t",
            "kg_source": "osm",
            "ground_truth": False,
            "ground_truth_type": "boolean",
        }
        a = Example.from_benchmark_row(row)
        b = Example.from_benchmark_row(row)
        assert a.model_dump() == b.model_dump()


# ============================================================================
# Example — integration test against the real sample
# ============================================================================


class TestExampleAgainstSampleBenchmark:
    """Loads every row in ``output/sample_benchmark.jsonl`` end-to-end.

    This is the smoke test that catches schema drift between the
    benchmark generator and the model section's loader. The sample
    is small (~270 rows) so the test runs in well under a second.
    """

    @pytest.fixture(scope="class")
    def rows(self) -> list[dict]:
        with SAMPLE_BENCHMARK_PATH.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_sample_is_nonempty(self, rows: list[dict]) -> None:
        assert len(rows) > 0

    def test_every_row_loads_into_example(self, rows: list[dict]) -> None:
        examples = [Example.from_benchmark_row(row) for row in rows]
        assert len(examples) == len(rows)

    def test_every_example_has_required_fields(self, rows: list[dict]) -> None:
        # ``base_template`` is intentionally not asserted non-empty:
        # the current sample has 120/270 rows (TIDs 2.5–2.8) with an
        # empty ``base_template`` value, and the model never reads it
        # directly. The field exists for analysis / template-shape
        # reporting only.
        for row in rows:
            ex = Example.from_benchmark_row(row)
            assert ex.question_id
            assert ex.tid
            assert ex.sfid
            assert ex.question
            assert ex.kg_source

    def test_every_example_has_at_least_one_entity(self, rows: list[dict]) -> None:
        # Every row in the current sample is a distance question with
        # entities A and B. Future samples may include questions with
        # more or fewer entities; this test will need updating then.
        for row in rows:
            ex = Example.from_benchmark_row(row)
            assert len(ex.entities) >= 1

    def test_no_gold_fields_leak_into_examples(self, rows: list[dict]) -> None:
        # ``ground_truth_type`` is intentionally exposed as schema
        # metadata (shape of the expected answer) — backends use it to
        # build the per-question response schema. It is *not* the
        # answer, so it does not count as a gold leak.
        for row in rows:
            ex = Example.from_benchmark_row(row)
            dumped = ex.model_dump()
            assert "ground_truth" not in dumped
            assert "is_valid" not in dumped
            assert "validation_errors" not in dumped
            assert "error" not in dumped
            assert "is_true_premise" not in dumped

    def test_strata_cell_propagates(self, rows: list[dict]) -> None:
        # Every row in the current sample has strata_cell = "HIC_high".
        examples = [Example.from_benchmark_row(row) for row in rows]
        non_null = [ex for ex in examples if ex.strata_cell is not None]
        assert len(non_null) > 0

    def test_round_trip_preserves_id(self, rows: list[dict]) -> None:
        # The set of question_ids on Example matches the set in the
        # source rows — no row is silently dropped.
        loaded_ids = {Example.from_benchmark_row(row).question_id for row in rows}
        source_ids = {row["question_id"] for row in rows}
        assert loaded_ids == source_ids


# ============================================================================
# Prediction / Usage / ToolCall
# ============================================================================


class TestPrediction:
    def _provenance_kwargs(self) -> dict:
        return {
            "question_id": "qid-1",
            "tid": "1.1",
            "sfid": "A",
            "answer_type": "Numeric (float)",
            "model_id": "qwen35-a3b-plain",
            "tier": "tier1",
            "language": "eng",
            "prompt_template_id": "tier1.canonical",
            "seed": 42,
            "snapshot_id": "abc123:v5",
        }

    def test_tier1_minimal_prediction(self) -> None:
        p = Prediction(answer=12.5, unit="km", **self._provenance_kwargs())
        assert p.refusal is False
        assert p.tool_calls is None
        assert p.cap_hit is None
        assert p.final_state is None
        assert p.usage.input_tokens == 0
        assert p.latency_ms == 0.0

    def test_tier3_with_tool_calls(self) -> None:
        prov = self._provenance_kwargs()
        prov["tier"] = "tier3"
        prov["model_id"] = "gpt-5.4-tools"
        p = Prediction(
            answer=True,
            tool_calls=[
                ToolCall(name="find_entity", args={"name": "Eiffel"}, result={"id": "99"}),
                ToolCall(
                    name="spatial_query",
                    args={"op": "distance", "a": "99", "b": "100"},
                    result=2.5,
                ),
            ],
            cap_hit=False,
            final_state="answered",
            usage=Usage(input_tokens=1500, output_tokens=200, tool_call_count=2, api_call_count=3),
            latency_ms=4250.0,
            **prov,
        )
        assert p.tool_calls is not None
        assert len(p.tool_calls) == 2
        assert p.usage.tool_call_count == 2
        assert p.final_state == "answered"

    def test_jsonl_round_trip(self) -> None:
        prov = self._provenance_kwargs()
        p = Prediction(answer=12.5, unit="km", latency_ms=123.4, **prov)
        line = p.model_dump_json()
        p2 = Prediction.model_validate_json(line)
        assert p2 == p


# ============================================================================
# Model — abstract base + a minimal subclass for round-trip
# ============================================================================


class TestModelABC:
    def test_model_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Model()  # type: ignore[abstract]

    def test_minimal_subclass_round_trips(self) -> None:
        class FakeModel(Model):
            model_id = "fake"
            tier = "tier1"

            def predict(self, example: Example) -> Prediction:
                return Prediction(
                    question_id=example.question_id,
                    tid=example.tid,
                    sfid=example.sfid,
                    answer_type="Numeric (float)",
                    model_id=self.model_id,
                    tier=self.tier,
                    language=example.language,
                    prompt_template_id="fake",
                    seed=0,
                    snapshot_id="fake",
                    answer=None,
                )

        ex = Example.from_benchmark_row(
            {
                "question_id": "qid",
                "tid": "1.1",
                "sfid": "A",
                "question": "Q",
                "base_template": "T",
                "entity_a_id": "1",
                "entity_a_name": "X",
                "entity_b_id": "2",
                "entity_b_name": "Y",
                "kg_source": "osmh3kg",
            }
        )
        model = FakeModel()
        preds = model.predict_batch([ex, ex])
        assert len(preds) == 2
        assert all(p.question_id == "qid" for p in preds)
