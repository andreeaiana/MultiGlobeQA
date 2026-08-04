"""Per-TID canonical overrides take precedence on the model side.

The resolution order in ``resolve_canonical_answer_type``:

  1. ``TID_TO_CANONICAL_OVERRIDE[tid]`` if present  ← highest
  2. ``Example.ground_truth_type`` (per-row label)
  3. ``TemplateInfo.answer_type`` (registry fallback)

The override map exists for two classes of TID where (2) alone gives
the wrong answer:

* **Overloaded benchmark types** (``categorical``, ``grid_cell_id``)
  where one tag covers multiple distinct canonical types — the per-TID
  override disambiguates which canonical applies.
* **Benchmark-side mislabels** where the row's ``ground_truth_type``
  literally describes the wrong shape. (None currently.)

TIDs are the released ids (parents 1..66). These tests pin the
behaviour: when ``tid`` is in the override map, it wins regardless of
the row label.
"""

from __future__ import annotations

import pytest

from benchmark_models.schemas.type_registry import (
    TID_TO_CANONICAL_OVERRIDE,
    resolve_canonical_answer_type,
)


class TestResolveWithOverride:
    def test_tid_62_1_ternary_override(self) -> None:
        """Overloaded ``categorical`` tag: TID 62.1 is Ternary
        (yes/no/plausibly), not whatever ``categorical`` would
        canonicalise to by default."""
        out = resolve_canonical_answer_type(
            ground_truth_type="categorical",
            fallback=None,
            tid="62.1",
        )
        assert out == "Ternary (true/false/plausibly)"

    def test_tid_50_1_predefined_left_right_override(self) -> None:
        """Overloaded ``categorical`` tag: TID 50.1 picks the
        left/right enum, not the cardinal-direction or
        upstream/downstream alternates."""
        out = resolve_canonical_answer_type(
            ground_truth_type="categorical",
            fallback=None,
            tid="50.1",
        )
        assert out == "Predefined value (left, right)"

    def test_tid_52_1_cardinal_direction_override(self) -> None:
        """Overloaded ``categorical`` tag: TID 52.1 is a cardinal
        direction (Northeast, West, ...)."""
        out = resolve_canonical_answer_type(
            ground_truth_type="categorical",
            fallback=None,
            tid="52.1",
        )
        assert out == "Cardinal direction (predefined)"

    def test_tid_64_2_s2_override(self) -> None:
        """Overloaded ``grid_cell_id`` tag: TID 64.2 is S2, not H3."""
        out = resolve_canonical_answer_type(
            ground_truth_type="grid_cell_id",
            fallback=None,
            tid="64.2",
        )
        assert out == "S2 cell index"

    def test_tid_64_4_coordinates_override(self) -> None:
        """Overloaded ``grid_cell_id`` tag: TID 64.4 is a
        coordinate-string (lat_lon), not actually a cell index at all."""
        out = resolve_canonical_answer_type(
            ground_truth_type="grid_cell_id",
            fallback=None,
            tid="64.4",
        )
        assert out == "Coordinates (lat/long pair)"

    def test_unknown_tid_falls_through_to_row_label(self) -> None:
        """When the TID has no override, the row label wins (per the
        existing precedence). TID 999.999 is not in the map."""
        out = resolve_canonical_answer_type(
            ground_truth_type="boolean",
            fallback="Numeric (float)",
            tid="999.999",
        )
        assert out == "Boolean (yes/no, true/false)"

    def test_no_tid_passed_falls_through_to_row_label(self) -> None:
        """Existing callers that don't pass ``tid`` keep their old
        behaviour — back-compat."""
        out = resolve_canonical_answer_type(
            ground_truth_type="coordinates",
            fallback="Numeric (float)",
        )
        assert out == "Coordinates (lat/long pair)"


class TestBackendsThreadTidThrough:
    """Each backend's schema-build call site must pass ``tid`` so
    the override map applies on the model side. We exercise the
    full ``(ground_truth_type, registry-fallback, tid)`` resolution
    chain for a representative overloaded TID, mirroring what the
    backends do at schema build time.
    """

    @pytest.fixture
    def tid_62_1_example(self):
        from benchmark_models.base import Example

        return Example(
            question_id="q1",
            tid="62.1",  # ternary (yes/no/plausibly), tagged 'categorical'
            sfid="L",
            language="eng",
            kg_source="kwg",
            question="Is X plausibly within reach of Y?",
            base_template="t",
            ground_truth_type="categorical",
            entities={},
        )

    def test_override_returns_ternary_for_tid_62_1(
        self, tid_62_1_example
    ) -> None:
        """End-to-end: ``Example.ground_truth_type=categorical`` plus
        ``tid='62.1'`` resolves to Ternary via the override map,
        identical to what the eval pipeline does."""
        from benchmark_models.schemas import load_template_registry

        registry = load_template_registry()
        info = registry.get("62.1")
        out = resolve_canonical_answer_type(
            tid_62_1_example.ground_truth_type,
            info.answer_type,
            tid=tid_62_1_example.tid,
        )
        assert out == "Ternary (true/false/plausibly)"

    def test_override_map_keys_present(self) -> None:
        """Pin every TID we know is currently overloaded, so a future
        regen of the registry can't silently drop entries. Keys are
        released ids in the 1..66 parent range."""
        expected_keys = {
            # overloaded `categorical`
            "50.1", "51.1", "52.1", "62.1",
            # overloaded `grid_cell_id` (parent 64 = grid-cell
            # encodings; 64.4 = Coordinates)
            "40.1", "41.1", "64.1", "64.2", "64.3", "64.4",
        }
        assert expected_keys.issubset(set(TID_TO_CANONICAL_OVERRIDE.keys()))

    def test_unreleased_tids_not_override_keys(self) -> None:
        """Defensive guard: none of the TID strings below may be a key
        in the override map. Some are not TIDs under the current ids at
        all; the others are current TIDs that are not overloaded and so
        must resolve from the row label rather than an override."""
        old_tids = {
            "91.1", "92.1", "93.1", "105.1",
            "67.1",
            "111.1", "111.2", "111.3", "111.4", "111.5",
            "115.1",
            "60.1", "61.1", "73.1",
            "49.1", "75.1", "75.2", "75.3", "75.4", "75.5",
            "58.1", "69.1", "47.1", "71.1", "71.2", "71.3", "71.4",
            "45.1", "46.1", "55.1", "56.1", "57.1", "68.1",
            "70.1", "70.2", "70.3", "70.4",
        }
        stale = old_tids & set(TID_TO_CANONICAL_OVERRIDE.keys())
        assert stale == set(), (
            f"Override map contains keys that are not released TIDs: "
            f"{sorted(stale)}"
        )
