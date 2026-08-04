"""Tests for ``benchmark_models.dummy.DummyModel``.

DummyModel is the harness's behavioral stand-in for a real backend.
The tests verify that:

* Each :class:`DummyStrategy` produces the expected answer shape for
  every canonical answer type.
* Dynamic-type templates emit the strict-error :class:`Prediction`
  the runner expects (per the plan's "Dynamic-type templates —
  deferred" section).
* Provenance fields (model_id, tier, snapshot_id, etc.) are
  populated from constructor args, and the per-instance shadowing
  of ``model_id`` / ``tier`` works.
* The runner-facing contract (every Prediction has a Usage,
  latency_ms, raw_completion) is satisfied even on the error path.
* ``predict_batch`` (inherited default from :class:`Model`) works
  end-to-end against the bundled registry.

The tests use the *bundled* template registry rather than synthetic
fixtures because (a) it's already loaded in process by other tests,
(b) DummyModel needs a real registry to look up tids, and (c) the
real registry is small enough that test setup time is negligible.
"""

from __future__ import annotations

from benchmark_models.base import Example, Prediction
from benchmark_models.dummy import DummyModel, DummyStrategy, _type_default
from benchmark_models.schemas import TYPE_REGISTRY


def _example(
    *,
    tid: str = "1.1",
    sfid: str = "A",
    entity_a_name: str | None = "Cafe Kanel",
    entity_b_name: str | None = "Seniorenzentrum St. Michael",
) -> Example:
    """Build a minimal Example via from_benchmark_row."""
    row = {
        "question_id": f"qid-{tid}",
        "tid": tid,
        "sfid": sfid,
        "question": "How far apart are X and Y?",
        "base_template": "What is the distance between {{entity_A}} and {{entity_B}}?",
        "kg_source": "osmh3kg",
    }
    if entity_a_name is not None:
        row["entity_a_id"] = "1"
        row["entity_a_name"] = entity_a_name
    if entity_b_name is not None:
        row["entity_b_id"] = "2"
        row["entity_b_name"] = entity_b_name
    return Example.from_benchmark_row(row)


# ============================================================================
# Strategy: ALWAYS_NULL
# ============================================================================


class TestAlwaysNullStrategy:
    def test_returns_null_answer(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.ALWAYS_NULL, registry=registry)
        pred = model.predict(_example())
        assert pred.answer is None
        assert pred.refusal is True

    def test_provenance_fields_set(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.ALWAYS_NULL, registry=registry)
        pred = model.predict(_example(tid="1.1", sfid="A"))
        assert pred.question_id == "qid-1.1"
        assert pred.tid == "1.1"
        assert pred.sfid == "A"  # looked up from the registry
        assert pred.model_id == "dummy"
        assert pred.tier == "tier1"
        assert pred.snapshot_id == "dummy"
        assert pred.prompt_template_id == "dummy.canonical"
        assert pred.seed == 0

    def test_usage_and_latency_always_present(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.ALWAYS_NULL, registry=registry)
        pred = model.predict(_example())
        assert pred.usage is not None
        assert pred.usage.api_call_count == 1
        assert pred.latency_ms == 0.0

    def test_raw_completion_records_strategy(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.ALWAYS_NULL, registry=registry)
        pred = model.predict(_example())
        assert "always_null" in pred.raw_completion


# ============================================================================
# Strategy: ECHO_ENTITY_A
# ============================================================================


class TestEchoEntityAStrategy:
    def test_returns_entity_a_name(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.ECHO_ENTITY_A, registry=registry)
        pred = model.predict(_example(entity_a_name="Eiffel Tower"))
        assert pred.answer == "Eiffel Tower"
        assert pred.refusal is False

    def test_returns_null_when_no_entity_a(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.ECHO_ENTITY_A, registry=registry)
        pred = model.predict(_example(entity_a_name=None))
        assert pred.answer is None
        assert pred.refusal is True


# ============================================================================
# Strategy: TYPE_DEFAULT
# ============================================================================


class TestTypeDefaultStrategy:
    def test_numeric_float(self, registry) -> None:
        # 1.1 has answer_type "Numeric (float)" (distance_kilometers in
        # the benchmark's snake_case vocabulary → canonical "Numeric
        # (float)" with unit "km" from DEFAULT_UNIT_BY_BENCHMARK_TYPE).
        model = DummyModel(strategy=DummyStrategy.TYPE_DEFAULT, registry=registry)
        pred = model.predict(_example(tid="1.1", sfid="A"))
        assert pred.answer == 0.0
        assert isinstance(pred.answer, float)
        assert pred.refusal is False
        assert pred.answer_type == "Numeric (float)"
        assert pred.unit == "km"

    def test_unit_propagates_from_registry(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.TYPE_DEFAULT, registry=registry)
        pred = model.predict(_example(tid="1.1", sfid="A"))
        assert pred.unit == "km"


class TestTypeDefaultLookup:
    """Direct tests for the ``_type_default`` helper.

    Doesn't go through DummyModel — just verifies the per-type
    mapping is sane and covers every canonical answer type in
    :data:`TYPE_REGISTRY`.
    """

    def test_boolean(self) -> None:
        assert _type_default("Boolean (yes/no, true/false)") is False

    def test_numeric_integer(self) -> None:
        v = _type_default("Numeric (integer)")
        assert v == 0
        assert isinstance(v, int)

    def test_numeric_float(self) -> None:
        v = _type_default("Numeric (float)")
        assert v == 0.0
        assert isinstance(v, float)

    def test_numeric_ambiguous(self) -> None:
        # The ambiguous "Numeric" answer type — same default as float.
        v = _type_default("Numeric")
        assert isinstance(v, float)

    def test_set_enumeration(self) -> None:
        assert _type_default("Set enumeration (list of entity names)") == []
        assert _type_default("Set enumeration (list of dates)") == []

    def test_coordinates(self) -> None:
        v = _type_default("Coordinates (lat/long pair)")
        assert v == {"lat": 0.0, "lon": 0.0}

    def test_geometry(self) -> None:
        assert _type_default("Geometry (list of lat/long pairs)") == []

    def test_cardinal_direction(self) -> None:
        assert _type_default("Cardinal direction (predefined)") == "north"

    def test_predefined_left_right(self) -> None:
        assert _type_default("Predefined value (left, right)") == "left"

    def test_predefined_upstream_downstream(self) -> None:
        assert _type_default("Predefined value (upstream, downstream)") == "upstream"

    def test_date(self) -> None:
        assert _type_default("Date") == "2020-01-01"

    def test_entity_name(self) -> None:
        assert _type_default("Entity name (single)") == ""

    def test_entity_hex_string(self) -> None:
        assert _type_default("Entity hex string (single)") == ""

    def test_unknown_falls_back_to_none(self) -> None:
        assert _type_default("Definitely Not A Type") is None

    def test_every_registered_type_has_a_non_none_default(self) -> None:
        # The fallback ``None`` only kicks in for types we forgot to
        # handle. Every canonical type in TYPE_REGISTRY should map to
        # a non-None default.
        for canonical_name in TYPE_REGISTRY:
            assert _type_default(canonical_name) is not None, (
                f"_type_default({canonical_name!r}) returned None — "
                f"add a case in benchmark_models/dummy.py::_type_default"
            )


# ============================================================================
# Unknown TID
# ============================================================================


class TestUnknownTid:
    def test_unknown_tid_emits_error_prediction(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.ALWAYS_NULL, registry=registry)
        pred = model.predict(_example(tid="999.999"))
        assert pred.final_state == "error"
        assert "unknown tid" in pred.raw_completion
        assert "999.999" in pred.raw_completion

    def test_unknown_tid_does_not_raise(self, registry) -> None:
        # The runner contract: predict() must not raise on per-example
        # failures. Verify directly.
        model = DummyModel(strategy=DummyStrategy.ALWAYS_NULL, registry=registry)
        # Just that this doesn't raise.
        model.predict(_example(tid="not-a-real-tid"))


# ============================================================================
# Per-instance overrides
# ============================================================================


class TestPerInstanceOverrides:
    def test_model_id_override(self, registry) -> None:
        model = DummyModel(model_id="dummy-v2", registry=registry)
        pred = model.predict(_example())
        assert pred.model_id == "dummy-v2"

    def test_tier_override(self, registry) -> None:
        model = DummyModel(tier="tier3", registry=registry)
        pred = model.predict(_example())
        assert pred.tier == "tier3"

    def test_snapshot_id_override(self, registry) -> None:
        model = DummyModel(snapshot_id="abcd1234@efgh5678", registry=registry)
        pred = model.predict(_example())
        assert pred.snapshot_id == "abcd1234@efgh5678"

    def test_seed_override(self, registry) -> None:
        model = DummyModel(seed=99, registry=registry)
        pred = model.predict(_example())
        assert pred.seed == 99

    def test_multiple_dummies_in_one_process(self, registry) -> None:
        # Per-instance shadowing of model_id / tier means two dummies
        # can coexist in one process with different identities.
        a = DummyModel(model_id="a", tier="tier1", registry=registry)
        b = DummyModel(model_id="b", tier="tier3", registry=registry)
        pa = a.predict(_example())
        pb = b.predict(_example())
        assert pa.model_id == "a"
        assert pa.tier == "tier1"
        assert pb.model_id == "b"
        assert pb.tier == "tier3"


# ============================================================================
# predict_batch via the inherited default
# ============================================================================


class TestPredictBatch:
    def test_predict_batch_returns_in_order(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.TYPE_DEFAULT, registry=registry)
        examples = [_example(tid="1.1") for _ in range(5)]
        preds = model.predict_batch(examples)
        assert len(preds) == 5
        assert all(p.tid == "1.1" for p in preds)
        assert all(p.answer == 0.0 for p in preds)

    def test_predict_batch_handles_mixed_tids(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.TYPE_DEFAULT, registry=registry)
        examples = [
            _example(tid="1.1"),
            _example(tid="82.1"),  # dynamic — should error
            _example(tid="999.999"),  # unknown — should error
        ]
        preds = model.predict_batch(examples)
        assert len(preds) == 3
        assert preds[0].answer == 0.0
        assert preds[1].final_state == "error"
        assert preds[2].final_state == "error"


# ============================================================================
# Default registry loading
# ============================================================================


class TestDefaultRegistryLoading:
    def test_no_registry_arg_loads_bundled(self) -> None:
        # When constructed with registry=None (the default), DummyModel
        # loads the bundled template registry via the no-arg form of
        # load_template_registry. This is the smoke-test path.
        # The TID / SFID pair below is one of the smallest
        # always-shipped families in the bundled production registry.
        model = DummyModel()
        pred = model.predict(_example(tid="10.1", sfid="B"))
        assert pred.tid == "10.1"
        assert pred.sfid == "B"


# ============================================================================
# JSONL round-trip
# ============================================================================


class TestJsonlRoundTrip:
    def test_dummy_predictions_serialize_and_load(self, registry) -> None:
        model = DummyModel(strategy=DummyStrategy.TYPE_DEFAULT, registry=registry)
        pred = model.predict(_example(tid="1.1"))
        line = pred.model_dump_json()
        reloaded = Prediction.model_validate_json(line)
        assert reloaded == pred
