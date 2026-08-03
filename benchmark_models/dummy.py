"""DummyModel — trivial Model implementation for harness tests.

Returns canned predictions without touching GPU or API. Used by:

* Unit tests of the runner, registry, JSONL writer, and tier loops
  (under ``tests/benchmark_models/``).
* Ad-hoc smoke runs via ``python -m benchmark_models smoke
  --variant dummy --n 10`` to exercise the end-to-end plumbing
  (Example → Model → Prediction → JSONL → eval pipeline) without
  provisioning real models.

DummyModel produces well-formed :class:`Prediction` records — fully
populated provenance, type-consistent ``answer`` values, and the
same error-prediction shape the real runner emits. It's a
*behavioural* stand-in, not just a mock, so test assertions about
runner contracts (every Prediction has a Usage, every answer is
type-appropriate) hold without touching a backend.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from benchmark_models.base import (
    Example,
    Model,
    Prediction,
    Tier,
    Usage,
)
from benchmark_models.schemas import (
    TemplateRegistry,
    load_template_registry,
)


class DummyStrategy(StrEnum):
    """Predefined response strategies for :class:`DummyModel`."""

    ALWAYS_NULL = "always_null"
    """Always return ``answer=None`` (a refusal). The simplest possible
    behavior; useful for testing the abstention / refusal-flag pathway."""

    ECHO_ENTITY_A = "echo_entity_a"
    """Return entity A's display name as the answer. Useful for tests
    that want a non-null but recognizable answer derived from the
    example. Returns ``None`` if the example has no entity A slot."""

    TYPE_DEFAULT = "type_default"
    """Return a type-appropriate non-null default for the question's
    declared answer type:

    * ``Boolean`` → ``False``
    * ``Numeric (integer)`` → ``0``
    * ``Numeric (float)`` / ``Numeric`` → ``0.0``
    * ``Set enumeration (...)`` → ``[]``
    * ``Coordinates`` → ``{"lat": 0.0, "lon": 0.0}``
    * ``Geometry`` → ``[]``
    * ``Cardinal direction (predefined)`` → ``"north"``
    * ``Predefined value (left, right)`` → ``"left"``
    * ``Predefined value (upstream, downstream)`` → ``"upstream"``
    * ``Date`` → ``"2020-01-01"``
    * ``Entity name (single)`` / ``Entity hex string (single)`` → ``""``

    Useful for tier-1 smoke runs that just need to exercise the runner
    without producing meaningful answers. The eval pipeline will mark
    every prediction as wrong, which is fine — the goal is plumbing
    validation, not accuracy.
    """


class DummyModel(Model):
    """Trivial :class:`Model` implementation for harness tests and smoke runs.

    Constructs fully-populated :class:`Prediction` records using a
    canned response strategy and template-registry lookups. Never
    raises on per-example failures; instead emits a Prediction with
    ``final_state="error"`` and the error in ``raw_completion``.
    """

    # Class-level defaults; instance constructor shadows these so a
    # single test can construct multiple dummy variants with different
    # ids/tiers in one process.
    model_id: str = "dummy"
    tier: Tier = "tier1"

    def __init__(
        self,
        *,
        strategy: DummyStrategy = DummyStrategy.ALWAYS_NULL,
        registry: TemplateRegistry | None = None,
        model_id: str = "dummy",
        tier: Tier = "tier1",
        snapshot_id: str = "dummy",
        prompt_template_id: str = "dummy.canonical",
        seed: int = 0,
    ) -> None:
        """Construct a DummyModel.

        Args:
            strategy: Which canned response strategy to use. See
                :class:`DummyStrategy`.
            registry: Optional :class:`TemplateRegistry` to use for
                ``sfid`` and ``answer_type`` lookups by ``tid``. If
                ``None``, loads the bundled registry via
                :func:`benchmark_models.schemas.load_template_registry`
                with no arguments.
            model_id: Identifier recorded in every :attr:`Prediction.model_id`.
                Override per instance to simulate multiple variants.
            tier: Tier identifier recorded in every :attr:`Prediction.tier`.
            snapshot_id: KG snapshot identifier recorded in every
                :attr:`Prediction.snapshot_id`. Defaults to ``"dummy"``
                since DummyModel doesn't query a real KG.
            prompt_template_id: Identifier recorded in every
                :attr:`Prediction.prompt_template_id`.
            seed: Recorded in every :attr:`Prediction.seed`. Doesn't
                affect output (DummyModel is deterministic regardless).
        """
        self.strategy = strategy
        self.registry: TemplateRegistry = (
            registry if registry is not None else load_template_registry()
        )
        # Shadow the class-level model_id / tier with per-instance values.
        self.model_id = model_id
        self.tier = tier
        self.snapshot_id = snapshot_id
        self.prompt_template_id = prompt_template_id
        self.seed = seed

    def predict(self, example: Example) -> Prediction:
        """Produce a canned :class:`Prediction` for one example."""
        try:
            info = self.registry.get(example.tid)
        except KeyError as exc:
            return self._error_prediction(
                example,
                sfid="?",
                answer_type="?",
                message=f"unknown tid {example.tid!r}: {exc}",
            )

        # Every TemplateInfo entry carries a non-None answer_type.
        assert info.answer_type is not None
        answer = self._compute_answer(example, info.answer_type)
        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=info.sfid,
            answer_type=info.answer_type,
            model_id=self.model_id,
            tier=self.tier,
            language=example.language,
            prompt_template_id=self.prompt_template_id,
            seed=self.seed,
            snapshot_id=self.snapshot_id,
            answer=answer,
            unit=info.answer_unit,
            refusal=(answer is None),
            usage=Usage(input_tokens=0, output_tokens=0, api_call_count=1),
            latency_ms=0.0,
            raw_completion=f"DummyModel(strategy={self.strategy.value})",
        )

    def _error_prediction(
        self,
        example: Example,
        *,
        sfid: str,
        answer_type: str,
        message: str,
        error_class: str = "config",
    ) -> Prediction:
        """Build a Prediction with ``final_state="error"`` and a message."""
        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=sfid,
            answer_type=answer_type,
            model_id=self.model_id,
            tier=self.tier,
            language=example.language,
            prompt_template_id=self.prompt_template_id,
            seed=self.seed,
            snapshot_id=self.snapshot_id,
            answer=None,
            refusal=True,
            final_state="error",
            error_class=error_class,  # type: ignore[arg-type]
            usage=Usage(api_call_count=0),
            latency_ms=0.0,
            raw_completion=f"DummyModel error: {message}",
        )

    def _compute_answer(self, example: Example, answer_type: str) -> Any:
        """Dispatch on the configured strategy."""
        if self.strategy is DummyStrategy.ALWAYS_NULL:
            return None
        if self.strategy is DummyStrategy.ECHO_ENTITY_A:
            return example.entity_a.name if example.entity_a is not None else None
        if self.strategy is DummyStrategy.TYPE_DEFAULT:
            return _type_default(answer_type)
        # Unreachable in practice — StrEnum protects against unknown values.
        raise ValueError(f"unknown strategy: {self.strategy!r}")


# ----------------------------------------------------------------------------
# Type-default lookup (used by DummyStrategy.TYPE_DEFAULT)
# ----------------------------------------------------------------------------


def _type_default(answer_type: str) -> Any:
    """Return a type-appropriate non-null default for ``answer_type``.

    The mapping mirrors the canonical Answer Type vocabulary in
    :data:`benchmark_models.schemas.TYPE_REGISTRY`. Unrecognized types
    fall back to ``None`` (which the caller may then mark as a refusal).
    """
    if answer_type == "Boolean (yes/no, true/false)":
        return False
    if answer_type == "Numeric (integer)":
        return 0
    if answer_type in {"Numeric (float)", "Numeric"}:
        return 0.0
    if answer_type in {
        "Set enumeration (list of entity names)",
        "Set enumeration (list of dates)",
        "Geometry (list of lat/long pairs)",
    }:
        return []
    if answer_type == "Coordinates (lat/long pair)":
        return {"lat": 0.0, "lon": 0.0}
    if answer_type == "Cardinal direction (predefined)":
        return "north"
    if answer_type == "Predefined value (left, right)":
        return "left"
    if answer_type == "Predefined value (upstream, downstream)":
        return "upstream"
    if answer_type == "Date":
        return "2020-01-01"
    if answer_type in {"Entity name (single)", "Entity hex string (single)"}:
        return ""
    if answer_type == "Ternary (true/false/plausibly)":
        return "false"
    if answer_type == "Geohash string":
        return "u000000"
    if answer_type == "Composite string (H3+S2)":
        return "H3(res=15): 0 | S2(lvl=20): 0"
    if answer_type == "H3 cell index":
        return "0"
    if answer_type == "S2 cell index":
        return "0"
    if answer_type == "MGRS-like grid cell":
        return "0N_0_0"
    if answer_type == "Bounding box (4-float)":
        return [0.0, 0.0, 0.0, 0.0]
    if answer_type == "Polar coordinates (angle, distance)":
        return {"angle_degrees": 0.0, "distance_km": 0.0}
    return None
