"""The :class:`Scorer` — entry point for scoring one prediction.

Given one :class:`~benchmark_models.base.Prediction` and its matching
:class:`~benchmark_eval.base.GoldRow`, :meth:`Scorer.score` returns a
:class:`~benchmark_eval.base.ScoreRecord` with EM, NE, coverage, and
per-type drill-down fields.

Dispatch is by ``answer_type`` string:

* ``"Boolean (yes/no, true/false)"`` → strict equality
* ``"Entity name (single)"`` → alias-backed :func:`resolve`
* ``"Entity hex string (single)"`` → strict equality (no aliases)
* ``"Set enumeration (list of entity names)"`` → per-element
  :func:`resolve` + Jaccard
* ``"Set enumeration (list of dates)"`` → per-element date
  canonicalization + Jaccard
* ``"Numeric"`` / ``"Numeric (float)"`` / ``"Numeric (integer)"`` →
  ``|ŷ − y*|`` with ``τ = max(δ, α|y*|)`` and strict unit match
* ``"Coordinates (lat/long pair)"`` → geodesic distance
* ``"Geometry (list of lat/long pairs)"`` → polygon IoU
* ``"Date"`` → components-match EM + day-distance NE
* ``"Predefined value (left, right)"`` / ``"Predefined value
  (upstream, downstream)"`` / ``"Cardinal direction (predefined)"``
  → alias-backed :func:`resolve` against the known canonical set

Any refusal (``prediction.answer is None``) short-circuits to
``em=0, ne=1, covered=False``. Any format error (failed numeric
parse, failed date parse, wrong Python type) is recorded as
``format_error=True`` with the same 0/1 scores and ``covered=False``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from benchmark_eval._paths import spec_path as _spec_path
from benchmark_eval.aliases import AliasStore
from benchmark_eval.base import (
    GoldRow,
    NumericFailureCategory,
    ScoreRecord,
)
from benchmark_eval.canonicalize import (
    canonicalize_string,
    canonicalize_unit,
    date_components_match,
    date_day_distance,
    date_precision_tolerance_days,
    parse_date,
    parse_numeric,
)
from benchmark_eval.distances import (
    NumericConfig,
    coordinate_distance_km,
    jaccard,
    line_hausdorff_km,
    numeric_distance,
    numeric_tolerance,
    polygon_iou,
    set_f1,
    set_precision,
    set_recall,
)
from benchmark_eval.units import convert_to_gold_unit
from benchmark_models.base import Prediction

SCORING_SPEC_PATH = _spec_path()
"""Canonical scoring rules. Every :class:`ScorerConfig` default is
sourced from this file when it exists; hardcoded fallbacks below are
used (with a loguru warning) only when the file is missing, so
development workflows don't break if the spec is deleted. Resolved
via the single :func:`benchmark_eval._paths.spec_path` helper."""

# Hardcoded fallbacks — kept in sync with ``scoring_spec.json``.
# Treated as authoritative only when the spec file is unavailable.
# TIDs whose gold is a dimensionless shape index (TID 57 family):
# perimeter-to-area ratio, shape index, convex-hull ratio, boundary
# roughness, vertices-per-km. Tagged bare ``numeric`` (no unit) in the
# release, so the unit-based dispatch would route them to ``count``;
# they are routed to the relative-tolerance ``ratio`` kind instead.
_RATIO_TIDS: frozenset[str] = frozenset(
    {"57.1", "57.2", "57.3", "57.4", "57.5"}  # boundary-irregularity ratio family
)


_FALLBACK_DEFAULTS: dict[str, Any] = {
    "alpha": 0.05,
    "default_delta_distance_km": 5.0,
    "default_delta_area_km2": 25.0,
    "default_delta_count": 1.0,
    "default_delta_ratio": 0.0,
    "use_scale_from_magnitude": True,
    "coordinate_delta_km": 5.0,
    "coordinate_d_scale_km": 200.0,
    "line_tolerance_km": 5.0,
    "polygon_iou_threshold": 0.5,
    "set_dates_strict": True,
    "exclude_refusals_from_coverage": True,
    "split_gold_missing_from_format_error": True,
    "composite_h3_containment": True,
    "polar_angle_tol_degrees": 5.0,
}


def load_scoring_spec(path: Path | str | None = None) -> dict[str, Any]:
    """Load the canonical scoring spec as a flat dict of field names.

    Returns the hardcoded fallback defaults (with a warning) when the
    spec file is missing, so dev workflows that delete or rename it
    keep working. The returned dict has exactly the keys in
    :data:`_FALLBACK_DEFAULTS`.
    """
    spec_path = Path(path) if path is not None else SCORING_SPEC_PATH
    if not spec_path.exists():
        logger.warning(
            "Scoring spec not found at {}; using hardcoded fallback "
            "defaults. Results may not be comparable across runs that "
            "use a different spec.",
            spec_path,
        )
        return dict(_FALLBACK_DEFAULTS)

    raw = json.loads(spec_path.read_text())
    numeric = raw.get("numeric", {})
    coords = raw.get("coordinates", {})
    geom = raw.get("geometry", {})
    coverage = raw.get("coverage", {})
    set_enum = raw.get("set_enumeration", {})
    grid = raw.get("grid_cell", {})
    polar = raw.get("polar", {})
    return {
        "alpha": numeric.get("alpha", _FALLBACK_DEFAULTS["alpha"]),
        "default_delta_distance_km": numeric.get(
            "default_delta_distance_km",
            _FALLBACK_DEFAULTS["default_delta_distance_km"],
        ),
        "default_delta_area_km2": numeric.get(
            "default_delta_area_km2",
            _FALLBACK_DEFAULTS["default_delta_area_km2"],
        ),
        "default_delta_count": numeric.get(
            "default_delta_count", _FALLBACK_DEFAULTS["default_delta_count"]
        ),
        "default_delta_ratio": numeric.get(
            "default_delta_ratio", _FALLBACK_DEFAULTS["default_delta_ratio"]
        ),
        "use_scale_from_magnitude": numeric.get(
            "use_scale_from_magnitude",
            _FALLBACK_DEFAULTS["use_scale_from_magnitude"],
        ),
        "coordinate_delta_km": coords.get(
            "delta_km", _FALLBACK_DEFAULTS["coordinate_delta_km"]
        ),
        "coordinate_d_scale_km": coords.get(
            "d_scale_km", _FALLBACK_DEFAULTS["coordinate_d_scale_km"]
        ),
        "line_tolerance_km": geom.get(
            "line_tolerance_km", _FALLBACK_DEFAULTS["line_tolerance_km"]
        ),
        "polygon_iou_threshold": geom.get(
            "polygon_iou_threshold", _FALLBACK_DEFAULTS["polygon_iou_threshold"]
        ),
        "set_dates_strict": set_enum.get(
            "strict_dates", _FALLBACK_DEFAULTS["set_dates_strict"]
        ),
        "exclude_refusals_from_coverage": coverage.get(
            "exclude_refusals_from_coverage",
            _FALLBACK_DEFAULTS["exclude_refusals_from_coverage"],
        ),
        "split_gold_missing_from_format_error": coverage.get(
            "split_gold_missing_from_format_error",
            _FALLBACK_DEFAULTS["split_gold_missing_from_format_error"],
        ),
        "composite_h3_containment": grid.get(
            "composite_h3_containment",
            _FALLBACK_DEFAULTS["composite_h3_containment"],
        ),
        "polar_angle_tol_degrees": polar.get(
            "angle_tol_degrees",
            _FALLBACK_DEFAULTS["polar_angle_tol_degrees"],
        ),
    }


def _spec_defaults() -> dict[str, Any]:
    """Lazy wrapper so dataclass field-default factories call it once."""
    return load_scoring_spec()


_SPEC = _spec_defaults()


@dataclass
class ScorerConfig:
    """Tolerance hyperparameters and feature toggles.

    Every default below is loaded from ``scoring_spec.json`` at
    import time. CLI overrides produce a *deviations* dict that the
    run logs to MLflow so a reader can tell whether a result used
    the canonical spec or a custom configuration.
    """

    # Numeric
    alpha: float = field(default_factory=lambda: _SPEC["alpha"])
    default_delta_distance_km: float = field(
        default_factory=lambda: _SPEC["default_delta_distance_km"]
    )
    default_delta_area_km2: float = field(
        default_factory=lambda: _SPEC["default_delta_area_km2"]
    )
    default_delta_count: float = field(
        default_factory=lambda: _SPEC["default_delta_count"]
    )
    default_delta_ratio: float = field(
        default_factory=lambda: _SPEC["default_delta_ratio"]
    )
    """Absolute floor for the dimensionless ``ratio`` numeric kind
    (TID 57.x). 0.0 → purely relative tolerance (``alpha*|y*|``)."""
    use_scale_from_magnitude: bool = field(
        default_factory=lambda: _SPEC["use_scale_from_magnitude"]
    )

    # Coordinates
    coordinate_delta_km: float = field(
        default_factory=lambda: _SPEC["coordinate_delta_km"]
    )
    coordinate_d_scale_km: float = field(
        default_factory=lambda: _SPEC["coordinate_d_scale_km"]
    )

    # Geometry
    line_tolerance_km: float = field(
        default_factory=lambda: _SPEC["line_tolerance_km"]
    )
    polygon_iou_threshold: float = field(
        default_factory=lambda: _SPEC["polygon_iou_threshold"]
    )

    # Set enumeration
    set_dates_strict: bool = field(
        default_factory=lambda: _SPEC["set_dates_strict"]
    )
    """When True (default), a set_enumeration of dates with any
    unparseable element is marked ``format_error=True`` rather than
    silently dropping the bad element. Symmetric with set_entities."""

    # Coverage policy
    exclude_refusals_from_coverage: bool = field(
        default_factory=lambda: _SPEC["exclude_refusals_from_coverage"]
    )
    """When True (default), refusals (``answer is None``) count
    as not-covered alongside format errors so ``coverage_rate``
    reflects only "produced a parseable, non-refusal answer matching
    the declared answer_type"."""

    # Grid-cell spatial proximity (companions em_spatial / ne_spatial).
    composite_h3_containment: bool = field(
        default_factory=lambda: _SPEC["composite_h3_containment"]
    )
    """When True (default), the Composite H3 half is scored by exact
    H3 hierarchy containment (with a centroid-distance fallback);
    when False, only the centroid-distance rule is used."""

    # Polar coordinates (angle, distance).
    polar_angle_tol_degrees: float = field(
        default_factory=lambda: _SPEC["polar_angle_tol_degrees"]
    )
    """Bearing tolerance (degrees) for Polar coordinates scoring
    (TID 63.1). EM=1 requires the bearing within this window AND the
    radial distance within the numeric-distance tolerance."""

    split_gold_missing_from_format_error: bool = field(
        default_factory=lambda: _SPEC["split_gold_missing_from_format_error"]
    )
    """When True (default), rows where the gold is missing
    (refusal on a row whose ``ground_truth is None`` AND
    ``is_true_premise`` is not False) are flagged ``gold_missing=True``
    rather than ``format_error=True``. Lets the aggregator separate
    upstream gold-defect counts from scorer-side parse failures."""

    def to_numeric_config(self) -> NumericConfig:
        return NumericConfig(
            alpha=self.alpha,
            default_delta_distance_km=self.default_delta_distance_km,
            default_delta_area_km2=self.default_delta_area_km2,
            default_delta_count=self.default_delta_count,
            default_delta_ratio=self.default_delta_ratio,
            use_scale_from_magnitude=self.use_scale_from_magnitude,
        )

    def deviations_from_spec(self) -> dict[str, Any]:
        """Return the subset of fields that differ from the canonical
        :data:`SCORING_SPEC_PATH`. Empty dict when this config matches
        the spec exactly.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            expected = _SPEC.get(f.name)
            actual = getattr(self, f.name)
            if expected is not None and actual != expected:
                out[f.name] = actual
        return out


# Mapping from canonicalized unit suffix to numeric "kind" used to
# pick the right default δ. Loaded from
# ``scoring_spec.json:numeric.kind_by_unit_suffix`` at module init;
# falls back to the hardcoded table only when the spec is missing
# (with a warning).
_FALLBACK_NUMERIC_KIND_BY_UNIT_SUFFIX: dict[str, str] = {
    "km": "distance",
    "km2": "area",
    "m": "distance",
    "m2": "area",
    "mi": "distance",
    "mi2": "area",
    "ha": "area",
    "nmi": "distance",
    "ft": "distance",
}


def _load_numeric_kind_map() -> dict[str, str]:
    """Load the unit→kind dict from the spec; fall back to the
    hardcoded table when the spec is missing or the field is
    absent."""
    if not SCORING_SPEC_PATH.exists():
        return dict(_FALLBACK_NUMERIC_KIND_BY_UNIT_SUFFIX)
    raw = json.loads(SCORING_SPEC_PATH.read_text())
    mapping = raw.get("numeric", {}).get("kind_by_unit_suffix", {})
    # The spec block may include a "description" key for documentation;
    # filter to actual unit→kind entries.
    out = {
        k: v for k, v in mapping.items()
        if isinstance(v, str) and v in ("distance", "area", "count")
    }
    if not out:
        logger.warning(
            "scoring_spec.json: numeric.kind_by_unit_suffix is empty / "
            "malformed; using hardcoded fallback (numeric kind dispatch "
            "may be incomplete for new units).",
        )
        return dict(_FALLBACK_NUMERIC_KIND_BY_UNIT_SUFFIX)
    return out


NUMERIC_KIND_BY_UNIT_SUFFIX: dict[str, str] = _load_numeric_kind_map()


# Coordinate-string separator chars loaded from the spec.
# Hardcoded fallback matches the spec exactly.
_FALLBACK_COORD_SEPARATOR_CHARS: str = r",;\s_"


def _load_coord_separator_chars() -> str:
    """Load the coordinate-separator character class from the spec.
    Falls back to ``,;\\s_`` when the spec is missing.
    """
    if not SCORING_SPEC_PATH.exists():
        return _FALLBACK_COORD_SEPARATOR_CHARS
    raw = json.loads(SCORING_SPEC_PATH.read_text())
    chars = raw.get("coordinates", {}).get("separator_chars")
    if not isinstance(chars, str) or not chars:
        return _FALLBACK_COORD_SEPARATOR_CHARS
    return chars


# ---------------------------------------------------------------------------
# Polygon-IoU planar-fallback warning state.
#
# When a polygon's EPSG:6933 reprojection fails the scorer falls back
# to planar (raw lat/lon) IoU and flags the row on ScoreRecord. We
# also emit a single loguru warning per run so the operator notices
# the issue quickly; the per-record flag remains for downstream
# rate reporting on AggregateReport.
# ---------------------------------------------------------------------------

_planar_fallback_warned = False


def _warn_planar_fallback_once() -> None:
    global _planar_fallback_warned
    if _planar_fallback_warned:
        return
    _planar_fallback_warned = True
    logger.warning(
        "polygon_iou: EPSG:6933 reprojection failed on at least one "
        "row; falling back to planar (raw lat/lon) IoU for affected "
        "rows. Check AggregateReport.polygon_planar_fallback_rate "
        "for the fraction of polygon rows affected."
    )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class Scorer:
    """Score predictions against gold rows.

    Stateless except for the :class:`AliasStore` and
    :class:`ScorerConfig` passed at construction; safe to reuse
    across threads if needed.
    """

    def __init__(
        self,
        *,
        alias_store: AliasStore | None = None,
        config: ScorerConfig | None = None,
    ) -> None:
        self._aliases = alias_store or AliasStore.from_entries([])
        self._config = config or ScorerConfig()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def score(self, prediction: Prediction, gold: GoldRow) -> ScoreRecord:
        """Score one prediction. Returns a :class:`ScoreRecord`."""
        # Tier-3 tool-call counters (None on tier 1 / tier 2).
        tool_call_count: int | None = None
        tool_error_count: int | None = None
        if prediction.tool_calls is not None:
            tool_call_count = len(prediction.tool_calls)
            tool_error_count = sum(
                1 for tc in prediction.tool_calls if tc.error is not None
            )

        # Self-consistency agreement (tier-2 SC=N only; None for N<2).
        sc_agreement = _sc_agreement(prediction.sc_samples)

        # Shared identity fields. ``final_state`` and ``error_class``
        # are denormalized so the aggregator can compute per-class
        # error counts and the adjusted-EM denominator without
        # rejoining against ``predictions.jsonl``. ``is_true_premise``,
        # ``tool_call_count`` / ``tool_error_count``, and
        # ``sc_agreement`` are denormalised so the aggregator can
        # stratify and average without re-reading the source files.
        base = {
            "question_id": gold.question_id,
            "tid": gold.tid,
            "sfid": gold.sfid,
            "answer_type": gold.answer_type,
            "model_id": prediction.model_id,
            "tier": prediction.tier,
            "language": prediction.language,
            "strata_cell": gold.strata_cell,
            "kg_source": gold.kg_source,
            "final_state": prediction.final_state,
            "error_class": prediction.error_class,
            "is_true_premise": gold.is_true_premise,
            # Normalized gold label, denormalized so the aggregator can
            # compute a label-free majority-class baseline (the EM a
            # guesser scores by always emitting each TID's modal gold)
            # without rejoining the benchmark. Pure gold property.
            "gold_label": _majority_gold_key(gold.ground_truth),
            "tool_call_count": tool_call_count,
            "tool_error_count": tool_error_count,
            "sc_agreement": sc_agreement,
            # Denormalize cost/latency + cached-token usage so the
            # aggregator can roll them up without rejoining the
            # predictions JSONL. ``cache_creation_tokens`` is
            # Anthropic-specific (the upfront premium when seeding a
            # new cache entry); other backends leave it at 0.
            "input_tokens": prediction.usage.input_tokens,
            "output_tokens": prediction.usage.output_tokens,
            "cached_tokens": prediction.usage.cached_tokens,
            "cache_creation_tokens": getattr(
                prediction.usage, "cache_creation_tokens", 0
            ),
            "latency_ms": prediction.latency_ms,
        }

        # Substrate-error short-circuit (must precede the refusal
        # branch). ANY ``final_state == "error"`` means the substrate
        # crashed — exhausted transport retries (``infra``), malformed
        # tier-3 Python (``code_gen``), a non-retryable provider 400 /
        # context-overflow, a schema violation, or an unclassified
        # failure — NOT that the model chose to abstain. Such a
        # prediction usually carries ``answer=None`` and would otherwise
        # fall into the refusal branch below — where, on a false-premise
        # row, it would be miscredited EM=1 as a "correct rejection",
        # and everywhere it would be miscounted in ``n_refusals``. Score
        # it EM=0 / covered=False / refusal=False so it is neither
        # credited nor counted as an abstention, regardless of
        # ``error_class``. ``em_micro`` still counts it as wrong
        # (headline "errors = failure" policy); ``em_adjusted``
        # separately excludes only the ``error_class`` values in
        # ``HARNESS_ERROR_CLASSES`` (see aggregate.py:non_harness_records).
        if prediction.final_state == "error":
            return ScoreRecord(**base, em=0, ne=1.0, covered=False)

        # Refusal short-circuit: model returned ``answer=None``.
        #
        # An abstention scores EM=1 only when the gold explicitly
        # marks the question as false-premise (``is_true_premise is
        # False``). On real-premise questions abstention is scored
        # as wrong (EM=0, ``covered=False``).
        #
        # Rows whose gold ``ground_truth is None`` *without* a
        # false-premise marker are flagged ``gold_missing=True``
        # (kept distinct from ``format_error``) so the aggregator can
        # separate upstream gold defects from scorer-side parse
        # failures.
        if prediction.answer is None or prediction.refusal:
            if gold.is_true_premise is False:
                # Correct abstention on a false-premise row:
                # ``covered=True`` (this counts as having answered),
                # with ``refusal=True`` recorded for diagnostics.
                return ScoreRecord(
                    **base, em=1, ne=0.0, covered=True, refusal=True,
                )
            if gold.ground_truth is None:
                if self._config.split_gold_missing_from_format_error:
                    return ScoreRecord(
                        **base, em=0, ne=1.0, covered=False,
                        gold_missing=True, refusal=True,
                    )
                return ScoreRecord(
                    **base, em=0, ne=1.0, covered=False,
                    format_error=True, refusal=True,
                )
            # Refusal on a real-premise row — wrong by policy.
            # ``coverage_rate`` excludes refusals (they aren't
            # "answers"); record ``refusal=True`` for diagnostics.
            return ScoreRecord(
                **base, em=0, ne=1.0, covered=False, refusal=True,
            )

        # Commission on a false-premise question: the model produced a
        # (non-refusal) answer of the declared type where the correct
        # action was to abstain (gold ``ground_truth is None``). It is a
        # parseable typed answer — so it counts as ``covered`` — but it
        # is wrong (EM=0). Without this short-circuit the type scorers
        # below hit ``float(None)`` / ``None`` gold and mislabel it a
        # ``format_error`` (deflating coverage, inflating the format-
        # error rate). Mirrors the false-premise refusal branch above.
        if gold.is_true_premise is False and gold.ground_truth is None:
            return ScoreRecord(**base, em=0, ne=1.0, covered=True)

        # Dispatch by answer type.
        answer_type = gold.answer_type
        if answer_type.startswith("Boolean"):
            return self._score_boolean(prediction, gold, base)
        if answer_type.startswith("Ternary"):
            return self._score_ternary(prediction, gold, base)
        if answer_type == "Entity name (single)":
            return self._score_entity_name(prediction, gold, base)
        if answer_type in {
            "Entity hex string (single)",
            "H3 cell index",
            "S2 cell index",
        }:
            # H3 / S2 cell indices share the existing entity-hex
            # matcher: ``_canonicalize_h3_hex`` already normalizes
            # hex / decimal-uint64 / signed-int64 forms (see its
            # docstring). Both sides round-trip through the same
            # canonicalization, so string equality after normalize
            # holds for both H3 and S2 cells.
            return self._score_entity_hex(prediction, gold, base)
        if answer_type == "Geohash string":
            return self._score_geohash(prediction, gold, base)
        if answer_type.startswith("Composite string"):
            return self._score_composite_string(prediction, gold, base)
        if answer_type.startswith("Set enumeration (list of entity"):
            return self._score_set_entities(prediction, gold, base)
        if answer_type.startswith("Set enumeration (list of date"):
            return self._score_set_dates(prediction, gold, base)
        if answer_type.startswith("Numeric"):
            return self._score_numeric(prediction, gold, base)
        if answer_type.startswith("Coordinates"):
            return self._score_coordinates(prediction, gold, base)
        if answer_type.startswith("Polar coordinates"):
            return self._score_polar_coordinates(prediction, gold, base)
        if answer_type.startswith("Bounding box"):
            return self._score_bounding_box(prediction, gold, base)
        if answer_type.startswith("Geometry"):
            return self._score_geometry(prediction, gold, base)
        if answer_type == "Date":
            return self._score_date(prediction, gold, base)
        if answer_type.startswith("Predefined value") or answer_type.startswith(
            "Cardinal direction"
        ):
            return self._score_enum_via_aliases(prediction, gold, base)

        # Unknown answer type — record a format error rather than a crash.
        return ScoreRecord(
            **base,
            em=0,
            ne=1.0,
            covered=False,
            format_error=True,
        )

    def score_batch(
        self,
        predictions: list[Prediction],
        golds_by_qid: dict[str, GoldRow],
    ) -> list[ScoreRecord]:
        """Score a batch of predictions.

        Predictions whose ``question_id`` is not in ``golds_by_qid``
        are skipped (logged, not raised). Predictions in the same
        order as input.

        Emits a single warning when duplicate ``question_id`` values
        appear in the prediction list — partial re-runs concatenated
        onto an old predictions JSONL would otherwise silently double
        the contribution of the duplicated rows. All rows are still
        scored so the aggregator sees the input as-is.
        """
        from loguru import logger

        # Deduplicate on question_id (last-wins, mirroring load_gold_rows).
        # A partial re-run concatenated onto an existing predictions JSONL
        # repeats qids; scoring every copy double-counts them in the
        # aggregate. Keep only the last occurrence per qid.
        by_qid: dict[str, Prediction] = {}
        for pred in predictions:
            by_qid[pred.question_id] = pred
        n_duplicate_qids = len(predictions) - len(by_qid)

        out: list[ScoreRecord] = []
        for pred in by_qid.values():
            gold = golds_by_qid.get(pred.question_id)
            if gold is None:
                logger.warning(
                    "No gold row for question_id={}; skipping.",
                    pred.question_id,
                )
                continue
            out.append(self.score(pred, gold))
        if n_duplicate_qids:
            logger.warning(
                "Scorer.score_batch: dropped {} duplicate question_id "
                "prediction row(s) via last-wins dedup. This usually "
                "indicates a partial re-run was concatenated onto an "
                "existing predictions JSONL; inspect the file if scoring "
                "looks off.",
                n_duplicate_qids,
            )
        return out

    # ------------------------------------------------------------------
    # Per-type scoring
    # ------------------------------------------------------------------

    def _score_boolean(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        pred_val = self._coerce_boolean(prediction.answer)
        gold_val = self._coerce_boolean(gold.ground_truth)
        if pred_val is None or gold_val is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        match = pred_val == gold_val
        return ScoreRecord(
            **base,
            em=int(match),
            ne=0.0 if match else 1.0,
            covered=True,
            raw_distance=0.0 if match else 1.0,
            raw_tolerance=1.0,
        )

    @staticmethod
    def _coerce_boolean(value: Any) -> bool | None:
        """Coerce a value to ``bool`` or return ``None`` when unparseable.

        Accepted forms (case-insensitive, whitespace-stripped):

        * Python booleans ``True`` / ``False`` (pass through).
        * Integers ``1`` / ``0``.
        * Strings ``"true"`` / ``"yes"`` / ``"1"`` / ``"y"`` (→ True).
        * Strings ``"false"`` / ``"no"`` / ``"0"`` / ``"n"`` (→ False).

        Everything else (including non-English Boolean tokens like
        ``"ja"`` / ``"nein"`` / ``"oui"`` / ``"non"``) returns
        ``None``. Known design limitation — the prompts ask the model
        to answer in English even when the question is in another
        language, so non-English Boolean tokens are treated as format
        errors. Add per-language tokens here if non-English models
        start systematically emitting native-language booleans.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().casefold()
            if s in ("true", "yes", "1", "y"):
                return True
            if s in ("false", "no", "0", "n"):
                return False
        return None

    def _score_entity_name(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """Entity-name match with alias-store resolution on **both** sides.

        Gold and prediction both flow through the alias store so
        different surface forms of the same entity resolve to the same
        canonical id. When the alias store can't resolve one side, a
        quote-stripped case-insensitive string equality fallback kicks
        in so short forms (e.g. model emits "Zeitgeist", gold is
        "Zeitgeist, Gärtnerstraße 15") still have a chance.
        """
        stripped_gold = _strip_quotes(gold.ground_truth)
        stripped_pred = _strip_quotes(prediction.answer)

        gold_canonical = self._normalize_gold_canonical(stripped_gold)
        gold_resolved = (
            self._aliases.resolve(stripped_gold)
            if isinstance(stripped_gold, str)
            else None
        ) or gold_canonical

        pred_resolved = self._aliases.resolve(
            stripped_pred, gold_canonical_id=gold_resolved
        )

        match = False
        resolve_failure = pred_resolved is None
        if pred_resolved is not None and gold_resolved is not None:
            match = pred_resolved == gold_resolved
        # Alias-free fallback: quote-stripped case-insensitive equality.
        # Preserves scoring when the alias store is incomplete or has
        # no entry for this surface form. Empty
        # strings on either side are excluded so a refusal-shaped
        # prediction (``""``) doesn't accidentally match an empty /
        # malformed gold.
        if (
            not match
            and isinstance(stripped_pred, str)
            and isinstance(stripped_gold, str)
            and stripped_pred
            and stripped_gold
        ):
            match = stripped_pred.casefold() == stripped_gold.casefold()
            if match:
                resolve_failure = False

        return ScoreRecord(
            **base,
            em=int(match),
            ne=0.0 if match else 1.0,
            covered=True,
            raw_distance=0.0 if match else 1.0,
            raw_tolerance=1.0,
            resolve_failure=resolve_failure,
        )

    @staticmethod
    def _normalize_gold_canonical(gold: Any) -> str | None:
        """Map a gold entity reference to its canonical alias-store id.

        Accepts:

        * already-prefixed id (``"osm:12345"``, ``"kwg:..."``,
          ``"wkg:..."``) — returned as-is.
        * bare numeric OSM id (``"12345"``) → ``"osm:12345"``.
        * full WorldKG resource URI
          (``"http://www.worldkg.org/resource/12345"``) → ``"wkg:12345"``.
        * full KnowWhereGraph LOD URI
          (``"http://stko-kwg.geog.ucsb.edu/lod/resource/<tail>"``) →
          ``"kwg:<tail>"``.
        * any other bare string (entity name) — returned unchanged
          for the alias store to resolve via :meth:`AliasStore.resolve`.

        Gold entity references arrive as full URIs (from the
        triples) but the per-language alias stores key off the
        ``<prefix>:<id>`` form, so the scorer must normalise both
        shapes to the same canonical id before alias-store lookup.
        """
        if gold is None:
            return None
        s = str(gold).strip()
        if s.startswith(("osm:", "kwg:", "wkg:")):
            return s
        # Recognise full URIs from the three KGs.
        if s.startswith("http://www.worldkg.org/resource/"):
            tail = s[len("http://www.worldkg.org/resource/"):]
            # WorldKG resource URIs include synthetic ``geo<digits>``
            # geometry nodes. We strip the ``geo`` prefix on bare
            # numerics so a triple's ``geo12345`` and the entity's
            # ``12345`` share one canonical id (mirrors the
            # subject-normalisation step in oracle.load_gold_triples).
            if tail.startswith("geo") and tail[3:].isdigit():
                tail = tail[3:]
            return f"wkg:{tail}"
        if s.startswith("http://stko-kwg.geog.ucsb.edu/lod/resource/"):
            tail = s[len("http://stko-kwg.geog.ucsb.edu/lod/resource/"):]
            return f"kwg:{tail}"
        if s.isdigit():
            return f"osm:{s}"
        # Bare names: caller resolves via the alias store.
        return s

    def _score_entity_hex(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """Match H3 cell ids across hex / decimal / signed-int forms.

        Gold may appear as a 14–16-char hex string, a decimal integer
        string, or a negative signed-int64 (resolution-15 cells
        overflow). Models typically emit lower-case hex. Normalise
        both sides to unsigned hex before comparison.
        """
        gold_hex = _canonicalize_h3_hex(gold.ground_truth)
        pred_hex = _canonicalize_h3_hex(prediction.answer)
        # Resolution-agnostic spatial companion (H3 only; S2 tokens
        # decode to None and get no companion). Credits a prediction
        # that localizes the target even when its H3 resolution /
        # encoding differs from gold — the common failure where the
        # model picks the right area at the wrong resolution.
        em_spatial, ne_spatial = self._h3_pair_spatial(
            _decode_h3_cell(gold.ground_truth),
            _decode_h3_cell(prediction.answer),
        )
        if gold_hex is None or pred_hex is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
                em_spatial=em_spatial,
                ne_spatial=ne_spatial,
            )
        match = pred_hex == gold_hex
        return ScoreRecord(
            **base,
            em=int(match),
            ne=0.0 if match else 1.0,
            covered=True,
            raw_distance=0.0 if match else 1.0,
            raw_tolerance=1.0,
            em_spatial=em_spatial,
            ne_spatial=ne_spatial,
        )

    def _score_set_entities(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        if not isinstance(prediction.answer, list):
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        gold_list = self._coerce_list(gold.ground_truth)
        if gold_list is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        # Resolve gold-side first; the prediction-side resolve then
        # passes the gold-resolved set as a hint so benign alias
        # collisions land on the right canonical.
        # Strip a single matched leading/trailing ``"`` pair from each
        # list element before resolution, so a quote-wrapped element
        # (e.g. ``'"Road segment with OBJECT ID 184598"'``) still
        # matches the bare-name form a model emits.
        gold_canon: set[str] = set()
        for n in gold_list:
            if n is None:
                continue
            n_stripped = _strip_quotes(n)
            normalized = self._normalize_gold_canonical(n_stripped)
            resolved = (
                self._aliases.resolve(normalized)
                if isinstance(normalized, str)
                else None
            )
            # Fallback chain, symmetric with the prediction side
            # (see the ``pred_canon.add(...)`` call below):
            #   1. alias-store hit -> the canonical id.
            #   2. no hit, but ``normalized`` is a prefixed canonical id
            #      (``osm:``/``kwg:``/``wkg:`` — a URI/numeric gold from
            #      the triples): keep it raw so it matches a prediction
            #      that resolved to that same id. Bare NAMES never hit
            #      this branch (``_normalize_gold_canonical`` returns them
            #      unchanged, without a prefix).
            #   3. bare name with no alias entry:
            #      ``canonicalize_string``, IDENTICAL to the
            #      prediction-side fallback. Both sides must be
            #      canonicalized the same way: keeping the raw
            #      ``normalized`` string here would leave the gold
            #      un-canonicalized while the prediction is
            #      casefolded/punctuation-stripped, so a byte-identical
            #      correct answer could never intersect the gold. Since
            #      100% of released set-gold members are bare names, that
            #      asymmetry would zero EM on ~65% of set rows.
            if resolved:
                gold_canon.add(resolved)
            elif isinstance(normalized, str) and normalized.startswith(
                ("osm:", "kwg:", "wkg:")
            ):
                gold_canon.add(normalized)
            else:
                gold_canon.add(canonicalize_string(str(n_stripped)))
        pred_canon: set[str] = set()
        for n in prediction.answer:
            if n is None:
                continue
            n_stripped = _strip_quotes(n)
            # Hint resolution with each gold canonical in turn so an
            # alias that's shared across multiple ids prefers the
            # gold-side one. Pick the first match; falls back to
            # bare resolution + string canonicalization.
            #
            # Determinism strategy: the
            # output of this branch must be byte-identical across
            # alias-store regenerations as long as the underlying
            # alias→canonical mappings are identical.
            #
            # The three resolution paths are stable because:
            #   1. ``sorted(gold_canon)`` orders by canonical id
            #      string — independent of set iteration order
            #      (Python's set iteration is hash-randomised when
            #      PYTHONHASHSEED varies).
            #   2. ``self._aliases.resolve(..., gold_canonical_id=gc)``
            #      returns ``gc`` if it's in the candidate list (no
            #      ordering needed), else ``candidates[0]``. The
            #      alias store sorts its candidate lists at build
            #      time (``aliases.py``: ``sorted(set(...))``), so
            #      ``candidates[0]`` is the alphabetically-first
            #      canonical id — stable across JSON key reorderings.
            #   3. The final ``canonicalize_string`` fallback is
            #      pure-function on the prediction string.
            resolved: str | None = None
            for gc in sorted(gold_canon):
                cand = self._aliases.resolve(n_stripped, gold_canonical_id=gc)
                if cand is not None:
                    resolved = cand
                    break
            if resolved is None:
                resolved = self._aliases.resolve(n_stripped)
            pred_canon.add(resolved or canonicalize_string(str(n_stripped)))
        return self._score_set_from_canonical(pred_canon, gold_canon, base)

    def _score_set_dates(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        if not isinstance(prediction.answer, list):
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        gold_list = self._coerce_list(gold.ground_truth)
        if gold_list is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        # Strict policy: if any prediction element fails to parse as a
        # date, mark the entire row as a format error. Silently
        # dropping malformed elements over-credits rows where the bad
        # elements happen not to overlap with gold.
        # Symmetric with set_entities, where unparseable elements
        # participate in the set and harm F1.
        pred_canon: set[tuple[int, int | None, int | None]] = set()
        for d in prediction.answer:
            dc = parse_date(d)
            if dc is None:
                if self._config.set_dates_strict:
                    return ScoreRecord(
                        **base,
                        em=0,
                        ne=1.0,
                        covered=False,
                        format_error=True,
                    )
                continue
            pred_canon.add((dc.year, dc.month, dc.day))
        gold_canon: set[tuple[int, int | None, int | None]] = set()
        for d in gold_list:
            dc = parse_date(d)
            if dc is None:
                # Bad gold dates are an upstream defect; flag rather
                # than silently dropping so the data team sees it.
                logger.warning(
                    "Skipping malformed gold date {!r} in set for "
                    "question_id={}",
                    d, gold.question_id,
                )
                continue
            gold_canon.add((dc.year, dc.month, dc.day))
        return self._score_set_from_canonical(pred_canon, gold_canon, base)

    def _score_set_from_canonical(
        self,
        pred_canon: set,
        gold_canon: set,
        base: dict,
    ) -> ScoreRecord:
        precision = set_precision(pred_canon, gold_canon)
        recall = set_recall(pred_canon, gold_canon)
        f1 = set_f1(precision, recall)
        jac = jaccard(pred_canon, gold_canon)
        ne = 1.0 - jac
        em = int(pred_canon == gold_canon)
        return ScoreRecord(
            **base,
            em=em,
            ne=ne,
            covered=True,
            raw_distance=ne,
            raw_tolerance=1.0,
            set_precision=precision,
            set_recall=recall,
            set_f1=f1,
        )

    @staticmethod
    def _coerce_list(value: Any) -> list | None:
        if isinstance(value, list):
            return value
        if isinstance(value, (set, tuple)):
            return list(value)
        # The benchmark stores set-enumeration gold as JSON-encoded
        # list strings (Spark's default array
        # serialization at write time). Try a defensive
        # ``json.loads`` so the scorer handles both shapes — already
        # parsed list (from predictions.jsonl) and JSON-string list
        # (from the released gold).
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, list):
                    return parsed
        return None

    def _score_numeric(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        # Parse prediction — the model's ``answer`` may be a native
        # number or a string with a trailing unit.
        parsed = parse_numeric(prediction.answer)
        if not parsed.ok:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
                # Surface the parse failure in the failure-breakdown
                # alongside the unit / value categories so analysis
                # code can see the parse-fail share without manually
                # joining ``format_error`` and ``answer_type``.
                numeric_failure_category="parse_error",
            )
        try:
            gold_val = float(gold.ground_truth)
        except (TypeError, ValueError):
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )

        # Unit comparison — canonicalize both sides.
        declared_unit = (
            prediction.unit if prediction.unit else parsed.declared_unit
        )
        pred_unit = canonicalize_unit(declared_unit)
        gold_unit = canonicalize_unit(gold.answer_unit)
        # Booleans for which side speaks about units at all.
        # When gold has no unit (e.g., Numeric (integer) count), unit
        # match defaults to True — the question didn't request a
        # specific unit.
        unit_match: bool
        if gold_unit is None:
            unit_match = True
        else:
            unit_match = pred_unit == gold_unit

        # Distance and tolerance. TID 57.x are dimensionless shape
        # indices tagged bare ``numeric`` (unit-less), so the unit-based
        # dispatch would route them to ``count`` (absolute δ=1.0, wider
        # than their entire value range). Route them to the ``ratio``
        # kind instead — a purely relative tolerance.
        if gold.tid in _RATIO_TIDS:
            kind = "ratio"
        else:
            kind = self._numeric_kind_for_unit(gold_unit)
        cfg = self._config.to_numeric_config()
        # Explicit None check — ``parsed.value or 0.0`` would also
        # treat a ``False`` / empty-collection value as 0.0 silently.
        # ``NumericParse.value`` is a float (or None on parse-failure);
        # the explicit form is robust against schema evolution.
        pred_val = parsed.value if parsed.value is not None else 0.0
        d = numeric_distance(pred_val, gold_val)
        tol = numeric_tolerance(gold_val, kind=kind, config=cfg)
        number_within_tolerance = d < tol
        em = int(number_within_tolerance and unit_match)
        ne = min(1.0, d / tol) if tol > 0 else (0.0 if d == 0 else 1.0)

        failure_category: NumericFailureCategory | None = None
        if em == 0:
            if number_within_tolerance and not unit_match:
                failure_category = "number_correct_unit_wrong"
            elif not number_within_tolerance and unit_match:
                failure_category = "number_wrong_unit_correct"
            else:
                failure_category = "both_wrong"

        # Unit-normalized companions. When pred_unit and gold_unit are
        # in the same dimensional group (distance / area), convert
        # pred_val into the gold unit and recompute d + EM/NE against
        # the same tolerance. Conversion impossible (unitless,
        # unknown, cross-group) → fall back to strict values, so the
        # normalized metrics are at least as lenient as strict, never
        # stricter. When conversion is possible, normalized EM ≥
        # strict EM and normalized NE ≤ strict NE by construction.
        em_normalized: int = em
        ne_normalized: float = ne
        if gold_unit is not None:
            converted = convert_to_gold_unit(pred_val, pred_unit, gold_unit)
            if converted is not None:
                d_norm = numeric_distance(converted, gold_val)
                within = d_norm < tol
                em_normalized = int(within)
                ne_normalized = (
                    min(1.0, d_norm / tol) if tol > 0 else (0.0 if d_norm == 0 else 1.0)
                )

        return ScoreRecord(
            **base,
            em=em,
            ne=ne,
            covered=True,
            raw_distance=d,
            raw_tolerance=tol,
            unit_match=unit_match,
            numeric_failure_category=failure_category,
            em_unit_normalized=em_normalized,
            ne_unit_normalized=ne_normalized,
        )

    @staticmethod
    def _numeric_kind_for_unit(gold_unit: str | None) -> str:
        """Map a canonicalized gold unit string to the numeric
        kind used for default δ selection."""
        if gold_unit is None:
            return "count"
        return NUMERIC_KIND_BY_UNIT_SUFFIX.get(gold_unit, "count")

    def _score_coordinates(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        pred_pt = self._coerce_latlon(prediction.answer)
        gold_pt = self._coerce_latlon(gold.ground_truth)
        if pred_pt is None or gold_pt is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        d = coordinate_distance_km(pred_pt, gold_pt)
        # TID 64.4 ("Convert the location to a lat/long grid at
        # {resolution} degrees") is typed Coordinates but the gold is a
        # coordinate SNAPPED to a fixed grid (e.g. "18.39_-66.02" at
        # 0.01 deg). The generic 5/10 km coordinate tolerance is ~9 grid
        # cells wide here, so a prediction in the wrong cell — or the
        # raw un-snapped location — scores EM=1 and the quantization is
        # never tested. Derive the cell size from the gold's decimal
        # precision and tighten the tolerance to a half-cell diagonal so
        # only a prediction in (or on the border of) the correct cell
        # passes. Latitude ~111 km/deg is used as a conservative upper
        # bound (longitude is shorter), so correct in-cell points are
        # never rejected. Falls back to the generic tolerance when the
        # gold precision can't be read.
        if gold.tid == "64.4":
            ndp = _coord_decimal_places(gold.ground_truth)
            if ndp is not None and ndp > 0:
                res_deg = 10.0 ** (-ndp)
                tol = res_deg * 111.0 * (2.0 ** 0.5) / 2.0
            else:
                tol = max(
                    self._config.coordinate_delta_km,
                    self._config.alpha * self._config.coordinate_d_scale_km,
                )
        else:
            tol = max(
                self._config.coordinate_delta_km,
                self._config.alpha * self._config.coordinate_d_scale_km,
            )
        em = int(d < tol)
        ne = min(1.0, d / tol) if tol > 0 else (0.0 if d == 0 else 1.0)
        return ScoreRecord(
            **base,
            em=em,
            ne=ne,
            covered=True,
            raw_distance=d,
            raw_tolerance=tol,
        )

    # Coordinate separator regex, built at import from
    # ``scoring_spec.json:coordinates.separator_chars``. Underscore is
    # in the default class for TID 64.4 (``'<lat>_<lon>'``
    # underscore-joined). The scorer only reaches this regex on rows
    # whose canonical answer-type is ``Coordinates``, so the looser
    # separator class can't accidentally capture unrelated string
    # values.
    _COORD_STRING_RE = re.compile(
        r"^\s*(-?\d+(?:\.\d+)?)\s*["
        + _load_coord_separator_chars()
        + r"]\s*(-?\d+(?:\.\d+)?)\s*$"
    )

    @staticmethod
    def _coerce_latlon(value: Any) -> tuple[float, float] | None:
        """Parse a lat/lon pair from any of the shapes the benchmark
        or models use: ``{"lat": …, "lon": …}`` dict, ``[lat, lon]``
        pair, or a ``"lat, lon"`` string. Returns ``None`` on
        unparseable input."""
        if value is None:
            return None
        if isinstance(value, dict):
            # Use ``in`` rather than truthiness so latitude / longitude
            # of ``0.0`` (equator / prime meridian) aren't misread as
            # missing.
            if "lat" in value:
                lat = value["lat"]
            elif "latitude" in value:
                lat = value["latitude"]
            else:
                return None
            if "lon" in value:
                lon = value["lon"]
            elif "lng" in value:
                lon = value["lng"]
            elif "longitude" in value:
                lon = value["longitude"]
            else:
                return None
            if lat is None or lon is None:
                return None
            try:
                return float(lat), float(lon)
            except (TypeError, ValueError):
                return None
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
        if isinstance(value, str):
            m = Scorer._COORD_STRING_RE.match(value)
            if m is None:
                return None
            try:
                return float(m.group(1)), float(m.group(2))
            except ValueError:
                return None
        return None

    def _score_geometry(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """Geometry covers lines and polygons. Polygon detection:
        treat as polygon iff (a) the gold is a closed ring (first
        coord == last coord) AND (b) the resulting polygon is valid
        and has non-trivial area in lon/lat space. A closed *route*
        (start == end on a touring path) would otherwise be silently
        misrouted to ``polygon_iou``; the area + validity check
        catches it because a route has near-zero enclosed area
        compared with a real polygon. A single ``(lat, lon)`` pair
        routes through coordinate scoring (caller shouldn't hit this
        path for that case but we handle it defensively)."""
        pred_coords = self._coerce_coord_list(prediction.answer)
        gold_coords = self._coerce_coord_list(gold.ground_truth)
        if pred_coords is None or gold_coords is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        # Empty pred and empty gold both coerce to ``[]`` and would
        # otherwise route into ``polygon_iou([], []) = 1.0`` for a
        # spurious EM=1. Treat both-empty as a format error since the
        # benchmark has no rows where empty geometry is the expected
        # answer.
        if not pred_coords and not gold_coords:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        # Closed-ring detection uses ``isclose`` so a JSON-serialised
        # tuple-vs-list mismatch (or float roundoff) doesn't classify
        # a polygon as a line.
        is_polygon = len(gold_coords) >= 4 and (
            math.isclose(gold_coords[0][0], gold_coords[-1][0], abs_tol=1e-9)
            and math.isclose(gold_coords[0][1], gold_coords[-1][1], abs_tol=1e-9)
        )
        # The closed-ring shape is necessary but not sufficient.
        # A closed route (path that loops back) also satisfies it but
        # has near-zero polygon area. Compute the polygon area; route
        # the row to the line matcher if it's degenerate.
        if is_polygon:
            try:
                from shapely.geometry import Polygon

                gold_poly = Polygon(
                    [(lon, lat) for (lat, lon) in gold_coords]
                )
                if not gold_poly.is_valid or gold_poly.area < 1e-12:
                    is_polygon = False
            except Exception:
                # Degenerate / non-importable: fall back to the
                # closed-ring heuristic alone.
                pass
        if is_polygon:
            iou, used_planar_fallback = polygon_iou(pred_coords, gold_coords)
            if used_planar_fallback:
                _warn_planar_fallback_once()
            em = int(iou >= self._config.polygon_iou_threshold)
            ne = 1.0 - iou
            return ScoreRecord(
                **base,
                em=em,
                ne=ne,
                covered=True,
                raw_distance=1.0 - iou,
                raw_tolerance=1.0,
                polygon_iou_used_planar_fallback=used_planar_fallback,
            )
        else:
            d = line_hausdorff_km(pred_coords, gold_coords)
            tol = self._config.line_tolerance_km
            em = int(d < tol)
            ne = min(1.0, d / tol) if tol > 0 else (0.0 if d == 0 else 1.0)
            return ScoreRecord(
                **base,
                em=em,
                ne=ne,
                covered=True,
                raw_distance=d,
                raw_tolerance=tol,
            )

    @staticmethod
    def _coerce_coord_list(value: Any) -> list[tuple[float, float]] | None:
        if not isinstance(value, list):
            return None
        out: list[tuple[float, float]] = []
        for p in value:
            pt = Scorer._coerce_latlon(p)
            if pt is None:
                return None
            out.append(pt)
        return out

    def _score_date(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        pred_dc = parse_date(prediction.answer)
        gold_dc = parse_date(gold.ground_truth)
        if pred_dc is None or gold_dc is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        em = int(date_components_match(pred_dc, gold_dc))
        d = date_day_distance(pred_dc, gold_dc)
        tol_days = date_precision_tolerance_days(pred_dc, gold_dc)
        ne = min(1.0, d / tol_days) if tol_days > 0 else (0.0 if d == 0 else 1.0)
        return ScoreRecord(
            **base,
            em=em,
            ne=ne,
            covered=True,
            raw_distance=float(d),
            raw_tolerance=float(tol_days),
        )

    def _score_enum_via_aliases(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """For Cardinal direction / Predefined value types: alias lookup
        against the known canonical vocabulary, with a quote-stripped
        casefold equality fallback when the alias store can't resolve.

        The fallback matches :meth:`_score_entity_name`'s behaviour:
        when the alias store is incomplete (or constructed empty, as
        for the default ``Scorer()``), surface equality is the best
        available signal. Without this fallback, a default-constructed
        ``Scorer()`` would silently score every cardinal/predefined row
        as EM=0.
        """
        gold_canonical = self._normalize_predefined_gold(
            gold.ground_truth, gold.answer_type
        )
        resolved = self._aliases.resolve(
            prediction.answer, gold_canonical_id=gold_canonical
        )
        match = resolved is not None and resolved == gold_canonical
        # Casefold fallback when the alias store is silent on either
        # side. Mirrors ``_score_entity_name``. Only kicks in when we
        # don't already have a positive alias-backed match.
        resolve_failure = resolved is None
        if not match and isinstance(prediction.answer, str):
            stripped_pred = _strip_quotes(prediction.answer)
            stripped_gold = (
                _strip_quotes(gold.ground_truth)
                if isinstance(gold.ground_truth, str)
                else None
            )
            if (
                isinstance(stripped_pred, str)
                and isinstance(stripped_gold, str)
                and stripped_pred
                and stripped_gold
                and stripped_pred.casefold() == stripped_gold.casefold()
            ):
                match = True
                resolve_failure = False
        return ScoreRecord(
            **base,
            em=int(match),
            ne=0.0 if match else 1.0,
            covered=True,
            raw_distance=0.0 if match else 1.0,
            raw_tolerance=1.0,
            resolve_failure=resolve_failure,
        )

    # The ``cardinal:`` / ``predef:`` prefixes are alias-store identity
    # conventions. They're hardcoded here AND in the alias store JSON
    # (e.g. ``"cardinal:N"``, ``"predef:left"``). They are not defined
    # in one shared constant, so changing the alias-store layout means
    # changing both this module and every existing alias file.
    _CARDINAL_PREFIX = "cardinal:"
    _PREDEFINED_PREFIX = "predef:"

    def _normalize_predefined_gold(
        self, gold: Any, answer_type: str
    ) -> str | None:
        """Map a gold value to its canonical cardinal: / predef: id.

        Gold rows store the value in natural form (``"N"``, ``"left"``).
        We resolve it through the alias store so the case / language
        of the gold doesn't matter — it only has to appear as an
        alias somewhere.
        """
        if gold is None:
            return None
        s = str(gold).strip()
        if s.startswith((self._CARDINAL_PREFIX, self._PREDEFINED_PREFIX)):
            return s
        resolved = self._aliases.resolve(s)
        if resolved is not None:
            return resolved
        # Fall back to a naively-constructed id (won't match anything
        # if the alias store is missing — scorer will flag
        # resolve_failure=True for the prediction side).
        prefix = (
            self._CARDINAL_PREFIX
            if answer_type.startswith("Cardinal direction")
            else self._PREDEFINED_PREFIX
        )
        return f"{prefix}{s}"

    # ------------------------------------------------------------------
    # Ternary / geohash / composite / bounding-box / polar-coordinates
    # ------------------------------------------------------------------

    def _score_ternary(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """Three-valued logic: ``true``/``false``/``plausibly``.

        Case-normalised string equality. Upstream gold uses
        ``"Plausibly"`` capitalised; we lowercase both sides. Models
        that emit a bool get coerced to ``"true"``/``"false"`` for
        comparison; ``"plausibly"`` stays as a string.
        """
        gold_val = _normalize_ternary(gold.ground_truth)
        pred_val = _normalize_ternary(prediction.answer)
        if gold_val is None or pred_val is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        match = gold_val == pred_val
        return ScoreRecord(
            **base,
            em=int(match),
            ne=0.0 if match else 1.0,
            covered=True,
            raw_distance=0.0 if match else 1.0,
            raw_tolerance=1.0,
        )

    # Base-32 alphabet used by geohash (Niemeyer's encoding — same as
    # OpenStreetMap and most geohash libraries). Excludes ``a``, ``i``,
    # ``l``, ``o`` to avoid visual confusion.
    _GEOHASH_ALPHABET = frozenset("0123456789bcdefghjkmnpqrstuvwxyz")

    def _score_geohash(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """Strict case-insensitive equality for base-32 geohashes."""
        gold_s = str(gold.ground_truth).strip().lower() if gold.ground_truth is not None else ""
        pred_s = str(prediction.answer).strip().lower() if prediction.answer is not None else ""
        if not gold_s or not pred_s:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        # Structural validation — a random 16-char alphanumeric string
        # should not be silently treated as a (mismatching) geohash.
        # Reject when any character is outside the base-32 alphabet.
        # Gold-side malformed values fall through to the equality
        # check (data defect surfaced as em=0).
        if any(c not in self._GEOHASH_ALPHABET for c in pred_s):
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        match = gold_s == pred_s
        return ScoreRecord(
            **base,
            em=int(match),
            ne=0.0 if match else 1.0,
            covered=True,
            raw_distance=0.0 if match else 1.0,
            raw_tolerance=1.0,
        )

    def _score_composite_string(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """``"H3(res=R): <int> | S2(lvl=L): <int>"`` — TID 42.1
        (multimodal "which grid cell contains [IMAGE_A]").

        ``em`` is strict equality of both pipe-separated parts after
        whitespace-normalization (the universal metric). A
        spatial companion (``em_spatial`` / ``ne_spatial``) additionally
        scores the H3 half by geographic containment / proximity so a
        prediction that localizes the entity is credited even when the
        exact ID / resolution / encoding differs.

        BENCHMARK-DEFECT NOTE (known limitation of the released data;
        the prompt ships as-is): the response
        schema's ``nl_description`` hardcodes ``H3(res=15) | S2(lvl=20)``
        in hex, but the gold uses variable low resolutions (H3 res 0-4,
        S2 lvl 2-9) in decimal, and the target resolution is neither
        stated in the question nor recoverable from it. So strict ``em``
        is structurally ~0 regardless of model skill; the spatial
        companion (resolution-agnostic) is the fair signal here. The S2
        half is not spatially scored (no S2 decoder dependency)."""
        gold_parts = _split_composite(gold.ground_truth)
        pred_parts = _split_composite(prediction.answer)
        if gold_parts is None or pred_parts is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        match = gold_parts == pred_parts
        em_spatial, ne_spatial = self._composite_h3_spatial(
            gold.ground_truth, prediction.answer
        )
        return ScoreRecord(
            **base,
            em=int(match),
            ne=0.0 if match else 1.0,
            covered=True,
            raw_distance=0.0 if match else 1.0,
            raw_tolerance=1.0,
            em_spatial=em_spatial,
            ne_spatial=ne_spatial,
        )

    def _composite_h3_spatial(
        self, gold_value: Any, pred_value: Any
    ) -> tuple[int | None, float | None]:
        """Spatial companion for Composite (H3 half only).

        Extracts the H3 token from each ``H3(res=R): X | S2(lvl=L): Y``
        string and defers to :meth:`_h3_pair_spatial` for the
        resolution-agnostic containment / proximity score. The S2 half
        has no decoder and is not spatially scored."""
        gold_h3 = _decode_h3_cell(_extract_composite_h3(gold_value))
        pred_h3 = _decode_h3_cell(_extract_composite_h3(pred_value))
        return self._h3_pair_spatial(gold_h3, pred_h3)

    def _h3_pair_spatial(
        self,
        gold_h3: tuple[str, int, tuple[float, float]] | None,
        pred_h3: tuple[str, int, tuple[float, float]] | None,
    ) -> tuple[int | None, float | None]:
        """Resolution-agnostic spatial score for a decoded gold/pred H3
        pair (each ``(cell_hex, resolution, (lat, lon))`` from
        :func:`_decode_h3_cell`, or ``None``).

        Credits containment in either direction — the finer cell's
        ancestor at the coarser cell's resolution equals the coarser
        cell — so a prediction that localizes the target is scored even
        when the resolution / encoding differs from gold. Falls back to
        centroid geodesic distance against the gold cell's edge length.

        Returns ``(em_spatial, ne_spatial)``: ``(None, None)`` when gold
        can't be decoded (no companion signal), ``(0, 1.0)`` when gold
        decodes but the prediction doesn't. Non-H3 ids (e.g. S2 tokens)
        decode to ``None`` and therefore receive no spatial companion."""
        if gold_h3 is None:
            return None, None
        if pred_h3 is None:
            return 0, 1.0
        g_cell, g_res, g_pt = gold_h3
        p_cell, p_res, p_pt = pred_h3
        if self._config.composite_h3_containment:
            try:
                import h3

                # Whichever cell is finer, map it to the coarser
                # resolution and check it lands in the coarser cell.
                if p_res >= g_res and h3.cell_to_parent(p_cell, g_res) == g_cell:
                    return 1, 0.0
                if g_res > p_res and h3.cell_to_parent(g_cell, p_res) == p_cell:
                    return 1, 0.0
            except Exception:
                pass
        # Centroid-distance fallback; tolerance = gold cell edge length.
        try:
            import h3

            tol = h3.average_hexagon_edge_length(g_res, unit="km")
        except Exception:
            tol = 0.0
        d = coordinate_distance_km(p_pt, g_pt)
        if tol <= 0:
            return int(d == 0), (0.0 if d == 0 else 1.0)
        return int(d < tol), min(1.0, d / tol)

    def _score_bounding_box(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """Axis-aligned bbox with IoU threshold.

        Gold format: ``"lat_min, lon_min, lat_max, lon_max"`` string.
        Predictions accepted as the same string, a 4-element list, or
        a dict with ``"lat_min"``/``"lon_min"``/``"lat_max"``/``"lon_max"``.
        EM = ``1[IoU ≥ threshold]``; NE = ``1 - IoU``.
        """
        gold_box = _coerce_bbox(gold.ground_truth)
        pred_box = _coerce_bbox(prediction.answer)
        if gold_box is None or pred_box is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        # Degenerate gold (zero-area point / line envelope). The
        # bounding box of a single-point facility *is* that point, so
        # 82% of TID 66.1 golds are zero-area. IoU against a zero-area
        # gold is 0 for any non-coincident prediction, which collapses
        # the documented ``IoU >= threshold`` rule to bit-exact float
        # equality — a correct enclosing box or an 11 cm miss both score
        # EM=0. Fall back to the same geodesic-proximity regime as
        # single-point Coordinates so semantically-identical content is
        # not scored with zero tolerance under one type and a 5 km
        # tolerance under another.
        g_la1, g_lo1, g_la2, g_lo2 = gold_box
        gold_area = (g_la2 - g_la1) * (g_lo2 - g_lo1)
        if gold_area <= 0.0:
            p_la1, p_lo1, p_la2, p_lo2 = pred_box
            gold_pt = ((g_la1 + g_la2) / 2.0, (g_lo1 + g_lo2) / 2.0)
            pred_pt = ((p_la1 + p_la2) / 2.0, (p_lo1 + p_lo2) / 2.0)
            d = coordinate_distance_km(gold_pt, pred_pt)
            tol = max(
                self._config.coordinate_delta_km,
                self._config.alpha * self._config.coordinate_d_scale_km,
            )
            em = int(d < tol)
            ne = min(1.0, d / tol) if tol > 0 else (0.0 if d == 0 else 1.0)
            return ScoreRecord(
                **base,
                em=em,
                ne=ne,
                covered=True,
                raw_distance=d,
                raw_tolerance=tol,
            )
        iou = _bbox_iou(gold_box, pred_box)
        em = int(iou >= self._config.polygon_iou_threshold)
        ne = max(0.0, min(1.0, 1.0 - iou))
        return ScoreRecord(
            **base,
            em=em,
            ne=ne,
            covered=True,
            raw_distance=1.0 - iou,
            raw_tolerance=1.0 - self._config.polygon_iou_threshold,
        )

    def _score_polar_coordinates(
        self, prediction: Prediction, gold: GoldRow, base: dict
    ) -> ScoreRecord:
        """(angle_degrees, distance_km) pair.

        EM requires the angle within ± 5° AND the radial distance
        within the distance tolerance (reusing ``coordinate_delta_km``
        as the additive δ). ``NE`` is the max of the two normalised
        errors, capped at 1.
        """
        gold_pol = _coerce_polar(gold.ground_truth)
        pred_pol = _coerce_polar(prediction.answer)
        if gold_pol is None or pred_pol is None:
            return ScoreRecord(
                **base,
                em=0,
                ne=1.0,
                covered=False,
                format_error=True,
            )
        g_ang, g_r = gold_pol
        p_ang, p_r = pred_pol
        # Angle diff modulo 360, wrapped to [0, 180].
        d_ang = abs((p_ang - g_ang + 180.0) % 360.0 - 180.0)
        d_r = abs(p_r - g_r)
        angle_tol = self._config.polar_angle_tol_degrees
        # Distance tolerance reuses the numeric-distance kind so the
        # scale-from-magnitude bucket logic kicks in (urban / regional
        # / national δ). This keeps polar radial scoring consistent
        # with single-point coordinate scoring on the same magnitudes.
        cfg = self._config.to_numeric_config()
        dist_tol = numeric_tolerance(g_r, kind="distance", config=cfg)
        # ``<`` on both halves, matching the documented hit-at-tolerance
        # rule (EM = 1[d < τ]) used by every other continuous type.
        em = int(d_ang < angle_tol and d_r < dist_tol)
        # Normalised error: the larger of the two ratios, capped at 1.
        ne = max(
            min(1.0, d_ang / max(angle_tol, 1e-9)),
            min(1.0, d_r / max(dist_tol, 1e-9)),
        )
        return ScoreRecord(
            **base,
            em=em,
            ne=ne,
            covered=True,
            raw_distance=d_r,
            raw_tolerance=dist_tol,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions — shared between scorer branches)
# ---------------------------------------------------------------------------


def _sc_agreement(samples: list[Any] | None) -> float | None:
    """Fraction of self-consistency samples matching the modal value.

    Returns ``None`` when ``samples`` is missing or has fewer than 2
    entries — agreement is undefined for N<2. Equality is computed on
    a stable JSON serialisation so dict / list answers compare
    correctly; on serialisation failure, the value falls back to
    ``repr(s)``.
    """
    if not samples or len(samples) < 2:
        return None
    counts: dict[str, int] = {}
    for s in samples:
        try:
            key = json.dumps(s, sort_keys=True, default=str)
        except (TypeError, ValueError):
            key = repr(s)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values()) / len(samples)


_COORD_SUFFIX_RE = re.compile(
    r"\s*\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)\s*$"
)


def _strip_quotes(value: Any) -> Any:
    """Strip wrapping ``"..."`` and/or a trailing ``(lat, lon)`` coordinate suffix.

    Upstream sometimes ships entity names with literal wrapping quotes
    (``'"Berlin"'``) and/or a disambiguating coordinate suffix appended by the
    entity-name sampler (``'"WALGREEN CO" (44.671, -85.655)'``). A correct
    bare-name prediction must still match, so we strip a wrapping quote pair and
    a trailing numeric ``(lat, lon)`` pair, iterating so ``'"NAME" (lat,lon)'``
    reduces to ``'NAME'``. The coordinate regex is anchored to the end and
    requires two numbers, so internal parentheticals (``'... (BPHC) ...'``,
    ``'Coco, PR'``) are preserved. This runs on both the single-name and the
    set-element comparison paths (both call this helper).
    """
    if isinstance(value, str):
        s = value.strip()
        for _ in range(3):
            before = s
            if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                s = s[1:-1].strip()
            m = _COORD_SUFFIX_RE.search(s)
            if m:
                s = s[: m.start()].strip()
            if s == before:
                break
        return s
    return value


_H3_HEX_RE = re.compile(r"^[0-9a-f]{14,16}$")


def _is_valid_h3_or_s2_cell(hx: str) -> bool:
    """Return True if ``hx`` is a structurally valid H3 OR S2 cell id.

    H3 has a strict bit structure (mode + resolution + base cell +
    digit chain); ``h3.is_valid_cell`` is authoritative. S2 uses the
    full 64-bit space without H3-style mode bits, but valid S2 cell
    ids have one specific trailing-bit pattern (the lowest set bit
    encodes level). Accept either.

    The matcher accepts any 14–16 hex string that survives one of
    the two structural checks. If ``h3`` raises (it does for
    out-of-range ints), the helper returns False.
    """
    try:
        import h3
    except ImportError:
        # ``h3`` is an optional dependency; without it, fall back to
        # the length check alone.
        return 14 <= len(hx) <= 16

    try:
        if h3.is_valid_cell(hx):
            return True
    except Exception:
        pass

    # S2 cell-id validity: lowest set bit (the "lsb") encodes the
    # level. A valid S2 cell id is non-zero and has exactly one set
    # bit at an odd position counted from the top of the 64-bit
    # value. ``s2sphere`` is not a hard dep, so fall back to the
    # bit-pattern check.
    try:
        n = int(hx, 16) & 0xFFFFFFFFFFFFFFFF
    except ValueError:
        return False
    if n == 0:
        return False
    # Lowest set bit position must be ≥ 0 and ≤ 60 (max level=30 →
    # lsb position 4; min level=0 → lsb at 60). Any bit position is
    # technically possible for arbitrary 64-bit junk, but additionally
    # require: face bits (top 3) ≤ 5 (six S2 cube faces).
    face = (n >> 61) & 0x7
    return face <= 5


def _canonicalize_h3_hex(value: Any) -> str | None:
    """Normalise an H3 / S2 cell id to lowercase hex, with structural
    validation.

    Accepts:

    * Lowercase hex string (14–16 chars; 15 is the typical H3 form;
      resolution-15 cells at ~16 hex digits appear too) — returned
      as-is after lowercasing **iff** :func:`_is_valid_h3_or_s2_cell`
      passes.
    * Decimal integer string — converted via ``int(s) → hex`` then
      validated.
    * Negative signed int64 string — unsigned-casts via
      ``value & 0xFFFFFFFFFFFFFFFF``. Upstream observed these on
      resolution-15 cells where the high bit is set.
    * Python int — same handling.

    Returns ``None`` when the input can't be parsed or doesn't pass
    structural validation. The structural check is what rejects a
    decimal that merely happens to fit in 14-16 hex digits
    after the int64 mask, such as a bare count like ``"99...99"``.
    """
    if value is None:
        return None

    def _int_to_hex(n: int) -> str | None:
        n = n & 0xFFFFFFFFFFFFFFFF
        hx = format(n, "x")
        if not (14 <= len(hx) <= 16):
            return None
        return hx if _is_valid_h3_or_s2_cell(hx) else None

    if isinstance(value, int) and not isinstance(value, bool):
        return _int_to_hex(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if _H3_HEX_RE.match(s):
            return s if _is_valid_h3_or_s2_cell(s) else None
        try:
            n = int(s)
        except ValueError:
            return None
        return _int_to_hex(n)
    return None


_TERNARY_VALUES: frozenset[str] = frozenset({"true", "false", "plausibly"})


def _normalize_ternary(value: Any) -> str | None:
    """Normalise to one of ``{true, false, plausibly}`` or ``None``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in _TERNARY_VALUES:
        return s
    # Fold yes/no to the boolean forms.
    if s in {"yes", "y"}:
        return "true"
    if s in {"no", "n"}:
        return "false"
    return None


def _split_composite(value: Any) -> tuple[str, ...] | None:
    """Split ``'H3(res=15): X | S2(lvl=20): Y'`` into normalised parts.

    Parts are sorted lexicographically before return so the matcher is
    insensitive to the order in which the model emits the halves —
    semantic equality only depends on the H3 and S2 ids, not on
    whether H3 was listed first or second.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    elif isinstance(value, str):
        parts = [p.strip() for p in value.split("|")]
    else:
        return None
    if not all(parts):
        return None
    return tuple(sorted(parts))


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Parse a bounding box into ``(lat_min, lon_min, lat_max, lon_max)``.

    Accepts a comma-separated string (gold shape), a 4-element list,
    or a dict with the four named keys. Normalises so ``lat_min ≤
    lat_max`` and ``lon_min ≤ lon_max`` — upstream occasionally
    produces swapped corners.
    """
    if value is None:
        return None
    nums: list[float] | None = None
    if isinstance(value, str):
        try:
            parts = [p.strip() for p in value.split(",")]
            nums = [float(p) for p in parts]
        except (ValueError, TypeError):
            return None
    elif isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            nums = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
    elif isinstance(value, dict):
        keys = ("lat_min", "lon_min", "lat_max", "lon_max")
        if not all(k in value for k in keys):
            return None
        try:
            nums = [float(value[k]) for k in keys]
        except (TypeError, ValueError):
            return None
    if nums is None or len(nums) != 4:
        return None
    la1, lo1, la2, lo2 = nums
    lat_min, lat_max = (la1, la2) if la1 <= la2 else (la2, la1)
    lon_min, lon_max = (lo1, lo2) if lo1 <= lo2 else (lo2, lo1)
    return (lat_min, lon_min, lat_max, lon_max)


def _majority_gold_key(gt: Any) -> str | None:
    """Stable, hashable normalization of a gold answer for the
    majority-class baseline. ``None`` gold (false-premise / missing) →
    ``None`` so those rows don't anchor a modal class. Booleans map to
    ``"true"``/``"false"``; lists/dicts to sorted JSON; scalars to a
    casefolded string. This is a coarse reference key, not the scoring
    canonicalization — it only needs to be consistent per gold value.
    """
    if gt is None:
        return None
    if isinstance(gt, bool):
        return "true" if gt else "false"
    if isinstance(gt, (list, dict)):
        try:
            return json.dumps(gt, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(gt)
    return str(gt).strip().casefold()


def _coord_decimal_places(value: Any) -> int | None:
    """Max fractional-digit count across the numbers in a coord string.

    Used to recover the grid resolution of a snapped-coordinate gold
    (TID 64.4): ``"18.39_-66.02"`` → 2 → 0.01 deg. Returns ``None`` when
    the value is not a string or carries no decimal point.
    """
    if not isinstance(value, str):
        return None
    fracs = re.findall(r"-?\d+\.(\d+)", value)
    if not fracs:
        return None
    return max(len(f) for f in fracs)


def _bbox_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Axis-aligned IoU in (lat, lon) degrees.

    Degenerate boxes (zero area on one or both sides) yield IoU=1.0
    iff they coincide exactly, else 0.0. Zero-area gold boxes are the
    *majority* of TID 66.1 (single-point facilities whose bounding box
    is the point itself, ~82% of released rows), so the scorer must NOT
    reach this exact-equality path for them: :meth:`Scorer._score_bounding_box`
    intercepts a zero-area gold and scores it by geodesic proximity
    instead. This helper's degenerate branch remains only for the rare
    case of a zero-area *prediction* against a non-degenerate gold.
    """
    a_la1, a_lo1, a_la2, a_lo2 = a
    b_la1, b_lo1, b_la2, b_lo2 = b
    ix_la1 = max(a_la1, b_la1)
    ix_lo1 = max(a_lo1, b_lo1)
    ix_la2 = min(a_la2, b_la2)
    ix_lo2 = min(a_lo2, b_lo2)
    if ix_la2 < ix_la1 or ix_lo2 < ix_lo1:
        return 0.0
    inter = (ix_la2 - ix_la1) * (ix_lo2 - ix_lo1)
    area_a = (a_la2 - a_la1) * (a_lo2 - a_lo1)
    area_b = (b_la2 - b_la1) * (b_lo2 - b_lo1)
    union = area_a + area_b - inter
    if union <= 0:
        # Degenerate case: both boxes are points / degenerate rectangles.
        return 1.0 if a == b else 0.0
    return inter / union


_POLAR_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:degrees?|°)?\s*[,;]\s*"
    r"(-?\d+(?:\.\d+)?)\s*(?:km|kilom(?:etres?|eters?))?\s*$",
    re.IGNORECASE,
)


def _coerce_polar(value: Any) -> tuple[float, float] | None:
    """Parse ``'θ degrees, r km'`` → ``(angle, distance)``.

    Accepts strings (gold shape), dicts with ``angle_degrees`` +
    ``distance_km``, or 2-element lists.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        # Membership rather than truthiness so an angle of ``0.0`` /
        # distance of ``0.0`` aren't misread as missing.
        ang = next(
            (value[k] for k in ("angle_degrees", "angle", "theta") if k in value),
            None,
        )
        dist = next(
            (value[k] for k in ("distance_km", "distance", "r") if k in value),
            None,
        )
        if ang is None or dist is None:
            return None
        try:
            return float(ang), float(dist)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        m = _POLAR_RE.match(value)
        if m is None:
            return None
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Grid-cell spatial-proximity helpers (em_spatial / ne_spatial companions).
#
# These decode a grid-cell answer to a geographic location so the scorer
# can credit a right-place prediction that used a different but valid
# grid-encoding convention (see ``scoring_spec.json:grid_cell``). Strict
# exact-ID EM is scored separately and unchanged; these only populate the
# ``em_spatial`` / ``ne_spatial`` companion fields. pyproj / h3 are
# imported lazily so a run that never scores a grid-cell row doesn't pay
# the import cost, and any decode failure degrades to "no spatial signal"
# rather than raising.
# ---------------------------------------------------------------------------

# ``H3(res=R): <id>`` half of a composite string (order-insensitive).
_COMPOSITE_H3_RE = re.compile(
    r"H3\s*\(\s*res\s*=\s*\d+\s*\)\s*:\s*([0-9A-Za-z]+)", re.IGNORECASE
)


def _extract_composite_h3(value: Any) -> str | None:
    """Pull the H3 id token out of a ``H3(res=R): <id> | S2(lvl=L): <id>``
    composite string. Returns the raw id string, or None when absent."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        s = " | ".join(str(p) for p in value)
    else:
        s = str(value)
    m = _COMPOSITE_H3_RE.search(s)
    return m.group(1) if m else None


def _decode_h3_cell(
    token: Any,
) -> tuple[str, int, tuple[float, float]] | None:
    """Normalise an H3 id (decimal or hex) to
    ``(cell_hex, resolution, (lat, lon))`` via the h3 library. Returns
    None when the token is not a genuine H3 cell (the shared
    :func:`_canonicalize_h3_hex` also accepts S2 ids, so an explicit
    ``h3.is_valid_cell`` check re-filters to H3 only)."""
    if token is None:
        return None
    hx = _canonicalize_h3_hex(token)
    if hx is None:
        return None
    try:
        import h3

        if not h3.is_valid_cell(hx):
            return None
        res = h3.get_resolution(hx)
        lat, lon = h3.cell_to_latlng(hx)
        return (hx, res, (lat, lon))
    except Exception:
        return None
