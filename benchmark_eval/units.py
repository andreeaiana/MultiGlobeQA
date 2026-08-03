"""Unit conversion for the ``em_unit_normalized`` / ``ne_unit_normalized``
scoring companions.

Loads the dimensionally-compatible unit groups from
``scoring_spec.json`` at import time. Hardcoded fallback matches
the spec exactly — used (with a warning) only when the spec file is
missing, so dev workflows don't break.

The public surface is :func:`convert_to_gold_unit`, which converts a
predicted value from its declared unit into the gold unit when both
are in the same dimensional group. Returns ``None`` for every
cross-group, unitless, or unknown-unit case — the caller treats that
as "conversion impossible" and falls back to strict failure for the
normalized metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from benchmark_eval.canonicalize import canonicalize_unit

from benchmark_eval._paths import spec_path as _spec_path

SCORING_SPEC_PATH = _spec_path()
"""Single source of truth for the spec path — see ``_paths.py`` (D6)."""

# Hardcoded fallbacks — kept in sync with ``scoring_spec.json``.
# Factors are multipliers to the canonical unit of each group.
_FALLBACK_GROUPS: dict[str, dict[str, float]] = {
    "distance": {
        "km": 1.0,
        "m": 0.001,
        "mi": 1.609344,
        "ft": 0.0003048,
        "nmi": 1.852,
    },
    "area": {
        "km2": 1.0,
        "m2": 1e-6,
        "ha": 0.01,
        "mi2": 2.589988110336,
    },
}


def _load_groups() -> dict[str, dict[str, float]]:
    """Load conversion groups from the spec. Falls back to the
    hardcoded table (with a warning) when the spec is missing.
    """
    if not SCORING_SPEC_PATH.exists():
        logger.warning(
            "Scoring spec not found at {}; using hardcoded unit "
            "conversion fallback.",
            SCORING_SPEC_PATH,
        )
        return {k: dict(v) for k, v in _FALLBACK_GROUPS.items()}
    raw: dict[str, Any] = json.loads(SCORING_SPEC_PATH.read_text())
    uc = raw.get("unit_conversion", {})
    groups_raw = uc.get("groups", {})
    if not groups_raw:
        logger.warning(
            "Scoring spec at {} has no unit_conversion.groups; "
            "using hardcoded fallback.",
            SCORING_SPEC_PATH,
        )
        return {k: dict(v) for k, v in _FALLBACK_GROUPS.items()}
    return {
        group_name: dict(group.get("factors_to_canonical", {}))
        for group_name, group in groups_raw.items()
    }


_GROUPS: dict[str, dict[str, float]] = _load_groups()


def _group_of(canonical_unit: str | None) -> str | None:
    """Return the dimensional group name (``"distance"`` / ``"area"``)
    of a canonicalized unit, or ``None`` if the unit is unknown or
    belongs to no conversion group.
    """
    if canonical_unit is None:
        return None
    for group_name, factors in _GROUPS.items():
        if canonical_unit in factors:
            return group_name
    return None


def convert_to_gold_unit(
    pred_value: float,
    pred_unit_canonical: str | None,
    gold_unit_canonical: str | None,
) -> float | None:
    """Convert ``pred_value`` from ``pred_unit_canonical`` to
    ``gold_unit_canonical``.

    Returns the converted value when both units are known and in the
    same dimensional group. Returns ``None`` in every other case:

    * either side is ``None`` / unitless
    * either side is unknown (not in any group)
    * the two sides are in different groups (e.g. distance vs area)

    Returning ``None`` is the caller's signal that conversion is
    impossible — for the normalized metrics, treat it as a strict
    failure (EM=0, NE=1).

    Args:
        pred_value: Numeric value produced by the model.
        pred_unit_canonical: Canonicalized prediction unit (already
            run through :func:`canonicalize_unit`).
        gold_unit_canonical: Canonicalized gold unit.

    Returns:
        ``pred_value`` expressed in ``gold_unit_canonical``, or
        ``None`` when conversion is impossible.
    """
    if pred_unit_canonical is None or gold_unit_canonical is None:
        return None
    if pred_unit_canonical == gold_unit_canonical:
        return pred_value
    pred_group = _group_of(pred_unit_canonical)
    gold_group = _group_of(gold_unit_canonical)
    if pred_group is None or gold_group is None or pred_group != gold_group:
        return None
    factors = _GROUPS[pred_group]
    pred_to_canonical = factors[pred_unit_canonical]
    gold_to_canonical = factors[gold_unit_canonical]
    if gold_to_canonical == 0:
        return None
    # pred_value [pred_unit] * pred_to_canonical → canonical unit
    # canonical_value / gold_to_canonical → gold unit
    return pred_value * pred_to_canonical / gold_to_canonical


def dimensional_group(unit_string: str | None) -> str | None:
    """Public helper: return the dimensional group of a (possibly
    non-canonicalized) unit string. Canonicalizes first, then looks
    up the group. ``None`` when the unit is unknown or unitless.
    """
    return _group_of(canonicalize_unit(unit_string))


__all__ = [
    "convert_to_gold_unit",
    "dimensional_group",
]
