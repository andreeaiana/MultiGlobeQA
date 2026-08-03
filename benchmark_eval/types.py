"""Answer-type vocabulary bridge (eval-side re-export).

The benchmark's ``ground_truth_type`` field uses snake_case names; the
eval pipeline's internal vocabulary uses the canonical display names
in :data:`benchmark_models.schemas.TYPE_REGISTRY`. The mapping lives
in :mod:`benchmark_models.schemas.type_registry` so both the eval
pipeline and the model section share one source of truth.

This module re-exports the bridge helpers so existing eval-side
imports (``from benchmark_eval.types import …``) continue to work.
"""

from __future__ import annotations

from benchmark_models.schemas import (
    BENCHMARK_TYPE_TO_CANONICAL,
    DEFAULT_UNIT_BY_BENCHMARK_TYPE,
    canonicalize_answer_type,
)
from benchmark_models.schemas import (
    default_unit_for_benchmark_type as default_unit_for,
)

__all__ = [
    "BENCHMARK_TYPE_TO_CANONICAL",
    "DEFAULT_UNIT_BY_BENCHMARK_TYPE",
    "canonicalize_answer_type",
    "default_unit_for",
]
