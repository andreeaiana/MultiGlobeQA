"""Schemas for benchmark_models.

Houses:
    - :mod:`type_registry`: canonical Answer Type → :class:`AnswerTypeSpec`
      mapping with JSON schema fragments, NL descriptions, and unit handling.
    - :mod:`spatial_functions`: canonical SFID and (SFID, SSFID) → name
      mapping. Source of truth for streamlined spatial-function names.
    - :mod:`template_registry`: pydantic loader for
      ``output/template_registry.json`` with strict cross-validation against
      both the type registry and the spatial-function registry.
"""

from benchmark_models.schemas.spatial_functions import (
    SPATIAL_FUNCTIONS,
    SPATIAL_SUBFUNCTIONS,
    SpatialFunctionInfo,
    SpatialSubFunctionInfo,
    get_spatial_function,
    get_spatial_subfunction,
    is_known_sfid,
    is_known_subfunction,
    list_spatial_functions,
    list_spatial_subfunctions,
)
from benchmark_models.schemas.template_registry import (
    TemplateInfo,
    TemplateRegistry,
    load_template_registry,
)
from benchmark_models.schemas.type_registry import (
    BENCHMARK_TYPE_TO_CANONICAL,
    DEFAULT_UNIT_BY_BENCHMARK_TYPE,
    TYPE_REGISTRY,
    AnswerTypeSpec,
    canonicalize_answer_type,
    default_unit_for_benchmark_type,
    get_type_spec,
    is_known_type,
    list_canonical_types,
    resolve_canonical_answer_type,
)

__all__ = [
    # Type registry
    "AnswerTypeSpec",
    "TYPE_REGISTRY",
    "BENCHMARK_TYPE_TO_CANONICAL",
    "DEFAULT_UNIT_BY_BENCHMARK_TYPE",
    "canonicalize_answer_type",
    "default_unit_for_benchmark_type",
    "resolve_canonical_answer_type",
    "get_type_spec",
    "is_known_type",
    "list_canonical_types",
    # Spatial-function registry
    "SpatialFunctionInfo",
    "SpatialSubFunctionInfo",
    "SPATIAL_FUNCTIONS",
    "SPATIAL_SUBFUNCTIONS",
    "get_spatial_function",
    "get_spatial_subfunction",
    "is_known_sfid",
    "is_known_subfunction",
    "list_spatial_functions",
    "list_spatial_subfunctions",
    # Template registry
    "TemplateInfo",
    "TemplateRegistry",
    "load_template_registry",
]
