"""Loader for ``benchmark_models/schemas/template_registry.json``.

The template registry is the per-TID metadata the runner, scorer, and
analysis code consult at startup: canonical answer type, SFID, and a
few optional tags. It is generated from the finalized ``benchmark.jsonl``
by ``scripts/build_template_registry_from_benchmark.py`` and versioned
in git.

This module provides:

* :class:`TemplateInfo`: pydantic model for one template entry.
* :class:`TemplateRegistry`: pydantic model wrapping the full file.
* :func:`load_template_registry`: read the JSON file from disk,
  validate against the type + spatial-function registries, and return
  a fully-typed :class:`TemplateRegistry`.

The loader **fails loudly** on missing required fields, unknown
``answer_type`` values, or invalid ``(sfid, ssfid)`` pairs. It does
**not** auto-canonicalize bad data — regenerate the registry from the
benchmark instead.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmark_models.schemas.spatial_functions import (
    SPATIAL_FUNCTIONS,
    SPATIAL_SUBFUNCTIONS,
    is_known_sfid,
    is_known_subfunction,
)
from benchmark_models.schemas.type_registry import TYPE_REGISTRY, is_known_type

# The bundled registry lives next to this module as a package resource.
DEFAULT_REGISTRY_FILENAME = "template_registry.json"


# ----------------------------------------------------------------------------
# TemplateInfo — one entry
# ----------------------------------------------------------------------------


class TemplateInfo(BaseModel):
    """Per-TID metadata derived from the benchmark.

    Carries the minimum the runtime needs to dispatch predictions and
    group scores: the template id, its spatial-function id, and the
    canonical answer type. ``ssfid`` and ``answer_unit`` are optional
    metadata with no runtime role — they're present for analysis code
    that wants per-sub-function or per-unit rollups.
    """

    model_config = ConfigDict(extra="forbid")

    tid: str
    """Template ID (e.g. ``"1.1"``). Matches the ``tid`` field on
    benchmark rows."""

    sfid: str
    """Spatial Function ID (``"A"`` – ``"P"``). Cross-validated against
    :data:`benchmark_models.schemas.spatial_functions.SPATIAL_FUNCTIONS`."""

    ssfid: str | None = None
    """Sub-function id within the SFID family (``"I"``, ``"II"``, …).
    Optional; the benchmark-derived exporter leaves it unset. When
    populated the ``(sfid, ssfid)`` pair is validated against
    :data:`benchmark_models.schemas.spatial_functions.SPATIAL_SUBFUNCTIONS`."""

    answer_type: str
    """Canonical Answer Type. Must appear in
    :data:`benchmark_models.schemas.type_registry.TYPE_REGISTRY`."""

    answer_unit: str | None = None
    """Informational unit for this TID (e.g. ``"km"`` for distance
    templates). The eval scorer derives the authoritative unit from the
    benchmark's per-type default (see
    :data:`DEFAULT_UNIT_BY_BENCHMARK_TYPE`); this field is kept only
    for per-TID analysis rollups."""

    @model_validator(mode="after")
    def _validate_spatial_function(self) -> TemplateInfo:
        """Cross-validate ``sfid`` (and ``ssfid`` when present)."""
        if not is_known_sfid(self.sfid):
            known = ", ".join(sorted(SPATIAL_FUNCTIONS.keys()))
            raise ValueError(
                f"TemplateInfo for tid={self.tid}: unknown SFID {self.sfid!r}. "
                f"Known SFIDs: [{known}]. Add the entry to "
                f"benchmark_models/schemas/spatial_functions.py if this is a "
                f"new spatial function family."
            )
        if self.ssfid is not None and not is_known_subfunction(
            self.sfid, self.ssfid
        ):
            ssfids_for_this_family = sorted(
                ssf for (sf, ssf) in SPATIAL_SUBFUNCTIONS if sf == self.sfid
            )
            raise ValueError(
                f"TemplateInfo for tid={self.tid}: unknown sub-function "
                f"{self.sfid}.{self.ssfid!r}. Known sub-functions under "
                f"SFID {self.sfid!r}: {ssfids_for_this_family}. Add the entry "
                f"to benchmark_models/schemas/spatial_functions.py if this is "
                f"a new sub-function."
            )
        return self

    @model_validator(mode="after")
    def _validate_answer_type(self) -> TemplateInfo:
        """Enforce that ``answer_type`` is in the canonical type registry."""
        if not is_known_type(self.answer_type):
            known = ", ".join(sorted(TYPE_REGISTRY.keys()))
            raise ValueError(
                f"TemplateInfo for tid={self.tid}: unknown answer type "
                f"{self.answer_type!r}. Known types: [{known}]. Either fix "
                f"the registry or add {self.answer_type!r} to the "
                f"type registry in benchmark_models/schemas/type_registry.py."
            )
        return self


# ----------------------------------------------------------------------------
# TemplateRegistry — full file wrapper
# ----------------------------------------------------------------------------


class TemplateRegistry(BaseModel):
    """The full ``template_registry.json`` file, parsed.

    Versioned so future schema migrations can be detected. The
    ``templates`` dict is keyed by tid (matching benchmark JSONL rows).
    """

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    """Schema version. Bumped when the structure changes incompatibly."""

    templates: dict[str, TemplateInfo] = Field(default_factory=dict)
    """tid → :class:`TemplateInfo` mapping."""

    def __len__(self) -> int:
        return len(self.templates)

    def __contains__(self, tid: object) -> bool:
        return tid in self.templates

    def get(self, tid: str) -> TemplateInfo:
        """Look up by tid. Raises ``KeyError`` if absent."""
        if tid not in self.templates:
            raise KeyError(tid)
        return self.templates[tid]


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


def load_template_registry(
    path: str | Path | None = None,
) -> TemplateRegistry:
    """Load + validate the template registry.

    Args:
        path: Optional override. Defaults to the bundled JSON next to
            this module.

    Returns:
        A validated :class:`TemplateRegistry`.
    """
    if path is None:
        resource = files("benchmark_models.schemas").joinpath(
            DEFAULT_REGISTRY_FILENAME
        )
        with resource.open("r", encoding="utf-8") as f:
            raw: Any = json.load(f)
    else:
        with Path(path).open("r", encoding="utf-8") as f:
            raw = json.load(f)
    return TemplateRegistry.model_validate(raw)


__all__ = [
    "DEFAULT_REGISTRY_FILENAME",
    "TemplateInfo",
    "TemplateRegistry",
    "load_template_registry",
]
