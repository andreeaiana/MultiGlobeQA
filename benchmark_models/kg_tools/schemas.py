"""Per-KG predicate vocabularies and subject-id canonicalisation.

Each of the three production KGs (kwg, osmh3kg, worldkg) uses a
different predicate set for the same conceptual relations: rdf:type,
naming, geometry link, geometry literal. The ``KGSchema`` dataclass
collects those per-KG conventions so the
:class:`~benchmark_models.kg_tools.triple_bridge.TripleBridge` and
:class:`~benchmark_models.kg_tools.entity_service.EntityService`
layers stay KG-agnostic.

Predicate constants are imported live from
``data_extraction/nt_to_delta/config.py`` via a one-shot
``NtToDeltaConfig`` materialisation per KG, so this module is the
single source-of-truth ↔ ETL alignment: if the data team adds a new
naming predicate to a future KG snapshot and updates the transformer
config, this module picks it up without duplication.

The benchmark's ``entity_a_id``/``entity_b_id`` fields carry already-
resolved KG identifiers but their lexical form varies per KG:

* kwg     — local-name suffix only, e.g. ``"zipCodeArea.27351"``.
            Full subject URI is the suffix prefixed with
            ``http://stko-kwg.geog.ucsb.edu/lod/resource/``.
* osmh3kg — bare numeric id, e.g. ``"2649308671"``. KG subjects appear
            in either bare-numeric or ``geo<id>`` form depending on
            whether the triple is about the entity itself or its
            geometry node. :func:`KGSchema.canonicalize_subject_id`
            strips the ``geo`` prefix back to bare-numeric.
* worldkg — bare numeric id, e.g. ``"3814104941"``. Full subject URI
            is ``http://www.worldkg.org/resource/<id>``.

The canonicalizer returns the **entity-level** subject URI/id. The
2-hop geometry resolution (in :class:`EntityService`) discovers the
geometry-node subject via the ``geometry_link_predicates`` triple
rather than guessing the geometry node's URI shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchmark_models.oracle import _normalize_subject
from data_extraction.nt_to_delta.config import NtToDeltaConfig

KGName = Literal["kwg", "osmh3kg", "worldkg"]


@dataclass(frozen=True)
class KGSchema:
    """Per-KG predicate vocabulary + subject-id canonicalisation.

    Instances are module-level singletons (see ``KWG_SCHEMA`` etc.);
    callers should look up by KG name via :func:`schema_for`. The
    predicate tuples are sourced from
    ``data_extraction/nt_to_delta/config.py`` to keep this module and
    the upstream ETL pipeline in lockstep.

    ``subject_uri_prefix`` is the namespace prepended to bare local
    names from benchmark-row ``entity_*_id`` fields to recover the
    full KG subject URI. Empty when subjects in the KG are stored as
    bare ids (osmh3kg).
    """

    name: KGName
    type_predicates: tuple[str, ...]
    name_predicates: tuple[str, ...]
    geometry_link_predicates: tuple[str, ...]
    geometry_data_predicates: tuple[str, ...]
    subject_uri_prefix: str

    def canonicalize_subject_id(self, benchmark_id: str) -> str:
        """Map a benchmark-row entity id to its KG-table subject form.

        For URI-prefixed KGs (kwg, worldkg) this prepends the
        namespace. For id-only KGs (osmh3kg) it strips a ``geo``
        prefix so e.g. ``"geo2649308671"`` and ``"2649308671"`` both
        map to ``"2649308671"`` — matching :func:`oracle._normalize_subject`
        which the gold-triples loader already uses.
        """
        if self.subject_uri_prefix:
            # Never re-prefix a value that is already an absolute IRI —
            # whether this KG's own resource prefix OR any other
            # namespace it emits (geometry-node URIs, ``rdf:type``
            # objects, cross-referenced resources). Agents routinely
            # pass such an IRI straight back into ``get_entity``.
            # Prepending the resource prefix to an ``http(s)://`` value
            # produced malformed ``.../resource/http://...`` subjects
            # that matched zero triples — a kwg-concentrated failure
            # that silently degraded the live-KG tiers. The broader
            # ``http(s)://`` guard subsumes the old exact-prefix check.
            if benchmark_id.startswith(("http://", "https://")):
                return benchmark_id
            return f"{self.subject_uri_prefix}{benchmark_id}"
        # osmh3kg — strip leading "geo" if present.
        return _normalize_subject(benchmark_id)


def _build_schema(kg_source: KGName) -> KGSchema:
    """Materialise an NtToDeltaConfig with KG-specific presets and
    extract its predicate tuples.

    The transformer's config object is the canonical predicate
    registry; constructing it for each KG (with throwaway path args)
    is the cleanest way to pick up KG-specific overrides applied by
    its ``__post_init__`` — without re-implementing the per-KG
    branching here.
    """
    cfg = NtToDeltaConfig(
        input_path=Path("/dev/null"),
        triples_delta_path=Path("/dev/null"),
        entities_delta_path=Path("/dev/null"),
        kg_source=kg_source,
    )
    # ``name_en_predicate`` takes priority in the transformer's entity
    # assembly; surface it alongside the label predicates so both are
    # consulted during fuzzy name search.
    name_predicates = (cfg.name_en_predicate, *cfg.label_predicates)
    subject_uri_prefix = _SUBJECT_URI_PREFIXES[kg_source]
    return KGSchema(
        name=kg_source,
        type_predicates=tuple(cfg.type_predicates),
        name_predicates=name_predicates,
        geometry_link_predicates=tuple(cfg.geometry_link_predicates),
        geometry_data_predicates=tuple(cfg.geometry_predicates),
        subject_uri_prefix=subject_uri_prefix,
    )


# Namespace prefixes prepended to bench-id local names to recover
# full KG subject URIs. Verified empirically against the production
# Delta tables — kwg subjects look like
# "http://stko-kwg.geog.ucsb.edu/lod/resource/zipCodeArea.27351" and
# benchmark rows carry just "zipCodeArea.27351". osmh3kg subjects are
# bare numeric, no namespace; the canonicalizer strips the optional
# "geo" prefix instead.
_SUBJECT_URI_PREFIXES: dict[KGName, str] = {
    "kwg": "http://stko-kwg.geog.ucsb.edu/lod/resource/",
    "osmh3kg": "",
    "worldkg": "http://www.worldkg.org/resource/",
}


KWG_SCHEMA: KGSchema = _build_schema("kwg")
OSMH3KG_SCHEMA: KGSchema = _build_schema("osmh3kg")
WORLDKG_SCHEMA: KGSchema = _build_schema("worldkg")


_SCHEMAS_BY_NAME: dict[KGName, KGSchema] = {
    "kwg": KWG_SCHEMA,
    "osmh3kg": OSMH3KG_SCHEMA,
    "worldkg": WORLDKG_SCHEMA,
}


def schema_for(kg_name: str) -> KGSchema:
    """Look up the ``KGSchema`` for a KG name. Fails loud on unknown
    KGs rather than silently returning a default — a wrong schema
    propagates through every triple query, so a missing entry is the
    kind of bug we want to surface at startup."""
    if kg_name not in _SCHEMAS_BY_NAME:
        raise ValueError(
            f"Unknown KG: {kg_name!r}. "
            f"Expected one of: {sorted(_SCHEMAS_BY_NAME.keys())}"
        )
    return _SCHEMAS_BY_NAME[kg_name]  # type: ignore[index]


__all__ = [
    "KGName",
    "KGSchema",
    "KWG_SCHEMA",
    "OSMH3KG_SCHEMA",
    "WORLDKG_SCHEMA",
    "schema_for",
]
