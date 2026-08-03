"""Entity-shaped API on top of :class:`TripleBridge`.

The :class:`TripleBridge` exposes raw triple-query primitives;
:class:`EntityService` adds per-KG predicate decoding so callers can
work with entity-shaped dicts (``{id, name, types, geometry_wkt,
category}``) without knowing how each KG encodes those relations.

What it does that the bridge doesn't:

* **2-hop geometry resolution**: production KGs encode geometry as
  ``entity → geo:hasGeometry → geometry_node → geo:asWKT →
  "<wkt>"``. This service combines a subject-keyed fetch with a
  follow-up subject-keyed fetch on the linked geometry node so
  callers see one ``geometry_wkt`` field on the entity.
* **Wrapper normalisation**: kwg wraps WKT in quotes, worldkg in
  typed literals, osmh3kg leaves it bare. Reuses
  :func:`benchmark_models.oracle._strip_typed_literal` to strip these
  before any downstream consumption.
* **WKT simplification**: geometry literals are passed through
  :func:`benchmark_models.oracle_simplify.simplify_wkt` (default
  tolerance ``0.001°`` matching the oracle simplifier). Without this
  a single ZCTA polygon (~164 K chars) consumes the agent's entire
  context budget.
* **Cross-form id handling**: ``get_entity("zipCodeArea.27351")``,
  ``get_entity("http://stko-kwg.../zipCodeArea.27351")``, or
  ``get_entity("zipCodeArea.27351", ...)`` all resolve to the same
  KG subject via :meth:`KGSchema.canonicalize_subject_id`.

Affordances explicitly NOT supported (because they don't exist in
the on-disk triples table):

* ``pagerank`` — produced by the transformer's entity build; absent
  from triples. The returned dict carries ``pagerank: None``.
* ``h3_index`` — same; ``h3_index: None``.

These can be re-added later if the transformer-built entity table
becomes available at a stable path; the service would grow a second
backend mode that joins against it. For now, tier-3 just doesn't
have those affordances on the agent's side — and the smoke runs we
inspected showed the agent doesn't use them anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from benchmark_models.kg_tools.schemas import KGSchema
from benchmark_models.kg_tools.triple_bridge import TripleBridge
from benchmark_models.oracle import _strip_typed_literal
from benchmark_models.oracle_simplify import simplify_wkt


# Default WKT simplification tolerance — same as oracle simplifier so
# the methodology footnote covers both surfaces uniformly. ε = 0.001°
# ≈ 111 m on the equator; reduces a typical ZCTA polygon from ~50 K
# chars to ~5 K chars with negligible accuracy impact on spatial-QA.
_DEFAULT_WKT_SIMPLIFY_TOLERANCE: float = 0.001


# find_entity defaults — top-K cap matters more than search recall once
# we have a ranker. The bridge over-fetches by ``_FIND_ENTITY_OVERFETCH``
# so the ranker has signal beyond the truncated head; without
# over-fetch, top-5 = first 5 in arbitrary scan order = noise.
_FIND_ENTITY_TOP_K: int = 5
_FIND_ENTITY_OVERFETCH: int = 25


@dataclass
class EntityService:
    """Per-KG entity decoder wrapping a TripleBridge.

    Stateless (no in-process cache) — every call hits the bridge.
    Construction is cheap; the heavy Spark startup lives on the
    bridge.
    """

    bridge: TripleBridge
    schema: KGSchema
    wkt_simplify_tolerance: float = _DEFAULT_WKT_SIMPLIFY_TOLERANCE

    # ------------------------------------------------------------------
    # Primary API used by the tool wrappers
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> dict[str, Any]:
        """Resolve an entity id (URI or benchmark-shape) to a dict.

        Schema of the return value:

        .. code-block:: python

            {
                "id":             <original entity_id, passthrough>,
                "subject":        <canonical KG subject URI>,
                "name":           <primary name str, or None>,
                "names":          [<all label literals>],
                "types":          [<list of rdf:type object values>],
                "category":       <first type, or None>,
                "geometry_wkt":   <simplified WKT, or None>,
                "geometry_node":  <subject URI of the geometry node, or None>,
                "pagerank":       None,    # not in triples table
                "h3_index":       None,    # not in triples table
                "raw_triples":    [<every triple about the entity>],
            }

        Returns ``{"error": "Entity not found", "id": ..., "subject": ...}``
        when the canonicalised subject has zero triples. (Callers that
        rely on the absence of an ``error`` key as a presence check
        should branch on ``"error" in result`` rather than checking
        truthiness of other fields.)
        """
        subject = self.schema.canonicalize_subject_id(entity_id)
        triples = self.bridge.fetch_triples_for_subject(subject)
        if not triples:
            return {
                "error": "Entity not found",
                "id": entity_id,
                "subject": subject,
            }

        # Decode predicates against the per-KG schema. Each predicate
        # match is one of:
        #   - rdf:type  → types list
        #   - name/label → names list (first wins for "name")
        #   - geometry_link → geometry_node subject (to follow next)
        types: list[str] = []
        names: list[str] = []
        geometry_node: str | None = None
        for t in triples:
            pred = t["predicate"]
            obj = t["object"]
            if pred in self.schema.type_predicates:
                types.append(obj)
            elif pred in self.schema.name_predicates:
                names.append(obj)
            elif pred in self.schema.geometry_link_predicates:
                # Multiple geometry links are rare; first-wins matches
                # the transformer's last-write-wins on the entity
                # build pipeline closely enough.
                if geometry_node is None:
                    geometry_node = obj

        # 2-hop: fetch the WKT literal from the geometry node.
        geometry_wkt: str | None = None
        if geometry_node is not None:
            wkt_triples = self.bridge.fetch_triples_for_subject(geometry_node)
            for t in wkt_triples:
                if t["predicate"] in self.schema.geometry_data_predicates:
                    raw_wkt = _strip_typed_literal(t["object"])
                    # Strip any remaining outer quotes the kwg path
                    # leaves on quote-wrapped WKT objects.
                    if (
                        len(raw_wkt) >= 2
                        and raw_wkt.startswith('"')
                        and raw_wkt.endswith('"')
                    ):
                        raw_wkt = raw_wkt[1:-1]
                    geometry_wkt = self._simplify_or_passthrough(raw_wkt)
                    break

        # Also check the 1-hop case where the entity itself carries
        # a geo:asWKT triple (some KG layouts skip the geometry node).
        # If the 2-hop path found a WKT, we already set geometry_wkt;
        # 1-hop only fires when the 2-hop didn't.
        if geometry_wkt is None:
            for t in triples:
                if t["predicate"] in self.schema.geometry_data_predicates:
                    raw_wkt = _strip_typed_literal(t["object"])
                    if (
                        len(raw_wkt) >= 2
                        and raw_wkt.startswith('"')
                        and raw_wkt.endswith('"')
                    ):
                        raw_wkt = raw_wkt[1:-1]
                    geometry_wkt = self._simplify_or_passthrough(raw_wkt)
                    break

        return {
            "id": entity_id,
            "subject": subject,
            "name": names[0] if names else None,
            "names": names,
            "types": types,
            "category": types[0] if types else None,
            "geometry_wkt": geometry_wkt,
            "geometry_node": geometry_node,
            "pagerank": None,
            "h3_index": None,
            "raw_triples": triples,
        }

    def find_entity_by_name(
        self,
        name_substring: str,
        *,
        limit: int = _FIND_ENTITY_TOP_K,
    ) -> list[dict[str, Any]]:
        """Top-K substring search over the KG's naming predicates.

        Returns lightweight dicts (no geometry) — caller follows up
        with :meth:`get_entity` for full hydration.

        Three knobs differ from a raw bridge call:

        * **Over-fetch + rank.** The bridge returns hits in scan
          order, which for numeric substrings (zip codes, OSM IDs)
          is essentially random — and dominated by noise (e.g.,
          "24230" matches dozens of ``earthquakeObservation.nc...``
          URIs containing those digits). We over-fetch
          (``_FIND_ENTITY_OVERFETCH=25``), then rank by
          ``(predicate_priority, len(label), subject)`` and truncate
          to ``limit``.
        * **Predicate priority** uses the schema's ordering of
          ``name_predicates`` — earlier predicates outrank later
          ones. For kwg this means matches via the primary label
          predicate beat matches via the alt-label predicate, which
          is the standard RAG signal for canonicality.
        * **Label-length ascending** is the highest-impact
          tie-break. A 5-char "24230" label is almost certainly more
          canonical than a 100-char observation description that
          happens to contain those digits as a substring.
        * **Subject prefix stripping**. The displayed ``subject``
          drops ``schema.subject_uri_prefix`` (when set) so a 80-char
          URI becomes a ~20-char local name. The agent can still
          round-trip the short form through :meth:`get_entity` —
          :meth:`KGSchema.canonicalize_subject_id` accepts either
          form. For osmh3kg (no prefix) the subject is unchanged.

        These three are RAG defaults, not bench secrets — the paper
        should document ``K=5``, the ranker tuple, and the
        prefix-stripping convention. The point of capping is
        bounding the agent's context budget across the session.
        """
        hits = self.bridge.search_object_substring(
            self.schema.name_predicates,
            name_substring,
            limit=_FIND_ENTITY_OVERFETCH,
        )
        # Predicate priority: earlier in schema.name_predicates = higher
        # priority. Build a dict for O(1) lookup; unknown predicates
        # (shouldn't happen given the bridge filter, but defensive)
        # sort last.
        pred_rank = {p: i for i, p in enumerate(self.schema.name_predicates)}
        unknown_rank = len(self.schema.name_predicates)
        prefix = self.schema.subject_uri_prefix

        def _rank_key(h: dict[str, Any]) -> tuple[int, int, str]:
            label = h.get("object") or ""
            return (
                pred_rank.get(h["predicate"], unknown_rank),
                len(label),
                h["subject"],
            )

        ranked = sorted(hits, key=_rank_key)[:limit]

        def _display_subject(subject: str) -> str:
            if prefix and subject.startswith(prefix):
                return subject[len(prefix):]
            return subject

        return [
            {
                "subject": _display_subject(h["subject"]),
                "name": h["object"],
                "predicate": h["predicate"],
            }
            for h in ranked
        ]

    def find_entities_by_type(
        self,
        type_substring: str,
        region: str | None = None,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find entities whose rdf:type object contains ``type_substring``.

        ``region`` is accepted for API parity with the old bridge but
        currently unused — region filtering at the triple level needs
        a second join against a name/within predicate which depends
        on per-KG conventions not encoded in :class:`KGSchema` yet.
        Future work; for now the parameter is a no-op with a warning.
        """
        if region is not None:
            logger.warning(
                "find_entities_by_type: region filter not implemented "
                "for triple-native bridge; ignoring region={!r}",
                region,
            )
        hits = self.bridge.search_object_substring(
            self.schema.type_predicates,
            type_substring,
            limit=limit,
        )
        # Each hit's subject is the entity; the object is the type
        # value matched. Look up the entity's primary name via a
        # follow-up subject fetch so the agent sees something more
        # informative than just IDs. For 50 entities this is 50 small
        # Spark queries — adequate for non-headline tool calls.
        out: list[dict[str, Any]] = []
        for h in hits:
            subj = h["subject"]
            subj_triples = self.bridge.fetch_triples_for_subject(subj)
            entity_name: str | None = None
            for t in subj_triples:
                if t["predicate"] in self.schema.name_predicates:
                    entity_name = t["object"]
                    break
            out.append(
                {
                    "subject": subj,
                    "name": entity_name,
                    "matched_type": h["object"],
                }
            )
        return out

    def list_types(self, *, limit: int = 50) -> dict[str, Any]:
        """Distinct rdf:type object values across the KG.

        Returns a paged view: at most ``limit`` types plus a
        ``has_more`` flag indicating whether additional types exist
        beyond the page. The default ``limit=50`` is sized to keep the
        tool output well under the per-step context budget on a 32K
        model: 50 URIs at ~60 chars/each ≈ 3 KB, vs. the previous
        ~12 KB at limit=200. Agents that need the full vocabulary can
        call again with a larger ``limit``; agents typically only need
        the head to discover canonical type names.
        """
        # Over-fetch by 1 so we can detect "more available" without a
        # second COUNT query — keeps cost at one Spark scan.
        types = self.bridge.query_distinct_objects(
            self.schema.type_predicates,
            limit=limit + 1,
        )
        has_more = len(types) > limit
        if has_more:
            types = types[:limit]
        return {"types": types, "has_more": has_more, "shown": len(types)}

    # ------------------------------------------------------------------
    # Lifecycle / passthroughs
    # ------------------------------------------------------------------

    def get_delta_versions(self) -> dict[str, int]:
        """Proxy to the underlying bridge for snapshot-id versioning.

        Tier-3 :func:`benchmark_models.snapshot.upgrade_to_versioned`
        consumes this via the ``SupportsDeltaVersions`` protocol.
        """
        return self.bridge.get_delta_versions()

    def close(self) -> None:
        """Stop the underlying Spark session."""
        self.bridge.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _simplify_or_passthrough(self, wkt: str) -> str:
        """Apply WKT simplification when tolerance is set and the
        input is non-trivial; otherwise pass through."""
        if not self.wkt_simplify_tolerance:
            return wkt
        simplified, meta = simplify_wkt(wkt, self.wkt_simplify_tolerance)
        if "error" in meta:
            # Don't fail the tool call on a parse error — return the
            # original WKT and let the agent's Python sandbox decide.
            logger.warning(
                "get_entity: WKT simplify failed ({}); using original",
                meta["error"],
            )
            return wkt
        return simplified


__all__ = ["EntityService"]
