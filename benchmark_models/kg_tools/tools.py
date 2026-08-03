"""KG access tools as Smolagents Tool subclasses.

Each tool follows the Smolagents ``Tool`` pattern: a class with
``name``, ``description``, ``inputs``, ``output_type``, and a
``forward()`` method.

Two implementations per tool:

- **Mock** (default, ``backend=None``): returns canned data shaped
  like the real ``EntityService`` returns. Used for testing and
  development without Spark / KG access.
- **Real** (``backend=EntityService(TripleBridge(...), schema)``):
  backed by the triple-native KG bridge over the actual KG Delta
  snapshot.

The mock implementations match the real-bridge return shape so the
agent loop sees consistent dicts regardless of whether tests run
against the mock or a SLURM job runs against the production data.

Output contract
---------------
Every ``forward()`` returns a **JSON-encoded string**. The agent is
expected to ``json.loads(result)`` before iterating. Returning
``str(results)`` (Python repr) silently violates the documented "list
of dicts" return contract — Python repr uses single quotes and
``None`` instead of JSON ``null``, so ``json.loads`` fails and the
naive ``for e in results: e['subject']`` iterates the string
character-by-character (the recurring Gemini failure mode on tier-3a).
``json.dumps`` produces a contract the model can rely on.
"""

from __future__ import annotations

import json
from typing import Any

from smolagents import Tool

# Cached separators tuple: no spaces, so JSON output stays compact in
# the agent's context window (saves ~5-10% per tool call vs default).
_JSON_SEP = (",", ":")


def _to_json(payload: Any) -> str:
    """Serialize a tool return value to JSON. ``default=str`` falls
    back to ``str()`` for any object pyarrow / shapely / numpy types
    that aren't JSON-native; this keeps the tool wrapper tolerant of
    backend-side quirks without each tool needing custom encoders."""
    return json.dumps(payload, default=str, separators=_JSON_SEP)


class FindEntityTool(Tool):
    """Look up entities in the knowledge graph by name."""

    name = "find_entity"
    description = (
        "Search for entities in the spatial knowledge graph by name. "
        "Returns a JSON string encoding a list of up to 5 matching "
        "entities — parse with ``json.loads(...)`` before iterating. "
        "Each entry has a ``subject`` (the KG identifier you pass to "
        "get_entity, in short form e.g. 'zipCodeArea.27351'), a "
        "``name``, and the ``predicate`` that matched. Results are "
        "ranked by (canonical-predicate first, then shortest label, "
        "then lexicographic), so the most likely match is at index 0. "
        "If the result list looks like noise (e.g., a numeric query "
        "matched lots of substring hits), make your query more "
        "specific rather than scrolling — the cap is intentional."
    )
    inputs = {
        "name": {
            "type": "string",
            "description": "The entity name to search for (e.g., 'Cafe Kanel', 'Brandenburg Gate').",
        },
    }
    output_type = "string"

    def __init__(self, backend: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend

    def forward(self, name: str) -> str:
        if self._backend is not None:
            results = self._backend.find_entity_by_name(name)
            return _to_json(results)
        return _to_json([
            {
                "subject": "mock_001",
                "name": name,
                "predicate": "http://www.w3.org/2000/01/rdf-schema#label",
            },
        ])


class FindEntitiesByTypeTool(Tool):
    """Search for entities by category/type in the knowledge graph."""

    name = "find_entities_by_type"
    description = (
        "Search for entities in the spatial knowledge graph whose "
        "rdf:type matches a given category substring (e.g., 'hospital', "
        "'park', 'cafe'). Optionally filter by region. Returns a JSON "
        "string encoding a list of matches — parse with "
        "``json.loads(...)`` before iterating. Each entry has "
        "``subject`` (the KG identifier), ``name`` (primary label), and "
        "``matched_type`` (the rdf:type value that matched). Use this "
        "for questions like 'find all hospitals near X' or 'how many "
        "cafes are in Y'."
    )
    inputs = {
        "category": {
            "type": "string",
            "description": "The entity category to search for (e.g., 'hospital', 'park', 'cafe').",
        },
        "region": {
            "type": "string",
            "description": "Optional region name or ID to filter results within. Pass null for no filter.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, backend: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend

    def forward(self, category: str, region: str | None = None) -> str:
        if self._backend is not None:
            results = self._backend.find_entities_by_type(category, region=region)
            return _to_json(results)
        return _to_json([
            {"subject": "mock_type_001", "name": f"Mock {category} 1", "matched_type": category},
            {"subject": "mock_type_002", "name": f"Mock {category} 2", "matched_type": category},
        ])


class GetEntityTool(Tool):
    """Fetch full properties and geometry for a single entity."""

    name = "get_entity"
    description = (
        "Retrieve the entity record for a single KG identifier. "
        "Returns a JSON string encoding a dict — parse with "
        "``json.loads(...)`` before accessing fields. The dict has: "
        "``subject`` (canonical KG URI), ``name`` (primary label), "
        "``names`` (all labels), ``types`` (list of rdf:type values), "
        "``category`` (first type), ``geometry_wkt`` (WKT string, "
        "automatically simplified to fit context — use as-is for "
        "spatial computation), ``geometry_node`` (the linked geometry "
        "node URI), and ``raw_triples`` (every triple about the entity, "
        "for cases where the agent needs predicates beyond the common "
        "ones). When no triples are found, the dict carries an "
        "``error`` key instead of these fields. Use this after "
        "find_entity to fetch full data for a specific entity, or "
        "directly when the benchmark question already names a KG "
        "identifier."
    )
    inputs = {
        "entity_id": {
            "type": "string",
            "description": "The entity identifier (KG subject URI or benchmark id).",
        },
    }
    output_type = "string"

    def __init__(self, backend: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend

    def forward(self, entity_id: str) -> str:
        if self._backend is not None:
            result = self._backend.get_entity(entity_id)
            return _to_json(result)
        # Mock: match the real EntityService return shape.
        return _to_json({
            "id": entity_id,
            "subject": entity_id,
            "name": f"Entity {entity_id}",
            "names": [f"Entity {entity_id}"],
            "types": ["http://example.org/poi"],
            "category": "http://example.org/poi",
            "geometry_wkt": "POINT (13.3248 52.5006)",
            "geometry_node": None,
            "pagerank": None,
            "h3_index": None,
            "raw_triples": [],
        })


class ListCategoriesTool(Tool):
    """Discover available entity categories (rdf:type values) in the KG."""

    name = "list_categories"
    description = (
        "List rdf:type values present in the knowledge graph. Returns "
        "a JSON string encoding a dict — parse with ``json.loads(...)`` "
        "before accessing fields. The dict has: ``types`` (a list of "
        "up to 50 type URIs/labels sorted lexicographically), ``shown`` "
        "(how many were returned), and ``has_more`` (True if more "
        "types exist beyond the returned head). Pass any of the "
        "returned type values to find_entities_by_type. Useful when "
        "the question asks about a category whose canonical name in "
        "this KG is unclear. The result is intentionally capped to "
        "keep agent context bounded; the head of the alphabetical "
        "list covers most common categories."
    )
    inputs: dict[str, Any] = {}
    output_type = "string"

    def __init__(self, backend: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend

    def forward(self) -> str:
        if self._backend is not None:
            results = self._backend.list_types()
            return _to_json(results)
        return _to_json(
            {
                "types": [
                    "cafe", "restaurant", "hospital", "school", "park",
                    "church", "museum", "hotel", "pharmacy", "supermarket",
                ],
                "shown": 10,
                "has_more": False,
            }
        )
