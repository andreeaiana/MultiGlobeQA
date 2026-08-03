"""KG-callable tools for tier 3 of benchmark_models.

Custom Smolagents ``Tool`` subclasses that wrap the KG access layer
(backed by the existing question_generator.executors infrastructure
and Spark/Sedona). These tools are passed to Smolagents' ``CodeAgent``
for tier 3a (KG + Python) and tier 3c (KG + Web + Python).

Tools:
- ``FindEntityTool`` — entity linking by name
- ``FindEntitiesByTypeTool`` — category-based entity search
- ``GetEntityTool`` — fetch full properties + WKT geometry
- ``ListCategoriesTool`` — discover available entity categories

Each tool has a mock implementation for testing without Spark / KG
access, and a real implementation backed by an ``EntityService``
(which itself wraps a ``TripleBridge`` over the triples Delta table,
requiring a Spark session and a KG snapshot).
"""
