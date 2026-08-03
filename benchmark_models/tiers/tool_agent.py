"""Tier 3 — Tool-augmented agentic LLM.

Tier 3 is implemented via Smolagents CodeAgent in
``benchmark_models.tier3.SmolagentsModel``. This module is kept as a
re-export for backwards compatibility with any code that imports from
``benchmark_models.tiers.tool_agent``.

See ``benchmark_models/tier3.py`` for the full implementation.
"""

from benchmark_models.tier3 import SmolagentsModel, SubTier

__all__ = ["SmolagentsModel", "SubTier"]
