"""Tier runtime loops for benchmark_models.

Each tier is a small mixin or callable that the registry composes with a
backend. Tier 1 / 2 are essentially "call the backend with the right
prompt and parse the response." Tier 3 is the agent loop with KG tools.
"""
