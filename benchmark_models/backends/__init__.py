"""Backend wrappers for benchmark_models.

Each backend is a generic class that handles all variants from a given
provider via feature flags (thinking, thinking_level, extended_thinking,
self_consistency_n, etc.) read from the variant's YAML config.
"""
