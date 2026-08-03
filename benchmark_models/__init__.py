"""Benchmark Models — systems-under-evaluation for MultiGlobeQA.

Implements the model section: a clean, model-agnostic abstraction for running
zero-shot QA across LLMs and (later) RAG systems on the MultiGlobeQA
benchmark, organized into capability tiers (plain LLM, reasoning,
tool-augmented). See README.md for the user-facing overview.

Subpackages:
- backends/  - per-provider client wrappers (vllm, openai, google)
- tiers/     - per-tier runtime loops (plain, reasoning, tool_agent)
- kg_tools/  - tier-3 KG-callable tools and the executor bridge
- prompts/   - master prompt templates per tier
- schemas/   - canonical answer-type registry
"""
