"""Tier 2 — Reasoning-enhanced LLM.

Two styles:
(a) Native thinking mode (when the variant's backend supports it: GPT-5.4
    Thinking, Sonnet 4.6 extended thinking, Gemini 3 Flash thinking_level=high,
    Qwen3.5-35B-A3B with ``thinking: true``, Gemma 4 31B with ``thinking: true``).
(b) Prompted CoT for variants without native thinking (Qwen3.5-27B dense):
    the schema is {"reasoning": <string>, "answer": ...} in a single response.

Self-consistency (N=3, decided): when the variant config has
self_consistency_n > 1, sample N reasoning chains at temperature > 0 and
take the majority answer. Records sc_samples in the Prediction record.
"""
