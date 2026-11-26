"""Tier 2 — Reasoning-enhanced LLM.

Two styles:
(a) Native reasoning: Gemini 3 Flash with ``thinking_level: high``, and the
    vLLM variants with ``schema_reasoning: false`` (Qwen 3.8, Gemma 4, gpt-oss,
    GLM-4.7-Flash).
(b) Prompted CoT for the vLLM variants with ``schema_reasoning: true``
    (Qwen 3.5, Gemma 3): the schema is {"reasoning": <string>, "answer": ...}
    in a single response.

Self-consistency (N=3, decided): when the variant config has
self_consistency_n > 1, sample N reasoning chains at temperature > 0 and
take the majority answer. Records sc_samples in the Prediction record.
"""
