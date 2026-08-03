"""Tier 1 — Plain LLM (parametric-knowledge floor).

Single instruction + question, no CoT trigger, no examples, no tools.
Renders the tier-1 master prompt with the question and the per-template
answer-type description, calls the backend, parses the typed JSON response.
"""
