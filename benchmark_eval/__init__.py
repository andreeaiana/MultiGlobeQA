"""Eval pipeline for SpatialBenchRAG — primary metrics.

Computes Exact Match (EM) and Normalized Error (NE) per prediction,
dispatching on the question's answer type, and aggregates the
per-question scores into micro / macro rollups with per-group
breakdowns.

Public surface:

* :mod:`benchmark_eval.base` — :class:`GoldRow`, :class:`ScoreRecord`,
  :class:`AggregateReport`, and supporting types.
* :mod:`benchmark_eval.canonicalize` — string / unit / numeric / date
  normalization helpers.
* :mod:`benchmark_eval.aliases` — :class:`AliasStore` backing the
  entity / predefined-value / cardinal-direction :func:`resolve`
  function.
* :mod:`benchmark_eval.distances` — per-type ``d(â, a*)`` and
  ``τ(a*)`` functions.
* :mod:`benchmark_eval.scorer` — :class:`Scorer` (top-level entry).
* :mod:`benchmark_eval.aggregate` — build an :class:`AggregateReport`
  from a list of :class:`ScoreRecord`.
* :mod:`benchmark_eval.loaders` — load gold rows from a benchmark
  JSONL and predictions from a model-section JSONL.
* :mod:`benchmark_eval.cli` — ``python -m benchmark_eval score ...``.
"""
