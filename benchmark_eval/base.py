"""Core pydantic models for the eval pipeline.

Defines:

* :class:`GoldRow` — a benchmark row reduced to the fields the eval
  pipeline needs. Loaded from the benchmark JSONL directly (the
  :class:`benchmark_models.base.Example` model strips gold fields by
  design, so the eval side reads the JSONL independently).
* :class:`ScoreRecord` — one scored prediction. Written as JSONL
  alongside the predictions file; one line per prediction.
* :class:`GroupStats` — per-group aggregate (a bucket of
  :class:`ScoreRecord`).
* :class:`AggregateReport` — top-level rollup with micro / macro EM
  and NE, plus per-answer-type / per-SFID / per-language breakdowns
  and diagnostic counters.

:class:`ScoreRecord` is deliberately a flat schema (nullable
per-type drill-down fields) so the JSONL file reads cleanly into a
pandas DataFrame for downstream analysis. An alternative dict-based
``drill_down`` field was rejected because it forces unnesting before
analysis.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Canonical answer-type names from ``benchmark_models.schemas.TYPE_REGISTRY``.
# Not imported as a ``Literal`` here because adding a new type shouldn't
# require an eval-side dependency bump — the scorer dispatches by string.


HARNESS_ERROR_CLASSES: frozenset[str] = frozenset(
    {"infra", "context_overflow", "code_gen"}
)
"""Error classes that count as *harness* failures — the model or its
execution substrate failed, not the model's capability on the task.

``infra`` (exhausted transport retries), ``context_overflow`` (the
provider rejected the request as too large / over the context limit, so
the model never produced an answer) and ``code_gen`` (tier-3 CodeAgent
produced malformed Python) are orthogonal to the capability being
measured. ``config`` (unknown tid, oracle coverage gap, backend
mis-route) and ``schema_violation`` (the model returned malformed
structured output) are left in the headline denominator because they
represent real model behaviour or upstream data issues the paper
should own.

``em_adjusted`` in :class:`AggregateReport` excludes this set from
the denominator so readers can compare a capability-only number to
the raw headline. Neither number is "the truth" — both are reported
alongside ``error_class_breakdown`` so the reader can pick."""


# ---------------------------------------------------------------------------
# Gold row
# ---------------------------------------------------------------------------


class GoldRow(BaseModel):
    """The benchmark-side data needed to score one prediction.

    The eval loader reads a benchmark JSONL and produces one
    :class:`GoldRow` per row, joined against the prediction by
    ``question_id``. Everything here is read from the benchmark row
    *except* ``answer_type`` and ``answer_unit`` which are resolved
    from the :class:`TemplateRegistry` via ``tid``.
    """

    model_config = {"extra": "allow"}

    question_id: str
    tid: str
    sfid: str
    answer_type: str
    """Canonical answer type from the type registry (e.g.,
    ``"Numeric (float)"``). The scorer dispatches on this string."""
    answer_unit: str | None = None
    """Expected unit string from the template registry (e.g.,
    ``"kilometres"``). ``None`` for non-numeric types or when the
    template doesn't pin a unit."""
    ground_truth: Any
    """The gold answer. Type depends on ``answer_type``:
    ``str`` for entity names and cardinal directions, ``float`` / ``int``
    for numeric, ``bool`` for boolean, ``dict`` / ``list`` for
    coordinates / geometry / sets, etc. Canonicalization happens in
    the scorer."""
    strata_cell: str | None = None
    language: str = "eng"
    """ISO 639-3 code. Default ``"eng"`` matches the multilingual
    benchmark layout (rows are emitted with three-letter codes)."""
    kg_source: str | None = None
    """KG bucket the question belongs to (``"kwg"`` / ``"osmh3kg"`` /
    ``"worldkg"`` / ``"multimodal"``). Denormalized from the benchmark
    row so the aggregator can produce per-KG breakdowns without
    rejoining against the benchmark JSONL. ``None`` only on synthetic
    fixtures that omit the field."""
    is_true_premise: bool | None = None
    """Parsed false-premise flag from the benchmark. ``False`` marks
    rows where the question's premise does not hold and the expected
    answer is abstention; ``True`` marks rows explicitly validated as
    real-premise; ``None`` when the benchmark row omits the field.
    Used by the scorer to flip abstention from EM=0 (old default, which
    penalised correct abstentions) to EM=1 when ``ground_truth is
    None``."""


# ---------------------------------------------------------------------------
# ScoreRecord — one scored prediction
# ---------------------------------------------------------------------------


NumericFailureCategory = Literal[
    "number_wrong_unit_correct",
    "number_correct_unit_wrong",
    "both_wrong",
    # Parse failure: the prediction couldn't be coerced to a number
    # at all (e.g. the model emitted a sentence). Surfaced on
    # ``format_error=True`` rows so the failure-breakdown distinguishes
    # parse-fail from the three numeric-mismatch cases.
    "parse_error",
]


class ScoreRecord(BaseModel):
    """The eval-side mirror of one :class:`benchmark_models.base.Prediction`.

    Everything a downstream analysis needs for grouping and for
    hyperparameter-sensitivity re-scoring is denormalized here so
    the JSONL file is self-contained. ``raw_distance`` and
    ``raw_tolerance`` preserve the underlying ``d`` and ``τ`` so a
    caller can recompute EM / NE under different tolerance settings
    without re-running the models.
    """

    model_config = {"extra": "forbid"}

    # Identity (denormalized from Prediction + GoldRow).
    question_id: str
    tid: str
    sfid: str
    answer_type: str
    model_id: str
    tier: str
    language: str
    strata_cell: str | None = None
    kg_source: str | None = None
    """KG bucket (denormalized from :attr:`GoldRow.kg_source`). Lets
    :class:`AggregateReport.per_kg` roll up without rejoining the
    benchmark JSONL. ``None`` only on synthetic fixtures."""

    # Per-prediction cost / latency (denormalized from Prediction.usage
    # and Prediction.latency_ms). Lets the aggregator surface
    # ``mean_latency_ms`` / ``total_input_tokens`` / ``total_output_tokens``
    # without joining against the predictions JSONL. Useful for paper
    # tables that report latency / cost alongside accuracy. Defaults
    # mirror :class:`benchmark_models.base.Usage` so older score-record
    # files deserialize cleanly.
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    """I: prompt-caching hits denormalised from
    :attr:`benchmark_models.base.Usage.cached_tokens`. ``0`` when the
    backend doesn't expose cache info (vLLM) or no cache was hit
    on this prediction. Rolls up to
    :attr:`AggregateReport.total_cached_tokens`."""
    cache_creation_tokens: int = 0
    """I: Anthropic-specific cache-seed premium. ``0`` for other
    backends. Rolls up to
    :attr:`AggregateReport.total_cache_creation_tokens`."""
    latency_ms: float = 0.0

    # Harness-error provenance (denormalized from Prediction). Needed
    # at the record level so :class:`AggregateReport` can compute
    # per-class error counts and the adjusted-EM denominator
    # (capability-only: harness failures excluded) without rejoining
    # against ``predictions.jsonl``.
    final_state: str | None = None
    """Mirror of ``Prediction.final_state``: ``answered``, ``abstained``,
    ``cap_hit``, ``error``, or ``no_final_answer``. ``None`` for tier 1 /
    tier 2 predictions that don't carry a terminal state."""
    error_class: str | None = None
    """Mirror of ``Prediction.error_class``: ``infra``, ``code_gen``,
    ``schema_violation``, ``config``, or ``unknown``. Populated only
    for predictions with ``final_state == "error"``."""

    # Primary scores.
    em: int
    """Exact Match in {0, 1}. For discrete types, strict equality.
    For continuous types, ``1[d(â, a*) < τ(a*)]`` (hit at tolerance).
    For dates, components-match. For polygons,
    ``1[IoU ≥ iou_threshold]``."""
    ne: float
    """Normalized Error in [0, 1]. ``min(1, d(â, a*) / τ(a*))`` for
    most types; ``1 − Jaccard`` for set enumeration; ``1 − IoU`` for
    polygons; day-distance normalized for dates."""
    covered: bool
    """``True`` if the prediction is not a refusal (``answer`` is
    not ``None``) and not a format error. Refusals get ``em=0`` and
    ``ne=1`` and ``covered=False``."""

    # Audit trail — raw scalars behind the metrics.
    raw_distance: float | None = None
    """``d(â, a*)`` in the units of the distance function for this
    answer type. ``None`` for dates (components-match is not a real
    distance) and format errors."""
    raw_tolerance: float | None = None
    """``τ(a*)`` used. ``None`` when a tolerance-based rule doesn't
    apply."""

    # Failure diagnostics.
    format_error: bool = False
    """``True`` when the prediction couldn't be parsed into the
    expected type. Implies ``em=0``, ``ne=1``, ``covered=False``.
    Distinct from :attr:`gold_missing` (v2): a format error blames the
    prediction; gold-missing blames the upstream gold."""
    gold_missing: bool = False
    """v2: ``True`` when the gold row carries no usable answer
    (refusal on a row with ``ground_truth is None`` AND
    ``is_true_premise`` is not False — i.e., neither a real-premise
    row with a gold answer nor an explicit false-premise marker).
    Split from ``format_error`` so the aggregator can separate
    upstream gold-defect counts from scorer-side parse failures.
    Implies ``em=0``, ``ne=1``, ``covered=False``."""
    resolve_failure: bool | None = None
    """For entity / cardinal / predefined answer types: ``True`` if
    :func:`benchmark_eval.aliases.resolve` returned ``None`` for the
    prediction. ``None`` when the type isn't alias-backed."""
    refusal: bool = False
    """v2: ``True`` when the prediction was a refusal
    (``answer is None`` or ``Prediction.refusal``). Counted as
    not-covered in v2 ``coverage_rate``. Distinct from ``format_error``
    and ``gold_missing``: a refusal is a deliberate model output, a
    format error is unparseable, gold-missing is a data defect."""

    # Numeric drill-down.
    numeric_failure_category: NumericFailureCategory | None = None
    """For numeric types where ``em=0``, the root cause — populated
    only when the failure could be categorized. ``None`` on
    non-numeric types and on numeric-but-correct rows."""
    unit_match: bool | None = None
    """``True`` if the predicted unit (after canonicalization)
    equals the requested gold unit. ``None`` for non-numeric types."""

    # Unit-normalized companions (numeric types only). ``None`` for
    # non-numeric rows; the aggregator filters those out when
    # averaging. When the prediction and gold units are in the same
    # dimensional group (distance or area), the prediction is
    # converted into the gold unit before recomputing EM/NE. When
    # conversion is impossible (unitless, unknown, or cross-group),
    # the normalized companions fall back to the strict EM/NE values
    # — always ≥ strict EM and ≤ strict NE, never stricter.
    em_unit_normalized: int | None = None
    ne_unit_normalized: float | None = None

    # Spatial-proximity companions (grid-cell types only: MGRS-like
    # grid cell, Composite string (H3+S2)). ``None`` for every other
    # answer type — the aggregator filters those out when averaging.
    # ``em`` stays the strict exact-ID match (backward-compatible; it
    # feeds the universal ``em_micro``), while these companions
    # recompute EM/NE after decoding both sides to a geographic
    # location and scoring by geodesic proximity. This credits a
    # prediction that identifies the right place under a different but
    # valid grid-encoding convention (e.g. standard-convention UTM
    # eastings vs the benchmark's offset MGRS encoding). Applied
    # identically to every model; see
    # :meth:`Scorer._score_mgrs_like_grid_cell` and
    # :meth:`Scorer._score_composite_string`. ``S2 cell index`` has no
    # spatial companion by design (no S2 decoder dependency; strict-ID
    # is the honest measure there).
    em_spatial: int | None = None
    ne_spatial: float | None = None

    # Set-enumeration drill-down.
    set_precision: float | None = None
    set_recall: float | None = None
    set_f1: float | None = None

    # False-premise stratification. Denormalised from
    # :attr:`GoldRow.is_true_premise` so the aggregator can split EM
    # by premise validity without rejoining against the benchmark.
    # ``None`` on rows where the benchmark omitted the field.
    is_true_premise: bool | None = None

    # Normalized gold label for the majority-class baseline. Coarse,
    # stable key for the row's gold value (booleans → "true"/"false",
    # sets → sorted JSON, scalars → casefolded string). ``None`` for
    # missing / false-premise gold. Lets the aggregator compute the EM
    # of a label-free per-TID majority guesser (the no-reasoning floor)
    # without rejoining the benchmark. NOT the scoring canonicalization.
    gold_label: str | None = None

    # Tier-3 operational counters. Denormalised from
    # :attr:`benchmark_models.base.Prediction.tool_calls` so the
    # aggregator can compute mean tool-call counts and tool-error
    # rates without re-reading the predictions file. ``None`` on
    # tier-1 / tier-2 records.
    tool_call_count: int | None = None
    tool_error_count: int | None = None

    # Self-consistency agreement (tier-2 SC=N only). Fraction of
    # ``Prediction.sc_samples`` that match the modal value, computed
    # with stable JSON-serialisation as the equality key. ``None`` on
    # records where ``sc_samples`` is absent or has fewer than 2
    # entries (the metric is undefined for N<2).
    sc_agreement: float | None = None

    # Polygon-IoU planar fallback flag. ``None`` for non-polygon
    # rows (the polygon branch in ``_score_geometry`` was not taken).
    # ``True`` for polygon rows where EPSG:6933 reprojection failed
    # and the IoU was computed on raw lon/lat coordinates as
    # Cartesian X/Y (biased at high latitudes). ``False`` for
    # polygon rows that reprojected cleanly. Surfaces in
    # ``AggregateReport`` as ``n_polygon_planar_fallback`` +
    # ``polygon_planar_fallback_rate`` so the paper can report what
    # fraction of polygon rows used the degraded metric.
    polygon_iou_used_planar_fallback: bool | None = None


# ---------------------------------------------------------------------------
# AggregateReport — rolled up over a list of ScoreRecords
# ---------------------------------------------------------------------------


class GroupStats(BaseModel):
    """Aggregate over one group of :class:`ScoreRecord` (e.g., one
    answer type, one SFID, one language).

    Kept small so the top-level :class:`AggregateReport` JSON stays
    human-readable. Detailed per-question data lives in the
    `score_records.jsonl` alongside.
    """

    n: int
    n_covered: int
    em: float | None
    """Mean EM over the group. Equivalent to "accuracy at tolerance τ"
    for the group's answer type. ``None`` for empty groups (B5 fix);
    in the normal aggregation path the group is non-empty by
    construction, so ``None`` only appears for defensive direct
    callers of :func:`_group_stats`.

    **Caveat (mirrors the global ``em_micro`` / ``em_conditional``):**
    this counts a correct refusal on a false-premise row as EM=1, so on
    any group that mixes true- and false-premise rows it blends
    answer accuracy with correct-rejection rate. For a clean
    "accuracy on attempted real questions" number per group, read
    :attr:`em_true_premise`."""
    ne: float | None
    """Mean NE over the group. Lower is better. ``None`` for empty
    groups."""
    em_conditional: float | None
    """Mean EM conditional on ``covered``. ``None`` when ``n_covered=0``
    or the group is empty. Same false-premise caveat as :attr:`em`."""
    ne_conditional: float | None = None
    """Mean NE conditional on ``covered`` — the "how close was it when it
    answered?" view for the group. Because NE is hard-set to 1.0 on every
    non-answer (refusal / format error / harness error / uncovered), the
    unconditional :attr:`ne` blends real error magnitude with fiat
    max-error padding; restricting to ``covered`` rows removes the padding
    and leaves the informative signal. ``None`` when ``n_covered=0`` or the
    group is empty. Default ``None`` keeps old score-record / report files
    deserializing cleanly."""

    # False-premise stratification, per group. Added so every per-group
    # rollup (per_answer_type / per_sfid / per_kg / per_strata_cell /
    # per_income_tier) exposes a true-premise-only EM, mirroring the
    # global ``em_true_premise`` / ``negative_rejection_rate``. Without
    # these, a group's headline ``em`` silently credited correct
    # false-premise refusals as capability. Defaults keep old
    # score-record / report files deserializing cleanly.
    n_true_premise: int = 0
    n_false_premise: int = 0
    em_true_premise: float | None = None
    """Mean EM over the group's real-premise rows only
    (``is_true_premise is True``). The "did the model answer the
    attempted questions correctly?" number, uncontaminated by
    correct-refusal credit. ``None`` when the group has no such rows."""
    ne_true_premise: float | None = None
    """Mean NE over the group's real-premise rows only
    (``is_true_premise is True``). The NE analogue of
    :attr:`em_true_premise`. ``None`` when the group has no such rows."""
    negative_rejection_rate: float | None = None
    """Mean EM over the group's false-premise rows only
    (``is_true_premise is False``) — i.e. the correct-abstention rate on
    bad questions in this group. ``None`` when the group has none."""
    false_refusal_rate: float | None = None
    """Over-abstention / false-refusal rate for this group: the fraction
    of the group's answerable (true-premise) rows on which the model
    refused. The confound for :attr:`negative_rejection_rate` — a model
    that abstains indiscriminately shows high values here. ``None`` when
    the group has no true-premise rows."""
    majority_em: float | None = None
    """EM of a label-free per-TID majority-class baseline restricted to
    this group: for each TID in the group, always emit that TID's modal
    gold; the fraction correct. The no-reasoning floor to print beside
    :attr:`em`. Model-independent (pure gold property). ``None`` when no
    row in the group carries a ``gold_label``."""


class AggregateReport(BaseModel):
    """Top-level rollup for one variant × run.

    Written to ``aggregate_report.json`` alongside
    ``score_records.jsonl``.
    """

    model_config = {"extra": "forbid"}

    # Provenance.
    model_id: str
    tier: str
    n_total: int
    """Total predictions scored (matches ``predictions.jsonl`` row
    count minus any rows the model section already flagged with an
    unknown-tid or other setup-time error)."""

    # Coverage / formatting.
    #
    # v2 redefinition: ``coverage_rate`` is the fraction of rows whose
    # prediction was parseable AND non-refusal AND not gold-missing.
    # Refusals, format errors, and gold-missing rows are counted
    # separately so the four categories ``answer_rate`` /
    # ``refusal_rate`` / ``format_error_rate`` / ``gold_missing_rate``
    # sum to 1.0 — EXCEPT when substrate-error rows
    # (``final_state=="error"``) are present: those form a fifth bucket
    # (covered=False / refusal=False / format_error=False), so the four
    # rates then sum to ``1 - n_harness_errors/n_total``.
    n_covered: int
    coverage_rate: float
    """Fraction of rows where the prediction was parseable AND
    non-refusal AND not gold-missing. **Important caveat (C3)**:
    on a *false-premise* row (``is_true_premise=False``) a refusal
    is the correct answer and counts as ``covered=True``. So
    ``coverage_rate`` does NOT mean "fraction of questions the model
    attempted an answer for" — it means "fraction of rows that
    produced a scoreable response", which includes deliberate
    refusals to false-premise questions. Read alongside
    ``refusal_rate`` (true-premise refusals only) and
    ``negative_rejection_rate`` (false-premise correct-refusal rate)."""
    n_format_errors: int
    format_error_rate: float
    n_refusals: int = 0
    """v2: count of rows where the prediction was a refusal (model
    returned ``answer=None``). Default 0 for backward compatibility
    when an old report is loaded by a v2 aggregator."""
    refusal_rate: float = 0.0
    """v2: ``n_refusals / n_total``."""
    n_gold_missing: int = 0
    """v2: count of rows where the gold answer is missing
    (``ground_truth is None`` without a false-premise marker).
    Separated from ``format_error`` so upstream gold defects don't
    inflate the scorer-side failure count."""
    gold_missing_rate: float = 0.0
    """v2: ``n_gold_missing / n_total``."""
    answer_rate: float = 0.0
    """v2 alias for ``coverage_rate`` (same value). Provided so
    paper-table writers can use the more descriptive name without
    re-deriving it. ``answer_rate + refusal_rate + format_error_rate
    + gold_missing_rate ≈ 1.0`` (within rounding) when no substrate-error
    rows are present; with harness errors they sum to
    ``1 - n_harness_errors/n_total`` (error rows are a fifth bucket)."""

    # Headline.
    em_micro: float
    """Overall EM across all predictions (one weight per question)."""
    em_macro: float
    """Mean of per-answer-type EMs (equal weight per type). Protects
    against a single high-volume type dominating the headline."""
    em_conditional: float
    """**Mixed-population metric**: overall EM conditional on
    ``covered=True``. ``covered=True`` includes correct false-premise
    refusals, so this number averages *attempted-answer accuracy on
    real-premise rows* together with *correct-refusal rate on
    false-premise rows*. Two qualitatively different signals
    collapsed into one. Read alongside:
    * :attr:`em_conditional_true_premise` — same metric restricted
      to real-premise rows; the "did the model answer attempted
      questions correctly?" number on its own.
    * :attr:`em_true_premise` — same denominator but unconditional
      (includes refusals on real-premise rows as em=0).
    * :attr:`negative_rejection_rate` — correct-refusal rate on
      false-premise rows alone.
    """
    em_conditional_true_premise: float | None = None
    """EM conditional on ``covered=True`` AND
    ``is_true_premise is not False``. Excludes false-premise rows
    (which would inflate the apparent "attempted-answer accuracy"
    via correct refusals). ``None`` when no covered true-premise
    rows exist."""
    ne_micro: float
    """Overall NE across all predictions. **Caveat:** NE is hard-set to
    1.0 on every failure (harness / format / refusal / uncovered), so this
    raw micro-average conflates real error magnitude with fiat max-error
    on non-answers. Read alongside :attr:`ne_conditional` (the "how close
    when it answers" view) and :attr:`ne_adjusted` (harness rows removed).
    """
    ne_macro: float
    ne_conditional: float | None = None
    """**Mixed-population metric, but the informative NE view**: overall NE
    conditional on ``covered=True``. NE is padded to 1.0 on every
    non-answer (refusal / format error / harness error / uncovered), so the
    raw :attr:`ne_micro` blends real error magnitude with that fiat
    max-error; restricting to ``covered`` rows drops the padding and leaves
    the "how close was the answer when the model actually answered?"
    signal. Like :attr:`em_conditional`, ``covered=True`` still includes
    correct false-premise refusals (NE=1 there), so on a mixed population
    this averages attempted-answer error together with false-premise
    non-answers; read alongside :attr:`ne_true_premise` for the
    real-premise-only view. Default ``None`` for backward compatibility —
    old reports without the field deserialize cleanly."""

    # Majority-class baseline (the no-reasoning floor). Model-independent
    # — computed from the gold label distribution alone: for each TID,
    # always emit its modal gold; the fraction correct. Printed beside
    # ``em_micro`` / ``em_macro`` so a reader can see how much of the
    # headline is reachable with zero spatial reasoning (e.g. booleans
    # give ~0.76). ``None`` when no record carries a ``gold_label``.
    majority_baseline_em: float | None = None
    """Per-TID majority-class EM (micro): for each TID always emit its
    most-frequent gold, computed over the LABELED rows only (those
    carrying a ``gold_label``) with exact casefolded-string equality. A
    no-reasoning reference point — compare against ``em_true_premise``
    (same population), NOT ``em_micro``: the latter also spans
    false-premise rows and scores continuous types with tolerances, so
    the baseline is not a strict lower bound of it."""
    majority_baseline_em_macro: float | None = None
    """Mean of per-answer-type ``majority_em`` (macro), with the same
    labeled-rows / exact-match caveat as ``majority_baseline_em``."""

    # Unit-normalized numeric companions.
    # Computed on numeric predictions only — the prediction is
    # converted into the gold unit when both are dimensionally
    # compatible (distance or area), and EM/NE are recomputed.
    # Cross-group, unitless, or unknown-unit predictions fall back to
    # the strict EM/NE (never more lenient than the strict pair).
    # Fields are ``None`` when the run has zero numeric predictions.
    #
    # DENOMINATOR POLICY (fixed): these are computed over **all
    # Numeric-typed rows**, scoring a refusal / format-error / harness
    # error as its actual EM (0 for a real-premise miss; a correct
    # false-premise refusal keeps EM=1, exactly as ``em_micro`` does).
    # This makes ``numeric_em_strict`` finally equal "``em_micro``
    # restricted to numeric rows" (its documented contract) and, crucially,
    # COMPARABLE across models: a model cannot inflate it by refusing the
    # hard rows. The old computation averaged only over the scored-success
    # subset (``unit_match is not None``), a model-chosen denominator that
    # produced 36× gaps and ranking inversions. The scored-subset numbers
    # are preserved as the ``*_conditional`` companions below, and
    # ``numeric_retained_fraction`` exposes how much of the population they
    # cover. ``None`` when the run has zero Numeric-typed rows.
    numeric_em_strict: float | None = None
    """Mean EM over ALL Numeric-typed rows under strict unit matching —
    ``em_micro`` restricted to numeric rows. Comparable across models."""
    numeric_em_unit_normalized: float | None = None
    """Mean EM over ALL Numeric-typed rows after converting
    dimensionally-compatible prediction units to the gold unit.
    Isolates spatial-reasoning capability from unit-instruction
    compliance; the gap ``numeric_em_unit_normalized −
    numeric_em_strict`` is the size of the 'correct answer in the
    wrong unit' effect."""
    numeric_ne_strict: float | None = None
    """Mean NE over ALL Numeric-typed rows under strict unit handling."""
    numeric_ne_unit_normalized: float | None = None
    """Mean NE over ALL Numeric-typed rows with unit-normalized
    distance. Lower than ``numeric_ne_strict`` when the model was
    off-by-a-unit (e.g. m vs km), since the per-row NE is no longer
    pinned to 1.0 by the unit-induced 1000× gap."""
    numeric_em_conditional: float | None = None
    """DIAGNOSTIC (model-chosen denominator — NOT comparable across
    models): mean EM over the scored-success subset only (numeric rows
    that produced a parseable number; refusals / format-errors
    excluded). The old ``numeric_em_strict``. Read as "accuracy when the
    model attempted a number", never as a headline. ``None`` when no
    numeric row was scored."""
    n_numeric_typed: int = 0
    """Count of Numeric-typed rows — the denominator of the unconditional
    numeric metrics above."""
    n_numeric_scored: int = 0
    """Count of Numeric-typed rows that reached the scoring success path
    (parseable number). The denominator of ``numeric_em_conditional``."""
    numeric_retained_fraction: float | None = None
    """``n_numeric_scored / n_numeric_typed`` — the share of numeric rows
    the conditional metric is computed over. Below ~1.0 means the
    conditional number is not comparable across models. ``None`` when no
    numeric rows."""

    # Spatial-proximity companions for grid-cell answer types (MGRS-like
    # grid cell, Composite string (H3+S2)). Computed only over rows
    # carrying a non-null ``em_spatial`` — the decodable grid types
    # (S2 cell index is excluded by design). ``grid_em_strict`` mirrors
    # ``em_micro`` restricted to those rows; ``grid_em_spatial``
    # recomputes EM after decoding both sides to a location and scoring
    # by geodesic proximity. The gap ``grid_em_spatial −
    # grid_em_strict`` is the "right place, non-canonical grid encoding"
    # effect. All ``None`` (and ``n_grid_spatial=0``) when the run has
    # zero spatially-scored grid predictions.
    grid_em_strict: float | None = None
    grid_em_spatial: float | None = None
    grid_ne_strict: float | None = None
    grid_ne_spatial: float | None = None
    n_grid_spatial: int = 0
    """Count of grid-cell predictions that received a spatial companion
    score (non-null ``em_spatial``). Default 0 for backward compat."""

    # Capability vs harness split.
    # ``em_micro`` counts errors as wrong so harness failures depress
    # the headline. ``em_adjusted`` excludes harness errors
    # (``error_class`` in :data:`HARNESS_ERROR_CLASSES`) from the
    # denominator so readers can separate model capability from
    # harness capability. Both are reported; which one is used as
    # "the number" is a paper-writing choice, not a scoring choice.
    em_adjusted: float = 0.0
    """EM over predictions that are not harness failures. Equivalent
    to ``em_micro`` over the subset where ``error_class`` is not in
    ``{infra, code_gen}``. Matches the "capability-only" number
    readers can apply their own adjustment to. Default 0.0 for
    backward compatibility — old reports without the field deserialize
    cleanly."""
    ne_adjusted: float = 0.0
    """Mean NE over predictions that are not harness failures — the NE
    analogue of :attr:`em_adjusted` (rows whose ``error_class`` is in
    ``{infra, context_overflow, code_gen}`` are excluded from the
    denominator). Since NE is padded to 1.0 on harness failures, dropping
    those rows separates model-side error magnitude from harness-induced
    fiat max-error. Default 0.0 for backward compatibility."""
    n_harness_errors: int = 0
    """Count of predictions classified as harness errors (``infra`` +
    ``context_overflow`` + ``code_gen``). Complementary to
    ``n_format_errors`` which counts scorer-side parse failures rather
    than model/runtime failures. Default 0 for backward compatibility."""
    n_missing_predictions: int = 0
    """Count of current-benchmark gold qids that have NO prediction in
    this cell (gold size minus scored rows). Scoring is prediction-driven,
    so these are silently absent from every rate above — this field makes
    the gap visible. Non-zero indicates an incomplete inference run (e.g.
    a truncated large-split run), NOT a benchmark defect: the questions
    are valid and scoreable, there is just no model output. Missing rows
    are NOT scored as EM=0 here; use this rate to flag / exclude
    incomplete cells from cross-model comparisons. 0 when ``n_gold`` is
    not supplied to ``build_report`` (e.g. old callers / unit fixtures)."""
    missing_prediction_rate: float = 0.0
    """``n_missing_predictions / n_gold`` (0.0 when ``n_gold`` unknown)."""
    error_class_breakdown: dict[str, int] = Field(default_factory=dict)
    """Raw counts by ``error_class`` across all error predictions.
    Keys are the canonical labels from
    :data:`benchmark_models.base.ErrorClass`. Empty dict when no
    errors occurred."""

    # Per-group breakdowns.
    per_answer_type: dict[str, GroupStats] = Field(default_factory=dict)
    per_sfid: dict[str, GroupStats] = Field(default_factory=dict)
    per_language: dict[str, GroupStats] = Field(default_factory=dict)
    per_kg: dict[str, GroupStats] = Field(default_factory=dict)
    """Per-KG-bucket breakdown keyed by ``ScoreRecord.kg_source`` (e.g.
    ``"kwg"``, ``"osmh3kg"``, ``"worldkg"``, ``"multimodal"``). Empty
    dict when no record carried a ``kg_source``. Paper-table material:
    "EM is 73% on worldkg but 58% on osmh3kg" is the kind of slice
    readers expect — surfacing it inline saves a notebook groupby."""
    per_strata_cell: dict[str, GroupStats] = Field(default_factory=dict)
    """Per-stratum breakdown keyed by the full ``strata_cell`` tag
    (e.g., ``"HIC_high"``, ``"LIC_low"``). Empty dict when no record
    carried a ``strata_cell``. Reviewers reading the headline EM
    table typically want to know whether the model performs
    uniformly across income tiers and density buckets — this rollup
    surfaces that without a notebook groupby."""

    per_income_tier: dict[str, GroupStats] = Field(default_factory=dict)
    """Per-income-tier breakdown keyed by the prefix before ``_`` in
    ``strata_cell`` — typically one of ``"HIC"`` (high-income),
    ``"UMC"`` (upper-middle), ``"LMC"`` (lower-middle), ``"LIC"``
    (low-income). Coarser companion to :attr:`per_strata_cell` for
    paper tables that don't need the full 12-cell matrix. Empty
    dict when no record carried a ``strata_cell``."""

    # False-premise stratification. ``negative_rejection_rate`` is the
    # mean EM over rows where the question's premise is false — by
    # construction, the only way to score EM=1 on those rows is to
    # abstain (``answer is None``), so this is the model's correct-
    # abstention rate on bad questions. ``em_true_premise`` is the
    # mean EM over real-premise rows; together with
    # ``negative_rejection_rate`` it separates capability-on-real-
    # questions from abstention-on-bad-questions, which the headline
    # ``em_micro`` confounds. ``false_refusal_rate`` completes the
    # detection 2x2 (premise validity x answered/abstained): it is the
    # over-abstention rate on answerable rows, the confound for
    # ``negative_rejection_rate``. ``None`` on either side when the
    # corresponding subset is empty or the benchmark omits
    # ``is_true_premise``.
    n_true_premise: int = 0
    n_false_premise: int = 0
    em_true_premise: float | None = None
    ne_true_premise: float | None = None
    """Mean NE over real-premise rows (``is_true_premise is True``) — the
    NE analogue of :attr:`em_true_premise`. The real-premise-only error
    magnitude, uncontaminated by false-premise non-answers (which pad NE to
    1.0). ``None`` when no real-premise rows exist. Default ``None`` for
    backward compatibility."""
    negative_rejection_rate: float | None = None
    false_refusal_rate: float | None = None
    """Over-abstention / false-refusal rate: the fraction of answerable
    (true-premise) rows on which the model refused
    (``mean(r.refusal for r in records if r.is_true_premise is True)``).
    The confound for :attr:`negative_rejection_rate` — a model that
    abstains indiscriminately scores high correct-rejection *and* high
    false-refusal, so the two must be read together to tell genuine
    false-premise detection from blanket abstention. Together with
    ``negative_rejection_rate`` (premise validity x answered/abstained)
    this forms the detection 2x2. ``None`` when no true-premise rows
    exist. Default ``None`` for backward compatibility."""

    # Set-enumeration aggregate roll-up. Per-record set P/R/F1 already
    # live on :class:`ScoreRecord`; these fields surface their means
    # at the report level so set-only types (entity-list, date-list)
    # have first-class headline numbers separate from ``em_micro``
    # (which counts strict set equality only). ``None`` when the run
    # has zero set-enumeration predictions.
    #
    # DENOMINATOR POLICY (fixed): ``mean_set_*`` are over **all
    # Set-enumeration-typed rows**, scoring a refusal / format-error as
    # P/R/F1 = 0 (it produced no set). Comparable across models. The old
    # scored-subset numbers (``set_f1 is not None``) are the
    # ``*_conditional`` companions; ``set_retained_fraction`` shows their
    # coverage. ``n_set_records`` keeps its historical meaning (scored
    # subset) for backward compatibility; ``n_set_typed`` is the new
    # unconditional denominator.
    n_set_records: int = 0
    """Count of set rows on the scored-success path (``set_f1 is not
    None``). Historical field — the denominator of the ``*_conditional``
    metrics, NOT of ``mean_set_*``."""
    n_set_typed: int = 0
    """Count of all Set-enumeration-typed rows — the denominator of the
    unconditional ``mean_set_*`` metrics."""
    set_retained_fraction: float | None = None
    """``n_set_records / n_set_typed``. Below ~1.0 means the conditional
    set metrics are not comparable across models."""
    mean_set_precision: float | None = None
    """Mean set precision over ALL set-typed rows (refusals / format
    errors = 0). Comparable across models."""
    mean_set_recall: float | None = None
    """Mean set recall over ALL set-typed rows (refusals / format
    errors = 0)."""
    mean_set_f1: float | None = None
    """Mean set F1 over ALL set-typed rows (refusals / format errors =
    0). Comparable across models."""
    mean_set_f1_conditional: float | None = None
    """DIAGNOSTIC (model-chosen denominator): mean set F1 over the
    scored-success subset only. The old ``mean_set_f1``. Read as "set
    quality when the model produced a set", never as a headline."""

    # Tier-3 operational metrics. Computed over records where
    # ``tool_call_count is not None`` (i.e., tier-3 only). ``None`` on
    # tier-1 / tier-2 only runs.
    #
    # * ``cap_hit_rate`` — fraction of agent runs that hit
    #   ``max_tool_calls`` without producing a final answer.
    # * ``tool_error_rate`` — fraction of agent runs with at least
    #   one failed tool call. Per-question rate, not error-per-call.
    # * ``no_final_answer_rate`` — fraction of agent runs that
    #   terminated with ``final_state=="no_final_answer"``.
    # * ``mean_tool_calls`` / ``mean_tool_errors`` — average count of
    #   tool calls / failed tool calls per agent run.
    n_tier3_records: int = 0
    cap_hit_rate: float | None = None
    tool_error_rate: float | None = None
    no_final_answer_rate: float | None = None
    n_no_final_answer: int = 0
    """Count of tier-3 records with ``final_state == "no_final_answer"``
    (agent loop terminated without producing a ``final_answer(...)`` call).
    Distinct from intentional refusals — useful to debug an agent-loop
    ceiling regression separately from genuine model-side abstentions.
    Stays ``0`` when no tier-3 records are present."""
    mean_tool_calls: float | None = None
    mean_tool_errors: float | None = None

    # Self-consistency agreement (tier-2 SC=N only). Mean
    # ``ScoreRecord.sc_agreement`` over records that have an
    # agreement value populated. A free per-question reliability
    # signal — ``1.0`` means every SC run was unanimous; ``2/3 ≈
    # 0.667`` is the lower bound for N=3 with majority vote.
    n_sc_records: int = 0
    sc_agreement_rate: float | None = None

    # Polygon-IoU planar-fallback diagnostics. Counts polygon rows
    # whose EPSG:6933 reprojection failed and fell through to
    # raw-lat/lon planar IoU (biased at high latitudes). When the
    # rate is < 1%, the equal-area metric is sound as-is; when > 10%,
    # the projection should be revisited (e.g. per-latitude-band
    # Albers fallback).
    n_polygon_planar_fallback: int = 0
    polygon_planar_fallback_rate: float | None = None
    """Fraction of polygon-typed rows whose IoU was computed with the
    planar fallback (reprojection failure). ``None`` on runs with no
    polygon rows; ``0.0`` when no fallbacks fired."""

    # Per-prediction cost / latency rollup. Denormalised from
    # :attr:`ScoreRecord.input_tokens` / ``.output_tokens`` /
    # ``.latency_ms`` so paper tables can quote a single
    # latency-per-question number and an end-to-end token-spend total
    # without joining against the predictions JSONL. ``None`` when
    # ``n_total == 0`` (degenerate empty run).
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    """I: sum of :attr:`ScoreRecord.cached_tokens` across all records.
    Lets paper tables compute real (post-cache) input cost rather
    than nominal — when the backend reports cache hits."""
    total_cache_creation_tokens: int = 0
    """I: sum of Anthropic ``cache_creation_input_tokens`` across
    records. The upfront premium when seeding new cache entries;
    pair with :attr:`total_cached_tokens` to compute cache ROI."""
    mean_latency_ms: float | None = None
    """Mean :attr:`ScoreRecord.latency_ms` across all records.
    ``None`` when no records are present. Includes harness-error
    rows (infra timeouts, code-gen failures) which often have
    ``latency_ms=0`` (immediate fail) or large-outlier timeouts;
    see :attr:`mean_latency_ms_adjusted` for the harness-filtered
    companion."""
    mean_latency_ms_adjusted: float | None = None
    """Mean :attr:`ScoreRecord.latency_ms` over **non-harness** records
    (``error_class not in HARNESS_ERROR_CLASSES``). Parallel to
    :attr:`em_adjusted`. Prefer this for paper-table "speed" columns —
    the raw mean is distorted by infra failures. ``None`` when no
    non-harness records are present."""

    # Diagnostics (nullable — populated only when the sub-metric has data).
    resolve_failure_rate: float | None = None
    """Fraction of alias-backed predictions (entity / cardinal /
    predefined) where ``resolve()`` returned ``None``. ``None`` when
    no alias-backed predictions are present."""
    unit_match_rate: float | None = None
    """Fraction of numeric predictions where the declared unit
    matched the gold unit. ``None`` when no numeric predictions are
    present."""
    numeric_failure_breakdown: dict[str, int] | None = None
    """Counts of numeric EM=0 failures by category
    (``number_wrong_unit_correct`` / ``number_correct_unit_wrong`` /
    ``both_wrong``). ``None`` when no numeric predictions are present."""

    # Scoring-spec provenance.
    scoring_spec_deviations: dict[str, Any] = Field(default_factory=dict)
    """Fields that differ from the canonical spec for this run — e.g.
    ``{"alpha": 0.01}`` when the operator used ``--alpha 0.01``. Empty
    when the run used spec defaults."""

    # Reproducibility hashes (E-3). Pin the report to the inputs it
    # was scored against so a re-run with a different alias store or
    # a re-issued benchmark share is detectable at a glance.
    alias_store_sha256: str | None = None
    """SHA256 hex digest of the alias-store JSON file contents.
    ``None`` when the run was scored without an alias store on disk
    (e.g., a synthetic-fixture test run)."""
    benchmark_snapshot_sha256: str | None = None
    """SHA256 hex digest of the ``(relative_path, size_bytes)``
    manifest for every JSONL shard under the benchmark root. Captures
    add / remove / size-change of any shard without reading row
    bytes. ``None`` when the run was scored against a synthetic
    benchmark without an on-disk shard layout."""
