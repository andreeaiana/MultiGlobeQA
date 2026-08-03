"""Build an :class:`AggregateReport` from a list of
:class:`ScoreRecord`.

Reports **both micro and macro** versions of EM and NE per the plan
decision:

* **Micro** (``em_micro``): simple mean across all predictions. One
  weight per question. Reflects the benchmark's actual distribution
  of answer types.
* **Macro** (``em_macro``): mean across per-answer-type means. Each
  answer type contributes equally. Protects against a single
  high-volume type dominating the headline.

Also reports **coverage** (fraction of predictions not refused and
not format-errored) and **conditional EM** (``em | covered``) per
the plan's Option E refusal policy.

Per-group breakdowns (``per_answer_type``, ``per_sfid``,
``per_language``) are included so downstream analysis can group
without rescoring. Diagnostic counters (``resolve_failure_rate``,
``unit_match_rate``, ``numeric_failure_breakdown``) are populated
only when the underlying data has relevant entries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from benchmark_eval.base import (
    HARNESS_ERROR_CLASSES,
    AggregateReport,
    GroupStats,
    ScoreRecord,
)


def _majority_baseline_em(records: list[ScoreRecord]) -> float | None:
    """EM of a label-free per-TID majority-class baseline over ``records``.

    For each TID present, the baseline always emits that TID's most
    common gold label; the returned value is the fraction of rows whose
    gold equals its TID's modal label. This is the no-reasoning floor —
    model-independent, a pure property of the gold distribution — printed
    beside model EM so a reader can see how much of the score is reachable
    with zero spatial reasoning. Rows without a ``gold_label`` (missing /
    false-premise gold) are excluded. ``None`` when none carry a label.
    """
    by_tid: dict[str, Counter[str]] = defaultdict(Counter)
    n = 0
    for r in records:
        if r.gold_label is None:
            continue
        by_tid[r.tid][r.gold_label] += 1
        n += 1
    if n == 0:
        return None
    hits = sum(c.most_common(1)[0][1] for c in by_tid.values())
    return hits / n


def _group_stats(records: list[ScoreRecord]) -> GroupStats:
    """Compute :class:`GroupStats` for a list of records.

    Returns ``em=ne=em_conditional=None`` when ``records`` is empty,
    so the empty-group case can't be confused with a real zero-score
    group ("perfect zero"). The normal aggregation path can't reach
    this branch (``_group_by`` filters keys before calling); the
    ``None`` is purely defensive for direct callers.
    """
    n = len(records)
    n_covered = sum(1 for r in records if r.covered)
    em = mean(r.em for r in records) if records else None
    ne = mean(r.ne for r in records) if records else None
    covered_records = [r for r in records if r.covered]
    em_conditional = (
        mean(r.em for r in covered_records) if covered_records else None
    )
    # NE|covered — the informative "how close when it answers" view. NE is
    # hard-set to 1.0 on every non-answer, so the unconditional ``ne`` above
    # blends real error magnitude with fiat max-error padding; restricting
    # to covered rows removes the padding. ``None`` when no covered rows.
    ne_conditional = (
        mean(r.ne for r in covered_records) if covered_records else None
    )
    # False-premise stratification, mirroring the global report fields
    # (aggregate.py build_report: ``true_premise`` uses ``is True``,
    # ``false_premise`` uses ``is False``; rows with an unknown premise
    # are in neither). ``em`` above counts a correct false-premise
    # refusal as EM=1, so a group that mixes premise validities blends
    # accuracy with correct-rejection rate. ``em_true_premise`` isolates
    # the attempted-answer accuracy; ``negative_rejection_rate`` the
    # correct-abstention rate on bad questions; ``false_refusal_rate`` the
    # over-abstention rate on answerable rows (confound for the former).
    true_premise = [r for r in records if r.is_true_premise is True]
    false_premise = [r for r in records if r.is_true_premise is False]
    em_true_premise = (
        mean(r.em for r in true_premise) if true_premise else None
    )
    ne_true_premise = (
        mean(r.ne for r in true_premise) if true_premise else None
    )
    negative_rejection_rate = (
        mean(r.em for r in false_premise) if false_premise else None
    )
    false_refusal_rate = (
        mean(1 if r.refusal else 0 for r in true_premise)
        if true_premise
        else None
    )
    return GroupStats(
        n=n,
        n_covered=n_covered,
        em=em,
        ne=ne,
        em_conditional=em_conditional,
        ne_conditional=ne_conditional,
        n_true_premise=len(true_premise),
        n_false_premise=len(false_premise),
        em_true_premise=em_true_premise,
        ne_true_premise=ne_true_premise,
        negative_rejection_rate=negative_rejection_rate,
        false_refusal_rate=false_refusal_rate,
        majority_em=_majority_baseline_em(records),
    )


# Keys that should never legitimately be ``None`` on a
# :class:`ScoreRecord` — silently dropping rows with these missing
# would let upstream bugs hide as a vanishing denominator.
_REQUIRED_GROUP_KEYS: frozenset[str] = frozenset({
    "tid", "sfid", "answer_type", "model_id", "tier", "language",
})


def _group_by(records: list[ScoreRecord], key: str) -> dict[str, list[ScoreRecord]]:
    """Group records by an attribute name.

    For required keys (``tid``, ``sfid``, ``answer_type``,
    ``model_id``, ``tier``, ``language``), records with ``None``
    values are dropped *with a warning* rather than silently — a
    None on these keys means upstream lost information and the rollup
    denominator would shrink without explanation. Optional keys
    (``strata_cell``) are dropped silently as before.
    """
    out: dict[str, list[ScoreRecord]] = {}
    n_dropped = 0
    for r in records:
        k = getattr(r, key, None)
        if k is None:
            if key in _REQUIRED_GROUP_KEYS:
                n_dropped += 1
            continue
        out.setdefault(str(k), []).append(r)
    if n_dropped:
        from loguru import logger

        logger.warning(
            "_group_by: dropped {} records with {!r}=None — this is "
            "a required grouping key. Investigate upstream pipeline.",
            n_dropped, key,
        )
    return out


def build_report(
    records: Iterable[ScoreRecord],
    *,
    model_id: str | None = None,
    tier: str | None = None,
    scoring_spec_deviations: dict[str, Any] | None = None,
    alias_store_sha256: str | None = None,
    benchmark_snapshot_sha256: str | None = None,
    n_gold: int | None = None,
) -> AggregateReport:
    """Roll a list of :class:`ScoreRecord` up into an :class:`AggregateReport`.

    Args:
        records: The scored predictions.
        model_id: Override for the report's ``model_id`` — defaults
            to the first record's value. Useful when scoring a
            subset that should still be labeled with the full run's
            identity.
        tier: Override for the report's ``tier`` field. Same pattern.
        scoring_spec_deviations: Fields where the run deviated from
            the canonical spec (e.g. from CLI overrides). Empty dict
            when the run used spec defaults.
        alias_store_sha256: Reproducibility hash of the alias store
            file the run scored against. ``None`` when the run used
            a synthetic store. Computed by :func:`benchmark_eval.
            _provenance.compute_alias_store_sha256`.
        benchmark_snapshot_sha256: Reproducibility hash of the
            benchmark shard manifest. ``None`` when the run scored
            against synthetic gold rows. Computed by
            :func:`benchmark_eval._provenance.compute_benchmark_snapshot_sha256`.

    Returns:
        An :class:`AggregateReport`. Every field is computed from
        ``records``; no MLflow / filesystem side effects.
    """
    rec_list = list(records)
    if not rec_list:
        raise ValueError("Cannot build an AggregateReport from zero records.")

    # Provenance (first-record fallback when not overridden).
    first = rec_list[0]
    m_id = model_id if model_id is not None else first.model_id
    m_tier = tier if tier is not None else first.tier

    # Totals.
    n_total = len(rec_list)
    n_covered = sum(1 for r in rec_list if r.covered)
    n_format_errors = sum(1 for r in rec_list if r.format_error)
    # v2: split refusals + gold-missing out of the format-error bucket.
    # Refusals that landed on false-premise rows have ``covered=True``
    # so they DON'T count against coverage; refusals on real-premise
    # rows have ``covered=False`` and ``refusal=True``. Gold-missing
    # rows always have ``refusal=True`` (the model abstained); we
    # report them under their own flag and exclude them from
    # ``n_refusals`` so the four buckets are mutually exclusive.
    n_gold_missing = sum(1 for r in rec_list if r.gold_missing)
    n_refusals = sum(
        1 for r in rec_list
        if r.refusal and not r.gold_missing and not r.covered
    )

    # Harness-error accounting: ``em_micro`` counts errors as wrong
    # (headline "errors = failure" policy). ``em_adjusted`` excludes
    # harness errors from the denominator so readers see the
    # capability-only number. Both are reported; choosing between
    # them is a paper-writing choice, not a scoring choice.
    error_class_counts: Counter[str] = Counter(
        r.error_class
        for r in rec_list
        if r.final_state == "error" and r.error_class is not None
    )
    n_harness_errors = sum(
        count
        for cls, count in error_class_counts.items()
        if cls in HARNESS_ERROR_CLASSES
    )
    non_harness_records = [
        r
        for r in rec_list
        if not (
            r.final_state == "error"
            and r.error_class in HARNESS_ERROR_CLASSES
        )
    ]
    em_adjusted = (
        mean(r.em for r in non_harness_records)
        if non_harness_records
        else 0.0
    )
    # NE analogue of ``em_adjusted``: NE is padded to 1.0 on harness
    # failures, so dropping those rows separates model-side error magnitude
    # from harness-induced fiat max-error. Mirrors ``em_adjusted``'s 0.0
    # empty-case convention.
    ne_adjusted = (
        mean(r.ne for r in non_harness_records)
        if non_harness_records
        else 0.0
    )

    # Headline.
    em_micro = mean(r.em for r in rec_list)
    ne_micro = mean(r.ne for r in rec_list)
    per_type_raw = _group_by(rec_list, "answer_type")
    per_type = {k: _group_stats(v) for k, v in per_type_raw.items()}
    per_sfid = {
        k: _group_stats(v) for k, v in _group_by(rec_list, "sfid").items()
    }
    per_language = {
        k: _group_stats(v) for k, v in _group_by(rec_list, "language").items()
    }
    # KG-source rollup. Skip records without ``kg_source`` (synthetic
    # fixtures) so the dict is empty rather than carrying a single
    # ``"None"`` bucket. With real data every record has a kg_source.
    kg_records = [r for r in rec_list if r.kg_source]
    per_kg = {
        k: _group_stats(v)
        for k, v in _group_by(kg_records, "kg_source").items()
    }
    # Strata-cell rollup. Skip records that don't carry a
    # ``strata_cell`` so the dict is empty (rather than a single
    # ``"None"`` bucket) for benchmarks without stratified sampling.
    strata_records = [r for r in rec_list if r.strata_cell]
    per_strata_cell = {
        k: _group_stats(v)
        for k, v in _group_by(strata_records, "strata_cell").items()
    }
    # Income-tier rollup: take the prefix before the first ``_`` in
    # the strata cell tag (``"HIC_high"`` -> ``"HIC"``). Coarser
    # companion to ``per_strata_cell`` — useful when the paper table
    # only wants the income axis without the density axis.
    per_income_tier_records: dict[str, list[ScoreRecord]] = defaultdict(list)
    for r in strata_records:
        # Defensive: if the tag has no ``_`` the whole string is the
        # tier (e.g., a malformed row). Better to surface that as a
        # single-tier bucket than drop it.
        tier = r.strata_cell.split("_", 1)[0] if r.strata_cell else None
        if tier:
            per_income_tier_records[tier].append(r)
    per_income_tier = {
        k: _group_stats(v) for k, v in per_income_tier_records.items()
    }

    # Macro = mean of per-answer-type means. Filter out None values
    # (empty groups carry em=ne=None defensively).
    _em_vals = [s.em for s in per_type.values() if s.em is not None]
    _ne_vals = [s.ne for s in per_type.values() if s.ne is not None]
    em_macro = mean(_em_vals) if _em_vals else 0.0
    ne_macro = mean(_ne_vals) if _ne_vals else 0.0
    # Majority-class baseline (no-reasoning floor). Micro = per-TID
    # majority over all rows; macro = mean of the per-answer-type
    # majority floors (parallel to em_macro).
    majority_baseline_em = _majority_baseline_em(rec_list)
    _maj_vals = [
        s.majority_em for s in per_type.values() if s.majority_em is not None
    ]
    majority_baseline_em_macro = mean(_maj_vals) if _maj_vals else None
    covered_records = [r for r in rec_list if r.covered]
    em_conditional = (
        mean(r.em for r in covered_records) if covered_records else 0.0
    )
    # NE|covered — the informative "how close when it answers" view. NE is
    # hard-set to 1.0 on every non-answer (refusal / format error / harness
    # error / uncovered), so the raw ``ne_micro`` blends real error
    # magnitude with fiat max-error padding; restricting to covered rows
    # drops the padding. Mirrors ``em_conditional``'s 0.0 empty-case.
    ne_conditional = (
        mean(r.ne for r in covered_records) if covered_records else 0.0
    )
    # ``em_conditional`` averages over ``covered=True``, but
    # ``covered=True`` includes correct false-premise refusals. The
    # headline metric therefore mixes "accuracy on attempted real
    # answers" with "correct-refusal rate on bad questions" — two
    # qualitatively different signals collapsed into one number.
    # ``em_conditional_true_premise`` excludes false-premise rows so
    # the "attempted answer accuracy" is readable on its own.
    covered_true_premise = [
        r for r in covered_records if r.is_true_premise is not False
    ]
    em_conditional_true_premise = (
        mean(r.em for r in covered_true_premise)
        if covered_true_premise
        else None
    )

    # Diagnostics — populated only when the relevant sub-population
    # is non-empty.
    alias_backed = [r for r in rec_list if r.resolve_failure is not None]
    resolve_failure_rate = (
        mean(1 if r.resolve_failure else 0 for r in alias_backed)
        if alias_backed
        else None
    )

    # Numeric populations. ``numeric_scored`` is the success-path subset
    # (a parseable number reached scoring, so ``unit_match`` was set);
    # ``numeric_typed`` is EVERY Numeric-typed row, including refusals /
    # format-errors / harness errors. The unconditional metrics average
    # over ``numeric_typed`` so they cannot be gamed by a model refusing
    # the hard rows (the old scored-subset denominator produced 36× gaps
    # and ranking inversions); the conditional metrics keep the
    # scored-subset behaviour as an explicit diagnostic.
    numeric_scored = [r for r in rec_list if r.unit_match is not None]
    numeric_typed = [
        r for r in rec_list if r.answer_type.startswith("Numeric")
    ]
    unit_match_rate = (
        mean(1 if r.unit_match else 0 for r in numeric_scored)
        if numeric_scored
        else None
    )

    numeric_failures = Counter(
        r.numeric_failure_category
        for r in rec_list
        if r.numeric_failure_category is not None
    )
    numeric_failure_breakdown = (
        dict(numeric_failures) if numeric_scored else None
    )

    # Numeric-only EM/NE summary. Unconditional over ``numeric_typed``
    # (refusals / format-errors contribute their actual EM/NE — 0 for a
    # real-premise miss). ``numeric_em_strict`` is thus "``em_micro``
    # restricted to numeric rows", comparable across models. ``None``
    # when the run has zero Numeric-typed rows.
    numeric_em_strict: float | None = None
    numeric_em_unit_normalized: float | None = None
    numeric_ne_strict: float | None = None
    numeric_ne_unit_normalized: float | None = None
    numeric_em_conditional: float | None = None
    numeric_retained_fraction: float | None = None
    if numeric_typed:
        numeric_em_strict = mean(r.em for r in numeric_typed)
        numeric_ne_strict = mean(r.ne for r in numeric_typed)
        numeric_em_unit_normalized = mean(
            (r.em_unit_normalized if r.em_unit_normalized is not None else r.em)
            for r in numeric_typed
        )
        numeric_ne_unit_normalized = mean(
            (r.ne_unit_normalized if r.ne_unit_normalized is not None else r.ne)
            for r in numeric_typed
        )
        numeric_retained_fraction = len(numeric_scored) / len(numeric_typed)
    if numeric_scored:
        # DIAGNOSTIC: model-chosen denominator (the old numeric_em_strict).
        numeric_em_conditional = mean(r.em for r in numeric_scored)

    # Grid-cell spatial companions. Rows carrying a non-null
    # ``em_spatial`` are the spatially-scored grid-cell types (MGRS-like
    # grid cell + the H3 half of Composite string; S2 cell index is
    # excluded by design). ``grid_em_strict`` restates the strict
    # exact-ID EM on those rows; ``grid_em_spatial`` is the
    # geodesic-proximity EM, so the gap between them is the "right place,
    # non-canonical grid encoding" effect. All None (n_grid_spatial=0)
    # when the run has no such rows.
    grid_records = [r for r in rec_list if r.em_spatial is not None]
    n_grid_spatial = len(grid_records)
    grid_em_strict: float | None = None
    grid_em_spatial: float | None = None
    grid_ne_strict: float | None = None
    grid_ne_spatial: float | None = None
    if grid_records:
        grid_em_strict = mean(r.em for r in grid_records)
        grid_em_spatial = mean(r.em_spatial for r in grid_records)
        grid_ne_strict = mean(r.ne for r in grid_records)
        grid_ne_spatial = mean(
            (r.ne_spatial if r.ne_spatial is not None else r.ne)
            for r in grid_records
        )

    # False-premise stratification. ``is_true_premise`` is denormalised
    # onto ScoreRecord by the scorer; rows where it is None are
    # excluded from both subsets (the benchmark didn't validate the
    # premise either way).
    true_premise = [r for r in rec_list if r.is_true_premise is True]
    false_premise = [r for r in rec_list if r.is_true_premise is False]
    em_true_premise = mean(r.em for r in true_premise) if true_premise else None
    # NE analogue of ``em_true_premise`` — real-premise-only error
    # magnitude, uncontaminated by false-premise non-answers (NE=1 padding).
    ne_true_premise = (
        mean(r.ne for r in true_premise) if true_premise else None
    )
    negative_rejection_rate = (
        mean(r.em for r in false_premise) if false_premise else None
    )
    # Over-abstention / false-refusal rate: fraction of ANSWERABLE
    # (true-premise) rows the model refused. Completes the detection 2x2
    # (premise validity x answered/abstained) and is the confound for
    # ``negative_rejection_rate`` — a model that abstains indiscriminately
    # shows high values on both. ``None`` when no true-premise rows.
    false_refusal_rate = (
        mean(1 if r.refusal else 0 for r in true_premise)
        if true_premise
        else None
    )

    # Set-enumeration aggregate. ``set_scored`` is the success-path subset
    # (``set_f1 is not None``); ``set_typed`` is EVERY
    # Set-enumeration-typed row. The unconditional ``mean_set_*`` average
    # over ``set_typed``, scoring a refusal / format-error as 0 (no set
    # produced) so they are comparable across models. The scored-subset
    # F1 is kept as ``mean_set_f1_conditional``.
    set_scored = [r for r in rec_list if r.set_f1 is not None]
    set_typed = [
        r for r in rec_list if r.answer_type.startswith("Set enumeration")
    ]
    n_set_records = len(set_scored)
    n_set_typed = len(set_typed)
    set_retained_fraction: float | None = (
        n_set_records / n_set_typed if n_set_typed else None
    )
    mean_set_precision: float | None = None
    mean_set_recall: float | None = None
    mean_set_f1: float | None = None
    mean_set_f1_conditional: float | None = None
    if set_typed:
        # ``or 0.0`` maps a refusal / format-error (P/R/F1 = None) to 0.
        mean_set_precision = mean(r.set_precision or 0.0 for r in set_typed)
        mean_set_recall = mean(r.set_recall or 0.0 for r in set_typed)
        mean_set_f1 = mean(r.set_f1 or 0.0 for r in set_typed)
    if set_scored:
        mean_set_f1_conditional = mean(r.set_f1 or 0.0 for r in set_scored)

    # Tier-3 operational metrics. Filter on ``tool_call_count is not
    # None`` (set only on tier-3 records). All five fields stay None
    # when the run is tier-1 / tier-2 only.
    tier3_records = [r for r in rec_list if r.tool_call_count is not None]
    n_tier3_records = len(tier3_records)
    cap_hit_rate: float | None = None
    tool_error_rate: float | None = None
    no_final_answer_rate: float | None = None
    n_no_final_answer = 0
    mean_tool_calls: float | None = None
    mean_tool_errors: float | None = None
    if tier3_records:
        cap_hit_rate = mean(
            1 if r.final_state == "cap_hit" else 0 for r in tier3_records
        )
        n_no_final_answer = sum(
            1 for r in tier3_records if r.final_state == "no_final_answer"
        )
        no_final_answer_rate = n_no_final_answer / len(tier3_records)
        # ``tool_error_rate`` is the fraction of agent runs with at
        # least one failed tool call (per-question rate) — distinct
        # from ``mean_tool_errors`` (the absolute mean count).
        tool_error_rate = mean(
            1 if (r.tool_error_count or 0) > 0 else 0 for r in tier3_records
        )
        mean_tool_calls = mean(r.tool_call_count or 0 for r in tier3_records)
        mean_tool_errors = mean(r.tool_error_count or 0 for r in tier3_records)

    # Self-consistency agreement. Filter on ``sc_agreement is not None``;
    # only populated for tier-2 records with N≥2 SC samples.
    sc_records = [r for r in rec_list if r.sc_agreement is not None]
    n_sc_records = len(sc_records)
    sc_agreement_rate = (
        mean(r.sc_agreement or 0.0 for r in sc_records) if sc_records else None
    )

    # Cost / latency rollup. Sum token totals (paper-table cost
    # column) and mean latency (paper-table speed column) across every
    # record. ``mean_latency_ms`` is ``None`` on empty runs to mirror
    # the rest of the report's "no data" sentinels.
    total_input_tokens = sum(r.input_tokens for r in rec_list)
    total_output_tokens = sum(r.output_tokens for r in rec_list)
    total_cached_tokens = sum(
        getattr(r, "cached_tokens", 0) for r in rec_list
    )
    total_cache_creation_tokens = sum(
        getattr(r, "cache_creation_tokens", 0) for r in rec_list
    )
    mean_latency_ms = (
        mean(r.latency_ms for r in rec_list) if rec_list else None
    )
    # Harness-error records (infra crash, code-gen failure) often
    # have ``latency_ms=0`` (immediate fast-fail) or huge values
    # (transport timeout), both of which distort the headline mean.
    # ``mean_latency_ms_adjusted`` filters them out — parallel to
    # ``em_adjusted``'s denominator. The "raw" mean stays alongside
    # for transparency.
    mean_latency_ms_adjusted = (
        mean(r.latency_ms for r in non_harness_records)
        if non_harness_records
        else None
    )

    # Polygon-IoU planar-fallback rate. Filter to polygon rows (those
    # where the flag is not None — None marks non-polygon rows that
    # never went through the polygon branch).
    polygon_rows = [
        r for r in rec_list if r.polygon_iou_used_planar_fallback is not None
    ]
    n_polygon_planar_fallback = sum(
        1 for r in polygon_rows if r.polygon_iou_used_planar_fallback
    )
    polygon_planar_fallback_rate = (
        n_polygon_planar_fallback / len(polygon_rows)
        if polygon_rows
        else None
    )

    coverage_rate = n_covered / n_total if n_total else 0.0

    # Missing-prediction gap (gold-driven): current-benchmark gold qids
    # with no prediction are silently absent from every rate above, since
    # scoring is prediction-driven. Surface the count/rate (do NOT penalize
    # them as EM=0) so incomplete cells can be flagged/excluded.
    n_missing_predictions = (
        max(0, n_gold - n_total) if n_gold is not None else 0
    )
    missing_prediction_rate = (
        n_missing_predictions / n_gold if n_gold else 0.0
    )
    if n_missing_predictions:
        from loguru import logger

        logger.warning(
            "{} of {} current-benchmark gold qids have NO prediction "
            "({:.1%}); prediction-driven scoring omits them from every "
            "rate. Likely an incomplete inference run — flag/exclude this "
            "cell from cross-model comparisons.",
            n_missing_predictions,
            n_gold,
            missing_prediction_rate,
        )

    return AggregateReport(
        model_id=m_id,
        tier=m_tier,
        n_total=n_total,
        n_covered=n_covered,
        coverage_rate=coverage_rate,
        answer_rate=coverage_rate,
        n_format_errors=n_format_errors,
        format_error_rate=n_format_errors / n_total if n_total else 0.0,
        n_refusals=n_refusals,
        refusal_rate=n_refusals / n_total if n_total else 0.0,
        n_gold_missing=n_gold_missing,
        gold_missing_rate=n_gold_missing / n_total if n_total else 0.0,
        em_micro=em_micro,
        em_macro=em_macro,
        em_conditional=em_conditional,
        em_conditional_true_premise=em_conditional_true_premise,
        ne_micro=ne_micro,
        ne_macro=ne_macro,
        ne_conditional=ne_conditional,
        majority_baseline_em=majority_baseline_em,
        majority_baseline_em_macro=majority_baseline_em_macro,
        em_adjusted=em_adjusted,
        ne_adjusted=ne_adjusted,
        n_harness_errors=n_harness_errors,
        n_missing_predictions=n_missing_predictions,
        missing_prediction_rate=missing_prediction_rate,
        error_class_breakdown=dict(error_class_counts),
        per_answer_type=per_type,
        per_sfid=per_sfid,
        per_language=per_language,
        per_kg=per_kg,
        per_strata_cell=per_strata_cell,
        per_income_tier=per_income_tier,
        resolve_failure_rate=resolve_failure_rate,
        unit_match_rate=unit_match_rate,
        numeric_failure_breakdown=numeric_failure_breakdown,
        numeric_em_strict=numeric_em_strict,
        numeric_em_unit_normalized=numeric_em_unit_normalized,
        numeric_ne_strict=numeric_ne_strict,
        numeric_ne_unit_normalized=numeric_ne_unit_normalized,
        numeric_em_conditional=numeric_em_conditional,
        n_numeric_typed=len(numeric_typed),
        n_numeric_scored=len(numeric_scored),
        numeric_retained_fraction=numeric_retained_fraction,
        grid_em_strict=grid_em_strict,
        grid_em_spatial=grid_em_spatial,
        grid_ne_strict=grid_ne_strict,
        grid_ne_spatial=grid_ne_spatial,
        n_grid_spatial=n_grid_spatial,
        n_true_premise=len(true_premise),
        n_false_premise=len(false_premise),
        em_true_premise=em_true_premise,
        ne_true_premise=ne_true_premise,
        negative_rejection_rate=negative_rejection_rate,
        false_refusal_rate=false_refusal_rate,
        n_set_records=n_set_records,
        n_set_typed=n_set_typed,
        set_retained_fraction=set_retained_fraction,
        mean_set_precision=mean_set_precision,
        mean_set_recall=mean_set_recall,
        mean_set_f1=mean_set_f1,
        mean_set_f1_conditional=mean_set_f1_conditional,
        n_tier3_records=n_tier3_records,
        cap_hit_rate=cap_hit_rate,
        tool_error_rate=tool_error_rate,
        no_final_answer_rate=no_final_answer_rate,
        n_no_final_answer=n_no_final_answer,
        mean_tool_calls=mean_tool_calls,
        mean_tool_errors=mean_tool_errors,
        n_sc_records=n_sc_records,
        sc_agreement_rate=sc_agreement_rate,
        n_polygon_planar_fallback=n_polygon_planar_fallback,
        polygon_planar_fallback_rate=polygon_planar_fallback_rate,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cached_tokens=total_cached_tokens,
        total_cache_creation_tokens=total_cache_creation_tokens,
        mean_latency_ms=mean_latency_ms,
        mean_latency_ms_adjusted=mean_latency_ms_adjusted,
        scoring_spec_deviations=dict(scoring_spec_deviations or {}),
        alias_store_sha256=alias_store_sha256,
        benchmark_snapshot_sha256=benchmark_snapshot_sha256,
    )


__all__ = ["build_report"]
