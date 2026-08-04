"""Tests for :func:`benchmark_eval.aggregate.build_report`.

Covers:

* Micro vs macro EM/NE math on synthetic records with skewed type
  distribution (macro differs from micro by design there).
* Coverage, conditional EM, format error rate.
* Per-group breakdowns (answer_type, sfid, language).
* Diagnostic nullability — diagnostic fields are ``None`` when
  the sub-population is empty.
"""

from __future__ import annotations

import pytest

from benchmark_eval.aggregate import build_report
from benchmark_eval.base import ScoreRecord


def _rec(
    *,
    qid: str = "q",
    answer_type: str = "Boolean (yes/no, true/false)",
    sfid: str = "A",
    language: str = "en",
    strata_cell: str | None = None,
    em: int = 1,
    ne: float = 0.0,
    covered: bool = True,
    refusal: bool = False,
    format_error: bool = False,
    resolve_failure: bool | None = None,
    unit_match: bool | None = None,
    numeric_failure_category: str | None = None,
    final_state: str | None = None,
    error_class: str | None = None,
    is_true_premise: bool | None = None,
    tool_call_count: int | None = None,
    tool_error_count: int | None = None,
    sc_agreement: float | None = None,
    set_precision: float | None = None,
    set_recall: float | None = None,
    set_f1: float | None = None,
    polygon_iou_used_planar_fallback: bool | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        question_id=qid,
        tid="1.1",
        sfid=sfid,
        answer_type=answer_type,
        model_id="m",
        tier="tier1",
        language=language,
        strata_cell=strata_cell,
        em=em,
        ne=ne,
        covered=covered,
        refusal=refusal,
        format_error=format_error,
        resolve_failure=resolve_failure,
        unit_match=unit_match,
        numeric_failure_category=numeric_failure_category,  # type: ignore[arg-type]
        final_state=final_state,
        error_class=error_class,
        is_true_premise=is_true_premise,
        tool_call_count=tool_call_count,
        tool_error_count=tool_error_count,
        sc_agreement=sc_agreement,
        set_precision=set_precision,
        set_recall=set_recall,
        set_f1=set_f1,
        polygon_iou_used_planar_fallback=polygon_iou_used_planar_fallback,
    )


class TestHeadline:
    def test_all_correct(self) -> None:
        recs = [_rec(qid=f"q{i}", em=1, ne=0.0) for i in range(5)]
        r = build_report(recs)
        assert r.em_micro == 1.0
        assert r.em_macro == 1.0
        assert r.ne_micro == 0.0
        assert r.ne_macro == 0.0
        assert r.coverage_rate == 1.0

    def test_half_correct(self) -> None:
        recs = [
            _rec(qid="q0", em=1, ne=0.0),
            _rec(qid="q1", em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert r.em_micro == 0.5
        assert r.ne_micro == 0.5

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError):
            build_report([])


class TestMicroVsMacro:
    def test_skewed_distribution(self) -> None:
        """Lots of correct numeric + one wrong boolean: micro stays
        high, macro drops.

        9 numeric correct (EM=1, NE=0) + 1 boolean wrong (EM=0, NE=1):

        * Micro EM = 9/10 = 0.9.
        * Macro EM = (1.0 + 0.0) / 2 = 0.5 (each type gets equal
          weight regardless of count).
        """
        recs = (
            [_rec(qid=f"q{i}", answer_type="Numeric", em=1, ne=0.0) for i in range(9)]
            + [_rec(qid="q9", answer_type="Boolean (yes/no, true/false)", em=0, ne=1.0)]
        )
        r = build_report(recs)
        assert r.em_micro == pytest.approx(0.9)
        assert r.em_macro == pytest.approx(0.5)
        assert r.ne_micro == pytest.approx(0.1)
        assert r.ne_macro == pytest.approx(0.5)


class TestCoverageAndConditional:
    def test_coverage_rate(self) -> None:
        recs = [
            _rec(qid="q0", em=1, covered=True),
            _rec(qid="q1", em=0, ne=1.0, covered=False),
            _rec(qid="q2", em=1, covered=True),
        ]
        r = build_report(recs)
        assert r.coverage_rate == pytest.approx(2 / 3)

    def test_conditional_excludes_refusals(self) -> None:
        recs = [
            _rec(qid="q0", em=1, covered=True),
            _rec(qid="q1", em=0, ne=1.0, covered=False),
            _rec(qid="q2", em=0, covered=True),
        ]
        r = build_report(recs)
        # Micro EM counts refusal as 0: (1+0+0)/3 = 1/3
        assert r.em_micro == pytest.approx(1 / 3)
        # Conditional EM only over covered: (1+0)/2 = 0.5
        assert r.em_conditional == pytest.approx(0.5)


class TestFormatErrors:
    def test_format_error_rate(self) -> None:
        recs = [
            _rec(qid="q0", em=1, covered=True),
            _rec(qid="q1", em=0, ne=1.0, covered=False, format_error=True),
        ]
        r = build_report(recs)
        assert r.format_error_rate == 0.5
        assert r.n_format_errors == 1


class TestPerGroupBreakdowns:
    def test_per_answer_type(self) -> None:
        recs = [
            _rec(qid="q0", answer_type="Numeric", em=1),
            _rec(qid="q1", answer_type="Numeric", em=0, ne=1.0),
            _rec(qid="q2", answer_type="Boolean (yes/no, true/false)", em=1),
        ]
        r = build_report(recs)
        assert set(r.per_answer_type) == {"Numeric", "Boolean (yes/no, true/false)"}
        assert r.per_answer_type["Numeric"].em == 0.5
        assert r.per_answer_type["Boolean (yes/no, true/false)"].em == 1.0

    def test_per_sfid(self) -> None:
        recs = [
            _rec(qid="q0", sfid="A", em=1),
            _rec(qid="q1", sfid="B", em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert r.per_sfid["A"].em == 1.0
        assert r.per_sfid["B"].em == 0.0

    def test_per_language(self) -> None:
        recs = [
            _rec(qid="q0", language="en", em=1),
            _rec(qid="q1", language="de", em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert r.per_language["en"].em == 1.0
        assert r.per_language["de"].em == 0.0

    def test_per_strata_cell_full_tag(self) -> None:
        """Per-stratum rollup keys on the full ``strata_cell`` tag
        (e.g. ``HIC_high``). Reviewers reading per-strata tables get
        the income-tier-by-density-bucket cross product."""
        recs = [
            _rec(qid="q0", strata_cell="HIC_high", em=1),
            _rec(qid="q1", strata_cell="HIC_high", em=1),
            _rec(qid="q2", strata_cell="LIC_low", em=0, ne=1.0),
            _rec(qid="q3", strata_cell="LIC_low", em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert set(r.per_strata_cell.keys()) == {"HIC_high", "LIC_low"}
        assert r.per_strata_cell["HIC_high"].em == 1.0
        assert r.per_strata_cell["LIC_low"].em == 0.0
        assert r.per_strata_cell["HIC_high"].n == 2

    def test_per_income_tier_prefix(self) -> None:
        """``per_income_tier`` rollup splits ``HIC_high`` and
        ``HIC_medium`` into the same ``HIC`` bucket."""
        recs = [
            _rec(qid="q0", strata_cell="HIC_high", em=1),
            _rec(qid="q1", strata_cell="HIC_medium", em=1),
            _rec(qid="q2", strata_cell="LIC_low", em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert set(r.per_income_tier.keys()) == {"HIC", "LIC"}
        assert r.per_income_tier["HIC"].em == 1.0
        assert r.per_income_tier["HIC"].n == 2
        assert r.per_income_tier["LIC"].em == 0.0
        assert r.per_income_tier["LIC"].n == 1

    def test_per_strata_empty_when_no_records_carry_strata(self) -> None:
        """Records without ``strata_cell`` produce an empty rollup
        rather than a single ``"None"`` bucket."""
        recs = [
            _rec(qid="q0", em=1),
            _rec(qid="q1", em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert r.per_strata_cell == {}
        assert r.per_income_tier == {}


class TestDiagnostics:
    def test_resolve_failure_rate_populated(self) -> None:
        recs = [
            _rec(qid="q0", resolve_failure=False),
            _rec(qid="q1", resolve_failure=True, em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert r.resolve_failure_rate == 0.5

    def test_resolve_failure_rate_none_when_no_alias_types(self) -> None:
        recs = [_rec(qid="q0", em=1)]  # boolean — no resolve_failure
        r = build_report(recs)
        assert r.resolve_failure_rate is None

    def test_unit_match_rate(self) -> None:
        recs = [
            _rec(qid="q0", unit_match=True),
            _rec(qid="q1", unit_match=False, em=0, ne=1.0),
        ]
        r = build_report(recs)
        assert r.unit_match_rate == 0.5

    def test_numeric_failure_breakdown(self) -> None:
        recs = [
            _rec(
                qid="q0",
                unit_match=True,
                em=0,
                ne=1.0,
                numeric_failure_category="number_wrong_unit_correct",
            ),
            _rec(
                qid="q1",
                unit_match=False,
                em=0,
                ne=0.5,
                numeric_failure_category="number_correct_unit_wrong",
            ),
            _rec(qid="q2", unit_match=True, em=1),
        ]
        r = build_report(recs)
        assert r.numeric_failure_breakdown == {
            "number_wrong_unit_correct": 1,
            "number_correct_unit_wrong": 1,
        }


class TestProvenance:
    def test_defaults_to_first_record(self) -> None:
        recs = [_rec(qid="q0")]
        r = build_report(recs)
        assert r.model_id == "m"
        assert r.tier == "tier1"

    def test_overrides_applied(self) -> None:
        recs = [_rec(qid="q0")]
        r = build_report(recs, model_id="override-variant", tier="tier2")
        assert r.model_id == "override-variant"
        assert r.tier == "tier2"


class TestHarnessErrorsSplit:
    """The paper reports raw ``em_micro`` (errors count as wrong) as the
    headline, plus ``em_adjusted`` (harness errors excluded from the
    denominator) for readers who want the capability-only view.
    Per-class counts accompany both numbers.
    """

    def test_no_errors_adjusted_equals_micro(self) -> None:
        recs = [_rec(qid=f"q{i}", em=1) for i in range(4)]
        recs.append(_rec(qid="q4", em=0))
        r = build_report(recs)
        # No harness errors: adjusted == micro.
        assert r.em_adjusted == r.em_micro
        assert r.n_harness_errors == 0
        assert r.error_class_breakdown == {}

    def test_code_gen_errors_excluded_from_adjusted(self) -> None:
        """3 correct + 2 code_gen errors: micro = 3/5 = 0.6;
        adjusted = 3/3 = 1.0."""
        recs = [
            _rec(qid="q0", em=1),
            _rec(qid="q1", em=1),
            _rec(qid="q2", em=1),
            _rec(
                qid="q3",
                em=0,
                ne=1.0,
                covered=False,
                final_state="error",
                error_class="code_gen",
            ),
            _rec(
                qid="q4",
                em=0,
                ne=1.0,
                covered=False,
                final_state="error",
                error_class="code_gen",
            ),
        ]
        r = build_report(recs)
        assert r.em_micro == pytest.approx(0.6)
        assert r.em_adjusted == pytest.approx(1.0)
        assert r.n_harness_errors == 2
        assert r.error_class_breakdown == {"code_gen": 2}

    def test_infra_errors_also_excluded(self) -> None:
        recs = [
            _rec(qid="q0", em=1),
            _rec(
                qid="q1",
                em=0,
                covered=False,
                final_state="error",
                error_class="infra",
            ),
        ]
        r = build_report(recs)
        assert r.em_micro == 0.5
        assert r.em_adjusted == 1.0
        assert r.error_class_breakdown == {"infra": 1}

    def test_config_error_stays_in_denominator(self) -> None:
        """``config`` is not a harness error — missing tids and
        coverage gaps are real data issues the paper should own."""
        recs = [
            _rec(qid="q0", em=1),
            _rec(
                qid="q1",
                em=0,
                covered=False,
                final_state="error",
                error_class="config",
            ),
        ]
        r = build_report(recs)
        assert r.em_micro == 0.5
        assert r.em_adjusted == 0.5   # config stays in denom
        assert r.n_harness_errors == 0
        assert r.error_class_breakdown == {"config": 1}

    def test_schema_violation_stays_in_denominator(self) -> None:
        recs = [
            _rec(qid="q0", em=1),
            _rec(
                qid="q1",
                em=0,
                covered=False,
                final_state="error",
                error_class="schema_violation",
            ),
        ]
        r = build_report(recs)
        assert r.em_adjusted == 0.5
        assert r.n_harness_errors == 0

    def test_mixed_breakdown(self) -> None:
        """Multiple error classes in one run — breakdown preserves counts."""
        recs = [
            _rec(qid="q0", em=1),
            _rec(qid="q1", em=1),
            *[
                _rec(
                    qid=f"cg{i}",
                    em=0,
                    covered=False,
                    final_state="error",
                    error_class="code_gen",
                )
                for i in range(3)
            ],
            _rec(
                qid="infra0",
                em=0,
                covered=False,
                final_state="error",
                error_class="infra",
            ),
            _rec(
                qid="cfg0",
                em=0,
                covered=False,
                final_state="error",
                error_class="config",
            ),
        ]
        r = build_report(recs)
        assert r.error_class_breakdown == {
            "code_gen": 3,
            "infra": 1,
            "config": 1,
        }
        assert r.n_harness_errors == 4  # 3 code_gen + 1 infra
        # micro = 2/7 ≈ 0.286; adjusted excludes code_gen+infra = 2/3 ≈ 0.667
        assert r.em_micro == pytest.approx(2 / 7)
        assert r.em_adjusted == pytest.approx(2 / 3)

    def test_error_class_missing_when_final_state_not_error(self) -> None:
        """A non-error prediction with a stray error_class value
        (shouldn't happen in practice) must not leak into the
        breakdown. Only error-state rows count."""
        recs = [
            _rec(qid="q0", em=1, final_state="answered"),
            _rec(
                qid="q1",
                em=0,
                covered=False,
                final_state="abstained",
                error_class="code_gen",  # defensive: shouldn't be set here
            ),
        ]
        r = build_report(recs)
        assert r.error_class_breakdown == {}  # filtered by final_state
        assert r.n_harness_errors == 0


# ============================================================================
# Scoring-spec provenance
# ============================================================================


class TestScoringSpecProvenance:
    def test_empty_deviations_by_default(self) -> None:
        recs = [_rec(qid="q0", em=1)]
        r = build_report(recs)
        assert r.scoring_spec_deviations == {}
        # ``scoring_spec_version`` was dropped — there is one canonical
        # spec, no versioning surface.
        assert not hasattr(r, "scoring_spec_version")

    def test_deviations_propagated(self) -> None:
        recs = [_rec(qid="q0", em=1)]
        r = build_report(
            recs,
            scoring_spec_deviations={"alpha": 0.1},
        )
        assert r.scoring_spec_deviations == {"alpha": 0.1}


# ============================================================================
# Numeric-only summary (em_unit_normalized dilution fix)
# ============================================================================


def _num_rec(
    *,
    qid: str,
    em: int,
    em_normalized: int,
    ne: float = 0.0,
    ne_normalized: float | None = None,
    unit_match: bool = True,
) -> ScoreRecord:
    """Numeric ScoreRecord helper — distinct from ``_rec`` so the
    non-None numeric fields (``unit_match`` + ``em_unit_normalized`` +
    ``ne_unit_normalized``) are always populated."""
    return ScoreRecord(
        question_id=qid,
        tid="1.1",
        sfid="A",
        answer_type="Numeric (float)",
        model_id="m",
        tier="tier1",
        language="en",
        em=em,
        ne=ne,
        covered=True,
        unit_match=unit_match,
        em_unit_normalized=em_normalized,
        ne_unit_normalized=ne_normalized if ne_normalized is not None else ne,
    )


class TestNumericOnlySummary:
    def test_absent_when_no_numeric_rows(self) -> None:
        r = build_report([_rec(qid="q0", em=1)])  # boolean, unit_match=None
        assert r.numeric_em_strict is None
        assert r.numeric_em_unit_normalized is None
        assert r.numeric_ne_strict is None
        assert r.numeric_ne_unit_normalized is None

    def test_reports_mean_over_numeric_only(self) -> None:
        # 2 numeric rows + 1 boolean row. Numeric-only summary should
        # ignore the boolean row entirely.
        recs = [
            _rec(qid="q0", em=1, answer_type="Boolean (yes/no, true/false)"),
            _num_rec(qid="q1", em=1, em_normalized=1),
            _num_rec(qid="q2", em=0, em_normalized=1, ne=1.0, ne_normalized=0.0),
        ]
        r = build_report(recs)
        # em_micro: 2/3 across all rows.
        assert r.em_micro == pytest.approx(2 / 3)
        # numeric-only strict: 1/2 numeric rows passed.
        assert r.numeric_em_strict == pytest.approx(0.5)
        # numeric-only normalized: 2/2 after unit conversion.
        assert r.numeric_em_unit_normalized == pytest.approx(1.0)
        # numeric-only ne strict vs normalized.
        assert r.numeric_ne_strict == pytest.approx(0.5)
        assert r.numeric_ne_unit_normalized == pytest.approx(0.0)

    def test_gap_surfaces_unit_confusion(self) -> None:
        # All three numeric rows had correct magnitudes but wrong
        # units. Gap (normalized − strict) = 1.0 = maximal.
        recs = [
            _num_rec(qid=f"q{i}", em=0, em_normalized=1, ne=1.0, ne_normalized=0.0)
            for i in range(3)
        ]
        r = build_report(recs)
        assert r.numeric_em_strict == 0.0
        assert r.numeric_em_unit_normalized == 1.0


def _grid_rec(
    *,
    qid: str = "q",
    answer_type: str = "MGRS-like grid cell",
    em: int = 0,
    ne: float = 1.0,
    em_spatial: int | None = 1,
    ne_spatial: float | None = 0.0,
) -> ScoreRecord:
    return ScoreRecord(
        question_id=qid,
        tid="75.4",
        sfid="A",
        answer_type=answer_type,
        model_id="m",
        tier="tier1",
        language="en",
        em=em,
        ne=ne,
        covered=True,
        em_spatial=em_spatial,
        ne_spatial=ne_spatial,
    )


class TestGridSpatialSummary:
    def test_absent_when_no_grid_rows(self) -> None:
        r = build_report([_rec(qid="q0", em=1)])  # boolean, no em_spatial
        assert r.grid_em_strict is None
        assert r.grid_em_spatial is None
        assert r.grid_ne_strict is None
        assert r.grid_ne_spatial is None
        assert r.n_grid_spatial == 0

    def test_reports_gap_between_strict_and_spatial(self) -> None:
        # Two MGRS rows (both strict-0: right place, wrong convention;
        # one spatial hit, one spatial miss) + one boolean row that the
        # grid-only summary must ignore.
        recs = [
            _rec(qid="q0", em=1, answer_type="Boolean (yes/no, true/false)"),
            _grid_rec(qid="q1", em=0, ne=1.0, em_spatial=1, ne_spatial=0.0),
            _grid_rec(qid="q2", em=0, ne=1.0, em_spatial=0, ne_spatial=1.0),
        ]
        r = build_report(recs)
        assert r.n_grid_spatial == 2
        assert r.grid_em_strict == pytest.approx(0.0)   # both strict-0
        assert r.grid_em_spatial == pytest.approx(0.5)  # 1 of 2 hit
        assert r.grid_ne_spatial == pytest.approx(0.5)
        # Universal em_micro is unchanged (strict) across all 3 rows.
        assert r.em_micro == pytest.approx(1 / 3)

    def test_s2_rows_excluded(self) -> None:
        # S2 rows carry em_spatial=None → excluded from the grid summary.
        recs = [
            ScoreRecord(
                question_id="s",
                tid="75.2",
                sfid="A",
                answer_type="S2 cell index",
                model_id="m",
                tier="tier1",
                language="en",
                em=0,
                ne=1.0,
                covered=True,
            ),
        ]
        r = build_report(recs)
        assert r.n_grid_spatial == 0
        assert r.grid_em_spatial is None


# ============================================================================
# False-premise stratification (item D)
# ============================================================================


class TestFalsePremiseStratification:
    """When ``is_true_premise`` is denormalised onto the records, the
    aggregator splits EM by premise validity. ``negative_rejection_rate``
    is the EM mean over false-premise rows (the only way to score 1
    there is to abstain, so it is the correct-abstention rate)."""

    def test_absent_when_no_premise_data(self) -> None:
        # Plain records (is_true_premise=None) — neither subset.
        r = build_report([_rec(qid="q0", em=1)])
        assert r.n_true_premise == 0
        assert r.n_false_premise == 0
        assert r.em_true_premise is None
        assert r.negative_rejection_rate is None

    def test_em_true_premise_only(self) -> None:
        recs = [
            _rec(qid="q0", em=1, is_true_premise=True),
            _rec(qid="q1", em=0, ne=1.0, is_true_premise=True),
        ]
        r = build_report(recs)
        assert r.n_true_premise == 2
        assert r.n_false_premise == 0
        assert r.em_true_premise == pytest.approx(0.5)
        assert r.negative_rejection_rate is None

    def test_negative_rejection_rate_only(self) -> None:
        # Two false-premise rows: one correctly abstained (EM=1),
        # one fabricated (EM=0). NR rate = 1/2.
        recs = [
            _rec(qid="q0", em=1, is_true_premise=False),
            _rec(qid="q1", em=0, ne=1.0, is_true_premise=False),
        ]
        r = build_report(recs)
        assert r.n_true_premise == 0
        assert r.n_false_premise == 2
        assert r.em_true_premise is None
        assert r.negative_rejection_rate == pytest.approx(0.5)

    def test_both_subsets(self) -> None:
        recs = [
            _rec(qid="q0", em=1, is_true_premise=True),
            _rec(qid="q1", em=1, is_true_premise=True),
            _rec(qid="q2", em=0, ne=1.0, is_true_premise=True),
            _rec(qid="q3", em=1, is_true_premise=False),
            _rec(qid="q4", em=0, ne=1.0, is_true_premise=False),
        ]
        r = build_report(recs)
        assert r.n_true_premise == 3
        assert r.n_false_premise == 2
        assert r.em_true_premise == pytest.approx(2 / 3)
        assert r.negative_rejection_rate == pytest.approx(0.5)

    def test_none_premise_excluded_from_both_subsets(self) -> None:
        # Mix of validated and unvalidated premise rows; the unvalidated
        # ones (is_true_premise=None) must not leak into either subset.
        recs = [
            _rec(qid="q0", em=1, is_true_premise=True),
            _rec(qid="q1", em=1, is_true_premise=None),  # excluded
            _rec(qid="q2", em=0, ne=1.0, is_true_premise=False),
        ]
        r = build_report(recs)
        assert r.n_true_premise == 1
        assert r.n_false_premise == 1


# ============================================================================
# Set-enumeration aggregate roll-up (item C)
# ============================================================================


class TestSetEnumAggregate:
    """Per-record set P/R/F1 are computed by the scorer; the report
    aggregates them only over set-enum rows."""

    def test_absent_when_no_set_records(self) -> None:
        r = build_report([_rec(qid="q0", em=1)])  # boolean — no set fields
        assert r.n_set_records == 0
        assert r.mean_set_precision is None
        assert r.mean_set_recall is None
        assert r.mean_set_f1 is None

    def test_means_over_set_rows_only(self) -> None:
        recs = [
            _rec(qid="q0", em=1),  # boolean — excluded
            _rec(
                qid="q1",
                answer_type="Set enumeration (list of entity names)",
                em=1,
                ne=0.0,
                set_precision=1.0,
                set_recall=1.0,
                set_f1=1.0,
            ),
            _rec(
                qid="q2",
                answer_type="Set enumeration (list of entity names)",
                em=0,
                ne=0.5,
                set_precision=0.5,
                set_recall=0.5,
                set_f1=0.5,
            ),
        ]
        r = build_report(recs)
        assert r.n_set_records == 2
        assert r.mean_set_precision == pytest.approx(0.75)
        assert r.mean_set_recall == pytest.approx(0.75)
        assert r.mean_set_f1 == pytest.approx(0.75)


# ============================================================================
# Tier-3 operational metrics (item A)
# ============================================================================


class TestTier3Operations:
    """Aggregates over records where ``tool_call_count is not None``."""

    def test_absent_when_no_tier3_records(self) -> None:
        r = build_report([_rec(qid="q0", em=1)])
        assert r.n_tier3_records == 0
        assert r.cap_hit_rate is None
        assert r.tool_error_rate is None
        assert r.no_final_answer_rate is None
        assert r.mean_tool_calls is None
        assert r.mean_tool_errors is None

    def test_cap_hit_rate(self) -> None:
        # 3 tier-3 records: one cap_hit, two answered.
        recs = [
            _rec(
                qid="q0",
                em=0,
                covered=False,
                final_state="cap_hit",
                tool_call_count=10,
                tool_error_count=0,
            ),
            _rec(
                qid="q1",
                em=1,
                final_state="answered",
                tool_call_count=4,
                tool_error_count=0,
            ),
            _rec(
                qid="q2",
                em=1,
                final_state="answered",
                tool_call_count=2,
                tool_error_count=1,
            ),
        ]
        r = build_report(recs)
        assert r.n_tier3_records == 3
        assert r.cap_hit_rate == pytest.approx(1 / 3)
        # 1 of 3 runs had ≥1 tool error.
        assert r.tool_error_rate == pytest.approx(1 / 3)
        # Mean tool calls = (10+4+2)/3.
        assert r.mean_tool_calls == pytest.approx(16 / 3)
        # Mean tool errors = (0+0+1)/3.
        assert r.mean_tool_errors == pytest.approx(1 / 3)

    def test_no_final_answer_rate(self) -> None:
        recs = [
            _rec(
                qid="q0",
                em=0,
                covered=False,
                final_state="no_final_answer",
                tool_call_count=3,
                tool_error_count=0,
            ),
            _rec(
                qid="q1",
                em=1,
                final_state="answered",
                tool_call_count=1,
                tool_error_count=0,
            ),
        ]
        r = build_report(recs)
        assert r.no_final_answer_rate == pytest.approx(0.5)
        assert r.n_no_final_answer == 1

    def test_n_no_final_answer_counts_tier3_only(self) -> None:
        """``n_no_final_answer`` must count only tier-3 records (those
        with ``tool_call_count is not None``). A tier-1 record with
        ``final_state="no_final_answer"`` (which shouldn't legally
        happen, but the safeguard matters anyway) must not leak in."""
        recs = [
            _rec(
                qid="q0",
                em=0,
                covered=False,
                final_state="no_final_answer",
                tool_call_count=4,
                tool_error_count=0,
            ),
            _rec(
                qid="q1",
                em=0,
                covered=False,
                final_state="no_final_answer",
                tool_call_count=4,
                tool_error_count=0,
            ),
            _rec(
                qid="q2",
                em=1,
                final_state="answered",
                tool_call_count=1,
                tool_error_count=0,
            ),
            # Tier-1 record (tool_call_count=None) — excluded.
            _rec(qid="q3", em=0, covered=False),
        ]
        r = build_report(recs)
        assert r.n_tier3_records == 3
        assert r.n_no_final_answer == 2
        assert r.no_final_answer_rate == pytest.approx(2 / 3)

    def test_n_no_final_answer_zero_when_no_tier3(self) -> None:
        """``n_no_final_answer`` defaults to 0 on tier-1/2-only runs."""
        recs = [_rec(qid="q0", em=1), _rec(qid="q1", em=0)]
        r = build_report(recs)
        assert r.n_tier3_records == 0
        assert r.n_no_final_answer == 0
        assert r.no_final_answer_rate is None

    def test_tier1_records_excluded(self) -> None:
        # Plain tier-1 record (tool_call_count=None) must not leak into
        # the tier-3 denominator.
        recs = [
            _rec(qid="q0", em=1),  # tier 1 — excluded
            _rec(
                qid="q1",
                em=1,
                final_state="answered",
                tool_call_count=2,
                tool_error_count=0,
            ),
        ]
        r = build_report(recs)
        assert r.n_tier3_records == 1
        assert r.mean_tool_calls == pytest.approx(2.0)


# ============================================================================
# Self-consistency agreement (item B)
# ============================================================================


class TestSelfConsistencyAgreement:
    def test_absent_when_no_sc_records(self) -> None:
        r = build_report([_rec(qid="q0", em=1)])
        assert r.n_sc_records == 0
        assert r.sc_agreement_rate is None

    def test_agreement_rate(self) -> None:
        # Three SC records with agreements 1.0, 2/3, 1.0.
        recs = [
            _rec(qid="q0", em=1, sc_agreement=1.0),
            _rec(qid="q1", em=0, ne=1.0, sc_agreement=2 / 3),
            _rec(qid="q2", em=1, sc_agreement=1.0),
        ]
        r = build_report(recs)
        assert r.n_sc_records == 3
        assert r.sc_agreement_rate == pytest.approx((1.0 + 2 / 3 + 1.0) / 3)


class TestGroupStatsEmpty:
    """``_group_stats`` on empty input returns ``None`` rather than
    ``0.0`` so the "no data" case can't be confused with a real
    "perfect zero" group. The normal aggregation path can't reach
    this branch (``_group_by`` filters keys before calling), so the
    None is purely defensive for direct callers."""

    def test_group_stats_returns_none_on_empty_records(self) -> None:
        from benchmark_eval.aggregate import _group_stats

        stats = _group_stats([])
        assert stats.n == 0
        assert stats.n_covered == 0
        assert stats.em is None
        assert stats.ne is None
        assert stats.em_conditional is None

    def test_group_stats_returns_em_conditional_none_when_no_covered(self) -> None:
        from benchmark_eval.aggregate import _group_stats

        # All records are format errors → no covered records → em_conditional is None.
        stats = _group_stats([_rec(qid="q0", em=0, ne=1.0, covered=False)])
        assert stats.n == 1
        assert stats.n_covered == 0
        assert stats.em == 0.0
        assert stats.em_conditional is None


class TestEmConditionalTruePremise:
    """``em_conditional`` averages over ``covered=True``, which
    includes correct false-premise refusals (em=1). The parallel
    ``em_conditional_true_premise`` excludes false-premise rows so
    the "attempted-answer accuracy" signal is readable alone."""

    def test_em_conditional_includes_false_premise_refusals(self) -> None:
        recs = [
            _rec(qid="q0", em=1, covered=True, is_true_premise=True),
            _rec(qid="q1", em=0, ne=1.0, covered=True, is_true_premise=True),
            # False-premise correct refusal: em=1, covered=True.
            _rec(qid="q2", em=1, covered=True, is_true_premise=False),
        ]
        r = build_report(recs)
        # Mixed metric: (1 + 0 + 1) / 3 = 0.667.
        assert r.em_conditional == pytest.approx(2 / 3)
        # True-premise-only: (1 + 0) / 2 = 0.5.
        assert r.em_conditional_true_premise == pytest.approx(0.5)

    def test_em_conditional_true_premise_none_when_no_covered_true_premise(
        self,
    ) -> None:
        recs = [
            _rec(qid="q0", em=1, covered=True, is_true_premise=False),  # bad qn
        ]
        r = build_report(recs)
        # ``em_conditional`` still computed (one covered row).
        assert r.em_conditional == 1.0
        # ``em_conditional_true_premise`` None — no covered real-premise rows.
        assert r.em_conditional_true_premise is None


class TestPolygonPlanarFallback:
    """AggregateReport surfaces planar-fallback diagnostics so the
    paper can report what fraction of polygon rows used the degraded
    IoU metric (raw lon/lat instead of EPSG:6933 equal-area)."""

    def test_absent_when_no_polygon_records(self) -> None:
        # All rows are non-polygon (flag stays None) — rate is None.
        r = build_report([_rec(qid="q0", em=1)])
        assert r.n_polygon_planar_fallback == 0
        assert r.polygon_planar_fallback_rate is None

    def test_zero_rate_when_all_polygon_rows_reproject_cleanly(self) -> None:
        recs = [
            _rec(qid="q0", em=1, polygon_iou_used_planar_fallback=False),
            _rec(qid="q1", em=0, ne=1.0, polygon_iou_used_planar_fallback=False),
        ]
        r = build_report(recs)
        assert r.n_polygon_planar_fallback == 0
        assert r.polygon_planar_fallback_rate == 0.0

    def test_rate_reflects_fallback_fraction(self) -> None:
        # 2 of 4 polygon rows fell back; rate = 0.5. Non-polygon row
        # (flag=None) is excluded from the denominator.
        recs = [
            _rec(qid="q0", polygon_iou_used_planar_fallback=False),
            _rec(qid="q1", polygon_iou_used_planar_fallback=True),
            _rec(qid="q2", polygon_iou_used_planar_fallback=True),
            _rec(qid="q3", polygon_iou_used_planar_fallback=False),
            _rec(qid="q4"),  # non-polygon row, flag stays None
        ]
        r = build_report(recs)
        assert r.n_polygon_planar_fallback == 2
        assert r.polygon_planar_fallback_rate == 0.5


def _numrec6(qid, em, *, unit_match, tid="1.1", gold_label=None, covered=True):
    """A Numeric-typed ScoreRecord. unit_match=None marks a non-scored
    (refusal / format-error) row."""
    return ScoreRecord(
        question_id=qid, tid=tid, sfid="A", answer_type="Numeric (float)",
        model_id="m", tier="tier1", language="en", em=em,
        ne=0.0 if em else 1.0, covered=covered, unit_match=unit_match,
        gold_label=gold_label,
    )


class TestDenominatorPolicyFix:
    """Numeric_em_strict / mean_set_f1 must be over ALL type rows so a
    model can't inflate them by refusing hard rows."""

    def test_numeric_unconditional_vs_conditional(self) -> None:
        # 2 scored-correct + 1 refusal (unit_match=None).
        recs = [
            _numrec6("a", 1, unit_match=True),
            _numrec6("b", 1, unit_match=True),
            _numrec6("c", 0, unit_match=None, covered=False),
        ]
        r = build_report(recs)
        # Unconditional over all 3 typed rows.
        assert r.numeric_em_strict == pytest.approx(2 / 3)
        assert r.n_numeric_typed == 3
        # Conditional over the 2 scored rows (the old, gameable number).
        assert r.numeric_em_conditional == pytest.approx(1.0)
        assert r.n_numeric_scored == 2
        assert r.numeric_retained_fraction == pytest.approx(2 / 3)

    def test_set_unconditional_scores_refusal_zero(self) -> None:
        set_type = "Set enumeration (list of entity names)"
        scored = ScoreRecord(
            question_id="a", tid="16.1", sfid="B", answer_type=set_type,
            model_id="m", tier="tier1", language="en", em=1, ne=0.0,
            covered=True, set_precision=1.0, set_recall=1.0, set_f1=1.0,
        )
        refusal = ScoreRecord(
            question_id="b", tid="16.1", sfid="B", answer_type=set_type,
            model_id="m", tier="tier1", language="en", em=0, ne=1.0,
            covered=False, refusal=True,  # set_f1 stays None
        )
        r = build_report([scored, refusal])
        assert r.n_set_typed == 2 and r.n_set_records == 1
        assert r.mean_set_f1 == pytest.approx(0.5)  # (1.0 + 0) / 2
        assert r.mean_set_f1_conditional == pytest.approx(1.0)  # scored only
        assert r.set_retained_fraction == pytest.approx(0.5)


class TestMajorityBaseline:
    """Part 1 #5 / audit #3: model-independent no-reasoning floor."""

    def test_per_tid_majority_micro(self) -> None:
        # TID X: 9 'true' / 1 'false' → majority guess right on 9/10.
        recs = [
            _numrec6(str(i), 0, unit_match=True, tid="18.1",
                     gold_label="true" if i < 9 else "false")
            for i in range(10)
        ]
        r = build_report(recs)
        assert r.majority_baseline_em == pytest.approx(0.9)

    def test_majority_is_model_independent(self) -> None:
        # Same golds, model gets everything wrong → majority unchanged.
        recs = [
            _numrec6(str(i), 0, unit_match=True, tid="30.1", gold_label="true")
            for i in range(5)
        ]
        r = build_report(recs)
        assert r.em_micro == 0.0
        assert r.majority_baseline_em == pytest.approx(1.0)  # constant gold
        gs = r.per_answer_type["Numeric (float)"]
        assert gs.majority_em == pytest.approx(1.0)

    def test_none_gold_label_excluded(self) -> None:
        # A false-premise row (gold_label=None) doesn't anchor a class.
        recs = [
            _numrec6("a", 0, unit_match=True, tid="1.1", gold_label="5.0"),
            _numrec6("b", 0, unit_match=None, tid="1.1", gold_label=None,
                     covered=False),
        ]
        r = build_report(recs)
        # Only the labeled row counts → 1/1.
        assert r.majority_baseline_em == pytest.approx(1.0)


class TestMissingPredictionsAndContextOverflow:
    def test_missing_predictions_surfaced_not_penalized(self) -> None:
        # 3 scored records; the current benchmark has 5 gold qids → 2 have
        # no prediction. They are surfaced, NOT scored EM=0.
        recs = [_rec(qid=f"q{i}", em=1, ne=0.0) for i in range(3)]
        r = build_report(recs, n_gold=5)
        assert r.n_missing_predictions == 2
        assert r.missing_prediction_rate == pytest.approx(2 / 5)
        # em_micro is over the 3 scored rows only (not deflated to 3/5).
        assert r.em_micro == 1.0
        assert r.n_total == 3

    def test_missing_predictions_zero_when_n_gold_omitted(self) -> None:
        recs = [_rec(qid=f"q{i}") for i in range(3)]
        r = build_report(recs)  # backward-compat: no n_gold
        assert r.n_missing_predictions == 0
        assert r.missing_prediction_rate == 0.0

    def test_context_overflow_excluded_from_em_adjusted(self) -> None:
        # One correct row + one context_overflow substrate error (EM=0).
        # em_micro counts the error as wrong (0.5); em_adjusted excludes
        # the harness error → 1.0 over the one real attempt.
        recs = [
            _rec(qid="q0", em=1),
            _rec(qid="q1", em=0, covered=False,
                 final_state="error", error_class="context_overflow"),
        ]
        r = build_report(recs)
        assert r.em_micro == pytest.approx(0.5)
        assert r.em_adjusted == pytest.approx(1.0)
        assert r.n_harness_errors == 1
        assert r.error_class_breakdown.get("context_overflow") == 1


# ============================================================================
# NE conditional / adjusted / true-premise variants (mirror the EM variants)
# ============================================================================


class TestNEVariants:
    """NE is hard-set to 1.0 on every non-answer, so ``ne_micro`` conflates
    real error magnitude with fiat max-error. The conditional / adjusted /
    true-premise variants strip that padding, mirroring the EM analogues."""

    def test_ne_conditional_over_covered_only(self) -> None:
        recs = [
            _rec(qid="q0", em=1, ne=0.0, covered=True),
            _rec(qid="q1", em=0, ne=0.4, covered=True),
            # Non-answer padded to NE=1.0 — excluded from the conditional.
            _rec(qid="q2", em=0, ne=1.0, covered=False),
        ]
        r = build_report(recs)
        assert r.ne_micro == pytest.approx((0.0 + 0.4 + 1.0) / 3)
        assert r.ne_conditional == pytest.approx(0.2)  # (0.0 + 0.4) / 2

    def test_ne_conditional_zero_when_no_covered(self) -> None:
        # Mirrors em_conditional's report-level 0.0 empty-case.
        r = build_report([_rec(qid="q0", em=0, ne=1.0, covered=False)])
        assert r.ne_conditional == 0.0

    def test_ne_adjusted_excludes_harness(self) -> None:
        recs = [
            _rec(qid="q0", em=1, ne=0.2),
            _rec(qid="q1", em=0, ne=1.0, covered=False,
                 final_state="error", error_class="code_gen"),
        ]
        r = build_report(recs)
        assert r.ne_micro == pytest.approx(0.6)      # (0.2 + 1.0) / 2
        assert r.ne_adjusted == pytest.approx(0.2)   # harness row dropped

    def test_ne_true_premise_over_true_premise_rows(self) -> None:
        recs = [
            _rec(qid="q0", em=1, ne=0.0, is_true_premise=True),
            _rec(qid="q1", em=0, ne=0.5, is_true_premise=True),
            # False-premise correct refusal (NE=1 padded) — excluded.
            _rec(qid="q2", em=1, ne=1.0, is_true_premise=False),
        ]
        r = build_report(recs)
        assert r.ne_true_premise == pytest.approx(0.25)  # (0.0 + 0.5) / 2

    def test_ne_true_premise_none_when_no_true_premise(self) -> None:
        r = build_report([_rec(qid="q0", em=1, ne=0.0)])  # premise unknown
        assert r.ne_true_premise is None

    def test_group_ne_variants(self) -> None:
        recs = [
            _rec(qid="q0", answer_type="Numeric", em=1, ne=0.0,
                 covered=True, is_true_premise=True),
            _rec(qid="q1", answer_type="Numeric", em=0, ne=0.6,
                 covered=False, is_true_premise=True),
        ]
        r = build_report(recs)
        gs = r.per_answer_type["Numeric"]
        assert gs.ne_conditional == pytest.approx(0.0)   # covered q0 only
        assert gs.ne_true_premise == pytest.approx(0.3)  # both tp rows

    def test_group_ne_conditional_none_when_no_covered(self) -> None:
        gs = build_report(
            [_rec(qid="q0", em=0, ne=1.0, covered=False)]
        ).per_answer_type["Boolean (yes/no, true/false)"]
        assert gs.ne_conditional is None  # group-level empty → None


# ============================================================================
# False-refusal (over-abstention) rate — completes the detection 2x2
# ============================================================================


class TestFalseRefusalRate:
    """``false_refusal_rate`` = abstention rate on answerable (true-premise)
    rows; the confound for ``negative_rejection_rate``."""

    def test_false_refusal_over_true_premise_rows(self) -> None:
        recs = [
            _rec(qid="q0", em=1, is_true_premise=True),
            _rec(qid="q1", em=1, is_true_premise=True),
            _rec(qid="q2", em=0, ne=1.0, covered=False, refusal=True,
                 is_true_premise=True),
        ]
        r = build_report(recs)
        assert r.false_refusal_rate == pytest.approx(1 / 3)

    def test_false_refusal_none_when_no_true_premise(self) -> None:
        recs = [_rec(qid="q0", em=1, covered=True, is_true_premise=False)]
        r = build_report(recs)
        assert r.false_refusal_rate is None

    def test_false_premise_refusal_not_counted(self) -> None:
        # A refusal on a FALSE-premise row is correct rejection, not an
        # over-abstention — it must not enter the true-premise denominator.
        recs = [
            _rec(qid="q0", em=1, is_true_premise=True),           # answered
            _rec(qid="q1", em=1, covered=True, refusal=True,
                 is_true_premise=False),                          # correct rej
        ]
        r = build_report(recs)
        assert r.false_refusal_rate == pytest.approx(0.0)
        assert r.negative_rejection_rate == pytest.approx(1.0)

    def test_group_false_refusal_rate(self) -> None:
        recs = [
            _rec(qid="q0", em=1, is_true_premise=True),
            _rec(qid="q1", em=0, ne=1.0, covered=False, refusal=True,
                 is_true_premise=True),
        ]
        r = build_report(recs)
        gs = r.per_answer_type["Boolean (yes/no, true/false)"]
        assert gs.false_refusal_rate == pytest.approx(0.5)
