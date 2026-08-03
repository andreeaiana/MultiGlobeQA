"""Tests for D3+D4: kg_source field + latency/cost on ScoreRecord.

D3 plumbs ``kg_source`` from the benchmark row through ``GoldRow`` →
``ScoreRecord`` → ``AggregateReport.per_kg``. D4 denormalizes
``Prediction.usage.input_tokens / output_tokens`` and ``latency_ms``
onto ``ScoreRecord`` and rolls up via
``AggregateReport.total_input_tokens / total_output_tokens / mean_latency_ms``.

These tests pin the join + rollup so future scorer or aggregator
refactors don't silently drop the fields.
"""

from __future__ import annotations

from benchmark_eval.aggregate import build_report
from benchmark_eval.base import GoldRow, ScoreRecord
from benchmark_eval.scorer import Scorer
from benchmark_models.base import Prediction, Usage


def _make_prediction(
    qid: str,
    *,
    answer: object = 1.0,
    input_tokens: int = 100,
    output_tokens: int = 20,
    latency_ms: float = 500.0,
    model_id: str = "test-model",
    tier: str = "tier1",
    language: str = "eng",
) -> Prediction:
    return Prediction(
        question_id=qid,
        tid="tid_0001",
        sfid="sfid_a",
        answer_type="numeric",
        model_id=model_id,
        tier=tier,
        language=language,
        prompt_template_id="tier1_plain",
        seed=42,
        snapshot_id="unset",
        answer=answer,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=latency_ms,
    )


def _make_gold(qid: str, *, kg_source: str | None = "kwg") -> GoldRow:
    return GoldRow(
        question_id=qid,
        tid="tid_0001",
        sfid="sfid_a",
        answer_type="Numeric (float)",
        answer_unit=None,
        ground_truth=1.0,
        language="eng",
        kg_source=kg_source,
        is_true_premise=True,
    )


class TestScoreRecordCarriesKgSource:
    def test_kg_source_propagates_to_score_record(self) -> None:
        scorer = Scorer()
        pred = _make_prediction("q1")
        gold = _make_gold("q1", kg_source="worldkg")
        record = scorer.score(pred, gold)
        assert record.kg_source == "worldkg"

    def test_kg_source_none_when_missing(self) -> None:
        scorer = Scorer()
        pred = _make_prediction("q1")
        gold = _make_gold("q1", kg_source=None)
        record = scorer.score(pred, gold)
        assert record.kg_source is None


class TestScoreRecordCarriesCostLatency:
    def test_input_output_tokens_propagate(self) -> None:
        scorer = Scorer()
        pred = _make_prediction("q1", input_tokens=250, output_tokens=42)
        gold = _make_gold("q1")
        record = scorer.score(pred, gold)
        assert record.input_tokens == 250
        assert record.output_tokens == 42

    def test_latency_ms_propagates(self) -> None:
        scorer = Scorer()
        pred = _make_prediction("q1", latency_ms=1234.5)
        gold = _make_gold("q1")
        record = scorer.score(pred, gold)
        assert record.latency_ms == 1234.5

    def test_refusal_path_also_carries_cost(self) -> None:
        # The short-circuit refusal path is a separate ScoreRecord
        # construction site — make sure base dict still carries
        # input_tokens / output_tokens / latency_ms there too.
        scorer = Scorer()
        pred = _make_prediction(
            "q1", answer=None, input_tokens=150, output_tokens=5, latency_ms=42.0,
        )
        # Real-premise refusal scores em=0.
        gold = _make_gold("q1")
        record = scorer.score(pred, gold)
        assert record.input_tokens == 150
        assert record.output_tokens == 5
        assert record.latency_ms == 42.0
        assert record.kg_source == "kwg"  # also pinned in this branch


# ----------------------------------------------------------------------------
# Aggregate report
# ----------------------------------------------------------------------------


def _make_record(
    qid: str,
    *,
    kg_source: str | None = "kwg",
    input_tokens: int = 100,
    output_tokens: int = 20,
    latency_ms: float = 500.0,
    em: int = 1,
    ne: float = 0.0,
    covered: bool = True,
    error_class: str | None = None,
    final_state: str | None = None,
    cached_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> ScoreRecord:
    return ScoreRecord(
        question_id=qid,
        tid="tid_0001",
        sfid="sfid_a",
        answer_type="Numeric (float)",
        model_id="test-model",
        tier="tier1",
        language="eng",
        kg_source=kg_source,
        em=em,
        ne=ne,
        covered=covered,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
        latency_ms=latency_ms,
        error_class=error_class,  # type: ignore[arg-type]
        final_state=final_state,
    )


class TestAggregateReportPerKg:
    def test_per_kg_breakdown_groups_by_kg_source(self) -> None:
        records = [
            _make_record("q1", kg_source="kwg", em=1),
            _make_record("q2", kg_source="kwg", em=0),
            _make_record("q3", kg_source="worldkg", em=1),
            _make_record("q4", kg_source="worldkg", em=1),
            _make_record("q5", kg_source="osmh3kg", em=0),
        ]
        report = build_report(records)
        assert set(report.per_kg.keys()) == {"kwg", "worldkg", "osmh3kg"}
        assert report.per_kg["kwg"].n == 2
        assert report.per_kg["kwg"].em == 0.5
        assert report.per_kg["worldkg"].em == 1.0
        assert report.per_kg["osmh3kg"].em == 0.0

    def test_per_kg_empty_when_no_records_carry_kg_source(self) -> None:
        records = [
            _make_record("q1", kg_source=None),
            _make_record("q2", kg_source=None),
        ]
        report = build_report(records)
        assert report.per_kg == {}

    def test_records_without_kg_excluded_but_dont_break_overall(self) -> None:
        # Mixed: some records carry kg_source, others don't. Overall
        # headline numbers cover ALL records; per_kg only buckets the
        # kg-bearing ones.
        records = [
            _make_record("q1", kg_source="kwg", em=1),
            _make_record("q2", kg_source=None, em=1),
        ]
        report = build_report(records)
        assert report.em_micro == 1.0  # all records
        assert "kwg" in report.per_kg
        assert report.per_kg["kwg"].n == 1


class TestAggregateReportCostLatency:
    def test_total_input_tokens_sums_across_records(self) -> None:
        records = [
            _make_record("q1", input_tokens=100, output_tokens=20),
            _make_record("q2", input_tokens=150, output_tokens=30),
            _make_record("q3", input_tokens=200, output_tokens=40),
        ]
        report = build_report(records)
        assert report.total_input_tokens == 450
        assert report.total_output_tokens == 90

    def test_mean_latency_ms_averages(self) -> None:
        records = [
            _make_record("q1", latency_ms=100.0),
            _make_record("q2", latency_ms=200.0),
            _make_record("q3", latency_ms=300.0),
        ]
        report = build_report(records)
        assert report.mean_latency_ms == 200.0

    def test_zero_latency_records_give_zero_mean(self) -> None:
        # build_report rejects empty input by design; instead verify the
        # mean is 0 (not None) when every record has zero latency.
        records = [
            _make_record("q1", latency_ms=0.0),
            _make_record("q2", latency_ms=0.0),
        ]
        report = build_report(records)
        assert report.mean_latency_ms == 0.0
        assert report.total_input_tokens == 200  # 2 × 100 default

    def test_mean_latency_ms_adjusted_excludes_harness_errors(self) -> None:
        """B2: harness errors (infra timeouts, code-gen failures)
        often have latency_ms=0 (immediate fast-fail) or massive
        outliers — both distort the raw mean. The ``_adjusted``
        companion filters them out parallel to ``em_adjusted``."""
        records = [
            _make_record("q1", latency_ms=200.0),
            _make_record("q2", latency_ms=300.0),
            # One harness-error row at 0 ms (immediate fast-fail) and
            # one at 30,000 ms (timeout) — both would distort the
            # raw mean.
            _make_record(
                "q3", latency_ms=0.0,
                error_class="infra", final_state="error",
            ),
            _make_record(
                "q4", latency_ms=30000.0,
                error_class="code_gen", final_state="error",
            ),
        ]
        report = build_report(records)
        # Raw mean includes the harness rows: (200+300+0+30000)/4 = 7625.
        assert report.mean_latency_ms == 7625.0
        # Adjusted excludes harness rows: (200+300)/2 = 250.
        assert report.mean_latency_ms_adjusted == 250.0

    def test_total_cached_tokens_sums_across_records(self) -> None:
        """I: AggregateReport.total_cached_tokens is the sum across
        records. Backends that report cache hits (OpenAI, Anthropic,
        Google, DeepSeek post-section-H) populate ScoreRecord;
        backends that don't (vLLM) leave it at 0."""
        records = [
            _make_record("q1", cached_tokens=80),
            _make_record("q2", cached_tokens=120),
            _make_record("q3", cached_tokens=0),  # vLLM-shaped record
        ]
        report = build_report(records)
        assert report.total_cached_tokens == 200
        assert report.total_cache_creation_tokens == 0

    def test_total_cache_creation_tokens_anthropic_only(self) -> None:
        """I: cache_creation_tokens is Anthropic-specific (upfront
        premium when seeding a new cache entry); other backends leave
        it at 0."""
        records = [
            _make_record("q1", cached_tokens=0, cache_creation_tokens=300),
            _make_record("q2", cached_tokens=240, cache_creation_tokens=0),
        ]
        report = build_report(records)
        assert report.total_cached_tokens == 240
        assert report.total_cache_creation_tokens == 300

    def test_mean_latency_ms_adjusted_none_when_all_harness(self) -> None:
        """All-harness-error run: adjusted mean is None (no clean data)."""
        records = [
            _make_record(
                "q1", latency_ms=0.0,
                error_class="infra", final_state="error",
            ),
            _make_record(
                "q2", latency_ms=10000.0,
                error_class="code_gen", final_state="error",
            ),
        ]
        report = build_report(records)
        assert report.mean_latency_ms == 5000.0  # raw mean still defined
        assert report.mean_latency_ms_adjusted is None


# ----------------------------------------------------------------------------
# End-to-end: scorer → record → aggregate
# ----------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline_preserves_per_kg_and_cost(self) -> None:
        scorer = Scorer()
        preds = [
            _make_prediction("q1", input_tokens=100, output_tokens=20, latency_ms=500.0),
            _make_prediction("q2", input_tokens=200, output_tokens=40, latency_ms=700.0),
        ]
        golds = {
            "q1": _make_gold("q1", kg_source="kwg"),
            "q2": _make_gold("q2", kg_source="worldkg"),
        }
        records = [scorer.score(p, golds[p.question_id]) for p in preds]
        report = build_report(records)
        assert set(report.per_kg.keys()) == {"kwg", "worldkg"}
        assert report.total_input_tokens == 300
        assert report.total_output_tokens == 60
        assert report.mean_latency_ms == 600.0
