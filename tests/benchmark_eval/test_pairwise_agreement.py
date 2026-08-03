"""Tests for the pairwise model-agreement helper.

The helper takes ``{model_name: {question_id: em}}`` and emits per-pair:
agreement rate, both-right / both-wrong / A-only / B-only rates, and
Cohen's kappa. The tests pin:

* The arithmetic (synthetic 0/1 grids).
* The intersection-only join behaviour (models with disjoint
  ``question_id`` sets must not produce inflated rates).
* Cohen's kappa edge cases (perfect agreement, both-uniform).
* Determinism: ``compute_all_pairs`` returns pairs in sorted
  model-name order.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_eval.pairwise_agreement import (
    PairwiseAgreement,
    compute_all_pairs,
    compute_pairwise,
    format_table,
    load_em_by_question_id,
    write_report,
)


class TestComputePairwise:
    def test_perfect_agreement_kappa_one(self) -> None:
        # Both models score the same on every question.
        em_a = {"q1": 1, "q2": 1, "q3": 0, "q4": 0}
        em_b = {"q1": 1, "q2": 1, "q3": 0, "q4": 0}
        p = compute_pairwise(em_a, em_b, model_a="a", model_b="b")
        assert p.n_intersection == 4
        assert p.agreement_rate == 1.0
        assert p.both_right_rate == 0.5
        assert p.both_wrong_rate == 0.5
        assert p.a_only_right_rate == 0.0
        assert p.b_only_right_rate == 0.0
        assert p.cohen_kappa == 1.0

    def test_complete_disagreement_kappa_negative(self) -> None:
        # Whatever A gets right, B gets wrong, and vice versa.
        em_a = {"q1": 1, "q2": 1, "q3": 0, "q4": 0}
        em_b = {"q1": 0, "q2": 0, "q3": 1, "q4": 1}
        p = compute_pairwise(em_a, em_b, model_a="a", model_b="b")
        assert p.n_intersection == 4
        assert p.agreement_rate == 0.0
        assert p.a_only_right_rate == 0.5
        assert p.b_only_right_rate == 0.5
        assert p.cohen_kappa == -1.0  # chance-corrected complete disagreement

    def test_random_50_50_kappa_near_zero(self) -> None:
        # A is half-right; B independently half-right. Agreement rate
        # is ~0.5 but kappa should be near zero (no agreement beyond
        # chance).
        em_a = {f"q{i}": (i % 2) for i in range(100)}
        em_b = {f"q{i}": ((i // 2) % 2) for i in range(100)}
        p = compute_pairwise(em_a, em_b, model_a="a", model_b="b")
        assert p.n_intersection == 100
        # Both are exactly 50% accurate; pe = 0.5 here.
        assert abs(p.agreement_rate - 0.5) < 1e-6
        assert abs(p.cohen_kappa - 0.0) < 1e-6

    def test_a_better_than_b_disagreement_breakdown(self) -> None:
        # A is right on all; B is right on half.
        em_a = {"q1": 1, "q2": 1, "q3": 1, "q4": 1}
        em_b = {"q1": 1, "q2": 1, "q3": 0, "q4": 0}
        p = compute_pairwise(em_a, em_b, model_a="a", model_b="b")
        assert p.agreement_rate == 0.5
        assert p.both_right_rate == 0.5
        assert p.both_wrong_rate == 0.0
        assert p.a_only_right_rate == 0.5
        assert p.b_only_right_rate == 0.0

    def test_intersection_only_no_inflation(self) -> None:
        # Models have disjoint question sets in addition to a small
        # overlap. The agreement must be computed only on the overlap.
        em_a = {"q1": 1, "q2": 1, "qA": 1, "qA2": 1}  # 4 questions
        em_b = {"q1": 1, "q2": 0, "qB": 0, "qB2": 0}  # 4 questions
        p = compute_pairwise(em_a, em_b, model_a="a", model_b="b")
        assert p.n_intersection == 2  # only q1 and q2 overlap
        # On the overlap: q1 agrees (both 1), q2 disagrees (A=1, B=0).
        assert p.agreement_rate == 0.5
        assert p.a_only_right_rate == 0.5

    def test_empty_intersection_returns_zeroed_record(self) -> None:
        em_a = {"q1": 1}
        em_b = {"q2": 1}
        p = compute_pairwise(em_a, em_b, model_a="a", model_b="b")
        assert p.n_intersection == 0
        assert p.agreement_rate == 0.0
        assert p.cohen_kappa == 0.0

    def test_both_models_always_right_kappa_one(self) -> None:
        # Degenerate case: po = 1.0, pe = 1.0. Kappa convention: 1.0.
        em_a = {"q1": 1, "q2": 1, "q3": 1}
        em_b = {"q1": 1, "q2": 1, "q3": 1}
        p = compute_pairwise(em_a, em_b, model_a="a", model_b="b")
        assert p.cohen_kappa == 1.0


class TestComputeAllPairs:
    def test_returns_n_choose_2_pairs(self) -> None:
        em_by_model = {
            "a": {"q1": 1, "q2": 0},
            "b": {"q1": 1, "q2": 0},
            "c": {"q1": 1, "q2": 1},
            "d": {"q1": 0, "q2": 0},
        }
        pairs = compute_all_pairs(em_by_model)
        # 4 choose 2 = 6 pairs.
        assert len(pairs) == 6
        # Sorted lex order ensures determinism.
        keys = [(p.model_a, p.model_b) for p in pairs]
        assert keys == sorted(keys)

    def test_two_model_input_produces_one_pair(self) -> None:
        em_by_model = {
            "alpha": {"q1": 1},
            "beta": {"q1": 0},
        }
        pairs = compute_all_pairs(em_by_model)
        assert len(pairs) == 1
        assert pairs[0].model_a == "alpha"
        assert pairs[0].model_b == "beta"


class TestLoadEmByQuestionId:
    def test_reads_em_field_only(self, tmp_path: Path) -> None:
        f = tmp_path / "score_records.jsonl"
        rows = [
            {"question_id": "q1", "em": 1, "ne": 0.0, "model_id": "x"},
            {"question_id": "q2", "em": 0, "ne": 1.0, "model_id": "x"},
        ]
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        em = load_em_by_question_id(f)
        assert em == {"q1": 1, "q2": 0}

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "score_records.jsonl"
        f.write_text(
            json.dumps({"question_id": "q1", "em": 1}) + "\n"
            "not-json\n"
            + json.dumps({"question_id": "q2", "em": 0}) + "\n"
            + json.dumps({"no_em": True}) + "\n"
        )
        em = load_em_by_question_id(f)
        # Skip the not-json line + the no-em row.
        assert em == {"q1": 1, "q2": 0}

    def test_duplicate_qid_last_wins(self, tmp_path: Path) -> None:
        """P1-3: duplicate ``question_id`` rows last-write-wins; the
        intended fix is to surface the duplicate via a warning, not to
        change the existing behaviour."""
        f = tmp_path / "score_records.jsonl"
        f.write_text(
            json.dumps({"question_id": "q1", "em": 0}) + "\n"
            + json.dumps({"question_id": "q1", "em": 1}) + "\n"  # duplicate
        )
        em = load_em_by_question_id(f)
        # Last value wins (1).
        assert em == {"q1": 1}

    def test_duplicate_qid_emits_warning(
        self, tmp_path: Path, caplog
    ) -> None:
        """P1-3: a single warning per file when duplicates are present
        — partial re-runs concatenated onto an old score file would
        otherwise silently corrupt the agreement matrix."""
        import logging

        f = tmp_path / "score_records.jsonl"
        f.write_text(
            json.dumps({"question_id": "q1", "em": 0}) + "\n"
            + json.dumps({"question_id": "q1", "em": 1}) + "\n"
            + json.dumps({"question_id": "q2", "em": 1}) + "\n"
            + json.dumps({"question_id": "q2", "em": 0}) + "\n"
        )
        with caplog.at_level(
            logging.WARNING, logger="benchmark_eval.pairwise_agreement"
        ):
            load_em_by_question_id(f)
        # One warning naming the file path and the duplicate count (2).
        assert any(
            "2 duplicate question_id" in rec.message
            for rec in caplog.records
        )

    def test_no_warning_when_no_duplicates(
        self, tmp_path: Path, caplog
    ) -> None:
        import logging

        f = tmp_path / "score_records.jsonl"
        f.write_text(
            json.dumps({"question_id": "q1", "em": 1}) + "\n"
            + json.dumps({"question_id": "q2", "em": 0}) + "\n"
        )
        with caplog.at_level(
            logging.WARNING, logger="benchmark_eval.pairwise_agreement"
        ):
            load_em_by_question_id(f)
        assert not any(
            "duplicate question_id" in rec.message
            for rec in caplog.records
        )


class TestReportFormats:
    def test_write_report_emits_valid_json(self, tmp_path: Path) -> None:
        pairs = [
            PairwiseAgreement(
                model_a="a", model_b="b",
                n_intersection=10,
                agreement_rate=0.8,
                both_right_rate=0.4,
                both_wrong_rate=0.4,
                a_only_right_rate=0.1,
                b_only_right_rate=0.1,
                cohen_kappa=0.6,
            ),
        ]
        out = tmp_path / "report.json"
        write_report(pairs, out)
        loaded = json.loads(out.read_text())
        assert isinstance(loaded, list)
        assert loaded[0]["model_a"] == "a"
        assert loaded[0]["cohen_kappa"] == 0.6

    def test_format_table_includes_header_and_pair_rows(self) -> None:
        pairs = [
            PairwiseAgreement(
                model_a="alpha", model_b="beta",
                n_intersection=10, agreement_rate=0.8,
                both_right_rate=0.4, both_wrong_rate=0.4,
                a_only_right_rate=0.1, b_only_right_rate=0.1,
                cohen_kappa=0.6,
            ),
        ]
        table = format_table(pairs)
        assert "model_a" in table
        assert "kappa" in table
        assert "alpha" in table
        assert "beta" in table
