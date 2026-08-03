"""Tests for ``benchmark_eval.loaders``.

Two new behaviours added 2026-05-05 alongside the new benchmark share:

1. ``load_gold_rows`` accepts a directory of Spark part files (the
   share's layout) in addition to a single JSONL file.
2. Canonical answer-type resolution falls back to the per-TID
   registry's ``answer_type`` when the per-row ``ground_truth_type``
   is overloaded (``categorical`` or ``grid_cell_id``) and absent
   from the flat type vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_eval.loaders import load_gold_rows
from benchmark_models.schemas import (
    TemplateInfo,
    TemplateRegistry,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(
    *,
    qid: str,
    tid: str,
    sfid: str,
    ground_truth: object,
    ground_truth_type: str,
    is_valid: bool = True,
) -> dict:
    return {
        "question_id": qid,
        "tid": tid,
        "sfid": sfid,
        "ground_truth": ground_truth,
        "ground_truth_type": ground_truth_type,
        "is_valid": is_valid,
    }


@pytest.fixture
def synthetic_registry() -> TemplateRegistry:
    """Registry covering one non-overloaded TID and the four
    ``categorical`` overloaded TIDs from the share."""
    return TemplateRegistry(
        templates={
            "20.1": TemplateInfo(
                tid="20.1", sfid="B",
                answer_type="Set enumeration (list of entity names)",
                answer_unit=None,
            ),
            # Overloaded `categorical` family
            "91.1": TemplateInfo(
                tid="91.1", sfid="K",
                answer_type="Predefined value (left, right)",
                answer_unit=None,
            ),
            "92.1": TemplateInfo(
                tid="92.1", sfid="K",
                answer_type="Predefined value (upstream, downstream)",
                answer_unit=None,
            ),
            "93.1": TemplateInfo(
                tid="93.1", sfid="K",
                answer_type="Cardinal direction (predefined)",
                answer_unit=None,
            ),
            "105.1": TemplateInfo(
                tid="105.1", sfid="M",
                answer_type="Ternary (true/false/plausibly)",
                answer_unit=None,
            ),
        }
    )


class TestLoadGoldRowsSingleFile:
    """Regression suite for the file-input mode."""

    def test_reads_single_file(
        self, tmp_path: Path, synthetic_registry: TemplateRegistry
    ) -> None:
        path = tmp_path / "benchmark.jsonl"
        _write_jsonl(
            path,
            [
                _row(
                    qid="q1", tid="20.1", sfid="B",
                    ground_truth=["a", "b"],
                    ground_truth_type="set_enumeration",
                ),
            ],
        )
        out = load_gold_rows(path, registry=synthetic_registry)
        assert "q1" in out
        assert out["q1"].answer_type == "Set enumeration (list of entity names)"


class TestLoadGoldRowsDirectory:
    """Spark-style directory of part files mirrors the share layout."""

    def test_reads_all_shards(
        self, tmp_path: Path, synthetic_registry: TemplateRegistry
    ) -> None:
        d = tmp_path / "benchmark.jsonl"
        d.mkdir()
        _write_jsonl(
            d / "part-00000.json",
            [
                _row(
                    qid="q1", tid="20.1", sfid="B",
                    ground_truth=["a"],
                    ground_truth_type="set_enumeration",
                ),
            ],
        )
        _write_jsonl(
            d / "part-00001.json",
            [
                _row(
                    qid="q2", tid="20.1", sfid="B",
                    ground_truth=["b"],
                    ground_truth_type="set_enumeration",
                ),
            ],
        )
        out = load_gold_rows(d, registry=synthetic_registry)
        assert set(out) == {"q1", "q2"}


class TestLoadGoldRowsPerTidFallback:
    """For overloaded benchmark vocab tags (``categorical``,
    ``grid_cell_id``), the per-row ``ground_truth_type`` does not
    map to a canonical answer-type. The loader must fall back to
    the per-TID registry entry."""

    def test_categorical_resolves_via_per_tid(
        self, tmp_path: Path, synthetic_registry: TemplateRegistry
    ) -> None:
        path = tmp_path / "benchmark.jsonl"
        _write_jsonl(
            path,
            [
                _row(
                    qid="q-91", tid="91.1", sfid="K",
                    ground_truth="left",
                    ground_truth_type="categorical",
                ),
                _row(
                    qid="q-92", tid="92.1", sfid="K",
                    ground_truth="upstream",
                    ground_truth_type="categorical",
                ),
                _row(
                    qid="q-93", tid="93.1", sfid="K",
                    ground_truth="north",
                    ground_truth_type="categorical",
                ),
                _row(
                    qid="q-105", tid="105.1", sfid="M",
                    ground_truth="No",
                    ground_truth_type="categorical",
                ),
            ],
        )
        out = load_gold_rows(path, registry=synthetic_registry)
        assert out["q-91"].answer_type == "Predefined value (left, right)"
        assert out["q-92"].answer_type == "Predefined value (upstream, downstream)"
        assert out["q-93"].answer_type == "Cardinal direction (predefined)"
        assert out["q-105"].answer_type == "Ternary (true/false/plausibly)"

    def test_unknown_tid_with_unknown_type_is_skipped(
        self, tmp_path: Path, synthetic_registry: TemplateRegistry
    ) -> None:
        # Unknown tid + unknown vocab tag → no fallback available →
        # the row is skipped (counted toward n_skipped_no_type).
        path = tmp_path / "benchmark.jsonl"
        _write_jsonl(
            path,
            [
                _row(
                    qid="q-bad", tid="999.999", sfid="?",
                    ground_truth="x",
                    ground_truth_type="some_unknown_tag",
                ),
            ],
        )
        out = load_gold_rows(path, registry=synthetic_registry)
        assert "q-bad" not in out

    def test_registry_known_tid_uses_registry_answer_type(
        self, tmp_path: Path, synthetic_registry: TemplateRegistry
    ) -> None:
        """Registry-known TIDs use the registry's answer_type. The
        registry is the eval-side authority — built from the
        benchmark via ``build_template_registry_from_benchmark``,
        which encodes per-TID overrides for overloaded vocab tags
        and benchmark-side mislabels."""
        path = tmp_path / "benchmark.jsonl"
        _write_jsonl(
            path,
            [
                _row(
                    qid="q1", tid="20.1", sfid="B",
                    ground_truth=["a"],
                    ground_truth_type="set_enumeration",
                ),
            ],
        )
        out = load_gold_rows(path, registry=synthetic_registry)
        assert out["q1"].answer_type == "Set enumeration (list of entity names)"

    def test_registry_overrides_mislabeled_per_row_type(
        self, tmp_path: Path
    ) -> None:
        """When the per-row ``ground_truth_type`` disagrees with the
        registry (e.g. TID 115.1 labels gold as ``coordinates`` but
        the data shape is a 4-number bbox), the registry's
        ``answer_type`` wins. This catches benchmark-side mislabels
        without waiting on a generator-side fix."""
        registry = TemplateRegistry(
            templates={
                "115.1": TemplateInfo(
                    tid="115.1", sfid="P",
                    answer_type="Bounding box (4-float)",
                    answer_unit="degrees",
                ),
            }
        )
        path = tmp_path / "benchmark.jsonl"
        _write_jsonl(
            path,
            [
                _row(
                    qid="q-bbox", tid="115.1", sfid="P",
                    # Per-row says coordinates (wrong) but the gold
                    # shape is a 4-number bbox.
                    ground_truth="45.18,-67.27,45.20,-67.25",
                    ground_truth_type="coordinates",
                ),
            ],
        )
        out = load_gold_rows(path, registry=registry)
        # Registry wins.
        assert out["q-bbox"].answer_type == "Bounding box (4-float)"
