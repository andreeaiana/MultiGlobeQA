"""Tests for the layout-aware benchmark/gold shard walkers in io.py.

The walkers exist to navigate the post-edited share layout
``benchmark/final/<kg>/<sfid>/<lang>/benchmark.jsonl/part-*.json`` (plus
the parallel ``<kg>/<sfid>/gold_triples.jsonl/``) without mixing the two
schemas. A blind ``rglob("part-*.json")`` would conflate benchmark rows
and gold-triple rows; these walkers stop at the leaf shard granularity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_models.io import (
    discover_benchmark_shards,
    discover_gold_shards,
)

# ----------------------------------------------------------------------------
# Layout fixture — mimics benchmark/final/<kg>/<sfid>/<lang>/benchmark.jsonl/
# ----------------------------------------------------------------------------


def _build_share_root(tmp_path: Path) -> Path:
    """Build a two-KG, two-sfid, two-lang share root with realistic shapes.

    Layout:
        root/
            kwg/
                sfid_a/
                    gold_triples.jsonl/part-00000.json
                    eng/benchmark.jsonl/part-00000.json
                    deu/benchmark.jsonl/part-00000.json
                sfid_b/
                    gold_triples.jsonl/part-00000.json
                    eng/benchmark.jsonl/part-00000.json
                    deu/benchmark.jsonl/part-00000.json
            worldkg/
                sfid_a/
                    gold_triples.jsonl/part-00000.json
                    eng/benchmark.jsonl/part-00000.json
                    deu/benchmark.jsonl/part-00000.json
    """
    root = tmp_path / "share"
    root.mkdir()
    kgs = ("kwg", "worldkg")
    sfids_per_kg = {"kwg": ("sfid_a", "sfid_b"), "worldkg": ("sfid_a",)}
    langs = ("eng", "deu")
    qid = 0
    for kg in kgs:
        for sfid in sfids_per_kg[kg]:
            cell_root = root / kg / sfid
            gold_dir = cell_root / "gold_triples.jsonl"
            gold_dir.mkdir(parents=True)
            (gold_dir / "part-00000.json").write_text(
                json.dumps({
                    "question_id": f"{kg}-{sfid}-q0",
                    "entity_role": "a",
                    "subject": "s",
                    "predicate": "p",
                    "object": "o",
                }) + "\n"
            )
            for lang in langs:
                qid += 1
                bench_dir = cell_root / lang / "benchmark.jsonl"
                bench_dir.mkdir(parents=True)
                (bench_dir / "part-00000.json").write_text(
                    json.dumps({
                        "question_id": f"{kg}-{sfid}-q{qid}",
                        "question_text": f"a question in {lang} for {kg}/{sfid}",
                        "language": lang,
                        "kg_source": kg,
                        "sfid": sfid,
                        "tid": 1,
                        "answer_type": "boolean",
                        "ground_truth": True,
                        "is_valid": True,
                        "is_true_premise": True,
                    }) + "\n"
                )
    return root


# ----------------------------------------------------------------------------
# discover_benchmark_shards
# ----------------------------------------------------------------------------


class TestDiscoverBenchmarkShards:
    def test_single_file_returns_path_unchanged(self, tmp_path: Path) -> None:
        f = tmp_path / "single.jsonl"
        f.write_text('{"question_id": "q1"}\n')
        assert discover_benchmark_shards(f) == [f]

    def test_leaf_shard_dir_returns_path_unchanged(self, tmp_path: Path) -> None:
        leaf = tmp_path / "benchmark.jsonl"
        leaf.mkdir()
        (leaf / "part-00000.json").write_text('{"question_id": "q1"}\n')
        assert discover_benchmark_shards(leaf) == [leaf]

    def test_share_root_walks_all_languages(self, tmp_path: Path) -> None:
        root = _build_share_root(tmp_path)
        shards = discover_benchmark_shards(root)
        # 2 langs × 3 cells = 6 leaves total
        assert len(shards) == 6
        # No gold-triples leaves leaked into the result.
        for s in shards:
            assert s.name == "benchmark.jsonl"

    def test_language_filter_restricts_walk(self, tmp_path: Path) -> None:
        root = _build_share_root(tmp_path)
        shards = discover_benchmark_shards(root, language="eng")
        assert len(shards) == 3
        for s in shards:
            assert s.parent.name == "eng"
            assert s.name == "benchmark.jsonl"

    def test_unknown_language_filter_raises_with_helpful_message(
        self, tmp_path: Path
    ) -> None:
        root = _build_share_root(tmp_path)
        with pytest.raises(ValueError, match="No benchmark.jsonl"):
            discover_benchmark_shards(root, language="zzz")

    def test_missing_path_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_benchmark_shards(tmp_path / "does_not_exist")

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        root = _build_share_root(tmp_path)
        shards_a = discover_benchmark_shards(root)
        shards_b = discover_benchmark_shards(root)
        assert shards_a == shards_b

    def test_rejects_leaf_with_wrong_name(self, tmp_path: Path) -> None:
        """A mis-pointed path at a ``gold_triples.jsonl/`` leaf must
        not be accepted as a benchmark shard — otherwise gold-triple
        rows (wrong schema) would be fed to the prompt renderer and
        the failure would surface much later as a confusing
        field-mapping error.
        """
        leaf = tmp_path / "gold_triples.jsonl"
        leaf.mkdir()
        (leaf / "part-00000.json").write_text(
            '{"question_id": "q1", "subject": "s", "predicate": "p", "object": "o"}\n'
        )
        with pytest.raises(ValueError, match="No benchmark.jsonl"):
            discover_benchmark_shards(leaf)


# ----------------------------------------------------------------------------
# discover_gold_shards
# ----------------------------------------------------------------------------


class TestDiscoverGoldShards:
    def test_single_file_returns_path_unchanged(self, tmp_path: Path) -> None:
        f = tmp_path / "gold.jsonl"
        f.write_text('{"question_id": "q1"}\n')
        assert discover_gold_shards(f) == [f]

    def test_leaf_shard_dir_returns_path_unchanged(self, tmp_path: Path) -> None:
        leaf = tmp_path / "gold_triples.jsonl"
        leaf.mkdir()
        (leaf / "part-00000.json").write_text('{"question_id": "q1"}\n')
        assert discover_gold_shards(leaf) == [leaf]

    def test_share_root_walks_all_kg_sfid_cells(self, tmp_path: Path) -> None:
        root = _build_share_root(tmp_path)
        shards = discover_gold_shards(root)
        # 3 (kg, sfid) cells in the fixture → 3 gold leaves.
        assert len(shards) == 3
        for s in shards:
            assert s.name == "gold_triples.jsonl"

    def test_rejects_leaf_with_wrong_name(self, tmp_path: Path) -> None:
        """A mis-pointed path at a ``benchmark.jsonl/`` leaf must not
        be accepted as a gold-triples shard — symmetric to the
        ``discover_benchmark_shards`` guard.
        """
        leaf = tmp_path / "benchmark.jsonl"
        leaf.mkdir()
        (leaf / "part-00000.json").write_text(
            '{"question_id": "q1", "question": "..."}\n'
        )
        with pytest.raises(ValueError, match="No gold_triples.jsonl"):
            discover_gold_shards(leaf)

    def test_gold_shards_do_not_leak_benchmark_shards(self, tmp_path: Path) -> None:
        """Ensures the two walkers don't cross-contaminate.

        ``rglob("benchmark.jsonl")`` and ``rglob("gold_triples.jsonl")``
        match exactly, but a regression that used a looser pattern (e.g.
        ``rglob("*.jsonl")``) would silently include both. Asserting
        names protects against that drift.
        """
        root = _build_share_root(tmp_path)
        bench_shards = discover_benchmark_shards(root)
        gold_shards = discover_gold_shards(root)
        assert set(s.name for s in bench_shards) == {"benchmark.jsonl"}
        assert set(s.name for s in gold_shards) == {"gold_triples.jsonl"}
        assert set(bench_shards).isdisjoint(set(gold_shards))


# ----------------------------------------------------------------------------
# Ambiguous size-root guard (benchmark/final/ holds both large/ and small/)
# ----------------------------------------------------------------------------


class TestAmbiguousSizeRootGuard:
    """Pointing a walker at a root that holds BOTH ``large/`` and
    ``small/`` benchmark trees is rejected: the two share a question_id
    space and a recursive walk would silently merge them (dedup
    last-wins). Callers must name the size explicitly."""

    def test_benchmark_walker_rejects_size_parent(self, tmp_path: Path) -> None:
        (tmp_path / "large").mkdir()
        (tmp_path / "small").mkdir()
        with pytest.raises(ValueError, match=r"both 'large/' and 'small/'"):
            discover_benchmark_shards(tmp_path)

    def test_gold_walker_rejects_size_parent(self, tmp_path: Path) -> None:
        (tmp_path / "large").mkdir()
        (tmp_path / "small").mkdir()
        with pytest.raises(ValueError, match=r"both 'large/' and 'small/'"):
            discover_gold_shards(tmp_path)

    def test_single_size_root_walks_normally(self, tmp_path: Path) -> None:
        # Only ``large/`` present (e.g. before small is materialized) — no
        # ambiguity, so the walk proceeds and finds the leaf as usual.
        bench = tmp_path / "large" / "kwg" / "sfid_a" / "eng" / "benchmark.jsonl"
        bench.mkdir(parents=True)
        (bench / "part-00000.json").write_text(
            '{"question_id": "q1", "question_text": "x"}\n'
        )
        shards = discover_benchmark_shards(tmp_path)
        assert len(shards) == 1
        assert shards[0].name == "benchmark.jsonl"


# ----------------------------------------------------------------------------
# load_benchmark / load_gold_triples integration with walkers
# ----------------------------------------------------------------------------


class TestLoadBenchmarkWalker:
    def test_load_share_root_concatenates_all_cells(self, tmp_path: Path) -> None:
        from benchmark_models.runner import load_benchmark

        root = _build_share_root(tmp_path)
        examples = load_benchmark(root)
        # 6 leaves × 1 row per leaf = 6 examples.
        assert len(examples) == 6

    def test_load_share_root_with_language_filter(self, tmp_path: Path) -> None:
        from benchmark_models.runner import load_benchmark

        root = _build_share_root(tmp_path)
        examples = load_benchmark(root, language="eng")
        assert len(examples) == 3
        for ex in examples:
            assert ex.language == "eng"


class TestLoadGoldTriplesWalker:
    def test_load_share_root_concatenates_all_kg_sfid_cells(
        self, tmp_path: Path
    ) -> None:
        from benchmark_models.oracle import load_gold_triples

        root = _build_share_root(tmp_path)
        gold = load_gold_triples(root)
        # 3 (kg, sfid) cells × 1 gold qid each = 3 questions covered.
        assert len(gold) == 3
        assert all(len(triples) >= 1 for triples in gold.values())
