"""Tests for ``benchmark_eval._provenance`` and the reproducibility
hashes plumbed through ``AggregateReport``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_eval._provenance import (
    compute_alias_store_sha256,
    compute_benchmark_snapshot_sha256,
)

# ---------------------------------------------------------------------------
# Alias store content hash
# ---------------------------------------------------------------------------


class TestAliasStoreHash:
    def test_stable_across_reads(self, tmp_path: Path) -> None:
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps({"osm:1": {"canonical": "X", "aliases": []}}))
        h1 = compute_alias_store_sha256(f)
        h2 = compute_alias_store_sha256(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex digest length

    def test_changes_on_content_change(self, tmp_path: Path) -> None:
        f = tmp_path / "aliases.json"
        f.write_text(json.dumps({"osm:1": {"canonical": "X", "aliases": []}}))
        h1 = compute_alias_store_sha256(f)
        f.write_text(json.dumps({"osm:2": {"canonical": "Y", "aliases": []}}))
        h2 = compute_alias_store_sha256(f)
        assert h1 != h2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_alias_store_sha256(tmp_path / "no.json")

    def test_directory_argument_raises(self, tmp_path: Path) -> None:
        # A directory should fail (alias store is a single file).
        with pytest.raises(FileNotFoundError):
            compute_alias_store_sha256(tmp_path)


# ---------------------------------------------------------------------------
# Benchmark snapshot manifest hash
# ---------------------------------------------------------------------------


def _make_shard(root: Path, kg: str, sfid: str, kind: str, payload: str) -> Path:
    """Create a synthetic benchmark JSONL shard."""
    leaf = root / kg / sfid / f"{kind}.jsonl"
    leaf.mkdir(parents=True, exist_ok=True)
    shard = leaf / "part-00000.json"
    shard.write_text(payload)
    return shard


class TestBenchmarkSnapshotHash:
    def test_deterministic_across_reads(self, tmp_path: Path) -> None:
        bench = tmp_path / "bench"
        _make_shard(bench, "kwg", "sfid_a", "benchmark", '{"q":"x"}\n')
        _make_shard(bench, "kwg", "sfid_a", "gold_triples", '{"q":"x"}\n')
        h1 = compute_benchmark_snapshot_sha256(bench)
        h2 = compute_benchmark_snapshot_sha256(bench)
        assert h1 == h2
        assert len(h1) == 64

    def test_changes_on_added_shard(self, tmp_path: Path) -> None:
        bench = tmp_path / "bench"
        _make_shard(bench, "kwg", "sfid_a", "benchmark", '{"q":"x"}\n')
        h_before = compute_benchmark_snapshot_sha256(bench)
        _make_shard(bench, "kwg", "sfid_b", "benchmark", '{"q":"y"}\n')
        h_after = compute_benchmark_snapshot_sha256(bench)
        assert h_before != h_after

    def test_changes_on_size_change(self, tmp_path: Path) -> None:
        bench = tmp_path / "bench"
        shard = _make_shard(
            bench, "kwg", "sfid_a", "benchmark", '{"q":"x"}\n'
        )
        h_before = compute_benchmark_snapshot_sha256(bench)
        # Append more bytes — size changes, manifest hash should change.
        shard.write_text('{"q":"x"}\n{"q":"y"}\n')
        h_after = compute_benchmark_snapshot_sha256(bench)
        assert h_before != h_after

    def test_changes_on_same_size_content_edit(self, tmp_path: Path) -> None:
        """Pre-fix, the manifest was ``(relpath, size)`` only,
        so a same-size content edit was invisible to the snapshot
        hash. Post-fix, the manifest hashes file contents in 1 MiB
        chunks, so any byte change shifts the hash."""
        bench = tmp_path / "bench"
        shard = _make_shard(
            bench, "kwg", "sfid_a", "benchmark", '{"q":"x"}\n'
        )
        h_before = compute_benchmark_snapshot_sha256(bench)
        # Same length, different content.
        shard.write_text('{"q":"y"}\n')
        assert shard.stat().st_size == len('{"q":"x"}\n')  # invariant
        h_after = compute_benchmark_snapshot_sha256(bench)
        assert h_before != h_after, (
            "Same-size content edit was silently absorbed — the manifest "
            "isn't hashing file contents."
        )

    def test_ignores_non_jsonl_artefacts(self, tmp_path: Path) -> None:
        bench = tmp_path / "bench"
        _make_shard(bench, "kwg", "sfid_a", "benchmark", '{"q":"x"}\n')
        h_before = compute_benchmark_snapshot_sha256(bench)
        # A parquet sibling under the same KG/SFID must not change the hash.
        parquet_leaf = bench / "kwg" / "sfid_a" / "benchmark.parquet"
        parquet_leaf.mkdir(parents=True, exist_ok=True)
        (parquet_leaf / "part-00000.snappy.parquet").write_text("dummy")
        h_after = compute_benchmark_snapshot_sha256(bench)
        assert h_before == h_after, "parquet artefacts must not affect snapshot hash"

    def test_order_independent_of_filesystem(self, tmp_path: Path) -> None:
        # Build two benchmarks with shards added in opposite orders.
        bench_a = tmp_path / "bench_a"
        _make_shard(bench_a, "kwg", "sfid_a", "benchmark", "x")
        _make_shard(bench_a, "kwg", "sfid_b", "benchmark", "y")
        bench_b = tmp_path / "bench_b"
        _make_shard(bench_b, "kwg", "sfid_b", "benchmark", "y")
        _make_shard(bench_b, "kwg", "sfid_a", "benchmark", "x")
        # Same shards, same sizes → same hash regardless of creation order.
        assert (
            compute_benchmark_snapshot_sha256(bench_a)
            == compute_benchmark_snapshot_sha256(bench_b)
        )

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_benchmark_snapshot_sha256(tmp_path / "missing")


# ---------------------------------------------------------------------------
# AggregateReport carries the hashes
# ---------------------------------------------------------------------------


class TestReportCarriesHashes:
    def test_build_report_propagates_hashes(self) -> None:
        from benchmark_eval.aggregate import build_report
        from benchmark_eval.base import ScoreRecord

        rec = ScoreRecord(
            question_id="q1",
            tid="1.1",
            sfid="A",
            answer_type="Boolean (yes/no, true/false)",
            language="en",
            tier="plain",
            model_id="dummy",
            em=1, ne=0.0, covered=True,
            format_error=False, refusal=False,
        )
        report = build_report(
            [rec],
            alias_store_sha256="a" * 64,
            benchmark_snapshot_sha256="b" * 64,
        )
        assert report.alias_store_sha256 == "a" * 64
        assert report.benchmark_snapshot_sha256 == "b" * 64

    def test_build_report_defaults_to_none(self) -> None:
        from benchmark_eval.aggregate import build_report
        from benchmark_eval.base import ScoreRecord

        rec = ScoreRecord(
            question_id="q1",
            tid="1.1",
            sfid="A",
            answer_type="Boolean (yes/no, true/false)",
            language="en",
            tier="plain",
            model_id="dummy",
            em=1, ne=0.0, covered=True,
            format_error=False, refusal=False,
        )
        report = build_report([rec])
        assert report.alias_store_sha256 is None
        assert report.benchmark_snapshot_sha256 is None
