"""Tests for ``benchmark_models.io.iter_jsonl_rows``.

The helper reads either a single JSONL file or a directory of Spark
``part-*.json`` shards. It's used by the model runner's
``load_benchmark``, the eval pipeline's ``load_gold_rows``, and the
template-registry exporter — covering it here keeps all three honest
without duplicating tests downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_models.io import iter_jsonl_rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestIterJsonlRowsSingleFile:
    def test_reads_all_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"id": 1}, {"id": 2}, {"id": 3}])
        rows = list(iter_jsonl_rows(path))
        assert [r["id"] for r in rows] == [1, 2, 3]

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"id": 1}\n\n{"id": 2}\n   \n{"id": 3}\n'
        )
        rows = list(iter_jsonl_rows(path))
        assert [r["id"] for r in rows] == [1, 2, 3]

    def test_malformed_line_skipped_with_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text(
            '{"id": 1}\n{ broken json }\n{"id": 2}\n'
        )
        rows = list(iter_jsonl_rows(path))
        # Malformed line is skipped; valid rows are preserved.
        assert [r["id"] for r in rows] == [1, 2]


class TestIterJsonlRowsDirectory:
    """The benchmark share lays rows out as Spark part files under
    ``benchmark/<kg>/<sfid>/benchmark.jsonl/part-*.json``. The helper
    walks that layout recursively."""

    def test_reads_all_shards(self, tmp_path: Path) -> None:
        d = tmp_path / "benchmark.jsonl"
        d.mkdir()
        _write_jsonl(d / "part-00000-abc-c000.json", [{"id": 1}, {"id": 2}])
        _write_jsonl(d / "part-00001-abc-c000.json", [{"id": 3}])
        rows = list(iter_jsonl_rows(d))
        # Sorted lexicographically, so part-00000 yields first.
        assert [r["id"] for r in rows] == [1, 2, 3]

    def test_walks_recursively_when_opted_in(self, tmp_path: Path) -> None:
        # Nested layout to mirror benchmark/<kg>/<sfid>/benchmark.jsonl/
        d = tmp_path / "share" / "kwg" / "sfid_b" / "benchmark.jsonl"
        d.mkdir(parents=True)
        _write_jsonl(d / "part-00000.json", [{"id": "a"}])
        # ``recursive=True`` is the opt-in for the registry-exporter
        # use case; the default (leaf-only) is what the runner / eval
        # loaders rely on to avoid mixing benchmark + gold_triples shards.
        rows = list(iter_jsonl_rows(tmp_path / "share", recursive=True))
        assert [r["id"] for r in rows] == ["a"]

    def test_leaf_only_default_rejects_parent_dir(self, tmp_path: Path) -> None:
        # Parent of a Spark shard dir — leaf-only refuses (would
        # otherwise silently mix sibling shard schemas).
        d = tmp_path / "share" / "kwg" / "sfid_b" / "benchmark.jsonl"
        d.mkdir(parents=True)
        _write_jsonl(d / "part-00000.json", [{"id": "a"}])
        with pytest.raises(ValueError, match="leaf-only"):
            list(iter_jsonl_rows(tmp_path / "share"))

    def test_leaf_only_default_skips_sibling_shard_dirs(
        self, tmp_path: Path
    ) -> None:
        # The actual hazard: benchmark.jsonl/ and gold_triples.jsonl/
        # under the same parent. Leaf-only on the parent rejects;
        # leaf-only on the leaf returns only that schema's rows.
        bench = tmp_path / "sfid_b" / "benchmark.jsonl"
        bench.mkdir(parents=True)
        _write_jsonl(bench / "part-00000.json", [{"schema": "bench"}])
        triples = tmp_path / "sfid_b" / "gold_triples.jsonl"
        triples.mkdir()
        _write_jsonl(triples / "part-00000.json", [{"schema": "triples"}])
        # Leaf path: only the matching schema's rows.
        rows = list(iter_jsonl_rows(bench))
        assert [r["schema"] for r in rows] == ["bench"]
        # Parent path: refused (would otherwise yield both schemas).
        with pytest.raises(ValueError, match="leaf-only"):
            list(iter_jsonl_rows(tmp_path / "sfid_b"))

    def test_accepts_jsonl_extension(self, tmp_path: Path) -> None:
        d = tmp_path / "data"
        d.mkdir()
        _write_jsonl(d / "part-00000.jsonl", [{"id": 1}])
        rows = list(iter_jsonl_rows(d))
        assert rows == [{"id": 1}]

    def test_empty_directory_raises_value_error(self, tmp_path: Path) -> None:
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(ValueError, match="No part-"):
            list(iter_jsonl_rows(d))

    def test_missing_path_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list(iter_jsonl_rows(tmp_path / "does_not_exist"))

    def test_deterministic_order_across_runs(self, tmp_path: Path) -> None:
        d = tmp_path / "data"
        d.mkdir()
        _write_jsonl(d / "part-00002.json", [{"id": 3}])
        _write_jsonl(d / "part-00000.json", [{"id": 1}])
        _write_jsonl(d / "part-00001.json", [{"id": 2}])
        rows = list(iter_jsonl_rows(d))
        assert [r["id"] for r in rows] == [1, 2, 3]
