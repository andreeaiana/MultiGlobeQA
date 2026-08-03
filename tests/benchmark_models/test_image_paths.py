"""Tests for :func:`benchmark_models.image_paths.resolve_image_path`."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_models.image_paths import resolve_image_path


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    """Build a minimal multimodal dataset tree under tmp_path.

    Layout mirrors the published convention:

        <root>/multimodal/<kg>/images/<file>
        <root>/multimodal/<kg>/<lang>/benchmark.jsonl/part-*.json
    """
    for kg in ("osmh3kg", "worldkg"):
        (tmp_path / "multimodal" / kg / "images").mkdir(parents=True)
        for lang in ("eng", "deu"):
            (tmp_path / "multimodal" / kg / lang / "benchmark.jsonl").mkdir(
                parents=True
            )
    return tmp_path


class TestCleanForm:
    """Stored path is the published convention: ``images/<file>``."""

    def test_resolves_against_dataset_root(self, dataset_root: Path) -> None:
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        out = resolve_image_path("images/abc.jpg", shard, "osmh3kg")
        assert out == str(
            dataset_root / "multimodal" / "osmh3kg" / "images" / "abc.jpg"
        )

    def test_uses_kg_source_not_path(self, dataset_root: Path) -> None:
        # source_path is under osmh3kg/, but row says worldkg → kg_source wins.
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        out = resolve_image_path("images/x.jpg", shard, "worldkg")
        assert out == str(
            dataset_root / "multimodal" / "worldkg" / "images" / "x.jpg"
        )

    def test_no_subdir_prefix(self, dataset_root: Path) -> None:
        # If the stored path is just a filename (no "images/"), the resolver
        # composes <root>/multimodal/<kg>/<filename> as documented.
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        out = resolve_image_path("abc.jpg", shard, "osmh3kg")
        assert out == str(dataset_root / "multimodal" / "osmh3kg" / "abc.jpg")

    def test_works_for_eng_layout_without_lang_subdir(
        self, dataset_root: Path
    ) -> None:
        # Some trees (raw_eng/) have <kg>/benchmark.jsonl/ directly,
        # no per-lang subdir.
        shard = dataset_root / "multimodal" / "osmh3kg" / "benchmark.jsonl"
        shard.mkdir(parents=True, exist_ok=True)
        out = resolve_image_path("images/a.jpg", shard, "osmh3kg")
        assert out == str(
            dataset_root / "multimodal" / "osmh3kg" / "images" / "a.jpg"
        )


class TestLegacyForm:
    """Stored path matches the generator's stub ``benchmark/eng/multimodal/<kg>/...``."""

    def test_strips_legacy_prefix(self, dataset_root: Path) -> None:
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        out = resolve_image_path(
            "benchmark/eng/multimodal/osmh3kg/images/abc.jpg",
            shard,
            "osmh3kg",
        )
        assert out == str(
            dataset_root / "multimodal" / "osmh3kg" / "images" / "abc.jpg"
        )

    def test_replaces_legacy_kg_with_kg_source(self, dataset_root: Path) -> None:
        # Even if the legacy path says osmh3kg, the row's kg_source wins.
        shard = dataset_root / "multimodal" / "worldkg" / "eng" / "benchmark.jsonl"
        out = resolve_image_path(
            "benchmark/eng/multimodal/osmh3kg/images/abc.jpg",
            shard,
            "worldkg",
        )
        assert out == str(
            dataset_root / "multimodal" / "worldkg" / "images" / "abc.jpg"
        )

    def test_handles_legacy_form_without_top_prefix(
        self, dataset_root: Path
    ) -> None:
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        out = resolve_image_path(
            "multimodal/osmh3kg/images/foo.png", shard, "osmh3kg"
        )
        assert out == str(
            dataset_root / "multimodal" / "osmh3kg" / "images" / "foo.png"
        )


class TestEdgeCases:
    def test_empty_stored_returns_empty(self, dataset_root: Path) -> None:
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        assert resolve_image_path("", shard, "osmh3kg") == ""

    def test_empty_kg_source_returns_stored(self, dataset_root: Path) -> None:
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        assert (
            resolve_image_path("images/abc.jpg", shard, "")
            == "images/abc.jpg"
        )

    def test_absolute_stored_path_passes_through(
        self, dataset_root: Path, tmp_path: Path
    ) -> None:
        shard = dataset_root / "multimodal" / "osmh3kg" / "eng" / "benchmark.jsonl"
        abs_path = str(tmp_path / "elsewhere" / "x.jpg")
        assert resolve_image_path(abs_path, shard, "osmh3kg") == abs_path

    def test_source_path_without_multimodal_ancestor_returns_stored(
        self, tmp_path: Path
    ) -> None:
        # No "multimodal" anywhere in source_path → cannot discover the
        # dataset root; preserve the stored value so the caller's
        # FileNotFoundError carries the original path.
        unrelated = tmp_path / "some" / "unrelated" / "place"
        unrelated.mkdir(parents=True)
        assert (
            resolve_image_path("images/a.jpg", unrelated, "osmh3kg")
            == "images/a.jpg"
        )

    def test_innermost_multimodal_wins(self, tmp_path: Path) -> None:
        # Pathological layout: "multimodal" appears twice. The innermost
        # (closest to the shard) is the correct dataset root.
        inner = (
            tmp_path
            / "multimodal"  # outer (decoy)
            / "wrapper"
            / "multimodal"  # inner — this is the real root marker
            / "osmh3kg"
            / "eng"
            / "benchmark.jsonl"
        )
        inner.mkdir(parents=True)
        out = resolve_image_path("images/a.jpg", inner, "osmh3kg")
        # Dataset root is the parent of the innermost ``multimodal/``.
        expected_root = tmp_path / "multimodal" / "wrapper"
        assert out == str(
            expected_root / "multimodal" / "osmh3kg" / "images" / "a.jpg"
        )
