"""Tests for :mod:`benchmark_eval.aliases`.

Covers:

* Programmatic construction via :meth:`AliasStore.from_entries`.
* Loading from the committed mock JSON (``output/entity_aliases.json``).
* Resolving exact / canonicalized / multilingual / unknown names.
* Collision behavior (``gold_canonical_id`` preference).
* Non-string inputs, empty strings, ``None``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_eval.aliases import AliasEntry, AliasStore


@pytest.fixture
def tiny_store() -> AliasStore:
    """A handcrafted :class:`AliasStore` with known collisions and
    multilingual entries.

    Two OSM entities share the spelling ``"Berlin"`` (CT, US vs
    Germany) — this fixture lets us exercise the ``gold_canonical_id``
    preference without depending on the real data.
    """
    return AliasStore.from_entries([
        AliasEntry(
            canonical_id="osm:berlin-de",
            display_name="Berlin",
            aliases={"Berlin", "Berlin, Germany", "Berlín", "柏林"},
        ),
        AliasEntry(
            canonical_id="osm:berlin-ct",
            display_name="Berlin",
            aliases={"Berlin", "Berlin, CT"},
        ),
        AliasEntry(
            canonical_id="cardinal:N",
            display_name="N",
            aliases={"N", "North", "Nord", "Norte"},
        ),
        AliasEntry(
            canonical_id="predef:left",
            display_name="left",
            aliases={"left", "gauche", "izquierda"},
        ),
    ])


class TestAliasStoreResolveBasic:
    def test_exact_match(self, tiny_store: AliasStore) -> None:
        # Without a gold hint, "Berlin" collides → returns the first
        # candidate in **sorted** id order (m11). Sorted is stable
        # across runs and across PYTHONHASHSEED variations; insertion
        # order would have depended on construction history.
        # ``osm:berlin-ct`` < ``osm:berlin-de`` lexicographically.
        assert tiny_store.resolve("Berlin") == "osm:berlin-ct"

    def test_canonicalized_match(self, tiny_store: AliasStore) -> None:
        # Case + whitespace should not matter.
        assert tiny_store.resolve("  BERLIN  ") == "osm:berlin-ct"

    def test_multilingual(self, tiny_store: AliasStore) -> None:
        assert tiny_store.resolve("柏林") == "osm:berlin-de"
        assert tiny_store.resolve("Berlín") == "osm:berlin-de"

    def test_cardinal(self, tiny_store: AliasStore) -> None:
        assert tiny_store.resolve("North") == "cardinal:N"
        assert tiny_store.resolve("Nord") == "cardinal:N"
        assert tiny_store.resolve("N") == "cardinal:N"

    def test_predefined(self, tiny_store: AliasStore) -> None:
        assert tiny_store.resolve("gauche") == "predef:left"
        assert tiny_store.resolve("izquierda") == "predef:left"


class TestAliasStoreResolveCollision:
    def test_gold_wins_on_collision(self, tiny_store: AliasStore) -> None:
        """When two canonical ids share an alias, the gold hint wins."""
        assert (
            tiny_store.resolve("Berlin", gold_canonical_id="osm:berlin-ct")
            == "osm:berlin-ct"
        )
        assert (
            tiny_store.resolve("Berlin", gold_canonical_id="osm:berlin-de")
            == "osm:berlin-de"
        )

    def test_gold_hint_without_alias_match_returns_none(
        self, tiny_store: AliasStore
    ) -> None:
        """Gold hint only tiebreaks — it doesn't make an unknown name resolve."""
        assert (
            tiny_store.resolve("Nonexistent", gold_canonical_id="osm:berlin-de")
            is None
        )


class TestAliasStoreResolveEdgeCases:
    def test_unknown_returns_none(self, tiny_store: AliasStore) -> None:
        assert tiny_store.resolve("Atlantis") is None

    def test_none_input(self, tiny_store: AliasStore) -> None:
        assert tiny_store.resolve(None) is None

    def test_empty_string(self, tiny_store: AliasStore) -> None:
        assert tiny_store.resolve("") is None
        assert tiny_store.resolve("   ") is None

    def test_non_string_input(self, tiny_store: AliasStore) -> None:
        assert tiny_store.resolve(42) is None  # type: ignore[arg-type]
        assert tiny_store.resolve([1, 2]) is None  # type: ignore[arg-type]


class TestAliasStoreLoadFromCommittedMock:
    """Shape + content assertions against the committed alias store.

    The alias store file is currently a minimal placeholder keyed on
    whatever entity names the 28-row sample carries. The cardinal /
    predefined / multilingual assertions below describe what the
    *real* alias file (arriving later, built from the full benchmark)
    must contain. They skip on the placeholder and start running once
    the real file lands.
    """

    ALIAS_FILE_SENTINEL = "North"

    def _load(self) -> AliasStore:
        path = Path(__file__).parents[2] / "output" / "entity_aliases.json"
        if not path.exists():
            pytest.skip(f"{path} not committed yet")
        store = AliasStore.load(path)
        # The sentinel isn't in the placeholder store built from a
        # 28-row sample. Skip if we're still on the placeholder.
        if store.resolve(self.ALIAS_FILE_SENTINEL) is None:
            pytest.skip(
                "Placeholder alias file in use — real alias file with "
                "cardinal / predefined / OSM-entity coverage not yet "
                "committed."
            )
        return store

    def test_loads_and_nonempty(self) -> None:
        path = Path(__file__).parents[2] / "output" / "entity_aliases.json"
        if not path.exists():
            pytest.skip(f"{path} not committed yet")
        store = AliasStore.load(path)
        assert len(store) > 0

    def test_contains_cardinals(self) -> None:
        store = self._load()
        for direction in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
            assert store.resolve(direction) == f"cardinal:{direction}"

    def test_cardinal_multilingual_resolves(self) -> None:
        store = self._load()
        assert store.resolve("North") == "cardinal:N"
        assert store.resolve("Nord") == "cardinal:N"

    def test_contains_predefined_values(self) -> None:
        store = self._load()
        assert store.resolve("left") == "predef:left"
        assert store.resolve("gauche") == "predef:left"
        assert store.resolve("upstream") == "predef:upstream"

    def test_contains_osm_entities_from_sample_benchmark(self) -> None:
        store = self._load()
        assert store.resolve("Cafe Kanel") == "osm:2476384356"
        assert store.resolve("Seniorenzentrum St. Michael") == "osm:2068224"


class TestAliasStoreMissingFile:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            AliasStore.load(tmp_path / "does_not_exist.json")

    def test_wrong_shape_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('["not", "an", "object"]')
        with pytest.raises(ValueError):
            AliasStore.load(path)


class TestAliasStoreForLanguage:
    """``AliasStore.for_language`` — multilingual entry point."""

    def _write_lang(
        self, dir_: Path, lang: str, body: dict[str, dict[str, object]]
    ) -> None:
        import json

        dir_.mkdir(parents=True, exist_ok=True)
        (dir_ / f"{lang}.json").write_text(json.dumps(body))

    def test_loads_target_language_when_present(
        self, tmp_path: Path
    ) -> None:
        self._write_lang(
            tmp_path, "deu",
            {"osm:1": {"canonical": "Berlin", "aliases": ["Berlín"]}},
        )
        store = AliasStore.for_language("deu", aliases_dir=tmp_path)
        assert store.resolve("Berlin") == "osm:1"
        assert store.resolve("Berlín") == "osm:1"

    def test_falls_back_to_en_when_lang_missing(
        self, tmp_path: Path
    ) -> None:
        # Only ``eng.json`` exists.
        self._write_lang(
            tmp_path, "eng",
            {"osm:1": {"canonical": "Berlin", "aliases": []}},
        )
        store = AliasStore.for_language(
            "jpn", aliases_dir=tmp_path, fallback="eng",
        )
        assert store.resolve("Berlin") == "osm:1"

    def test_no_lang_no_fallback_raises(
        self, tmp_path: Path
    ) -> None:
        # Neither deu.json nor eng.json exists.
        with pytest.raises(FileNotFoundError, match="No alias store"):
            AliasStore.for_language("deu", aliases_dir=tmp_path)

    def test_explicit_no_fallback(self, tmp_path: Path) -> None:
        # eng.json exists but caller passed fallback=None.
        self._write_lang(
            tmp_path, "eng",
            {"osm:1": {"canonical": "Berlin", "aliases": []}},
        )
        with pytest.raises(FileNotFoundError):
            AliasStore.for_language(
                "jpn", aliases_dir=tmp_path, fallback=None,
            )

    def test_target_present_does_not_load_fallback(
        self, tmp_path: Path
    ) -> None:
        # Both files exist; target lang should win.
        self._write_lang(
            tmp_path, "deu",
            {"osm:de": {"canonical": "deutsch", "aliases": []}},
        )
        self._write_lang(
            tmp_path, "eng",
            {"osm:en": {"canonical": "english", "aliases": []}},
        )
        store = AliasStore.for_language("deu", aliases_dir=tmp_path)
        assert "osm:de" in store.entries
        assert "osm:en" not in store.entries

    def test_aliases_empty_works_for_lookup(self, tmp_path: Path) -> None:
        """Q1 policy: ``aliases = []`` still resolves via the
        canonical/display name (auto-indexed in
        :meth:`AliasStore.from_entries`)."""
        self._write_lang(
            tmp_path, "deu",
            {"osm:1": {"canonical": "Berlin", "aliases": []}},
        )
        store = AliasStore.for_language("deu", aliases_dir=tmp_path)
        assert store.resolve("Berlin") == "osm:1"
