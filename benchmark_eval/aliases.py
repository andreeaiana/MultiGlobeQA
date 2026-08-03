"""Alias-backed entity / predefined-value / cardinal-direction resolution.

For answer types that name an entity (``Entity name (single)``, the
two ``Predefined value`` types, and ``Cardinal direction``), a
prediction can be expressed in many equivalent forms:

* Different languages (``"Berlin"`` / ``"Berlín"`` / ``"柏林"``).
* Different spellings (``"St. Michael"`` / ``"Sankt Michael"``).
* Different cardinal-direction abbreviations (``"N"`` / ``"North"``
  / ``"Nord"``).

:class:`AliasStore` is a simple keyed lookup: each canonical id
(e.g. ``"osm:12147663074"`` or ``"cardinal:N"``) maps to a set of
accepted names, and :meth:`resolve` returns the canonical id (or
``None``) for a free-text prediction.

The JSON file format is flat and easy to edit by hand. Each top-
level key is a canonical id; each value has an ``aliases`` list
and an optional ``canonical`` display name. A ``languages`` subfield
is optional and currently unused at lookup time (canonicalization
collapses language variants into one lookup index), but is
preserved so tools can regenerate the set of aliases from a
filtered subset.

Multiple canonical ids may share the same alias. In that case
:meth:`resolve` returns the id that equals ``gold_canonical_id`` if
present, otherwise the first id in insertion order. This handles
benign ambiguity (two entities accidentally sharing a name) without
making the eval fail loudly.

Entity hex strings (``Entity hex string (single)``) bypass the
alias store and go through strict string equality per the plan
decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark_eval.canonicalize import canonicalize_string


@dataclass
class AliasEntry:
    """One canonical entry in an :class:`AliasStore`."""

    canonical_id: str
    """Stable identifier used by the benchmark / KG
    (e.g., ``"osm:12147663074"``, ``"cardinal:N"``, ``"predef:left"``)."""
    display_name: str | None = None
    """Human-readable label for the canonical id, preserved through
    loads even though :meth:`resolve` doesn't use it."""
    aliases: set[str] = field(default_factory=set)
    """All accepted names for this id, stored in their original
    form. Lookup keys are built by canonicalizing each alias."""


@dataclass
class AliasStore:
    """A loaded alias lookup.

    Construction is via :meth:`load` from a JSON file; programmatic
    construction via :meth:`from_entries` is also available for
    tests.
    """

    entries: dict[str, AliasEntry]
    """Canonical id → :class:`AliasEntry`."""
    _by_canonical_alias: dict[str, list[str]]
    """Canonicalized alias string → list of canonical ids that
    claim that alias (in insertion order). Lookup here is O(1)."""

    @classmethod
    def for_language(
        cls,
        lang: str,
        *,
        aliases_dir: Path | str = "output/aliases",
        fallback: str | None = "en",
    ) -> AliasStore:
        """Load the per-language alias store for ``lang``.

        Loads ``<aliases_dir>/<lang>.json`` if present (built by
        ``scripts/build_alias_stores.py`` from the multilingual
        entity-name parquets). When the requested language file is
        missing, falls back to ``<aliases_dir>/<fallback>.json``
        (English source by default) so the eval still runs on
        languages whose target-lang store hasn't been built yet.

        Raises:
            FileNotFoundError: if neither ``<lang>.json`` nor the
                fallback file exists.
        """
        aliases_dir = Path(aliases_dir)
        target = aliases_dir / f"{lang}.json"
        if target.exists():
            return cls.load(target)
        if fallback is not None:
            fb = aliases_dir / f"{fallback}.json"
            if fb.exists():
                # Fall through silently — caller wants an alias store
                # at any cost. Logging a warning would be too noisy
                # for the common case where the eval pipeline iterates
                # over multiple languages and only some have stores.
                return cls.load(fb)
        raise FileNotFoundError(
            f"No alias store for language {lang!r} at {target} and no "
            f"fallback {fallback!r} either. Build with "
            f"``scripts.build_alias_stores`` first."
        )

    @classmethod
    def load(cls, path: Path | str) -> AliasStore:
        """Load an :class:`AliasStore` from a JSON file.

        Expected format::

            {
              "osm:12147663074": {
                "canonical": "Cafe Kanel",
                "aliases": ["Cafe Kanel", "Café Kanel"]
              },
              "cardinal:N": {
                "canonical": "N",
                "aliases": ["N", "North", "Nord", "Norte"]
              }
            }

        Unknown / extra keys are ignored. A missing ``aliases`` list
        is treated as an empty list. An empty file is a valid empty
        store.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Alias store file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Alias store {path} must be a JSON object keyed by "
                f"canonical id; got {type(raw).__name__}."
            )
        entries: dict[str, Any] = {}
        for canonical_id, body in raw.items():
            if not isinstance(body, dict):
                continue
            entries[canonical_id] = AliasEntry(
                canonical_id=canonical_id,
                display_name=body.get("canonical"),
                aliases=set(body.get("aliases", [])),
            )
        return cls.from_entries(list(entries.values()))

    @classmethod
    def from_entries(cls, entries: list[AliasEntry]) -> AliasStore:
        """Build an :class:`AliasStore` from a list of entries.

        m11: candidate-id lists per alias are *sorted* (deterministic
        across runs) rather than left in insertion order. Insertion
        order depends on JSON file order and on how the entries list
        was assembled — runs against an alias file edited between two
        commits could otherwise resolve the same prediction to
        different canonical ids on different days. Sorted order is
        stable and reproducible.
        """
        by_alias: dict[str, list[str]] = {}
        entries_dict: dict[str, AliasEntry] = {}
        for entry in entries:
            entries_dict[entry.canonical_id] = entry
            candidates = set(entry.aliases)
            if entry.display_name:
                candidates.add(entry.display_name)
            candidates.add(entry.canonical_id)
            for alias in candidates:
                key = canonicalize_string(alias)
                if not key:
                    continue
                by_alias.setdefault(key, []).append(entry.canonical_id)
        # Deterministic order across runs.
        for key in by_alias:
            by_alias[key] = sorted(set(by_alias[key]))
        return cls(entries=entries_dict, _by_canonical_alias=by_alias)

    def __len__(self) -> int:
        return len(self.entries)

    def resolve(
        self,
        prediction: Any,
        gold_canonical_id: str | None = None,
    ) -> str | None:
        """Return the canonical id that ``prediction`` refers to, or ``None``.

        Lookup order:

        1. Canonicalize ``prediction``.
        2. If the canonicalized key exists in the alias index, pick
           an id from the candidate list: ``gold_canonical_id`` if
           present (handles alias collisions in favor of the gold),
           otherwise the first id in insertion order.
        3. Otherwise return ``None``. The scorer treats ``None`` as
           ``em=0`` and flags ``resolve_failure=True`` so the
           aggregator can compute ``resolve_failure_rate``.

        Non-string predictions (``None``, numbers, etc.) short-circuit
        to ``None``.
        """
        if prediction is None or not isinstance(prediction, str):
            return None
        key = canonicalize_string(prediction)
        if not key:
            return None
        candidates = self._by_canonical_alias.get(key)
        if not candidates:
            return None
        if gold_canonical_id is not None and gold_canonical_id in candidates:
            return gold_canonical_id
        return candidates[0]

    def all_canonical_ids(self) -> list[str]:
        """Return every canonical id known to the store."""
        return list(self.entries.keys())


__all__ = [
    "AliasEntry",
    "AliasStore",
]
