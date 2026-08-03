"""Build per-language entity-alias JSONs for the eval pipeline.

The 2026-05-07 share carries multilingual entity-name dumps under
``benchmark/raw_eng/entity_names/{osmh3kg,worldkg}_entity_names.parquet``
(36M + 17M rows total). These are not in
:class:`benchmark_eval.aliases.AliasStore` shape directly. This
script transforms them into 17 per-language alias JSON files
(``output/aliases/<iso639-1>.json``) — one per target language plus
English source — that the eval pipeline loads via
``AliasStore.for_language``.

Algorithm (per target language ``L``):

1. Read both entity-names parquets (column-pruned: just
   ``entity_id``, ``lang``, ``name``, ``source_predicate``).
2. Strip a single matched leading/trailing ``"…"`` pair from each
   WorldKG name (the source data wraps every name in literal
   quote characters; OSMH3KG names are clean).
3. Group rows by ``(kg_prefix, entity_id)``.
4. **Pick the canonical name** with the priority cascade:
       target lang ``L`` → ``en`` → ``default``
   Within a language, prefer the most-canonical-looking
   ``source_predicate`` (e.g. ``name`` / ``rdfs:label`` over
   ``alt_name`` / ``oldName``); ties broken by sorted predicate
   for determinism.
5. **Aliases**: every other (lang, source_predicate, name) tuple's
   name, deduplicated, with the chosen canonical removed. When the
   set is empty, ``aliases = []`` (Q1 decision: empty when no
   variants — saves ~15-25% file size at our scale).
6. **kwg fallback**: kwg has no entity-names parquet (its IDs are
   semantic URIs like ``zipCodeArea.12054`` that double as the
   display name). Pull every distinct ``(entity_<letter>_id,
   entity_<letter>_name)`` pair from
   ``benchmark/raw_eng/kwg/sfid_*/benchmark.jsonl/`` and emit one
   ``"kwg:<id>": {"canonical": <name>, "aliases": []}`` entry into
   every per-language JSON (the same canonical across all 17
   languages — kwg has no per-language names).
7. Output JSON has stable sorted-key ordering for reproducibility.

Coverage (per target language) varies wildly:

* OSMH3KG: 16 / 16 target languages have entries (538 distinct
  ``lang`` values overall, with ja=3.4M, es=228k, de=193k as the
  high-coverage end and sq=702 as the lowest).
* WorldKG: 5 / 16 (de, fr, ja, ru, vi). The other 11 fall back to
  English / default.

Usage::

    uv run python -m scripts.build_alias_stores

    # Single language (debugging)
    uv run python -m scripts.build_alias_stores --languages en de

    # Custom output dir
    uv run python -m scripts.build_alias_stores --output-dir /tmp/aliases
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import click
import pyarrow.parquet as pq
from loguru import logger

from benchmark_models.io import iter_jsonl_rows


# Target languages (ISO 639-3) — output filename = key + ".json".
# Names match the potato annotation directory convention, with
# ``por_PT`` / ``zho_CN`` keeping the regional suffix. Hindi was
# dropped per the multilingual benchmark plan; see
# ``project_hindi_dropped`` memory entry.
TARGET_LANGUAGES: tuple[str, ...] = (
    "bul", "deu", "ell", "spa", "fra", "ita", "jpn", "kat",
    "por_PT", "ron", "rus", "sqi", "tur", "urd", "vie", "zho_CN",
)

# English is always built as a 17th store (the source-language
# baseline; covers entities for which we have no target-lang names).
EN_LANGUAGE: str = "eng"

ALL_LANGUAGES: tuple[str, ...] = TARGET_LANGUAGES + (EN_LANGUAGE,)


# Bridge: each ISO 639-3 output key → set of ``lang`` values that
# the entity_names parquet uses (which are ISO 639-1 / BCP-47 in
# lowercased form). The OSMH3KG and WorldKG entity_names data
# predates this rename, so the input side stays ISO 639-1; only the
# output side changes.
_LANG_ALIASES: dict[str, tuple[str, ...]] = {
    "bul": ("bg",),
    "deu": ("de",),
    "ell": ("el",),
    "spa": ("es",),
    "fra": ("fr",),
    "ita": ("it",),
    "jpn": ("ja",),
    "kat": ("ka",),
    "por_PT": ("pt-PT", "pt-pt", "pt"),
    "ron": ("ro",),
    "rus": ("ru",),
    "sqi": ("sq",),
    "tur": ("tr",),
    "urd": ("ur",),
    "vie": ("vi",),
    "zho_CN": ("zh-CN", "zh-cn", "zh", "zh-hans"),
    # English in the entity_names parquet uses bare "en".
    "eng": ("en",),
}


def _lang_match_set(target: str) -> tuple[str, ...]:
    """Return the set of entity_names ``lang`` values that count as
    ``target`` for canonical-name selection."""
    return _LANG_ALIASES.get(target, (target,))


# Predicate-priority for canonical-name selection within a language.
# Lower number = stronger preference. The OSMH3KG side uses bare
# OSM tag names (``name``, ``alt_name``, ...); WorldKG uses URIs
# (``http://www.worldkg.org/schema/name<Lang>`` / ``rdfs:label``).
_PRED_PRIORITY: dict[str, int] = {
    # OSMH3KG primary
    "name": 0,
    # WorldKG primary
    "http://www.w3.org/2000/01/rdf-schema#label": 0,
    # OSMH3KG official / native variants — count as canonical-tier
    "official_name": 1,
    "loc_name": 2,
    # WorldKG per-lang Capitalised forms (nameEn, nameDe, …) —
    # canonical-tier within that language
    # All other ``http://www.worldkg.org/schema/name*`` URIs match
    # this pattern via the predicate-prefix check below.
    # Secondary
    "alt_name": 3,
    "http://www.worldkg.org/schema/altName": 3,
    "short_name": 4,
    # Historical / other
    "old_name": 5,
    "http://www.worldkg.org/schema/oldName": 5,
}

_WORLDKG_NAME_PRED_RE = re.compile(
    r"^http://www\.worldkg\.org/schema/name[A-Z][a-zA-Z]*$"
)


def _predicate_priority(pred: str) -> int:
    """Lower = preferred when picking a canonical name."""
    if pred in _PRED_PRIORITY:
        return _PRED_PRIORITY[pred]
    if _WORLDKG_NAME_PRED_RE.match(pred):
        # ``worldkg:nameEn`` / ``nameDe`` / ``nameRu`` — primary
        # within the row's ``lang``.
        return 0
    if pred.startswith("name:"):
        # OSMH3KG ``name:de`` / ``name:ja`` — primary within lang.
        return 0
    # Unknown predicates land last so they never override a known
    # primary predicate.
    return 100


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _NameRecord:
    """One (lang, name, source_predicate) tuple for an entity."""

    lang: str
    name: str
    source_predicate: str


# ---------------------------------------------------------------------------
# Loading + grouping
# ---------------------------------------------------------------------------


def _strip_quote_pair(s: str) -> str:
    """Strip exactly one matched leading/trailing ``"…"`` pair."""
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


def _load_entity_names(
    path: Path,
    *,
    strip_quotes: bool,
    keep_only: set[str] | None = None,
) -> dict[str, list[_NameRecord]]:
    """Read one entity-names parquet and group records by ``entity_id``.

    Column-pruned read; result is one dict entry per distinct
    ``entity_id`` mapping to a list of ``_NameRecord`` (preserving
    parquet row order for determinism). When ``strip_quotes`` is
    ``True``, each name has one matched ``"…"`` pair stripped — the
    WorldKG quote pathology fix applied at load-time.

    ``keep_only`` filters to a set of entity ids; rows for entities
    not in the set are dropped at load time. Drops storage from
    ~13M entries to ~30k for the benchmark-referenced subset, and
    prevents materialising the full parquet's grouped dict in
    memory.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing entity-names parquet: {path}")

    logger.info("loading {} ...", path.name)
    table = pq.read_table(
        path, columns=["entity_id", "lang", "name", "source_predicate"]
    )
    logger.info("  {:,} rows", table.num_rows)

    grouped: dict[str, list[_NameRecord]] = defaultdict(list)
    eid_col = table.column("entity_id").to_pylist()
    lang_col = table.column("lang").to_pylist()
    name_col = table.column("name").to_pylist()
    pred_col = table.column("source_predicate").to_pylist()
    # Pre-compute the geo-stripped form of each filter id so a single
    # parquet eid can match either bare or geo<digits> forms.
    if keep_only is not None:
        keep_with_geo = set(keep_only)
        for kid in list(keep_only):
            if kid.startswith("geo") and kid[3:].isdigit():
                keep_with_geo.add(kid[3:])
            elif kid.isdigit():
                keep_with_geo.add(f"geo{kid}")
    else:
        keep_with_geo = None
    for eid, lang, name, pred in zip(eid_col, lang_col, name_col, pred_col):
        if eid is None or name is None:
            continue
        eid_str = str(eid)
        if keep_with_geo is not None and eid_str not in keep_with_geo:
            continue
        clean_name = _strip_quote_pair(name) if strip_quotes else name
        if not clean_name:
            continue
        grouped[eid_str].append(_NameRecord(
            lang=str(lang) if lang is not None else "default",
            name=str(clean_name),
            source_predicate=str(pred) if pred is not None else "",
        ))
    logger.info(
        "  {:,} distinct entity_ids in {} (filter applied: {})",
        len(grouped), path.name,
        keep_only is not None,
    )
    return grouped


# ---------------------------------------------------------------------------
# Canonical-name selection
# ---------------------------------------------------------------------------


def _pick_canonical(
    records: Iterable[_NameRecord],
    target_lang: str,
) -> str | None:
    """Return the canonical name for an entity in ``target_lang``, or
    ``None`` if no candidate exists in target / en / default langs.

    Priority cascade: target → ``en`` → ``default``. Ties within a
    language broken by lowest ``_predicate_priority``, then by sorted
    name (deterministic).
    """
    cascade: list[tuple[str, ...]] = [
        _lang_match_set(target_lang),
        ("en",),
        ("default",),
    ]
    by_lang: dict[str, list[_NameRecord]] = defaultdict(list)
    for r in records:
        by_lang[r.lang].append(r)

    for lang_set in cascade:
        candidates: list[_NameRecord] = []
        for l in lang_set:
            candidates.extend(by_lang.get(l, ()))
        if not candidates:
            continue
        # Stable canonical pick: lowest predicate priority, then
        # alphabetical name tie-break.
        candidates.sort(key=lambda r: (_predicate_priority(r.source_predicate), r.name))
        return candidates[0].name
    return None


def _aliases_excluding_canonical(
    records: Iterable[_NameRecord],
    canonical: str,
) -> list[str]:
    """Return the deduplicated, sorted list of names other than
    ``canonical``. Q1 policy: empty list when no variants exist."""
    other = {r.name for r in records if r.name != canonical}
    return sorted(other)


# ---------------------------------------------------------------------------
# Benchmark-referenced entity ids
# ---------------------------------------------------------------------------


def _collect_benchmark_entity_ids(
    benchmark_root: Path,
) -> tuple[set[str], set[str]]:
    """Walk every benchmark.jsonl shard and collect distinct entity ids
    referenced as ``entity_<letter>_id``, partitioned by KG.

    Returns ``(osmh3kg_ids, worldkg_ids)``. kwg ids are handled
    separately by :func:`_collect_kwg_entities` (same source, but
    we also need the per-row name there).

    This is the filter set: the alias-store output only includes
    entities that appear in at least one benchmark row, dropping the
    ~99% of entity-names parquet entries that are never referenced
    by the benchmark. Output size goes from ~20 GB total to ~50 MB
    total across the 17 per-language JSONs.
    """
    osmh3kg_ids: set[str] = set()
    worldkg_ids: set[str] = set()

    def _walk(root: Path, target_set: set[str]) -> None:
        if not root.is_dir():
            return
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            # Both per-SFID dirs (regular shares) and flat KG roots
            # (multimodal share) carry a ``benchmark.jsonl/`` dir.
            bj = entry / "benchmark.jsonl"
            if not bj.is_dir():
                # Maybe this *is* the multimodal flat layout.
                if entry.name == "benchmark.jsonl":
                    bj = entry
                else:
                    continue
            for row in iter_jsonl_rows(bj):
                for letter in ("a", "b", "c", "d"):
                    eid = row.get(f"entity_{letter}_id")
                    if eid:
                        target_set.add(str(eid))

    _walk(benchmark_root / "osmh3kg", osmh3kg_ids)
    _walk(benchmark_root / "worldkg", worldkg_ids)
    # Multimodal subset adds entity ids too (each KG has its own).
    mm_root = benchmark_root / "multimodal"
    if mm_root.is_dir():
        _walk(mm_root / "osmh3kg", osmh3kg_ids)
        _walk(mm_root / "worldkg", worldkg_ids)

    logger.info(
        "benchmark-referenced entity ids: osmh3kg={:,} worldkg={:,}",
        len(osmh3kg_ids), len(worldkg_ids),
    )
    return osmh3kg_ids, worldkg_ids


# ---------------------------------------------------------------------------
# kwg fallback
# ---------------------------------------------------------------------------


def _collect_kwg_entities(benchmark_root: Path) -> dict[str, str]:
    """Return ``{entity_id: entity_name}`` for every distinct
    kwg entity referenced in the benchmark.

    kwg has no entity-names parquet — its IDs (``zipCodeArea.12054``,
    ``noaaTropicalCycloneSegment.2018279N11069...``) are themselves
    descriptive labels. We pull the
    ``entity_<letter>_id`` / ``_name`` pairs straight from the
    benchmark JSON shards.

    A subset of kwg ``entity_<letter>_name`` values arrive
    quote-wrapped (``'"Barbour County, Alabama"'``) — same
    upstream pathology as WorldKG's ``rdfs:label``. Strip a single
    matched leading/trailing ``"…"`` pair so the canonical name is
    clean.
    """
    kwg_root = benchmark_root / "kwg"
    if not kwg_root.is_dir():
        logger.warning("no kwg dir under {}; skipping kwg fallback", benchmark_root)
        return {}

    seen: dict[str, str] = {}
    for sf in kwg_root.iterdir():
        if not sf.is_dir() or not sf.name.startswith("sfid_"):
            continue
        bj = sf / "benchmark.jsonl"
        if not bj.is_dir():
            continue
        for row in iter_jsonl_rows(bj):
            for letter in ("a", "b", "c", "d"):
                eid = row.get(f"entity_{letter}_id")
                ename = row.get(f"entity_{letter}_name")
                if not eid or not ename:
                    continue
                # First-seen wins — kwg ids are already canonical
                # so all rows referencing one id should share the
                # same name. Defensive against drift.
                seen.setdefault(str(eid), _strip_quote_pair(str(ename)))
    logger.info("kwg fallback: {} distinct (entity_id, name) pairs", len(seen))
    return seen


# ---------------------------------------------------------------------------
# Per-language store assembly
# ---------------------------------------------------------------------------


def build_alias_store_for_language(
    target_lang: str,
    *,
    osmh3kg: dict[str, list[_NameRecord]],
    worldkg: dict[str, list[_NameRecord]],
    kwg_entities: dict[str, str],
) -> dict[str, dict[str, object]]:
    """Build the alias-store dict for one target language.

    Output shape (matches existing ``AliasStore.load`` JSON contract):

        {"<canonical_id>": {"canonical": "<name>", "aliases": [...]}}

    Where ``canonical_id`` is one of:

    * ``osm:<id>`` — osmh3kg entity (id from entity-names parquet).
    * ``wkg:<id>`` — worldkg entity. Synthetic ``geo<digits>``
      geometry-node ids are stripped to bare digits to match the
      gold-triples loader's subject normalisation.
    * ``kwg:<id>`` — kwg entity (id is the semantic URI tail, also
      used as the canonical name).
    """
    out: dict[str, dict[str, object]] = {}

    # OSMH3KG entries
    for eid, recs in osmh3kg.items():
        canonical = _pick_canonical(recs, target_lang)
        if canonical is None:
            continue
        cid = f"osm:{eid}"
        out[cid] = {
            "canonical": canonical,
            "aliases": _aliases_excluding_canonical(recs, canonical),
        }

    # WorldKG entries (with geo<digits> normalisation)
    for eid, recs in worldkg.items():
        canonical = _pick_canonical(recs, target_lang)
        if canonical is None:
            continue
        # Mirror oracle.load_gold_triples / _normalize_gold_canonical
        # geo-prefix stripping so wkg:<id> is the same canonical id
        # the gold-triples loader produces.
        normalized = eid
        if normalized.startswith("geo") and normalized[3:].isdigit():
            normalized = normalized[3:]
        cid = f"wkg:{normalized}"
        # First-seen wins on collision (same numeric id reachable via
        # both bare and ``geo`` form — both should resolve to the
        # same union of names; merge defensively).
        if cid in out:
            existing_aliases = set(out[cid]["aliases"])  # type: ignore[arg-type]
            new_aliases = set(_aliases_excluding_canonical(recs, canonical))
            existing_aliases |= new_aliases
            existing_aliases.discard(out[cid]["canonical"])  # type: ignore[arg-type]
            out[cid]["aliases"] = sorted(existing_aliases)
        else:
            out[cid] = {
                "canonical": canonical,
                "aliases": _aliases_excluding_canonical(recs, canonical),
            }

    # kwg fallback (Q6 default — every language gets the same canonical)
    for eid, ename in kwg_entities.items():
        cid = f"kwg:{eid}"
        # If somehow already present (shouldn't happen — disjoint
        # id namespace), don't overwrite.
        out.setdefault(cid, {"canonical": ename, "aliases": []})

    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _build_partial_for_kg(
    grouped: dict[str, list[_NameRecord]],
    *,
    cid_prefix: str,
    target_lang: str,
    geo_strip: bool = False,
) -> dict[str, dict[str, object]]:
    """Build per-language alias entries from ONE KG's grouped names.

    ``cid_prefix`` is ``"osm"`` or ``"wkg"``. ``geo_strip=True`` strips
    a leading ``geo`` from numeric ids (worldkg synthetic geometry
    nodes share an id with their owning entity once normalised).
    """
    out: dict[str, dict[str, object]] = {}
    for eid, recs in grouped.items():
        canonical = _pick_canonical(recs, target_lang)
        if canonical is None:
            continue
        normalized = eid
        if geo_strip and normalized.startswith("geo") and normalized[3:].isdigit():
            normalized = normalized[3:]
        cid = f"{cid_prefix}:{normalized}"
        aliases = _aliases_excluding_canonical(recs, canonical)
        if cid in out:
            # Same canonical id reachable via both bare and ``geo``
            # forms — merge alias sets defensively.
            existing = out[cid]
            existing_aliases = set(existing["aliases"])  # type: ignore[arg-type]
            existing_aliases.update(aliases)
            existing_aliases.discard(existing["canonical"])  # type: ignore[arg-type]
            out[cid]["aliases"] = sorted(existing_aliases)
        else:
            out[cid] = {"canonical": canonical, "aliases": aliases}
    return out


def _merge_partial_into_disk(
    partial: dict[str, dict[str, object]],
    out_path: Path,
) -> int:
    """Read-modify-write merge of ``partial`` into ``out_path``.

    On collision (same canonical_id): keep the existing canonical name,
    union the aliases.
    Returns the final entry count.
    """
    existing: dict[str, dict[str, object]] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    for cid, body in partial.items():
        if cid in existing:
            ex_canonical = existing[cid]["canonical"]
            ex_aliases = set(existing[cid]["aliases"])  # type: ignore[arg-type]
            ex_aliases.update(body["aliases"])  # type: ignore[arg-type]
            # The new partial's canonical name (if different) becomes
            # an alias rather than overriding the existing canonical.
            if body["canonical"] != ex_canonical:
                ex_aliases.add(body["canonical"])  # type: ignore[arg-type]
            ex_aliases.discard(ex_canonical)  # type: ignore[arg-type]
            existing[cid] = {
                "canonical": ex_canonical,
                "aliases": sorted(ex_aliases),
            }
        else:
            existing[cid] = body
    out_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return len(existing)


def build_all(
    benchmark_root: Path,
    output_dir: Path,
    languages: tuple[str, ...] = ALL_LANGUAGES,
) -> dict[str, int]:
    """Build alias JSONs for every requested language.

    Memory-safe two-pass design: load each entity-names parquet
    individually, materialise per-language partial stores, merge each
    into its on-disk JSON, then drop the in-memory grouped dict
    before loading the next KG. Peak memory stays bounded at
    one-KG-grouped-dict + one-language-partial-dict — never both
    KGs' grouped dicts simultaneously.

    Returns a summary ``{lang: entry_count}`` dict.
    """
    import gc

    output_dir.mkdir(parents=True, exist_ok=True)

    # Always start from a clean slate so reruns produce
    # byte-identical files (no leftover entries from previous runs
    # over different vocabulary).
    for lang in languages:
        out_path = output_dir / f"{lang}.json"
        if out_path.exists():
            out_path.unlink()

    # Restrict the parquet load to entity ids actually referenced in
    # the benchmark — drops ~99% of entries that would otherwise bloat
    # the per-language JSONs to >1 GB each.
    osmh3kg_ids, worldkg_ids = _collect_benchmark_entity_ids(benchmark_root)

    # Pass 1: osmh3kg
    osmh3kg = _load_entity_names(
        benchmark_root / "entity_names" / "osmh3kg_entity_names.parquet",
        strip_quotes=False,
        keep_only=osmh3kg_ids,
    )
    summary: dict[str, int] = {}
    for lang in languages:
        partial = _build_partial_for_kg(
            osmh3kg, cid_prefix="osm", target_lang=lang,
        )
        count = _merge_partial_into_disk(partial, output_dir / f"{lang}.json")
        summary[lang] = count
        logger.info("{}: osmh3kg → {:,} entries", lang, count)
    del osmh3kg
    gc.collect()

    # Pass 2: worldkg
    worldkg = _load_entity_names(
        benchmark_root / "entity_names" / "worldkg_entity_names.parquet",
        strip_quotes=True,
        keep_only=worldkg_ids,
    )
    for lang in languages:
        partial = _build_partial_for_kg(
            worldkg, cid_prefix="wkg", target_lang=lang, geo_strip=True,
        )
        count = _merge_partial_into_disk(partial, output_dir / f"{lang}.json")
        summary[lang] = count
        logger.info("{}: + worldkg → {:,} entries", lang, count)
    del worldkg
    gc.collect()

    # Pass 3: kwg fallback (every language gets the same canonical)
    kwg_entities = _collect_kwg_entities(benchmark_root)
    kwg_partial = {
        f"kwg:{eid}": {"canonical": ename, "aliases": []}
        for eid, ename in kwg_entities.items()
    }
    for lang in languages:
        count = _merge_partial_into_disk(kwg_partial, output_dir / f"{lang}.json")
        summary[lang] = count
        logger.info("{}: + kwg → {:,} entries (final)", lang, count)

    return summary


@click.command()
@click.option(
    "--benchmark-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("benchmark/raw_eng"),
    show_default=True,
    help="Root containing entity_names/ + per-KG benchmark.jsonl shards.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("output/aliases"),
    show_default=True,
    help="Where to write the per-language alias JSON files.",
)
@click.option(
    "--languages",
    multiple=True,
    default=None,
    help=(
        "Target languages (ISO 639-1). Repeat the flag for multiple "
        "(e.g. ``--languages en --languages de``). Defaults to all 17 "
        "(16 target + en)."
    ),
)
def main(
    benchmark_root: Path,
    output_dir: Path,
    languages: tuple[str, ...],
) -> None:
    """Build per-language alias JSON files."""
    langs = languages if languages else ALL_LANGUAGES
    summary = build_all(benchmark_root, output_dir, languages=langs)
    click.echo("")
    click.echo("=== Summary ===")
    for lang, count in sorted(summary.items()):
        click.echo(f"  {lang}: {count:,} entries")


if __name__ == "__main__":
    main()
