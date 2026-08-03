"""Build the multilingual benchmark — 17 sibling languages under
``benchmark/raw_multilingual/<kg>/<sfid>/<lang>/``.

For every row in ``benchmark/raw_eng/<kg>/<sfid>/benchmark.jsonl/``:

1. Read the row's ``vid`` (template-variant id, set upstream).
2. Look up the source-language template for that ``vid`` in
   ``benchmark/raw_eng/templates.csv``. Run the regex matcher once to
   recover the non-entity placeholder captures (e.g. ``{value}``,
   ``{cardinal_direction}``, ``{grid_cell_ID}``) from the rendered
   ``question`` text — these slot values aren't kept in row fields,
   so we have to read them out of the question.
3. For each of the 17 sibling languages (``eng`` + 16 targets),
   look up the per-language template column from the same CSV row.
   Render it with:
     * Per-language entity names from
       ``output/aliases/<lang>.json`` (canonical id → display name);
       fall back to the row's English ``entity_<x>_name`` when the
       canonical id has no entry in that language.
     * The captures from step 2 (same values across languages —
       they're locale-neutral, e.g. cardinal directions, numbers,
       time periods).
     * ``[IMAGE_<slot>]`` literals on multimodal slots (mirrors the
       D1 fix in :mod:`scripts.fix_data_defects`).
4. Emit a row clone with:
     * ``question`` = rendered per-lang text
     * ``language`` = ISO 639-3 lang code
     * ``entity_<x>_name`` = per-language form (or English fallback)
     * all other fields copy through unchanged

Gold triples are language-agnostic: each ``(kg, sfid)`` cell has one
``gold_triples.{jsonl,parquet}/`` shipped at
``benchmark/raw_multilingual/<kg>/<sfid>/gold_triples.*`` (a copy of
``benchmark/raw_eng/<kg>/<sfid>/gold_triples.*``).

Output layout (per-language sibling tree):

    benchmark/raw_multilingual/
      <kg>/<sfid>/
        gold_triples.{jsonl,parquet}/   # one copy per cell
        <lang>/
          benchmark.{jsonl,parquet}/    # one per language

Multimodal:

    benchmark/raw_multilingual/multimodal/<kg>/
      gold_triples.{jsonl,parquet}/
      <lang>/benchmark.{jsonl,parquet}/

Idempotent. Atomic per-shard writes via ``.tmp + os.replace``.

Usage::

    # Dry run — counts only, no writes
    uv run python -m pipelines.translation.build_multilingual_benchmark --dry-run

    # Real run — emits all 17 languages
    uv run python -m pipelines.translation.build_multilingual_benchmark

    # One language only (debugging)
    uv run python -m pipelines.translation.build_multilingual_benchmark --lang deu

    # Skip multimodal cells
    uv run python -m pipelines.translation.build_multilingual_benchmark \\
        --skip-multimodal
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
from loguru import logger

from pipelines.translation._template_matcher import (
    ENTITY_PLACEHOLDER_TO_FIELD,
    PLACEHOLDER_RE,
    find_variant,
    get_image_slots,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "benchmark" / "raw_eng"
DEFAULT_FINAL_ROOT = REPO_ROOT / "benchmark" / "raw_multilingual"
DEFAULT_TEMPLATES_CSV = REPO_ROOT / "benchmark" / "raw_eng" / "templates.csv"
DEFAULT_ALIASES_DIR = REPO_ROOT / "output" / "aliases"

# 17 sibling languages: English + 16 target languages.
ALL_LANGUAGES: tuple[str, ...] = (
    "eng",
    "bul", "deu", "ell", "spa", "fra", "ita", "jpn", "kat",
    "por_PT", "ron", "rus", "sqi", "tur", "urd", "vie", "zho_CN",
)

# kg_source value → alias-store id prefix.
_KG_PREFIX: dict[str, str] = {
    "kwg": "kwg",
    "osmh3kg": "osm",
    "worldkg": "wkg",
}


# ---------------------------------------------------------------------------
# Templates + alias-store loaders
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TemplateRow:
    """One row of ``templates.csv``: ftidv plus per-language template strings."""
    ftidv: str
    ftid: str
    base_template: str
    eng: str
    by_lang: dict[str, str]  # lang code → template string


def load_templates_by_ftidv(path: Path) -> dict[str, TemplateRow]:
    """Read ``templates.csv`` and return ``{ftidv → TemplateRow}``."""
    out: dict[str, TemplateRow] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ftidv = r["ftidv"]
            by_lang: dict[str, str] = {}
            for lang in ALL_LANGUAGES:
                by_lang[lang] = (r.get(lang) or "").strip()
            out[ftidv] = TemplateRow(
                ftidv=ftidv,
                ftid=r.get("ftid", ""),
                base_template=r.get("base_template", ""),
                eng=r.get("eng", ""),
                by_lang=by_lang,
            )
    return out


def load_templates_by_tid(
    path: Path,
) -> dict[str, list[tuple[str, str]]]:
    """Group ``(ftidv, eng_template)`` by ``tid`` for the matcher."""
    out: dict[str, list[tuple[str, str]]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            tid = r.get("tid") or ""
            ftidv = r.get("ftidv") or ""
            eng = r.get("eng") or ""
            if tid:
                out.setdefault(tid, []).append((ftidv, eng))
    return out


def load_alias_store(path: Path) -> dict[str, str]:
    """Load a per-language alias JSON and return ``{canonical_id → name}``.

    Only the ``canonical`` field is read — that's the per-language
    name (or English fallback if the source data has no name in
    this language). The ``aliases`` list is ignored; we always pick
    the one canonical name per entity per language.
    """
    if not path.is_file():
        logger.warning("alias store missing: {}", path)
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for cid, body in raw.items():
        if isinstance(body, dict):
            name = body.get("canonical")
            if isinstance(name, str) and name:
                out[cid] = name
    return out


# ---------------------------------------------------------------------------
# Per-row rendering
# ---------------------------------------------------------------------------


def _canonical_id_for_slot(
    row: dict[str, Any],
    slot_letter: str,
) -> str | None:
    """Build the alias-store canonical id for ``entity_<slot>`` on this row.

    Mirrors :func:`benchmark_eval.scorer._normalize_gold_canonical`:

    * Already-prefixed id (``"osm:..."``, ``"kwg:..."``, ``"wkg:..."``) →
      pass through.
    * Bare id + ``kg_source`` → prefix-construct (``kwg:<id>``,
      ``osm:<id>``, ``wkg:<id>``). For worldkg, strip a leading
      ``geo`` from numeric ids (synthetic geometry-node convention).

    Returns ``None`` when the slot has no id on this row, or when the
    row's ``kg_source`` is unrecognised.
    """
    eid = row.get(f"entity_{slot_letter.lower()}_id")
    if not eid:
        return None
    eid_str = str(eid).strip()
    if not eid_str:
        return None
    if eid_str.startswith(("kwg:", "osm:", "wkg:")):
        return eid_str
    kg = row.get("kg_source")
    if kg not in _KG_PREFIX:
        return None
    prefix = _KG_PREFIX[kg]
    if prefix == "wkg":
        if eid_str.startswith("geo") and eid_str[3:].isdigit():
            eid_str = eid_str[3:]
    return f"{prefix}:{eid_str}"


def _per_lang_entity_name(
    row: dict[str, Any],
    slot_letter: str,
    lang_aliases: dict[str, str],
) -> str | None:
    """Return the per-language display name for ``entity_<slot>``.

    Cascade:

    1. Look up the canonical id in ``lang_aliases``.
    2. Fall back to the row's source ``entity_<slot>_name`` (English
       label produced upstream).

    Returns ``None`` only if BOTH lookups fail (slot is genuinely empty).
    """
    cid = _canonical_id_for_slot(row, slot_letter)
    if cid is not None:
        name = lang_aliases.get(cid)
        if name:
            return name
    fallback = row.get(f"entity_{slot_letter.lower()}_name")
    if isinstance(fallback, str) and fallback:
        return fallback
    return None


def render_template_for_lang(
    template: str,
    row: dict[str, Any],
    lang_aliases: dict[str, str],
    non_entity_values: dict[str, str],
    *,
    image_slots: set[str] | None = None,
) -> tuple[str | None, dict[str, str]]:
    """Render ``template`` (which is in some target language) using:

    * Per-language entity names for ``{entity_<X>}`` /
      ``{event_<X>}`` slots (looked up via ``lang_aliases``,
      English fallback through the row).
    * Literal ``[IMAGE_<X>]`` for slots in ``image_slots``.
    * ``non_entity_values`` for non-entity slots like ``{value}``,
      ``{time_period}`` (locale-neutral — same value across all
      languages).

    Returns ``(rendered_question, per_lang_entity_names)``.
    ``per_lang_entity_names`` is the name that landed in each entity
    slot keyed by lower-case slot letter (so the caller can write
    them back into the row's ``entity_<x>_name`` fields).

    Returns ``(None, {})`` if a required slot can't be resolved
    (missing entity name AND missing fallback) — should never
    happen on the post-D9/D10/D11/D14 data, but kept as a defensive
    no-emit signal.
    """
    image_slots = image_slots or set()
    out_parts: list[str] = []
    last_end = 0
    per_lang_names: dict[str, str] = {}
    for m in PLACEHOLDER_RE.finditer(template):
        out_parts.append(template[last_end:m.start()])
        ph = m.group(1)
        if ph in ENTITY_PLACEHOLDER_TO_FIELD:
            # entity_A → A; event_A → A; "entity A" (typo) → A
            slot_letter = ph[-1].upper() if ph[-1].isalpha() else "A"
            if slot_letter in image_slots:
                out_parts.append(f"[IMAGE_{slot_letter}]")
            else:
                # Event slots can also have aliases in principle, but
                # event ids share the alias-store key space (see
                # earthquake.* keys in the stores) so the same lookup
                # logic works.
                if ph.startswith("event_"):
                    # Event alias lookup keyed off event_<x>_id.
                    cid = None
                    eid = row.get(f"event_{slot_letter.lower()}_id")
                    if eid:
                        eid_str = str(eid).strip()
                        if eid_str.startswith(("kwg:", "osm:", "wkg:")):
                            cid = eid_str
                        else:
                            kg = row.get("kg_source")
                            prefix = _KG_PREFIX.get(kg or "")
                            if prefix:
                                cid = f"{prefix}:{eid_str}"
                    name = lang_aliases.get(cid) if cid else None
                    if not name:
                        name = row.get(f"event_{slot_letter.lower()}_name")
                else:
                    name = _per_lang_entity_name(
                        row, slot_letter, lang_aliases,
                    )
                if not name:
                    return None, {}
                out_parts.append(name)
                per_lang_names[slot_letter.lower()] = name
        else:
            v = non_entity_values.get(ph)
            if v is None:
                # Some templates may not reference all non-entity
                # placeholders that were captured; that's fine.
                # But if the template DOES reference one we didn't
                # capture, we can't render — this means the matcher
                # didn't see this placeholder in the source row.
                return None, {}
            out_parts.append(v)
        last_end = m.end()
    out_parts.append(template[last_end:])
    return "".join(out_parts), per_lang_names


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------


def discover_cells(raw_root: Path) -> list[Path]:
    """Yield each ``(kg, sfid)`` cell directory under ``raw/``.

    Returns directories with a ``benchmark.jsonl/`` child (matches
    both regular ``<kg>/<sfid>/`` and multimodal ``multimodal/<kg>/``).
    """
    out: list[Path] = []
    for kg_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        kg = kg_dir.name
        if kg == "multimodal":
            for inner in sorted(p for p in kg_dir.iterdir() if p.is_dir()):
                if (inner / "benchmark.jsonl").is_dir():
                    out.append(inner)
        else:
            for sfid_dir in sorted(
                p for p in kg_dir.iterdir() if p.is_dir() and p.name.startswith("sfid_")
            ):
                if (sfid_dir / "benchmark.jsonl").is_dir():
                    out.append(sfid_dir)
    return out


def cell_relative(cell_dir: Path, raw_root: Path) -> Path:
    """``benchmark/raw_eng/kwg/sfid_a`` → ``kwg/sfid_a``;
    ``benchmark/raw_eng/multimodal/osmh3kg`` → ``multimodal/osmh3kg``."""
    return cell_dir.relative_to(raw_root)


def iter_jsonl_shard(shard_dir: Path) -> Iterable[dict[str, Any]]:
    for part in sorted(shard_dir.glob("part-*")):
        if not part.is_file():
            continue
        with part.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                yield json.loads(line)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    os.replace(tmp, path)


def copy_gold_triples(cell_dir: Path, dst_cell_dir: Path) -> None:
    """Copy ``cell_dir/gold_triples.{jsonl,parquet}/`` to ``dst_cell_dir``.

    Atomic per-shard rewrite via temp + replace. Idempotent: skips
    files that exist with the same size + mtime."""
    for sub in ("gold_triples.jsonl", "gold_triples.parquet"):
        src = cell_dir / sub
        if not src.is_dir():
            continue
        dst = dst_cell_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        for src_file in src.iterdir():
            if not src_file.is_file():
                continue
            dst_file = dst / src_file.name
            # Idempotent skip: if dest exists with same size, leave it.
            if dst_file.is_file() and dst_file.stat().st_size == src_file.stat().st_size:
                continue
            shutil.copy2(src_file, dst_file)


@dataclass
class CellGenStats:
    cell: str
    rows_in: int = 0
    rows_emitted: dict[str, int] = None  # type: ignore[assignment]
    fallbacks: dict[str, int] = None  # type: ignore[assignment]
    skipped_render_failure: int = 0

    def __post_init__(self):
        if self.rows_emitted is None:
            self.rows_emitted = dict.fromkeys(ALL_LANGUAGES, 0)
        if self.fallbacks is None:
            self.fallbacks = dict.fromkeys(ALL_LANGUAGES, 0)


def generate_for_cell(
    cell_dir: Path,
    raw_root: Path,
    final_root: Path,
    templates_by_ftidv: dict[str, TemplateRow],
    templates_by_tid: dict[str, list[tuple[str, str]]],
    aliases: dict[str, dict[str, str]],
    languages: tuple[str, ...],
    *,
    dry_run: bool = False,
) -> CellGenStats:
    """Render all ``languages`` for one ``(kg, sfid)`` (or multimodal) cell.

    Reads source rows from ``cell_dir/benchmark.jsonl/``, renders each
    × each lang, and writes per-language shards under
    ``final_root/<rel>/<lang>/benchmark.jsonl/``.

    Also copies ``gold_triples.{jsonl,parquet}/`` from raw to final
    (cell-level, language-agnostic).
    """
    rel = cell_relative(cell_dir, raw_root)
    stats = CellGenStats(cell=str(rel))
    src_bench_dir = cell_dir / "benchmark.jsonl"

    # Read all source rows once so we can do the per-lang fan-out
    # without re-reading. Cells max out around 15k rows so memory
    # is bounded.
    src_rows = list(iter_jsonl_shard(src_bench_dir))
    stats.rows_in = len(src_rows)
    if not src_rows:
        return stats

    # Per-language output buffers.
    out_buffers: dict[str, list[dict[str, Any]]] = {l: [] for l in languages}

    for row in src_rows:
        vid = row.get("vid")
        if not vid or vid not in templates_by_ftidv:
            stats.skipped_render_failure += 1
            continue
        tpl_row = templates_by_ftidv[vid]
        tid = row.get("tid") or ""
        # Run the matcher to recover non-entity placeholder values
        # (e.g. {value}, {grid_cell_ID}) from the source question.
        # These values are language-neutral — same string used in
        # all per-lang renders.
        candidates = templates_by_tid.get(tid, [])
        image_slots = get_image_slots(row)
        matches = find_variant(
            candidates, row, quote_tolerant=False, image_slots=image_slots,
        )
        if not matches:
            # Fallback to quote-tolerant for any residual edge case
            matches = find_variant(
                candidates, row, quote_tolerant=True, image_slots=image_slots,
            )
        captures: dict[str, str] = {}
        if matches:
            # Pick the match whose ftidv matches the upstream vid
            # (defensive — matcher should always return exactly one).
            for ftidv, _tmpl, caps in matches:
                if ftidv == vid:
                    captures = caps
                    break
            else:
                captures = matches[0][2]
        # Render once per language.
        for lang in languages:
            tmpl = tpl_row.by_lang.get(lang) or ""
            if not tmpl:
                # No translation for this lang → skip emission for
                # this row in this lang. Should not happen given the
                # 100% per-lang fill rate from the templates table.
                continue
            rendered, per_lang_names = render_template_for_lang(
                tmpl, row, aliases.get(lang, {}), captures,
                image_slots=image_slots,
            )
            if rendered is None:
                stats.skipped_render_failure += 1
                continue
            new_row = dict(row)
            new_row["question"] = rendered
            new_row["language"] = lang
            for slot_letter, name in per_lang_names.items():
                new_row[f"entity_{slot_letter}_name"] = name
                # Track when we used the English fallback (per-lang
                # alias missing).
                cid = _canonical_id_for_slot(row, slot_letter)
                if cid is None or aliases.get(lang, {}).get(cid) is None:
                    stats.fallbacks[lang] += 1
            out_buffers[lang].append(new_row)
            stats.rows_emitted[lang] += 1

    # Flush per-lang buffers.
    if not dry_run:
        for lang, rows_out in out_buffers.items():
            if not rows_out:
                continue
            out_path = (
                final_root / rel / lang
                / "benchmark.jsonl" / "part-00000.json"
            )
            _atomic_write_jsonl(out_path, rows_out)

        # Copy gold triples cell-wide (language-agnostic).
        copy_gold_triples(cell_dir, final_root / rel)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--raw-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_RAW_ROOT,
    show_default=True,
)
@click.option(
    "--final-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_FINAL_ROOT,
    show_default=True,
)
@click.option(
    "--templates-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_TEMPLATES_CSV,
    show_default=True,
)
@click.option(
    "--aliases-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_ALIASES_DIR,
    show_default=True,
)
@click.option(
    "--lang", "languages",
    multiple=True,
    type=click.Choice(list(ALL_LANGUAGES)),
    help=(
        "Restrict to one or more languages (default: emit all 17). "
        "Pass repeatedly for multiple, e.g. ``--lang eng --lang deu``."
    ),
)
@click.option(
    "--skip-multimodal", is_flag=True, default=False,
    help="Skip the multimodal cells (osmh3kg + worldkg subsets).",
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Compute counts only; write nothing.",
)
def main(
    raw_root: Path,
    final_root: Path,
    templates_csv: Path,
    aliases_dir: Path,
    languages: tuple[str, ...],
    skip_multimodal: bool,
    dry_run: bool,
) -> None:
    if not languages:
        languages = ALL_LANGUAGES

    logger.info("Loading templates registry from {}", templates_csv)
    templates_by_ftidv = load_templates_by_ftidv(templates_csv)
    templates_by_tid = load_templates_by_tid(templates_csv)
    logger.info("  {} ftidvs, {} tids", len(templates_by_ftidv), len(templates_by_tid))

    logger.info("Loading {} alias stores from {}", len(languages), aliases_dir)
    aliases: dict[str, dict[str, str]] = {}
    for lang in languages:
        path = aliases_dir / f"{lang}.json"
        aliases[lang] = load_alias_store(path)
        logger.info("  {}: {} canonical entries", lang, len(aliases[lang]))

    cells = discover_cells(raw_root)
    if skip_multimodal:
        cells = [c for c in cells if "multimodal" not in c.parts]
    logger.info("Discovered {} cells", len(cells))

    grand_in = 0
    grand_emitted = Counter()
    grand_fallbacks = Counter()
    grand_skipped = 0
    for cell_dir in cells:
        stats = generate_for_cell(
            cell_dir, raw_root, final_root,
            templates_by_ftidv, templates_by_tid, aliases, languages,
            dry_run=dry_run,
        )
        grand_in += stats.rows_in
        for lang, n in stats.rows_emitted.items():
            grand_emitted[lang] += n
        for lang, n in stats.fallbacks.items():
            grand_fallbacks[lang] += n
        grand_skipped += stats.skipped_render_failure
        logger.info(
            "{}: {} rows → {} per-lang emits, {} skipped",
            stats.cell, stats.rows_in,
            sum(stats.rows_emitted.values()),
            stats.skipped_render_failure,
        )

    print("\n=== Summary ===")
    print(f"Cells processed: {len(cells)}")
    print(f"Source rows scanned: {grand_in:,}")
    print(f"Render failures (skipped): {grand_skipped}")
    print("\nPer-language emission counts (and English-fallback share):")
    print(f'{"lang":<8} {"emitted":>10} {"fallbacks":>10} {"%fb":>6}')
    for lang in languages:
        emitted = grand_emitted[lang]
        fb = grand_fallbacks[lang]
        pct = 100.0 * fb / emitted if emitted else 0.0
        print(f"{lang:<8} {emitted:>10,} {fb:>10,} {pct:>5.1f}%")
    total_out = sum(grand_emitted.values())
    print(f"\nTOTAL output rows: {total_out:,}")
    if dry_run:
        print("(dry-run — no files written)")


if __name__ == "__main__":
    main()
