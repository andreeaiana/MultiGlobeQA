"""Asymmetric cap=N stratified sampler for the multilingual benchmark.

Reads ``benchmark/raw_multilingual_renumbered/`` and writes a sampled
parallel tree at ``benchmark/sampled_multilingual/`` (default), keeping
multimodal rows and false-premise rows entirely intact and capping the
true-premise non-multimodal pool at ``--cap`` rows per
``(kg_source, sfid, tid, strata_cell, is_true_premise)`` cell.

The sample manifest is built once from the English split (question_ids
are identical across all 17 language tracks), then applied to every
language so per-question cross-language joins still work.

Usage::

    uv run python scripts/sample_benchmark.py
    uv run python scripts/sample_benchmark.py --cap 30 --seed 42
    uv run python scripts/sample_benchmark.py --dry-run

Outputs::

    benchmark/sampled_multilingual/
      kwg/sfid_a/
        gold_triples.jsonl/             # filtered by manifest (no orphans)
        eng/benchmark.jsonl/part-00000.json
        deu/benchmark.jsonl/part-00000.json
        ...
      multimodal/osmh3kg/eng/...         # multimodal subtree
      ...
      templates.csv  templates.json  templates.parquet  # copied from raw_eng_renumbered
      sample_manifest.csv                # per-row kept_reason audit
      row_counts.txt                     # sampled-vs-full summary

The sampler does NOT regenerate ``gold_triples.parquet/`` or
``benchmark.parquet/``. If a downstream reader needs parquet, run
``scripts/regenerate_benchmark_parquet.py --benchmark-root
benchmark/sampled_multilingual`` afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC_ROOT = PROJECT_ROOT / "benchmark" / "raw_multilingual_renumbered"
DEFAULT_TEMPLATES_SRC = PROJECT_ROOT / "benchmark" / "raw_eng_renumbered"
DEFAULT_DST_ROOT = PROJECT_ROOT / "benchmark" / "sampled_multilingual"
DEFAULT_CAP = 30
DEFAULT_SEED = 42

LANGUAGES = (
    "eng", "bul", "deu", "ell", "spa", "fra", "ita", "jpn", "kat",
    "por_PT", "ron", "rus", "sqi", "tur", "urd", "vie", "zho_CN",
)
KG_DIRS = ("kwg", "osmh3kg", "worldkg", "multimodal")

KEPT_MULTIMODAL = "multimodal"
KEPT_FALSE_PREMISE = "false_premise"
KEPT_SPARSE_WHOLE = "sparse_kept_whole"
KEPT_CAPPED = "capped"


def _is_multimodal_row(row: dict) -> bool:
    """Robust boolean coercion — JSONL flag may be 'true'/'false' string or bool."""
    v = row.get("is_multimodal")
    return v is True or (isinstance(v, str) and v.lower() == "true")


def _is_true_premise_row(row: dict) -> bool:
    """Robust boolean coercion. Treat missing / null as 'true'."""
    v = row.get("is_true_premise")
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    return isinstance(v, str) and v.lower() == "true"


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL atomically via .tmp + os.replace, matching the pattern at
    scripts/migrations/migrate_to_new_taxonomy.py:260-267."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _iter_jsonl_shards(bench_dir: Path) -> Iterator[dict]:
    """Yield rows from every part-* shard under a benchmark.jsonl/ dir."""
    for shard in sorted(bench_dir.glob("part-*")):
        if not shard.is_file():
            continue
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                yield json.loads(line)


def _cell_key(row: dict) -> tuple[str, str, str, str, str]:
    """Cell identity used by the sampling rule:
    (kg_source, sfid, tid, strata_cell, is_true_premise_str)."""
    return (
        str(row.get("kg_source", "")),
        str(row.get("sfid", "")),
        str(row.get("tid", "")),
        str(row.get("strata_cell", "")),
        "true" if _is_true_premise_row(row) else "false",
    )


def build_manifest(
    src_root: Path, *, cap: int, seed: int
) -> dict[str, str]:
    """Build the question_id → kept_reason map from the eng split.

    Per-row classification (NOT per-cell):
      1. Multimodal rows → kept as multimodal (highest priority).
      2. False-premise non-multimodal rows → kept as false_premise.
      3. True-premise non-multimodal rows → grouped by
         (kg_source, sfid, tid, strata_cell) cell, capped at ``cap``.

    Multimodal rows share ``kg_source`` with non-multimodal rows from
    the regular tree, so a per-cell rule (cell-level membership test
    for is_multimodal) would absorb non-multimodal rows that happen to
    share a cell key. Per-row classification is the correct
    interpretation of the sampling rule.

    Returns a dict keyed by question_id; values are one of the
    ``KEPT_*`` constants. Question_ids absent from this dict are
    excluded from the sample.
    """
    multimodal_rows: list[dict] = []
    false_premise_rows: list[dict] = []
    tp_non_mm_cells: dict[tuple, list[dict]] = defaultdict(list)

    for kg_dir_name in KG_DIRS:
        kg_dir = src_root / kg_dir_name
        if not kg_dir.is_dir():
            continue
        if kg_dir_name == "multimodal":
            inner_dirs = sorted(p for p in kg_dir.iterdir() if p.is_dir())
        else:
            inner_dirs = sorted(
                p for p in kg_dir.iterdir()
                if p.is_dir() and p.name.startswith("sfid_")
            )
        for inner_dir in inner_dirs:
            bench_dir = inner_dir / "eng" / "benchmark.jsonl"
            if not bench_dir.is_dir():
                continue
            for row in _iter_jsonl_shards(bench_dir):
                if _is_multimodal_row(row):
                    multimodal_rows.append(row)
                elif not _is_true_premise_row(row):
                    false_premise_rows.append(row)
                else:
                    tp_non_mm_cells[_cell_key(row)].append(row)

    logger.info(
        "Pool sizes: multimodal={}, false_premise={}, "
        "true-premise non-multimodal cells={} ({} rows)",
        len(multimodal_rows), len(false_premise_rows),
        len(tp_non_mm_cells),
        sum(len(rs) for rs in tp_non_mm_cells.values()),
    )

    rng = random.Random(seed)
    manifest: dict[str, str] = {}

    # Note on duplicate question_ids: ~105 qids appear in BOTH the
    # multimodal subtree AND the regular tree (multimodal rerender of
    # the same logical question). We tag in REVERSE priority order
    # (sparse/capped first, then false_premise, then multimodal last)
    # so the multimodal label wins for the summary breakdown. The
    # downstream filter only uses set membership; the labels here
    # affect the sample_manifest.csv audit, not the kept-row set.

    # Rule 3 (lowest priority): cap each TP-non-MM cell at ``cap`` rows.
    for _key, rows in tp_non_mm_cells.items():
        rows_sorted = sorted(rows, key=lambda r: r.get("question_id", ""))
        if len(rows_sorted) <= cap:
            for r in rows_sorted:
                manifest[r["question_id"]] = KEPT_SPARSE_WHOLE
        else:
            chosen = rng.sample(rows_sorted, cap)
            for r in chosen:
                manifest[r["question_id"]] = KEPT_CAPPED

    # Rule 2: keep every false-premise non-multimodal row.
    for r in false_premise_rows:
        manifest[r["question_id"]] = KEPT_FALSE_PREMISE

    # Rule 1 (highest priority): keep every multimodal row.
    for r in multimodal_rows:
        manifest[r["question_id"]] = KEPT_MULTIMODAL

    logger.info("Manifest built: {} unique question_ids kept", len(manifest))
    return manifest


def write_sampled_tree(
    src_root: Path,
    dst_root: Path,
    manifest: dict[str, str],
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Walk every (kg, sfid_or_inner, lang) leaf, filter by manifest, write."""
    stats: dict[str, int] = defaultdict(int)
    kept_qids = set(manifest)

    for kg_dir_name in KG_DIRS:
        src_kg_dir = src_root / kg_dir_name
        if not src_kg_dir.is_dir():
            continue
        if kg_dir_name == "multimodal":
            inner_dirs = sorted(p for p in src_kg_dir.iterdir() if p.is_dir())
        else:
            inner_dirs = sorted(
                p for p in src_kg_dir.iterdir()
                if p.is_dir() and p.name.startswith("sfid_")
            )

        for src_inner in inner_dirs:
            dst_inner = dst_root / kg_dir_name / src_inner.name

            # Per-language benchmark shards.
            for lang in LANGUAGES:
                src_bench = src_inner / lang / "benchmark.jsonl"
                if not src_bench.is_dir():
                    continue
                kept_rows = [
                    row for row in _iter_jsonl_shards(src_bench)
                    if row.get("question_id") in kept_qids
                ]
                stats[f"{lang}_rows"] += len(kept_rows)
                if not dry_run:
                    out_path = (
                        dst_inner / lang / "benchmark.jsonl" / "part-00000.json"
                    )
                    _atomic_write_jsonl(out_path, kept_rows)

            # Gold triples — language-agnostic, filter by manifest.
            # Orphan triples (rows whose question_id was sampled out) are
            # dropped here so the sampled tree contains only triples that
            # are actually reachable from a sampled question. Parquets are
            # NOT regenerated; run scripts/regenerate_benchmark_parquet.py
            # against the sampled tree if any downstream reader needs them.
            src_gold = src_inner / "gold_triples.jsonl"
            dst_gold = dst_inner / "gold_triples.jsonl"
            if src_gold.is_dir() and not dry_run:
                if dst_gold.exists():
                    shutil.rmtree(dst_gold)
                kept, dropped = _filter_gold_shards(src_gold, dst_gold, kept_qids)
                stats["gold_triples_kept"] += kept
                stats["gold_triples_dropped"] += dropped
                stats["gold_dirs_filtered"] += 1

    return stats


def _filter_gold_shards(
    src_dir: Path, dst_dir: Path, kept_qids: set[str],
) -> tuple[int, int]:
    """Copy ``gold_triples.jsonl`` shards keeping only rows whose
    ``question_id`` is in ``kept_qids``.

    Mirrors the shard filenames so any downstream reader globbing
    ``part-*`` picks up the filtered set identically. Returns
    ``(rows_kept, rows_dropped)`` aggregated across all shards.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    for shard in sorted(src_dir.glob("part-*")):
        if not shard.is_file():
            continue
        out_rows: list[dict] = []
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("question_id") in kept_qids:
                    out_rows.append(row)
                    kept += 1
                else:
                    dropped += 1
        # Atomic write: tmp + os.replace so a kill mid-shard leaves the
        # original (if any) intact. Filename preserved.
        out_path = dst_dir / shard.name
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, out_path)
    return kept, dropped


def copy_registry(templates_src: Path, dst_root: Path, *, dry_run: bool) -> None:
    """Copy templates.csv / .json / .parquet to the dst root."""
    for name in ("templates.csv", "templates.json"):
        src = templates_src / name
        if src.is_file() and not dry_run:
            dst_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_root / name)
    parquet_dir = templates_src / "templates.parquet"
    if parquet_dir.is_dir() and not dry_run:
        target = dst_root / "templates.parquet"
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.copytree(parquet_dir, target)
    # Some checkouts emit a single-file parquet instead of a dir.
    parquet_file = templates_src / "templates.parquet"
    if parquet_file.is_file() and not dry_run:
        shutil.copy2(parquet_file, dst_root / "templates.parquet")


def write_manifest_csv(
    src_root: Path,
    dst_root: Path,
    manifest: dict[str, str],
    *,
    dry_run: bool,
) -> None:
    """Write sample_manifest.csv at dst_root with row metadata + kept_reason.

    Re-reads the eng split to recover (kg_source, sfid, tid, strata_cell,
    is_true_premise, is_multimodal) for each kept question_id.
    """
    if dry_run:
        return

    out_path = dst_root / "sample_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id", "kg_source", "sfid", "tid", "strata_cell",
        "is_true_premise", "is_multimodal", "kept_reason",
    ]
    rows_to_write: list[dict] = []
    for kg_dir_name in KG_DIRS:
        kg_dir = src_root / kg_dir_name
        if not kg_dir.is_dir():
            continue
        inner_dirs = (
            sorted(p for p in kg_dir.iterdir() if p.is_dir())
            if kg_dir_name == "multimodal"
            else sorted(
                p for p in kg_dir.iterdir()
                if p.is_dir() and p.name.startswith("sfid_")
            )
        )
        for inner_dir in inner_dirs:
            bench_dir = inner_dir / "eng" / "benchmark.jsonl"
            if not bench_dir.is_dir():
                continue
            for row in _iter_jsonl_shards(bench_dir):
                qid = row.get("question_id")
                if qid not in manifest:
                    continue
                rows_to_write.append({
                    "question_id": qid,
                    "kg_source": row.get("kg_source", ""),
                    "sfid": row.get("sfid", ""),
                    "tid": row.get("tid", ""),
                    "strata_cell": row.get("strata_cell", ""),
                    "is_true_premise":
                        "true" if _is_true_premise_row(row) else "false",
                    "is_multimodal":
                        "true" if _is_multimodal_row(row) else "false",
                    "kept_reason": manifest[qid],
                })
    rows_to_write.sort(key=lambda r: r["question_id"])
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows_to_write:
            writer.writerow(row)


def write_row_counts(
    dst_root: Path,
    eng_kept: int,
    eng_full: int,
    *,
    cap: int,
    seed: int,
    per_kg_sfid: dict[tuple[str, str], int],
    dry_run: bool,
) -> None:
    """Mirror the format of raw_eng_renumbered/row_counts.txt for the sample."""
    if dry_run:
        return
    out_path = dst_root / "row_counts.txt"
    pct = 100 * eng_kept / eng_full if eng_full else 0.0
    lines = [
        f"# Sampled benchmark — asymmetric cap={cap}, seed={seed}, "
        f"multimodal + false-premise kept whole",
        f"# eng rows kept: {eng_kept} of {eng_full} ({pct:.1f}%)",
        "",
    ]
    for (kg, sfid), n in sorted(per_kg_sfid.items()):
        lines.append(f"{kg:<10} {sfid:<15} {n:>6}")
    lines.append("")
    lines.append(f"TOTAL: {eng_kept} rows")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-root", type=Path, default=DEFAULT_SRC_ROOT,
                   help=f"Source benchmark root (default: {DEFAULT_SRC_ROOT})")
    p.add_argument("--templates-src", type=Path, default=DEFAULT_TEMPLATES_SRC,
                   help=f"Templates source root (default: {DEFAULT_TEMPLATES_SRC})")
    p.add_argument("--dst-root", type=Path, default=DEFAULT_DST_ROOT,
                   help=f"Destination sampled root (default: {DEFAULT_DST_ROOT})")
    p.add_argument("--cap", type=int, default=DEFAULT_CAP,
                   help=f"Max rows per cell (default: {DEFAULT_CAP})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Random seed (default: {DEFAULT_SEED})")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute the manifest + counts but write no files.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> "
                      "<level>{level: <8}</level> {message}")

    if not args.src_root.is_dir():
        logger.error("Source root not found: {}", args.src_root)
        return 1

    logger.info("Building manifest from eng split (cap={}, seed={})",
                args.cap, args.seed)
    manifest = build_manifest(args.src_root, cap=args.cap, seed=args.seed)

    eng_kept = len(manifest)
    breakdown = defaultdict(int)
    for reason in manifest.values():
        breakdown[reason] += 1
    logger.info("Kept counts by reason: {}", dict(breakdown))

    logger.info("Writing sampled tree to {} (dry_run={})",
                args.dst_root, args.dry_run)
    stats = write_sampled_tree(args.src_root, args.dst_root, manifest,
                               dry_run=args.dry_run)

    logger.info("Per-language eng-equivalent kept rows: {} (target: same as "
                "eng for every lang)", stats.get("eng_rows", 0))
    gold_kept = stats.get("gold_triples_kept", 0)
    gold_dropped = stats.get("gold_triples_dropped", 0)
    gold_total = gold_kept + gold_dropped
    if gold_total:
        logger.info(
            "Gold triples filtered: {} kept / {} dropped ({:.1f}% orphans removed)",
            gold_kept, gold_dropped, 100 * gold_dropped / gold_total,
        )

    # Per-(kg, sfid) row counts from the sampled eng tree
    # (for row_counts.txt). Computed by re-walking the eng dst tree —
    # cheap given it's already on disk.
    per_kg_sfid: dict[tuple[str, str], int] = defaultdict(int)
    if not args.dry_run:
        for kg_dir_name in KG_DIRS:
            kg_dir = args.dst_root / kg_dir_name
            if not kg_dir.is_dir():
                continue
            inner_dirs = (
                sorted(p for p in kg_dir.iterdir() if p.is_dir())
                if kg_dir_name == "multimodal"
                else sorted(
                    p for p in kg_dir.iterdir()
                    if p.is_dir() and p.name.startswith("sfid_")
                )
            )
            for inner_dir in inner_dirs:
                bench = inner_dir / "eng" / "benchmark.jsonl"
                if bench.is_dir():
                    n = sum(1 for _ in _iter_jsonl_shards(bench))
                    per_kg_sfid[(kg_dir_name, inner_dir.name)] = n

    logger.info("Copying templates registry from {}", args.templates_src)
    copy_registry(args.templates_src, args.dst_root, dry_run=args.dry_run)

    logger.info("Writing sample_manifest.csv ({} rows)", eng_kept)
    write_manifest_csv(args.src_root, args.dst_root, manifest,
                       dry_run=args.dry_run)

    # We don't know the exact "eng_full" without re-counting; use the
    # well-known value from row_counts.txt for the comparison header.
    write_row_counts(args.dst_root, eng_kept, 96756,
                     cap=args.cap, seed=args.seed,
                     per_kg_sfid=per_kg_sfid, dry_run=args.dry_run)

    logger.success(
        "Done. eng kept = {} / 96756 ({:.1f}%). "
        "Multimodal: {}; false-premise: {}; sparse: {}; capped: {}",
        eng_kept, 100 * eng_kept / 96756,
        breakdown[KEPT_MULTIMODAL], breakdown[KEPT_FALSE_PREMISE],
        breakdown[KEPT_SPARSE_WHOLE], breakdown[KEPT_CAPPED],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
