#!/usr/bin/env python3
"""Materialize the **small** (canonical) benchmark from the **large** (full) one.

``benchmark/final/`` holds two peers with identical structure::

    benchmark/final/large/<kg>/<sfid>/<lang>/benchmark.jsonl/part-*.json   (full)
    benchmark/final/large/<kg>/<sfid>/gold_triples.jsonl/part-*.json
    benchmark/final/large/multimodal/<kg>/<lang>/...  + multimodal/<kg>/images/
    benchmark/final/small/<...same shape...>                                (canonical subset)

``small`` is ``large`` filtered to the ``question_id``\\ s listed in
``benchmark/final/canonical_qids.jsonl`` (a frozen, eql-allocation manifest:
~200 per ``(kg, sfid)`` text cell + 2 multimodal cells). The manifest is
``eng``-only, but ``question_id``\\ s are language-agnostic, so the same qid
set is applied across **every** language directory found under each cell.

For each manifest cell we subset, in lock-step:
  * ``benchmark.jsonl`` rows (per language),
  * ``gold_triples.jsonl`` rows (per ``(kg, sfid)``, language-agnostic),
  * the referenced multimodal ``images/`` (by the basenames the kept rows
    reference — image files are ``<qid>_<slot>.<ext>``).

``small`` is a **pure function** of ``large`` + the manifest, so re-running
after ``large`` changes keeps the two in sync (no drift). The run wipes and
rebuilds ``small`` by default.

Run::

    uv run python scripts/build_canonical_benchmark.py
    uv run python scripts/build_canonical_benchmark.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = REPO_ROOT / "benchmark" / "final"
DEFAULT_LARGE_ROOT = FINAL_ROOT / "large"
DEFAULT_SMALL_ROOT = FINAL_ROOT / "small"
DEFAULT_MANIFEST = FINAL_ROOT / "canonical_qids.jsonl"


def _read_jsonl_parts(leaf_dir: Path) -> list[dict]:
    """Concatenate every ``part-*.json`` line under a shard directory."""
    rows: list[dict] = []
    for pf in sorted(leaf_dir.glob("part-*.json")):
        with pf.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` as one ``part-00000.json`` shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def _iter_lang_dirs(cell_dir: Path):
    """Language subdirectories of a cell that hold a ``benchmark.jsonl/`` leaf."""
    if not cell_dir.is_dir():
        return
    for d in sorted(cell_dir.iterdir()):
        if d.is_dir() and (d / "benchmark.jsonl").is_dir():
            yield d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--large-root", type=Path, default=DEFAULT_LARGE_ROOT,
                    help=f"Source full benchmark (default: {DEFAULT_LARGE_ROOT})")
    ap.add_argument("--small-root", type=Path, default=DEFAULT_SMALL_ROOT,
                    help=f"Destination canonical benchmark (default: {DEFAULT_SMALL_ROOT})")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"Canonical qid manifest (default: {DEFAULT_MANIFEST})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be written without touching disk.")
    ap.add_argument("--keep", action="store_true",
                    help="Do NOT wipe an existing small-root first (default wipes).")
    args = ap.parse_args()

    large: Path = args.large_root
    small: Path = args.small_root
    if not large.is_dir():
        print(f"ERROR: large root not found: {large}", file=sys.stderr)
        return 2
    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    cells = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    print(f"Manifest: {args.manifest.relative_to(REPO_ROOT)}  ({len(cells)} cells)")
    print(f"large -> small: {large.relative_to(REPO_ROOT)} -> {small.relative_to(REPO_ROOT)}")
    if args.dry_run:
        print("(dry-run: no writes)\n")
    elif small.exists() and not args.keep:
        shutil.rmtree(small)
        print(f"Wiped existing {small.relative_to(REPO_ROOT)}\n")

    warnings: list[str] = []
    total_bench_rows = 0
    total_gt_rows = 0
    total_images = 0

    for cell in cells:
        kg = cell["kg"]
        mode = cell["mode"]
        qids = set(cell["qids"])
        expected = len(qids)

        if mode == "multimodal":
            src_cell = large / "multimodal" / kg
            dst_cell = small / "multimodal" / kg
            label = f"multimodal/{kg}"
        else:  # text
            sfid = cell["sfid"]
            src_cell = large / kg / sfid
            dst_cell = small / kg / sfid
            label = f"{kg}/{sfid}"

        if not src_cell.is_dir():
            warnings.append(f"{label}: source cell missing ({src_cell}) — skipped")
            continue

        # --- benchmark.jsonl, every language ---
        lang_counts: dict[str, int] = {}
        img_names: set[str] = set()
        for ld in _iter_lang_dirs(src_cell):
            lang = ld.name
            rows = _read_jsonl_parts(ld / "benchmark.jsonl")
            kept = [r for r in rows if r.get("question_id") in qids]
            lang_counts[lang] = len(kept)
            total_bench_rows += len(kept)
            if not args.dry_run:
                _write_jsonl(dst_cell / lang / "benchmark.jsonl" / "part-00000.json", kept)
            if mode == "multimodal":
                for r in kept:
                    for img in (r.get("images") or []):
                        p = img.get("path")
                        if p:
                            img_names.add(Path(p).name)

        # --- gold_triples.jsonl, language-agnostic ---
        gt_kept_n = 0
        gt_src = src_cell / "gold_triples.jsonl"
        if gt_src.is_dir():
            gt_rows = _read_jsonl_parts(gt_src)
            gt_kept = [r for r in gt_rows if r.get("question_id") in qids]
            gt_kept_n = len(gt_kept)
            total_gt_rows += gt_kept_n
            if not args.dry_run:
                _write_jsonl(dst_cell / "gold_triples.jsonl" / "part-00000.json", gt_kept)
        else:
            warnings.append(f"{label}: no gold_triples.jsonl/ in source")

        # --- images (multimodal only) ---
        img_copied = 0
        if mode == "multimodal" and img_names:
            src_img = src_cell / "images"
            dst_img = dst_cell / "images"
            for name in sorted(img_names):
                s = src_img / name
                if s.is_file():
                    if not args.dry_run:
                        dst_img.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(s, dst_img / name)
                    img_copied += 1
                else:
                    warnings.append(f"{label}: image missing in source: {name}")
            total_images += img_copied

        # --- per-cell sanity ---
        langs = sorted(lang_counts)
        eng_n = lang_counts.get("eng")
        if eng_n is not None and eng_n != expected:
            warnings.append(f"{label}: eng kept {eng_n} != manifest n_qids {expected}")
        off = {l: n for l, n in lang_counts.items() if n != (eng_n if eng_n is not None else expected)}
        if off:
            warnings.append(f"{label}: per-language count mismatch vs eng: {off}")

        img_note = f"  images={img_copied}" if mode == "multimodal" else ""
        print(f"  {label:24s} qids={expected:<4d} langs={len(langs):<2d} "
              f"eng={eng_n}  gold_triples={gt_kept_n}{img_note}")

    print(f"\nTotals: benchmark rows={total_bench_rows:,}  "
          f"gold_triples rows={total_gt_rows:,}  images={total_images:,}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  WARN: {w}")
        return 1
    print("\nOK — no warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
