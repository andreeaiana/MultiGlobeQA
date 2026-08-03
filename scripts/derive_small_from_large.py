#!/usr/bin/env python3
"""Derive ``results/runs/small/`` leaves by filtering ``results/runs/large/``
to the canonical qids — gap-fill only.

The canonical (``small``) qids are a strict subset of the full (``large``)
qids, and predictions are per-question independent (no tier mixes questions
within a run). So the canonical-qid rows of a ``large`` run ARE valid model
predictions on the ``small`` benchmark — identical to a native small run under
greedy decoding; a valid equivalent sample under temperature > 0. This script
*selects* those rows (no new inference) and writes them to the mirror path
under ``results/runs/small/``.

Gap-fill: a ``large`` leaf is derived only if its mirror path under
``small`` does NOT already exist — so the native ``martin_gpu_results`` tier-3
runs are never overwritten, and re-running is a no-op.

Scoring is a separate step (the derived leaf gets only ``predictions.jsonl``
plus a ``derived_from.txt`` provenance marker; run
``scripts/score_variant.py --size small --skip-done`` afterwards to produce
``scores/``).

Run:
    uv run python scripts/derive_small_from_large.py --dry-run
    uv run python scripts/derive_small_from_large.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LARGE = REPO_ROOT / "results" / "runs" / "large"
DEFAULT_SMALL = REPO_ROOT / "results" / "runs" / "small"
DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "final" / "canonical_qids.jsonl"


def load_canonical(manifest: Path) -> set[str]:
    q: set[str] = set()
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line:
            q |= set(json.loads(line)["qids"])
    return q


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--large-root", type=Path, default=DEFAULT_LARGE)
    ap.add_argument("--small-root", type=Path, default=DEFAULT_SMALL)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.large_root.is_dir():
        print(f"ERROR: large root not found: {args.large_root}")
        return 2
    canon = load_canonical(args.manifest)
    print(f"canonical qids: {len(canon)}")

    leaves = sorted(args.large_root.rglob("predictions.jsonl"))
    derived: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # variant -> [leaves, rows]
    skipped: dict[str, int] = defaultdict(int)                   # already present in small
    empty: list[str] = []

    for pf in leaves:
        rel = pf.relative_to(args.large_root)        # <variant>/.../predictions.jsonl
        variant = rel.parts[0]
        dst = args.small_root / rel
        if dst.exists():
            skipped[variant] += 1
            continue
        if args.dry_run:
            derived[variant][0] += 1
            continue
        kept: list[str] = []
        with pf.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("question_id") in canon:
                    kept.append(line)
        if not kept:
            empty.append(str(rel))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8") as out:
            out.write("\n".join(kept) + "\n")
        (dst.parent / "derived_from.txt").write_text(
            "Derived by scripts/derive_small_from_large.py — canonical-qid filter of\n"
            f"  results/runs/large/{rel}\n"
            f"kept {len(kept)} rows whose question_id is in benchmark/final/"
            "canonical_qids.jsonl.\nNo new inference; re-score against "
            "benchmark/final/small/ via scripts/score_variant.py --size small.\n")
        derived[variant][0] += 1
        derived[variant][1] += len(kept)

    # ---- report ----
    all_variants = sorted(set(derived) | set(skipped))
    print(f"\n{'variant':44s} {'derived':>8s} {'rows':>9s} {'skipped(in small)':>18s}")
    for v in all_variants:
        d_leaves, d_rows = derived.get(v, [0, 0])
        print(f"  {v:42s} {d_leaves:8d} {d_rows:9d} {skipped.get(v, 0):18d}")
    tot_leaves = sum(d[0] for d in derived.values())
    tot_rows = sum(d[1] for d in derived.values())
    tot_skip = sum(skipped.values())
    print(f"\n  TOTAL derived leaves={tot_leaves}  rows={tot_rows}  "
          f"skipped(already in small)={tot_skip}")
    if empty:
        print(f"\n{len(empty)} leaf(s) had 0 canonical rows (skipped): "
              f"{', '.join(empty[:8])}{' ...' if len(empty) > 8 else ''}")
    affected = sorted(v for v in derived if derived[v][0] > 0)
    if affected:
        print(f"\naffected variants (score with score_variant.py --size small "
              f"--skip-done):\n  {' '.join(affected)}")
    if args.dry_run:
        print("\n(dry-run: nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
