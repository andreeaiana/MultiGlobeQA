#!/usr/bin/env python3
"""Score every predictions.jsonl under one variant+language in one shot.

Walks ``results/runs/<size>/<variant>/**/<language>/.../predictions.jsonl``,
derives the matching benchmark + output paths for each cell, and
dispatches ``benchmark_eval score`` one cell at a time. Outputs land
in a sibling ``scores/`` directory next to each predictions file.

``<size>`` is ``large`` (the full benchmark) or ``small`` (the canonical
sub-sample); it selects both the results sub-tree to walk and the
matching ``benchmark/final/<size>/`` gold tree.

Auto-resolved per cell:
  * --aliases   = output/aliases/<language>.json
  * --benchmark = benchmark/final/<size>/<rest>/   where <rest> is the path
    segment(s) AFTER the <language> directory in the predictions path
    (e.g. predictions at results/runs/small/<variant>/.../eng/kwg/predictions.jsonl
     -> benchmark/final/small/kwg/)
  * --output    = <predictions_dir>/scores/

Usage:
  uv run python scripts/score_variant.py --variant qwen35-27b-plain --language eng
  uv run python scripts/score_variant.py --variant qwen35-a3b-tier3b --language eng --size small
  uv run python scripts/score_variant.py --variant qwen35-27b-image --language eng --dry-run

Pass extra flags through to ``benchmark_eval score`` via --extra (can repeat):
  uv run python scripts/score_variant.py --variant X --language eng --extra --no-mlflow
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "runs"
BENCHMARK_ROOT = REPO_ROOT / "benchmark" / "final"
ALIASES_ROOT = REPO_ROOT / "output" / "aliases"

SIZES = ("large", "small")


def derive_benchmark_path(pred_path: Path, language: str, size: str) -> Path:
    """Map ``.../<language>/<rest>/predictions.jsonl`` -> ``benchmark/final/<size>/<rest>/``.

    Convention: variant prediction paths nest as
    ``results/runs/<size>/<variant>/[<mode>/][seed<N>/]<lang>/<rest>/predictions.jsonl``.
    Everything after the language segment mirrors the benchmark tree under
    the matching ``<size>``.
    """
    parts = pred_path.parts
    try:
        lang_idx = parts.index(language)
    except ValueError:
        raise ValueError(
            f"Language segment {language!r} not found in path: {pred_path}"
        )
    rest = parts[lang_idx + 1 : -1]
    if not rest:
        raise ValueError(
            f"No benchmark sub-tree after {language!r} in path: {pred_path}"
        )
    return (BENCHMARK_ROOT / size).joinpath(*rest)


def find_prediction_files(variant: str, language: str, size: str) -> list[Path]:
    """Find every predictions.jsonl under ``results/runs/<size>/<variant>``
    whose path contains the language segment."""
    variant_dir = RESULTS_ROOT / size / variant
    if not variant_dir.is_dir():
        return []
    return sorted(
        p
        for p in variant_dir.rglob("predictions.jsonl")
        if language in p.parts
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--variant",
        required=True,
        help="Variant name (e.g. qwen35-27b-plain).",
    )
    ap.add_argument(
        "--language",
        required=True,
        help="ISO 639-3 language code (e.g. eng).",
    )
    ap.add_argument(
        "--size",
        choices=SIZES,
        default="large",
        help="Benchmark scope: 'large' (full) or 'small' (canonical sample). "
        "Selects results/runs/<size>/ and benchmark/final/<size>/. "
        "Default: large.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without invoking the scorer.",
    )
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip cells where scores/aggregate_report.json already exists.",
    )
    ap.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra flag to pass through to ``benchmark_eval score`` "
        "(can be repeated; e.g. --extra --no-mlflow).",
    )
    args = ap.parse_args()

    aliases_path = ALIASES_ROOT / f"{args.language}.json"
    if not aliases_path.is_file():
        print(f"ERROR: aliases file not found: {aliases_path}", file=sys.stderr)
        return 2

    preds = find_prediction_files(args.variant, args.language, args.size)
    if not preds:
        print(
            f"No predictions.jsonl found under results/runs/{args.size}/{args.variant}/ "
            f"for language={args.language!r}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Found {len(preds)} prediction file(s) for "
        f"size={args.size!r} variant={args.variant!r} language={args.language!r}"
    )
    print(f"Aliases: {aliases_path.relative_to(REPO_ROOT)}")
    print()

    n_scored = 0
    n_skipped = 0
    n_failed = 0

    for pred in preds:
        out_dir = pred.parent / "scores"
        try:
            bm = derive_benchmark_path(pred, args.language, args.size)
        except ValueError as e:
            print(f"FAIL (path parse): {pred} -- {e}", file=sys.stderr)
            n_failed += 1
            continue
        if not bm.is_dir():
            print(f"FAIL (benchmark missing): {bm} for {pred}", file=sys.stderr)
            n_failed += 1
            continue
        if args.skip_done and (out_dir / "aggregate_report.json").is_file():
            print(f"SKIP (already scored): {pred.relative_to(REPO_ROOT)}")
            n_skipped += 1
            continue

        cmd = [
            "uv", "run", "python", "-m", "benchmark_eval", "score",
            "--predictions", str(pred),
            "--benchmark", f"{bm}/",
            "--aliases", str(aliases_path),
            "--output", str(out_dir),
            "--language", args.language,
            *args.extra,
        ]
        rel = pred.relative_to(REPO_ROOT)
        if args.dry_run:
            print(f"WOULD RUN: {rel}")
            print(f"  benchmark = {bm.relative_to(REPO_ROOT)}/")
            print(f"  output    = {out_dir.relative_to(REPO_ROOT)}/")
            continue

        print(f"\n=== Scoring: {rel} ===")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAIL (scorer exit {result.returncode}): {rel}", file=sys.stderr)
            n_failed += 1
        else:
            n_scored += 1

    print(
        f"\nDone. scored={n_scored} skipped={n_skipped} failed={n_failed}"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
