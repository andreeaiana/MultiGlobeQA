#!/usr/bin/env python3
"""Score every predictions.jsonl under results/runs/ in parallel.

Reuses the exact ``benchmark_eval score`` CLI per cell (correct-by-
construction) but runs cells across a process pool, with ``--language``
(load one language's gold, not all 17) and ``--no-mlflow`` for speed.
Outputs land in a sibling ``scores/`` dir next to each predictions file.

Usage:
    python scripts/score_all.py --validate   # dry-run: derivation check only
    python scripts/score_all.py --workers 10 # score everything
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "results" / "runs"
PY = str(REPO / ".venv" / "bin" / "python")
LANGS = {
    "eng", "bul", "deu", "ell", "fra", "ita", "jpn", "kat", "por_PT",
    "ron", "rus", "spa", "sqi", "tur", "urd", "vie", "zho_CN",
}


def derive(pred: Path):
    """(size, language, benchmark_dir, aliases, out) or None if not derivable."""
    parts = pred.relative_to(RUNS).parts  # <size>/<variant>/.../<lang>/<rest>/predictions.jsonl
    size = parts[0]
    lang_idx = next((i for i in range(len(parts) - 1, -1, -1) if parts[i] in LANGS), None)
    if lang_idx is None:
        return None
    lang = parts[lang_idx]
    rest = parts[lang_idx + 1:-1]  # after lang, before predictions.jsonl
    if not rest:
        return None
    bm = REPO / "benchmark" / "final" / size / Path(*rest)
    aliases = REPO / "output" / "aliases" / f"{lang}.json"
    out = pred.parent / "scores"
    return size, lang, bm, aliases, out


def score_one(task):
    pred, size, lang, bm, aliases, out = task
    cmd = [
        PY, "-m", "benchmark_eval", "score",
        "--predictions", str(pred),
        "--benchmark", f"{bm}/",
        "--aliases", str(aliases),
        "--output", str(out),
        "--language", lang,
        "--no-mlflow",
    ]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    return (str(pred), r.returncode, (r.stderr or "")[-400:] if r.returncode else "")


def build_tasks():
    preds = [Path(p) for p in glob.glob(str(RUNS / "**" / "predictions.jsonl"), recursive=True)]
    tasks, skipped = [], []
    for p in sorted(preds):
        d = derive(p)
        if d is None or not d[2].is_dir() or not d[3].is_file():
            skipped.append((p, d))
            continue
        tasks.append((p, *d))
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--validate", action="store_true", help="derivation dry-run only")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip cells whose scores/aggregate_report.json exists")
    a = ap.parse_args()

    tasks, skipped = build_tasks()
    print(f"{len(tasks)} cells derivable, {len(skipped)} skipped", flush=True)
    for p, d in skipped[:15]:
        reason = "no lang/rest" if d is None else (
            "no benchmark dir" if not d[2].is_dir() else "no aliases")
        print(f"  SKIP ({reason}): {p.relative_to(RUNS)}", flush=True)
    if a.validate:
        # Summarize benchmark dirs hit, to confirm derivation is sane.
        from collections import Counter
        bms = Counter(str(t[3].relative_to(REPO)) for t in tasks)
        print(f"\nDistinct benchmark dirs: {len(bms)}")
        for k, v in sorted(bms.items()):
            print(f"  {v:>4}  {k}")
        return

    if a.skip_done:
        before = len(tasks)
        tasks = [t for t in tasks if not (t[5] / "aggregate_report.json").is_file()]
        print(f"skip-done: {before - len(tasks)} already scored, {len(tasks)} remain", flush=True)

    t0 = time.time()
    done = failed = 0
    n = len(tasks)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(score_one, t) for t in tasks]
        for f in as_completed(futs):
            pred, rc, err = f.result()
            done += 1
            if rc != 0:
                failed += 1
                print(f"FAIL {Path(pred).relative_to(RUNS)}: {err.strip()[:200]}", flush=True)
            if done % 200 == 0 or done == n:
                el = time.time() - t0
                eta = el / done * (n - done)
                print(f"  {done}/{n} ({100*done/n:.0f}%)  {el:.0f}s elapsed  "
                      f"{failed} failed  ETA {eta:.0f}s", flush=True)
    print(f"DONE: {done} cells scored, {failed} failed, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
