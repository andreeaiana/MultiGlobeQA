"""Hand-curated smoke-sample selector for the post-edit pipeline.

Picks ~50 representative questions per language from the
multilingual benchmark, mixed across three buckets:

* **NO-EDIT (20 rows)** — templates with single entity, ASCII-only
  names, simple grammatical structure. The LLM should mostly return
  ``{"needs_edit": false}``. Useful as a regression check (if the
  prompt over-edits these, something's wrong with the system prompt).

* **SINGLE-EDIT (20 rows)** — templates that typically need exactly
  one fix when entities are substituted: gender-bearing articles,
  prepositions before vowel-initial entities, postpositive articles
  in Romanian, case agreement, etc. Sampled across multiple TID
  families so we exercise different fix categories.

* **HARD / AMBIGUOUS (10 rows)** — multi-entity questions, entities
  with embedded punctuation (parentheses, commas, decimal coordinates),
  long names, templates with both ``entity_*`` and ``value``
  placeholders. These stress the validation + brace-stripping logic.

Output:

* ``output/postedit/smoke/<lang>.jsonl`` — the sampled rows in their
  original schema (drop-in for the pipeline's ``--input``).
* ``output/postedit/smoke/<lang>_manifest.csv`` — bucket + heuristic
  reasoning per qid, so you can review what got picked.

Determinism: fixed seed; re-running produces the same sample.

Usage::

    uv run python pipelines/postedit/select_smoke_sample.py
    uv run python pipelines/postedit/select_smoke_sample.py --languages eng,ron
    uv run python pipelines/postedit/select_smoke_sample.py --per-bucket 20,20,10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from loguru import logger

from pipelines.postedit.cell_walker import iter_cells, iter_records
from pipelines.postedit.config import DEFAULT_BENCHMARK_ROOT


SMOKE_OUT_DIR = Path(__file__).resolve().parents[2] / "output" / "postedit" / "smoke"


# Heuristic regexes for "this entity name is going to stress the
# pipeline" — coordinates in parens, embedded commas, very long names.
_HARD_ENTITY_RE = re.compile(
    r"\([\-\d.]+,\s*[\-\d.]+\)|"          # "(12.34, 56.78)" coordinates
    r",\s+\d|"                             # commas followed by digits
    r"^.{60,}$"                            # ≥60 chars
)

# TID prefixes that tend to NEED an edit on non-English
# (gender flips, preposition contractions, etc.). Used as a hint for
# the SINGLE-EDIT bucket.
_EDIT_PRONE_TID_PREFIXES = (
    "41.", "43.", "44.", "45.", "47.", "60.", "61.", "62.",
)

# TID prefixes that almost never need an edit (simple distance /
# containment / coordinate readout). Used as a hint for the NO-EDIT
# bucket.
_NO_EDIT_FAVOURED_TID_PREFIXES = (
    "1.", "2.", "3.", "24.",
)


def _entity_count(record: dict) -> int:
    """How many entity_* slots are populated."""
    return sum(
        1 for slot in ("a", "b", "c", "d")
        if record.get(f"entity_{slot}_name")
    )


def _bucket(record: dict) -> str:
    """Classify a row into one of the three smoke buckets.

    The classification is heuristic only — when we run the LLM the
    actual decision may not match. The point is to get a *spread* of
    plausibly-different cases, not perfect labels.
    """
    tid = str(record.get("tid", ""))
    n_entities = _entity_count(record)
    names = " ".join(
        str(record.get(f"entity_{s}_name") or "") for s in ("a", "b", "c", "d")
    )

    # Hard: multi-entity (≥3) OR any entity has weird punctuation /
    # coordinates / very long.
    if n_entities >= 3 or _HARD_ENTITY_RE.search(names):
        return "hard"

    # Edit-prone TID family is a likely SINGLE-EDIT candidate.
    if any(tid.startswith(p) for p in _EDIT_PRONE_TID_PREFIXES):
        return "single_edit"

    # Default to NO-EDIT for the simple-template families.
    if any(tid.startswith(p) for p in _NO_EDIT_FAVOURED_TID_PREFIXES):
        return "no_edit"

    # Everything else falls into single_edit (slightly conservative —
    # we'd rather over-pick edit candidates than miss them).
    return "single_edit"


def _classification_reason(record: dict, bucket: str) -> str:
    tid = str(record.get("tid", ""))
    n_entities = _entity_count(record)
    if bucket == "hard":
        if n_entities >= 3:
            return f"multi-entity (n={n_entities})"
        return "entity name has punctuation / coords / is long"
    if bucket == "single_edit":
        return f"edit-prone TID family ({tid})"
    return f"simple TID family ({tid}) + ASCII-only short names"


def _walk_lang(root: Path, lang: str) -> Iterator[dict]:
    """Yield every record across all cells for ``lang`` (skipping
    multimodal — out of scope for normalize.py at the moment)."""
    for cell in iter_cells(root, lang):
        if cell.kg == "multimodal":
            continue
        yield from iter_records(cell)


def select(
    *,
    benchmark_root: Path,
    language: str,
    n_no_edit: int,
    n_single_edit: int,
    n_hard: int,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Return (sampled_rows, manifest_rows) for ``language``."""
    pools: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    for record in _walk_lang(benchmark_root, language):
        n_total += 1
        bucket = _bucket(record)
        pools[bucket].append(record)
    logger.info(
        "Language={}: walked {} rows (no_edit={}, single_edit={}, hard={})",
        language, n_total,
        len(pools["no_edit"]), len(pools["single_edit"]), len(pools["hard"]),
    )

    rng = random.Random(seed)
    sampled: list[dict] = []
    manifest: list[dict] = []

    for bucket, n in (
        ("no_edit", n_no_edit),
        ("single_edit", n_single_edit),
        ("hard", n_hard),
    ):
        candidates = pools.get(bucket) or []
        if len(candidates) < n:
            logger.warning(
                "Language={}: bucket {} has only {} rows, requested {}; "
                "taking all available", language, bucket, len(candidates), n,
            )
            picked = list(candidates)
        else:
            picked = rng.sample(candidates, n)
        for record in picked:
            sampled.append(record)
            manifest.append({
                "question_id": record.get("question_id", ""),
                "vid": record.get("vid", ""),
                "tid": record.get("tid", ""),
                "sfid": record.get("sfid", ""),
                "kg_source": record.get("kg_source", ""),
                "language": record.get("language", language),
                "bucket": bucket,
                "n_entities": _entity_count(record),
                "reason": _classification_reason(record, bucket),
                "question": (record.get("question") or "")[:160],
            })
    return sampled, manifest


def write_outputs(
    language: str,
    sampled: list[dict],
    manifest: list[dict],
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{language}.jsonl"
    csv_path = out_dir / f"{language}_manifest.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if manifest:
        fieldnames = list(manifest[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in manifest:
                writer.writerow(row)

    logger.info("Wrote {} rows → {} (manifest at {})",
                len(sampled), jsonl_path, csv_path)
    return jsonl_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT,
        help="Root of the multilingual benchmark "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--languages", type=str, default="eng,ron",
        help="Comma-separated ISO 639-3 codes to sample from "
             "(default: eng,ron)",
    )
    parser.add_argument(
        "--per-bucket", type=str, default="20,20,10",
        help="Comma-separated counts: no_edit,single_edit,hard "
             "(default: 20,20,10 → 50 total per language)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--out-dir", type=Path, default=SMOKE_OUT_DIR,
        help=f"Output directory (default: {SMOKE_OUT_DIR})",
    )
    args = parser.parse_args()

    counts = [int(x) for x in args.per_bucket.split(",")]
    if len(counts) != 3:
        logger.error("--per-bucket must be three comma-separated ints "
                     "(no_edit,single_edit,hard)")
        return 1
    n_no_edit, n_single_edit, n_hard = counts

    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    for language in languages:
        sampled, manifest = select(
            benchmark_root=args.benchmark_root,
            language=language,
            n_no_edit=n_no_edit,
            n_single_edit=n_single_edit,
            n_hard=n_hard,
            seed=args.seed,
        )
        write_outputs(language, sampled, manifest, args.out_dir)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
