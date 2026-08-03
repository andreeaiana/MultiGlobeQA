"""Recover the template-variant FTIDv (paraphrase identity) used to
render each English benchmark row's ``question`` text, and write it
back as ``variant_ftidv``. Recovers the values that filled non-entity
placeholders (``{value}``, ``{cardinal_direction}``, ``{time_period}``,
``{attribute_A}``, ``{value_1}``, ``{value_2}``, ``{projection_1}``,
``{projection_2}``, ``{grid_cell_ID}``) into explicit row fields too —
closing the data loophole where those slot values were lost from the
structured row but baked into the question text.

Inputs:
* ``benchmark/raw_eng/templates.csv`` — the multilingual template lookup
  (one row per FTIDv with ``ftid``, ``eng``, plus all 16 target-lang
  columns). Built by ``scripts.build_template_translation_table``.
* English benchmark shards under ``benchmark/raw_eng/<kg>/<sfid>/
  benchmark.jsonl/part-*.json``.

Output (when ``--apply``):
* Each English benchmark row gains:
  - ``variant_ftidv`` — e.g. ``"1.1.3"`` (str)
  - ``variant_recovery_status`` — one of ``"matched"``, ``"fallback_canonical"``,
    ``"unrenderable"``
  - For non-entity placeholders the template uses, the captured slot
    value (e.g. ``"value": "5"`` for a TID whose template has ``{value}``).
* Existing ``question`` text is left unchanged. Atomic shard rewrite,
  with timestamped backup unless ``--no-backup``.

Default mode is ``--dry-run`` (writes a report at
``output/variant_recovery_report.json``; benchmark JSONL untouched).
``--apply`` writes back into the shards.

Usage::

    # Default: dry-run, generate report only
    uv run python -m pipelines.translation.recover_template_variant

    # Apply (writes back into benchmark.jsonl shards)
    uv run python -m pipelines.translation.recover_template_variant --apply
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from loguru import logger

from pipelines.translation._template_matcher import (
    PLACEHOLDER_RE,
    ENTITY_PLACEHOLDER_TO_FIELD,
    find_variant,
    get_image_slots,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATES_CSV = REPO_ROOT / "benchmark" / "templates.csv"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "benchmark" / "raw_eng"
DEFAULT_REPORT_PATH = REPO_ROOT / "output" / "variant_recovery_report.json"


def load_templates(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Read templates.csv and group ``(ftidv, eng)`` by FTID."""
    out: dict[str, list[tuple[str, str]]] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["ftid"], []).append((r["ftidv"], r["eng"]))
    return out


def iter_shards(benchmark_root: Path) -> Iterable[Path]:
    """Walk every benchmark.jsonl shard, including multimodal.

    Multimodal shards under ``multimodal/<kg>/benchmark.jsonl/`` use
    the same template registry as regular shards; they only differ in
    that some entity slots have images attached, in which case the
    renderer keeps ``[IMAGE_<slot>]`` literal in the question text
    rather than substituting the entity name. The matcher handles
    this via :func:`_template_matcher.get_image_slots`.
    """
    for bj in sorted(benchmark_root.rglob("benchmark.jsonl")):
        if not bj.is_dir():
            continue
        for part in sorted(bj.glob("part-*")):
            if part.is_file():
                yield part


def template_uses_only_entity_placeholders(template: str) -> bool:
    return all(
        ph in ENTITY_PLACEHOLDER_TO_FIELD
        for ph in PLACEHOLDER_RE.findall(template)
    )


def canonical_ftidv(tid: str) -> str:
    """Pick the canonical FTIDv for fallback when no variant matches.
    Convention: the first variant of the first (sub)template — i.e.,
    ``<TID>.1`` (e.g. for TID ``"1.1"`` → ``"1.1.1"``)."""
    return f"{tid}.1"


def recover_for_row(
    row: dict[str, Any],
    candidates: list[tuple[str, str]],
) -> dict[str, Any]:
    """Run matching on one row, return a dict of fields to add (or
    ``{}`` when the row has no template candidates).
    """
    if not candidates:
        return {}
    image_slots = get_image_slots(row)  # empty set for non-multimodal
    # Strict match first.
    matches = find_variant(
        candidates, row, quote_tolerant=False, image_slots=image_slots,
    )
    if len(matches) == 1:
        ftidv, _template, captures = matches[0]
        out = {
            "variant_ftidv": ftidv,
            "variant_recovery_status": "matched",
        }
        out.update(captures)
        return out
    if len(matches) > 1:
        # Multi-match: ambiguous. Conservative — flag and use canonical.
        return {
            "variant_ftidv": canonical_ftidv(row["tid"]),
            "variant_recovery_status": "ambiguous_fallback_canonical",
        }
    # Strict failed — quote-tolerant pass for any residual rows we
    # haven't already fixed via D8.
    matches = find_variant(
        candidates, row, quote_tolerant=True, image_slots=image_slots,
    )
    if len(matches) == 1:
        ftidv, _template, captures = matches[0]
        out = {
            "variant_ftidv": ftidv,
            "variant_recovery_status": "matched_quote_tolerant",
        }
        out.update(captures)
        return out
    if len(matches) > 1:
        return {
            "variant_ftidv": canonical_ftidv(row["tid"]),
            "variant_recovery_status": "ambiguous_fallback_canonical",
        }
    # Truly un-renderable (e.g. TID 104.1 with NULL entity_a_name).
    return {
        "variant_ftidv": canonical_ftidv(row["tid"]),
        "variant_recovery_status": "unrenderable_fallback_canonical",
    }


def process(
    benchmark_root: Path,
    templates_by_tid: dict[str, list[tuple[str, str]]],
    *,
    apply: bool,
    no_backup: bool,
) -> dict[str, Any]:
    """Walk every shard (regular + multimodal), recover variants,
    optionally write back."""
    status_counts = Counter()
    by_tid_status = {}
    n_total = 0
    n_changed = 0
    captured_keys = Counter()
    shards_touched = 0

    for shard in iter_shards(benchmark_root):
        new_rows: list[dict[str, Any]] = []
        shard_changed = 0
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                row = json.loads(line)
                tid = row.get("tid")
                n_total += 1
                if not tid:
                    new_rows.append(row)
                    continue
                candidates = templates_by_tid.get(tid)
                fields = recover_for_row(row, candidates or [])
                if not fields:
                    new_rows.append(row)
                    continue
                status = fields["variant_recovery_status"]
                status_counts[status] += 1
                by_tid_status.setdefault(tid, Counter())[status] += 1
                for k in fields:
                    if k not in {"variant_ftidv", "variant_recovery_status"}:
                        captured_keys[k] += 1
                # Idempotent: if the row already has this exact data,
                # don't count as a change.
                already = all(
                    row.get(k) == v for k, v in fields.items()
                )
                if not already:
                    new_row = dict(row)
                    new_row.update(fields)
                    shard_changed += 1
                    new_rows.append(new_row)
                else:
                    new_rows.append(row)

        if shard_changed > 0:
            shards_touched += 1
            n_changed += shard_changed
            if apply:
                if not no_backup:
                    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                    backup = shard.with_suffix(shard.suffix + f".bak.{ts}")
                    shutil.copy2(shard, backup)
                tmp = shard.with_suffix(shard.suffix + ".tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for r in new_rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                os.replace(tmp, shard)

    return {
        "n_total": n_total,
        "n_changed": n_changed,
        "shards_touched": shards_touched,
        "status_counts": dict(status_counts),
        "captured_field_counts": dict(captured_keys),
        "by_tid_status": {
            tid: dict(c) for tid, c in by_tid_status.items()
        },
    }


@click.command()
@click.option(
    "--templates-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_TEMPLATES_CSV,
    show_default=True,
    help="Multilingual template lookup CSV.",
)
@click.option(
    "--benchmark-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_BENCHMARK_ROOT,
    show_default=True,
    help="Root of the English benchmark tree.",
)
@click.option(
    "--report-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_REPORT_PATH,
    show_default=True,
    help="Where to write the variant-recovery report.",
)
@click.option(
    "--apply", is_flag=True, default=False,
    help="Write recovered fields back into the benchmark.jsonl shards. "
         "Default is dry-run (report only; shards untouched).",
)
@click.option(
    "--no-backup", is_flag=True, default=False,
    help="Skip the timestamped backup before each shard's rewrite. "
         "Only meaningful with --apply.",
)
def main(
    templates_csv: Path,
    benchmark_root: Path,
    report_path: Path,
    apply: bool,
    no_backup: bool,
) -> None:
    """Recover template-variant FTIDv (and slot values) for every row."""
    click.echo(f"Loading templates: {templates_csv}")
    templates_by_tid = load_templates(templates_csv)
    click.echo(f"Walking benchmark: {benchmark_root}")
    if not apply:
        click.echo("[dry-run] no shards will be written.")

    summary = process(
        benchmark_root, templates_by_tid,
        apply=apply, no_backup=no_backup,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    from datetime import timezone as _tz
    tmp.write_text(json.dumps({
        "applied_at": datetime.now(_tz.utc).isoformat(),
        "apply": apply,
        **summary,
    }, indent=2))
    os.replace(tmp, report_path)

    click.echo("")
    click.echo(f"Total rows scanned : {summary['n_total']}")
    click.echo(f"Rows with new fields: {summary['n_changed']}")
    click.echo(f"Shards touched     : {summary['shards_touched']}")
    click.echo("Status breakdown:")
    for s, n in sorted(
        summary["status_counts"].items(), key=lambda x: -x[1]
    ):
        pct = n / summary['n_total'] * 100 if summary['n_total'] else 0
        click.echo(f"  {s:<35} {n:>7} ({pct:.3f}%)")
    if summary["captured_field_counts"]:
        click.echo("Captured slot values:")
        for k, n in sorted(
            summary["captured_field_counts"].items(), key=lambda x: -x[1]
        ):
            click.echo(f"  {k:<25} {n}")
    click.echo("")
    click.echo(f"Report: {report_path}")
    if not apply:
        click.echo("Re-run with --apply to write fields back into the shards.")


if __name__ == "__main__":
    main()
