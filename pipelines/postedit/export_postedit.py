"""Export post-editing pipeline results to JSONL/CSV/XLSX, split by language.

By default discovers every per-cell ``results.jsonl`` under
``output/postedit/<lang>/`` — all KG buckets including the
``multimodal/<inner_kg>/`` subtree — via
:func:`cell_walker.iter_output_results`, groups records by language, and writes
a flattened set of files per language plus a combined "all languages" file.

Usage:
    uv run python pipelines/postedit/export_postedit.py
    uv run python pipelines/postedit/export_postedit.py --input output/postedit/results_en.jsonl
    uv run python pipelines/postedit/export_postedit.py --languages en,fr,de
    uv run python pipelines/postedit/export_postedit.py --format csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from pipelines.postedit.cell_walker import iter_output_results
from pipelines.postedit.config import DEFAULT_OUTPUT_DIR

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import openpyxl  # noqa: F401
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


# Columns for the flat row schema (CSV/XLSX)
FLAT_COLUMNS = [
    "question_id",
    "language",
    "vid",
    "tid",
    "sfid",
    "kg_source",
    "original_question",
    "base_template",
    "final_question",
    "decision",
    "edit_made",
    "final_ops",
    "gemini_ops",
    "gemini_final",
    "gemini_ok",
    "deepseek_ops",
    "deepseek_final",
    "deepseek_ok",
    "qwen_ops",
    "qwen_final",
    "qwen_ok",
    "entity_a",
    "entity_b",
]


def load_jsonl(path: Path) -> Iterable[dict]:
    """Yield each JSON object from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def discover_input_files(input_arg: Optional[str], default_dir: Path) -> list[Path]:
    """Resolve the post-edit result files to read.

    With ``--input`` set, use that single file. Otherwise discover every
    per-cell ``results.jsonl`` under ``default_dir/<lang>/`` (all KGs including
    the multimodal subtree) via :func:`iter_output_results`; stale top-level
    ``results_*.jsonl`` concatenations are intentionally ignored.
    """
    if input_arg:
        return [Path(input_arg)]
    return list(iter_output_results(default_dir))


def flatten_record(record: dict) -> dict:
    """Convert one EnsembleRecord JSON into a flat row dict."""
    instantiations = record.get("instantiations") or {}
    entity_values = list(instantiations.values())

    model_outputs = record.get("model_outputs") or {}
    gemini = model_outputs.get("gemini") or {}
    deepseek = model_outputs.get("deepseek") or {}
    qwen = model_outputs.get("qwen") or {}

    final_text = record.get("final_question", "")

    # Determine which model's ops became the final ops. Prefer the explicit
    # selected_model field set by the ensemble. Fall back to ok=True text
    # matching for backwards compatibility with older output files.
    final_ops: list[str] = []
    selected = record.get("selected_model")
    if selected:
        sel_m = model_outputs.get(selected) or {}
        if sel_m.get("ok"):
            final_ops = sel_m.get("ops") or []
    else:
        for m in (gemini, deepseek, qwen):
            if m.get("ok") and m.get("final_text") == final_text:
                final_ops = m.get("ops") or []
                break

    src_meta = record.get("source_metadata") or {}
    return {
        "question_id": record.get("question_id", ""),
        "language": record.get("language", ""),
        "vid": src_meta.get("vid", ""),
        "tid": src_meta.get("tid", ""),
        "sfid": src_meta.get("sfid", ""),
        "kg_source": src_meta.get("kg_source", ""),
        "original_question": record.get("original_question", ""),
        "base_template": record.get("base_template", ""),
        "final_question": final_text,
        "decision": record.get("decision", ""),
        "edit_made": final_text != record.get("original_question", ""),
        "final_ops": ",".join(final_ops),
        "gemini_ops": ",".join(gemini.get("ops") or []),
        "gemini_final": gemini.get("final_text", ""),
        "gemini_ok": gemini.get("ok", False),
        "deepseek_ops": ",".join(deepseek.get("ops") or []),
        "deepseek_final": deepseek.get("final_text", ""),
        "deepseek_ok": deepseek.get("ok", False),
        "qwen_ops": ",".join(qwen.get("ops") or []),
        "qwen_final": qwen.get("final_text", ""),
        "qwen_ok": qwen.get("ok", False),
        "entity_a": entity_values[0] if len(entity_values) > 0 else "",
        "entity_b": entity_values[1] if len(entity_values) > 1 else "",
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    """Write the original (nested) records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict], path: Path) -> None:
    """Write flat rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FLAT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_xlsx_per_language(rows: list[dict], path: Path, language: str) -> None:
    """Write a single-language XLSX with two sheets: Questions, Disagreements.

    The "Disagreements" sheet filters rows where the ensemble fell
    back to Gemini (i.e. ``decision == 'fallback_gemini'``) — those
    are the low-confidence rows worth sampling for spot-review.
    """
    if not HAS_PANDAS or not HAS_OPENPYXL:
        print(f"  (skipping XLSX for {language}: pandas/openpyxl not available)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=FLAT_COLUMNS)
    disagreements = df[df["decision"] == "fallback_gemini"]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Questions", index=False)
        disagreements.to_excel(writer, sheet_name="Disagreements", index=False)


def write_xlsx_all_languages(
    all_rows: list[dict],
    by_language: dict[str, list[dict]],
    path: Path,
) -> None:
    """Write a combined XLSX with All Questions + Per Language Summary sheets."""
    if not HAS_PANDAS or not HAS_OPENPYXL:
        print("  (skipping all-languages XLSX: pandas/openpyxl not available)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=FLAT_COLUMNS)
    summary_data = []
    for lang, rows in sorted(by_language.items()):
        n = len(rows)
        edited = sum(1 for r in rows if r["edit_made"])
        decisions = defaultdict(int)
        for r in rows:
            decisions[r["decision"]] += 1
        fallback = decisions.get("fallback_gemini", 0)
        summary_data.append({
            "language": lang,
            "total": n,
            "edited": edited,
            "edit_rate": round(edited / n, 4) if n else 0.0,
            "unanimous": decisions.get("unanimous", 0),
            "majority": decisions.get("majority", 0),
            "fallback_gemini": fallback,
            "fallback_rate": round(fallback / n, 4) if n else 0.0,
        })
    summary_df = pd.DataFrame(summary_data)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Questions", index=False)
        summary_df.to_excel(writer, sheet_name="Per Language Summary", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file (default: every results_*.jsonl in output/postedit/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR / "exports"),
        help="Output directory (default: output/postedit/exports)",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language codes to include (default: all found)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["jsonl", "csv", "xlsx", "all"],
        help="Output format (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(args.input, DEFAULT_OUTPUT_DIR)
    if not input_files:
        print("Error: no input files found.")
        return 1

    print(f"Reading {len(input_files)} input file(s)...")

    # Load all records and group by language
    by_language: dict[str, list[dict]] = defaultdict(list)
    for path in input_files:
        for record in load_jsonl(path):
            lang = record.get("language", "unknown")
            by_language[lang].append(record)
        print(f"  {path.name}: {sum(len(v) for v in by_language.values())} records so far")

    # Filter languages if requested
    if args.languages:
        wanted = {l.strip() for l in args.languages.split(",")}
        by_language = {k: v for k, v in by_language.items() if k in wanted}

    if not by_language:
        print("Error: no records to export after filtering.")
        return 1

    formats = {"jsonl", "csv", "xlsx"} if args.format == "all" else {args.format}

    # Per-language exports
    all_flat_rows: list[dict] = []
    by_language_flat: dict[str, list[dict]] = {}
    for lang in sorted(by_language):
        records = by_language[lang]
        flat_rows = [flatten_record(r) for r in records]
        by_language_flat[lang] = flat_rows
        all_flat_rows.extend(flat_rows)

        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{lang}] {len(records)} records")
        if "jsonl" in formats:
            jsonl_path = lang_dir / f"postedit_{lang}.jsonl"
            write_jsonl(records, jsonl_path)
            print(f"  wrote {jsonl_path}")
        if "csv" in formats:
            csv_path = lang_dir / f"postedit_{lang}.csv"
            write_csv(flat_rows, csv_path)
            print(f"  wrote {csv_path}")
        if "xlsx" in formats:
            xlsx_path = lang_dir / f"postedit_{lang}.xlsx"
            write_xlsx_per_language(flat_rows, xlsx_path, lang)
            print(f"  wrote {xlsx_path}")

    # Combined all-languages files
    print(f"\n[all] {len(all_flat_rows)} records across {len(by_language)} languages")
    if "jsonl" in formats:
        all_jsonl = output_dir / "all_languages.jsonl"
        # Write the original nested records (not flat) for the JSONL combined version
        all_records = [r for lang in sorted(by_language) for r in by_language[lang]]
        write_jsonl(all_records, all_jsonl)
        print(f"  wrote {all_jsonl}")
    if "csv" in formats:
        all_csv = output_dir / "all_languages.csv"
        write_csv(all_flat_rows, all_csv)
        print(f"  wrote {all_csv}")
    if "xlsx" in formats:
        all_xlsx = output_dir / "all_languages.xlsx"
        write_xlsx_all_languages(all_flat_rows, by_language_flat, all_xlsx)
        print(f"  wrote {all_xlsx}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
