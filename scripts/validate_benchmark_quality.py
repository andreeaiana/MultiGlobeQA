#!/usr/bin/env python3
"""Benchmark quality validation — checks question text, ground truth,
gold triples, and entity consistency per TID × KG.

Reads from data/benchmark_output/ (JSONL export).

Usage:
    uv run python scripts/validate_benchmark_quality.py [--output-dir data/benchmark_output]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    rows = []
    if not os.path.isdir(path):
        return rows
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(path, fname)) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


# ── Check functions ──────────────────────────────────────────────────

PLACEHOLDER_RE = re.compile(r"\{\{[a-zA-Z_0-9]+\}\}")


def check_question(row: dict) -> list[str]:
    """Check question text quality."""
    issues = []
    q = row.get("question", "")
    if not q:
        issues.append("empty question")
        return issues

    # Remaining placeholders
    found = PLACEHOLDER_RE.findall(q)
    if found:
        issues.append(f"placeholders: {found}")

    # "N/A" in question (failed param interpolation)
    if "N/A" in q:
        issues.append('"N/A" in question')

    # Generic fallback names
    for fallback in ("Entity A", "Entity B", "Entity C", "Event A", "Event B"):
        if fallback in q:
            issues.append(f'generic "{fallback}" in question')

    # "the time period" fallback
    if "the time period" in q.lower():
        issues.append('"the time period" not interpolated')

    return issues


def check_ground_truth(row: dict) -> list[str]:
    """Check ground truth value against ground_truth_type."""
    issues = []
    gt = row.get("ground_truth")
    gt_type = row.get("ground_truth_type", "")
    is_true = row.get("is_true_premise")

    # NULL ground truth
    if gt is None or gt == "":
        # OK for false-premise rows
        if is_true == "false" or is_true is None:
            return []
        issues.append("ground_truth is NULL")
        return issues

    # Type-specific checks
    if gt_type in ("distance_meters", "distance_kilometers"):
        try:
            val = float(gt)
            if val < 0:
                issues.append(f"negative distance: {val}")
            if val > 40_075:  # Earth circumference in km
                issues.append(f"distance exceeds Earth circumference: {val}")
        except (ValueError, TypeError):
            issues.append(f"distance not numeric: {gt[:50]}")

    elif gt_type in ("count", "numeric_integer"):
        try:
            val = int(float(gt))
            if val < 0:
                issues.append(f"negative count: {val}")
        except (ValueError, TypeError):
            issues.append(f"count not integer: {gt[:50]}")

    elif gt_type == "boolean":
        if gt.lower() not in ("true", "false"):
            issues.append(f"boolean not true/false: {gt}")

    elif gt_type == "coordinates":
        if "," not in gt:
            issues.append(f"coordinates missing comma: {gt[:50]}")

    elif gt_type in ("entity_list", "set_enumeration"):
        if not gt.startswith("["):
            issues.append(f"list doesn't start with [: {gt[:50]}")

    return issues


def check_entities(row: dict) -> list[str]:
    """Check entity ID/name consistency."""
    issues = []
    a_id = row.get("entity_a_id")
    a_name = row.get("entity_a_name")
    b_id = row.get("entity_b_id")
    b_name = row.get("entity_b_name")

    # For single-entity TIDs, only entity_a matters
    if a_id and a_name:
        if a_id == a_name:
            issues.append("entity_a_id == entity_a_name (missing name resolution)")

    # Duplicate entity pair
    if a_id and b_id and a_id == b_id:
        issues.append("entity_a_id == entity_b_id (self-reference)")

    return issues


def check_gold_triples(row: dict, gt_by_qid: dict) -> list[str]:
    """Check gold triples presence and predicate coverage."""
    issues = []
    qid = row.get("question_id")

    if not qid or qid not in gt_by_qid:
        issues.append("no gold triples")
        return issues

    gt_data = gt_by_qid[qid].get("gold_triples", "")
    if not gt_data:
        issues.append("gold_triples is empty")
        return issues

    # Parse gold triples JSON
    try:
        triples = json.loads(gt_data)
    except (json.JSONDecodeError, TypeError):
        issues.append("gold_triples not valid JSON")
        return issues

    if not triples:
        issues.append("gold_triples is empty list")
        return issues

    # Check predicate coverage
    predicates = {t.get("predicate", "") for t in triples}

    # Should have at least geometry and type predicates
    has_geometry = any(
        "hasGeometry" in p or "spatialObject" in p for p in predicates
    )
    has_wkt = any("asWKT" in p for p in predicates)
    has_type = any("type" in p.lower() for p in predicates)

    if not has_geometry and not has_wkt:
        issues.append("no geometry predicate in gold triples")
    if not has_type:
        issues.append("no rdf:type predicate in gold triples")

    return issues


# ── Main validation ──────────────────────────────────────────────────


def validate(output_dir: str) -> None:
    bm_rows = load_jsonl(os.path.join(output_dir, "benchmark.jsonl"))
    gt_rows = load_jsonl(os.path.join(output_dir, "gold_triples.jsonl"))

    if not bm_rows:
        print("ERROR: No benchmark rows found")
        sys.exit(1)

    gt_by_qid = {r["question_id"]: r for r in gt_rows if "question_id" in r}

    print(f"Loaded {len(bm_rows)} benchmark rows, {len(gt_rows)} gold triple rows\n")

    # Per TID × KG validation
    tid_kg_results: dict[tuple, dict] = defaultdict(
        lambda: {
            "total": 0,
            "question_issues": [],
            "gt_issues": [],
            "entity_issues": [],
            "gold_issues": [],
        }
    )

    for row in bm_rows:
        tid = row.get("tid", "?")
        kg = row.get("kg_source", "?")
        key = (tid, kg)
        r = tid_kg_results[key]
        r["total"] += 1

        q_issues = check_question(row)
        gt_issues = check_ground_truth(row)
        e_issues = check_entities(row)
        g_issues = check_gold_triples(row, gt_by_qid)

        r["question_issues"].extend(q_issues)
        r["gt_issues"].extend(gt_issues)
        r["entity_issues"].extend(e_issues)
        r["gold_issues"].extend(g_issues)

    # Print report
    header = f"{'TID':<8} {'KG':<10} {'Rows':>4} {'Q':>3} {'GT':>3} {'Ent':>3} {'Gold':>4} Status   Issues"
    print("=" * 100)
    print(header)
    print("-" * 100)

    pass_count = 0
    issue_count = 0

    for (tid, kg), r in sorted(tid_kg_results.items()):
        q_bad = len(r["question_issues"])
        gt_bad = len(r["gt_issues"])
        e_bad = len(r["entity_issues"])
        g_bad = len(r["gold_issues"])
        total = r["total"]

        all_ok = q_bad == 0 and gt_bad == 0 and e_bad == 0 and g_bad == 0
        status = "PASS" if all_ok else "ISSUES"

        if all_ok:
            pass_count += 1
        else:
            issue_count += 1

        # Summarize unique issues
        all_issues = set(
            r["question_issues"]
            + r["gt_issues"]
            + r["entity_issues"]
            + r["gold_issues"]
        )
        issue_str = "; ".join(sorted(all_issues)[:3])
        if len(all_issues) > 3:
            issue_str += f" (+{len(all_issues)-3} more)"

        q_mark = "✓" if q_bad == 0 else f"{q_bad}"
        gt_mark = "✓" if gt_bad == 0 else f"{gt_bad}"
        e_mark = "✓" if e_bad == 0 else f"{e_bad}"
        g_mark = "✓" if g_bad == 0 else f"{g_bad}"

        print(
            f"{tid:<8} {kg:<10} {total:>4} {q_mark:>3} {gt_mark:>3} {e_mark:>3} {g_mark:>4} {status:<8} {issue_str}"
        )

    # Summary
    total_rows = len(bm_rows)
    total_gt_match = sum(
        1 for r in bm_rows if r.get("question_id") in gt_by_qid
    )

    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"  TID × KG pairs: {pass_count} PASS, {issue_count} ISSUES")
    print(f"  Total rows: {total_rows}")
    print(
        f"  Gold triples matched: {total_gt_match}/{total_rows}"
        f" ({100*total_gt_match/max(total_rows,1):.0f}%)"
    )
    print(
        f"  Overall: {'ALL PASS' if issue_count == 0 else f'{issue_count} pairs have issues'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate benchmark quality")
    parser.add_argument(
        "--output-dir",
        default="data/benchmark_output",
        help="Path to benchmark output directory",
    )
    args = parser.parse_args()
    validate(args.output_dir)
