"""Compute per-language and aggregate statistics from post-editing pipeline output.

By default this discovers every per-cell ``results.jsonl`` the pipeline wrote
under ``output/postedit/<lang>/`` — covering all four KG buckets **including**
the ``multimodal/<inner_kg>/`` subtree — via
:func:`cell_walker.iter_output_results`.

It then joins against the released benchmark (``--final-root``, default
``benchmark/final``) on ``question_id`` to recover each language's true row
universe. Rows present in ``final`` but absent from ``output/postedit`` are
``normalize.py``-skips that were kept as the original (un-post-edited) text;
together with the post-edited rows whose ``final_question`` equals the
original, they make up the "kept as original" count reported against the full
benchmark denominator. The ``selected_model`` / ``decision`` attribution comes
from ``output/postedit`` (``final`` carries neither field).

Usage:
    uv run python scripts/postedit/postedit_statistics.py
    uv run python scripts/postedit/postedit_statistics.py --languages deu,fra
    uv run python scripts/postedit/postedit_statistics.py --no-final   # skip the join
    uv run python scripts/postedit/postedit_statistics.py --input output/postedit/results_en.jsonl
"""

import argparse
import json
import statistics as stats_mod
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

from pipelines.postedit.cell_walker import (
    iter_cells,
    iter_output_results,
    iter_records,
    output_path_for,
)
from pipelines.postedit.config import DEFAULT_FINAL_ROOT, DEFAULT_OUTPUT_DIR, MODEL_NAMES, TAXONOMY
from pipelines.postedit.ensemble import normalize_for_compare


# Edit-distance buckets
EDIT_DISTANCE_BUCKETS = [
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-20", 6, 20),
    ("21+", 21, float("inf")),
]


def load_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def edit_distance(a: str, b: str) -> int:
    """Approximate edit distance via SequenceMatcher (number of changed chars)."""
    if a == b:
        return 0
    sm = SequenceMatcher(None, a, b)
    # Sum up the lengths of replaced/inserted/deleted spans
    distance = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            distance += max(i2 - i1, j2 - j1)
        elif tag == "delete":
            distance += i2 - i1
        elif tag == "insert":
            distance += j2 - j1
    return distance


def bucket_for(distance: int) -> str:
    for label, lo, hi in EDIT_DISTANCE_BUCKETS:
        if lo <= distance <= hi:
            return label
    return "21+"


# Window radius (in characters) used when comparing a neighbourhood around an
# entity marker in the marked input vs. in the model's corrected output.
# Edits more than this many characters from a marker will not be attributed
# to that entity; they land in `both_or_other`.
ENTITY_WINDOW_RADIUS = 10


def classify_entity_position(record: dict, buckets: dict) -> None:
    """Bump one bucket in ``buckets`` based on which entity's neighbourhood
    was edited in ``record``.

    Uses the marker-preserving strings so absolute offsets don't drift after
    an upstream insert/delete: `marked_question` on the input side,
    `model_outputs[selected_model].raw.corrected_question` on the output
    side. Both strings use the `{{<entity-value>}}` marker convention, so
    the 10-char windows around each marker are directly comparable via
    independently-found indices.

    Every edited_record with ≥2 entities lands in exactly one bucket.
    Records with 0 or 1 entities are a separate case (handled by the caller
    via an early return here).
    """
    instantiations = record.get("instantiations") or {}
    values = list(instantiations.values())
    if len(values) < 2:
        return  # single-entity templates are out of scope for this statistic.

    selected = record.get("selected_model")
    if not selected:
        buckets["no_valid_corrected_question"] += 1
        return
    model_out = (record.get("model_outputs") or {}).get(selected) or {}
    corrected_marked = ((model_out.get("raw") or {}).get("corrected_question") or "")
    if not corrected_marked:
        buckets["no_valid_corrected_question"] += 1
        return

    marked = record.get("marked_question", "")
    marker1 = "{{" + values[0] + "}}"
    marker2 = "{{" + values[1] + "}}"

    # Ambiguity: the same marker appears more than once in the marked input.
    # Can't uniquely pair the input marker with the corrected-side marker,
    # so skip with a reason.
    if marked.count(marker1) > 1 or marked.count(marker2) > 1:
        buckets["entity_marker_ambiguous"] += 1
        return

    idx1_m = marked.find(marker1)
    idx2_m = marked.find(marker2)
    if idx1_m < 0 or idx2_m < 0:
        # Marker missing from the marked input — the record is malformed.
        # (Guard is defensive; in practice marked_question is generated from
        # the same template/instantiation pair, so markers should always be
        # present.)
        buckets["malformed_marked_input"] += 1
        return

    idx1_c = corrected_marked.find(marker1)
    idx2_c = corrected_marked.find(marker2)
    if idx1_c < 0 or idx2_c < 0:
        # Selected model dropped one of the entity markers — a real failure
        # mode worth tracking separately from the position outcomes.
        buckets["placeholder_missing_in_final"] += 1
        return

    w1_orig = marked[max(0, idx1_m - ENTITY_WINDOW_RADIUS): idx1_m + len(marker1) + ENTITY_WINDOW_RADIUS]
    w1_final = corrected_marked[max(0, idx1_c - ENTITY_WINDOW_RADIUS): idx1_c + len(marker1) + ENTITY_WINDOW_RADIUS]
    w2_orig = marked[max(0, idx2_m - ENTITY_WINDOW_RADIUS): idx2_m + len(marker2) + ENTITY_WINDOW_RADIUS]
    w2_final = corrected_marked[max(0, idx2_c - ENTITY_WINDOW_RADIUS): idx2_c + len(marker2) + ENTITY_WINDOW_RADIUS]

    d1 = edit_distance(w1_orig, w1_final)
    d2 = edit_distance(w2_orig, w2_final)

    if d1 > 0 and d2 == 0:
        buckets["first_entity_changes"] += 1
    elif d2 > 0 and d1 == 0:
        buckets["second_entity_changes"] += 1
    else:
        # Both windows differ (edit spans / straddles both entities), or
        # both unchanged (edit lies outside both windows → nothing to
        # attribute to either entity). Both cases aggregate here.
        buckets["both_or_other"] += 1


def get_final_ops(record: dict) -> list[str]:
    """Return the ops of the model whose output became the final question.

    Uses ``selected_model`` from the record if present (the explicit name set
    by the ensemble). When a record carries no ``selected_model``, matches
    ``final_question`` against each model's text instead; that path only
    matches successful models so failed-model fallback text never gets
    attributed.
    """
    model_outputs = record.get("model_outputs") or {}

    selected = record.get("selected_model")
    if selected:
        m = model_outputs.get(selected) or {}
        if m.get("ok"):
            return m.get("ops") or []
        return []

    # Text-match fallback for output files that carry no selected_model
    final_text = record.get("final_question", "")
    for name in MODEL_NAMES:
        m = model_outputs.get(name) or {}
        if m.get("ok") and m.get("final_text") == final_text:
            return m.get("ops") or []
    return []


def disagreement_pattern(record: dict) -> Optional[str]:
    """Classify the disagreement pattern among the 3 models, or None if agreed.

    Agreement is measured on normalised text (same rule as the ensemble
    vote: NFC + collapsed whitespace + strip). Whitespace or unicode drift
    between otherwise-identical model outputs therefore does not inflate
    the outlier / three-way-split buckets.
    """
    model_outputs = record.get("model_outputs") or {}
    finals = []
    for name in MODEL_NAMES:
        m = model_outputs.get(name) or {}
        if m.get("ok"):
            finals.append((name, normalize_for_compare(m.get("final_text", ""))))
    if len(finals) < 2:
        return "fewer_than_2_models_succeeded"

    texts = [t for _, t in finals]
    distinct = set(texts)
    if len(distinct) == 1:
        return None  # all agreed
    if len(distinct) == 2 and len(finals) == 3:
        # Find which 2 agreed and which 1 disagreed
        c = Counter(texts)
        majority_text = c.most_common(1)[0][0]
        outlier = next(name for name, t in finals if t != majority_text)
        return f"{outlier}_outlier"
    if len(distinct) == 3:
        return "all_three_different"
    return f"split_{len(distinct)}_of_{len(finals)}"


# ---------------------------------------------------------------------------
# Per-language statistics
# ---------------------------------------------------------------------------
def compute_language_stats(records: list[dict], total_final: Optional[int] = None) -> dict:
    """Compute statistics over a language's post-edit records.

    ``records`` are the ensemble rows from ``output/postedit`` (rows that
    actually went through post-editing). When ``total_final`` is given — the
    language's full row count in the released benchmark — the result also
    reports the benchmark-relative view: how many rows were ``normalize.py``-
    skipped (present in ``final`` but never post-edited, hence kept as the
    original) and how many of the released rows kept their original text.
    """
    n = len(records)
    if n == 0:
        return {"total_questions": 0}

    # 1. Counts
    edited_records = [r for r in records if r.get("final_question") != r.get("original_question")]
    edited_count = len(edited_records)
    # Rows where the whole ensemble failed (no model produced an ok result):
    # decide() returns selected_model=None and falls the text back to the
    # original, so these are always kept-as-original.
    all_models_failed = sum(1 for r in records if r.get("selected_model") is None)

    # 2. Decision distribution
    decision_counts = Counter(r.get("decision", "unknown") for r in records)
    fallback_count = decision_counts.get("fallback_gemini", 0)

    # 3. Edit-type counts (from final ops)
    op_counter: Counter = Counter()
    ops_per_edited: list[int] = []
    for r in edited_records:
        ops = get_final_ops(r)
        op_counter.update(ops)
        ops_per_edited.append(len(ops))

    # Ensure all 13 categories appear (with 0 if absent)
    op_counts = {op: op_counter.get(op, 0) for op in TAXONOMY}

    # 4. Per-model stats
    per_model = {}
    for name in MODEL_NAMES:
        proposed_edit = 0
        ops_dist: Counter = Counter()
        failures = 0
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        for r in records:
            m = (r.get("model_outputs") or {}).get(name) or {}
            if not m.get("ok"):
                failures += 1
            ops = m.get("ops") or []
            if ops:
                proposed_edit += 1
            ops_dist.update(ops)
            prompt_tokens.append(m.get("prompt_tokens", 0))
            completion_tokens.append(m.get("completion_tokens", 0))
        per_model[name] = {
            "proposed_edit_rate": round(proposed_edit / n, 4) if n else 0.0,
            "failure_rate": round(failures / n, 4) if n else 0.0,
            "ops_distribution": {op: ops_dist.get(op, 0) for op in TAXONOMY},
            "avg_prompt_tokens": round(sum(prompt_tokens) / n, 1) if n else 0,
            "avg_completion_tokens": round(sum(completion_tokens) / n, 1) if n else 0,
            "total_prompt_tokens": sum(prompt_tokens),
            "total_completion_tokens": sum(completion_tokens),
        }

    # 5. Pairwise model agreement
    #
    # Use the same normalisation as the ensemble's vote
    # (ensemble.normalize_for_compare: NFC + collapsed whitespace + strip),
    # so that "these two models agreed" means the same thing at stats time
    # as it did at voting time. Raw equality would under-report agreement
    # whenever one model's output differs only by whitespace or unicode
    # normalisation.
    pairwise_agreement = {}
    for a, b in combinations(MODEL_NAMES, 2):
        agreed = 0
        comparable = 0
        for r in records:
            ma = (r.get("model_outputs") or {}).get(a) or {}
            mb = (r.get("model_outputs") or {}).get(b) or {}
            if ma.get("ok") and mb.get("ok"):
                comparable += 1
                if (normalize_for_compare(ma.get("final_text", ""))
                        == normalize_for_compare(mb.get("final_text", ""))):
                    agreed += 1
        pairwise_agreement[f"{a}_vs_{b}"] = {
            "agreed": agreed,
            "comparable": comparable,
            "agreement_rate": round(agreed / comparable, 4) if comparable else 0.0,
        }

    # 6. Per-template stats (tid)
    by_tid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        tid = (r.get("source_metadata") or {}).get("tid", "unknown")
        by_tid[tid].append(r)
    per_tid = []
    for tid, tid_records in by_tid.items():
        tid_n = len(tid_records)
        tid_edited = sum(1 for r in tid_records if r.get("final_question") != r.get("original_question"))
        per_tid.append({
            "tid": tid,
            "total": tid_n,
            "edited": tid_edited,
            "edit_rate": round(tid_edited / tid_n, 4) if tid_n else 0.0,
        })
    per_tid.sort(key=lambda x: x["edit_rate"], reverse=True)

    # 7. Edit distance histogram and average
    distances = []
    bucket_counts: Counter = Counter()
    for r in records:
        d = edit_distance(r.get("original_question", ""), r.get("final_question", ""))
        distances.append(d)
        bucket_counts[bucket_for(d)] += 1
    avg_edit_distance = round(sum(distances) / n, 2) if n else 0.0
    median_edit_distance = round(stats_mod.median(distances), 2) if distances else 0.0

    # 8. Op co-occurrence (within the same edited question's final ops)
    cooccurrence: Counter = Counter()
    for r in edited_records:
        ops = get_final_ops(r)
        unique_ops = sorted(set(ops))
        for a, b in combinations(unique_ops, 2):
            cooccurrence[f"{a}+{b}"] += 1

    # 9. Disagreement pattern tally — only over rows that fell back
    # to Gemini (the ensemble's "low-confidence" bucket), i.e. the rows
    # whose decision == "fallback_gemini".
    disagreement_tally: Counter = Counter()
    for r in records:
        if r.get("decision") != "fallback_gemini":
            continue
        pattern = disagreement_pattern(r) or "agreed"
        disagreement_tally[pattern] += 1

    # 10. Entity-position analysis (which entity's neighbourhood was edited).
    #
    # Operates on marker-preserving strings: `marked_question` for the input
    # side, and the selected model's `raw.corrected_question` for the output
    # side. Both contain `{{<entity-value>}}` markers around each entity, so
    # we can locate each entity's position independently in each string and
    # compare the 10-char windows around the markers without relying on
    # absolute offsets staying aligned after an upstream insert/delete.
    #
    # Every edited_record with ≥2 entities lands in exactly one bucket.
    # Diagnostic buckets (no_valid_corrected_question, placeholder_missing_
    # in_final, entity_marker_ambiguous, malformed_marked_input) account
    # for records we cannot classify, so the sum of all buckets equals the
    # number of edited_records with two or more entities.
    entity_position: dict = {
        "first_entity_changes": 0,
        "second_entity_changes": 0,
        "both_or_other": 0,
        "no_valid_corrected_question": 0,
        "placeholder_missing_in_final": 0,
        "entity_marker_ambiguous": 0,
        "malformed_marked_input": 0,
    }
    for r in edited_records:
        classify_entity_position(r, entity_position)

    # 11. Op transitions (different models proposing different ops for the same question)
    op_transitions: Counter = Counter()
    for r in records:
        per_model_ops = {}
        for name in MODEL_NAMES:
            m = (r.get("model_outputs") or {}).get(name) or {}
            if m.get("ok") and m.get("ops"):
                per_model_ops[name] = tuple(sorted(set(m["ops"])))
        # Compare pairs
        for (a, ops_a), (b, ops_b) in combinations(per_model_ops.items(), 2):
            if ops_a != ops_b:
                key = f"{','.join(ops_a) or 'none'} -> {','.join(ops_b) or 'none'}"
                op_transitions[key] += 1

    # 12. Entity polish audit (from validate_and_strip's relaxation).
    # Each ModelResult carries an ``entity_edits`` list of accepted polish
    # edits — pluralisation, capitalisation, diacritic-fold, bounded
    # edit-distance typo fixes. Empty list when every marker matched
    # byte-identically (the common case). Surface aggregate counts so
    # the audit story is visible without ad-hoc jq.
    entity_polish_total = 0
    entity_polish_by_tier: Counter = Counter()
    entity_polish_by_voter: Counter = Counter()
    entity_polish_entity_counter: Counter = Counter()
    for r in records:
        for name in MODEL_NAMES:
            m = (r.get("model_outputs") or {}).get(name) or {}
            for edit in (m.get("entity_edits") or []):
                entity_polish_total += 1
                tier = edit.get("tier", "unknown")
                entity_polish_by_tier[tier] += 1
                entity_polish_by_voter[name] += 1
                expected = edit.get("expected", "")
                if expected:
                    entity_polish_entity_counter[expected] += 1
    entity_polish_top_entities = [
        {"entity": e, "polish_count": c}
        for e, c in entity_polish_entity_counter.most_common(20)
    ]

    result = {
        "total_questions": n,
        "questions_with_edits": edited_count,
        "edit_rate": round(edited_count / n, 4),
        "all_models_failed": all_models_failed,
        "fallback_gemini_count": fallback_count,
        "fallback_rate": round(fallback_count / n, 4),
        "decision_distribution": dict(decision_counts),
        "edit_op_counts": op_counts,
        "avg_ops_per_edited_question": round(sum(ops_per_edited) / len(ops_per_edited), 2)
            if ops_per_edited else 0.0,
        "per_model": per_model,
        "pairwise_agreement": pairwise_agreement,
        "per_tid_top10": per_tid[:10],
        # Full per-TID totals. Internal (leading underscore) — consumed by
        # compute_aggregate_stats so the cross-language top-10 is computed
        # from every template, not just per-language top-10 lists. Stripped
        # before serialization in main().
        "_per_tid_full": per_tid,
        "edit_distance_avg": avg_edit_distance,
        "edit_distance_median": median_edit_distance,
        "edit_distance_buckets": dict(bucket_counts),
        "op_cooccurrence_top20": dict(cooccurrence.most_common(20)),
        "disagreement_patterns": dict(disagreement_tally),
        "entity_position_changes": entity_position,
        "op_transitions_top10": dict(op_transitions.most_common(10)),
        "entity_polish_total": entity_polish_total,
        "entity_polish_by_tier": dict(entity_polish_by_tier),
        "entity_polish_by_voter": dict(entity_polish_by_voter),
        "entity_polish_top_entities": entity_polish_top_entities,
    }

    # Benchmark-relative view (only when joined against final). ``final`` keeps
    # every source row, so total_final >= n; the difference is normalize-skips
    # that were never post-edited (kept as original). edited_count + all rows
    # that kept their original text == total_final.
    if total_final is not None:
        not_edited = n - edited_count
        normalize_skipped = max(0, total_final - n)
        result["total_final"] = total_final
        result["normalize_skipped"] = normalize_skipped
        result["kept_original_total"] = normalize_skipped + not_edited
        result["edit_rate_over_final"] = (
            round(edited_count / total_final, 4) if total_final else 0.0
        )

    return result


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------
def compute_aggregate_stats(by_language: dict[str, dict]) -> dict:
    """Combine per-language stats into a cross-language report."""
    if not by_language:
        return {}

    total_questions = sum(s["total_questions"] for s in by_language.values())
    total_edited = sum(s["questions_with_edits"] for s in by_language.values())
    total_fallback = sum(s["fallback_gemini_count"] for s in by_language.values())
    total_all_failed = sum(s.get("all_models_failed", 0) for s in by_language.values())
    # Benchmark-relative totals, present only when stats were joined against
    # final (every per-language dict then carries ``total_final``).
    has_final = any("total_final" in s for s in by_language.values())

    # Edit-type heatmap: 13 ops × N languages
    op_heatmap: dict = {op: {} for op in TAXONOMY}
    for lang, s in by_language.items():
        for op, count in s.get("edit_op_counts", {}).items():
            op_heatmap[op][lang] = count

    # Decision breakdown across languages
    decision_by_lang = {
        lang: s.get("decision_distribution", {}) for lang, s in by_language.items()
    }

    # Per-model performance across languages
    model_perf_by_lang = {}
    for name in MODEL_NAMES:
        model_perf_by_lang[name] = {
            lang: {
                "proposed_edit_rate": s["per_model"][name]["proposed_edit_rate"],
                "failure_rate": s["per_model"][name]["failure_rate"],
                "total_prompt_tokens": s["per_model"][name]["total_prompt_tokens"],
                "total_completion_tokens": s["per_model"][name]["total_completion_tokens"],
            }
            for lang, s in by_language.items()
        }

    # Top 10 templates by global edit rate, computed from every TID — not
    # from each language's truncated per_tid_top10. Reads the internal
    # `_per_tid_full` emitted by compute_language_stats; falls back to
    # `per_tid_top10` when a caller-supplied dict carries only the
    # public fields.
    tid_totals: dict = defaultdict(lambda: {"total": 0, "edited": 0})
    for s in by_language.values():
        source = s.get("_per_tid_full") or s.get("per_tid_top10", [])
        for entry in source:
            tid_totals[entry["tid"]]["total"] += entry["total"]
            tid_totals[entry["tid"]]["edited"] += entry["edited"]
    top_templates = [
        {
            "tid": tid,
            "total": d["total"],
            "edited": d["edited"],
            "edit_rate": round(d["edited"] / d["total"], 4) if d["total"] else 0.0,
        }
        for tid, d in tid_totals.items()
    ]
    top_templates.sort(key=lambda x: x["edit_rate"], reverse=True)

    # Cross-language polish-edit aggregation. Each per-language stats
    # dict carries entity_polish_total + by-tier + by-voter + top-N
    # entities; roll them up here. The top-N global list is recomputed
    # by merging per-language counters (so a globally-common polished
    # entity surfaces even if it was rank-11 in every language).
    entity_polish_total_all = sum(
        s.get("entity_polish_total", 0) for s in by_language.values()
    )
    entity_polish_by_tier_all: Counter = Counter()
    entity_polish_by_voter_all: Counter = Counter()
    for s in by_language.values():
        entity_polish_by_tier_all.update(s.get("entity_polish_by_tier", {}))
        entity_polish_by_voter_all.update(s.get("entity_polish_by_voter", {}))
    global_entity_counter: Counter = Counter()
    for s in by_language.values():
        for entry in s.get("entity_polish_top_entities", []):
            global_entity_counter[entry["entity"]] += entry["polish_count"]
    entity_polish_top_global = [
        {"entity": e, "polish_count": c}
        for e, c in global_entity_counter.most_common(20)
    ]

    aggregate = {
        "total_questions": total_questions,
        "total_edited": total_edited,
        "overall_edit_rate": round(total_edited / total_questions, 4) if total_questions else 0.0,
        "total_all_models_failed": total_all_failed,
        "total_fallback_gemini": total_fallback,
        "overall_fallback_rate": round(total_fallback / total_questions, 4) if total_questions else 0.0,
        "languages": sorted(by_language.keys()),
        "language_count": len(by_language),
        "edit_op_heatmap": op_heatmap,
        "decision_by_language": decision_by_lang,
        "per_model_by_language": model_perf_by_lang,
        "top_templates_by_edit_rate": top_templates[:10],
        "entity_polish_total": entity_polish_total_all,
        "entity_polish_by_tier": dict(entity_polish_by_tier_all),
        "entity_polish_by_voter": dict(entity_polish_by_voter_all),
        "entity_polish_top_entities": entity_polish_top_global,
    }

    if has_final:
        total_final = sum(s.get("total_final", s["total_questions"])
                          for s in by_language.values())
        total_skipped = sum(s.get("normalize_skipped", 0) for s in by_language.values())
        total_kept = sum(s.get("kept_original_total", 0) for s in by_language.values())
        aggregate["total_final"] = total_final
        aggregate["total_normalize_skipped"] = total_skipped
        aggregate["total_kept_original"] = total_kept
        aggregate["overall_edit_rate_over_final"] = (
            round(total_edited / total_final, 4) if total_final else 0.0
        )

    return aggregate


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------
def print_console_summary(by_language: dict[str, dict], aggregate: dict) -> None:
    print()
    print("=" * 70)
    print("POST-EDITING STATISTICS")
    print("=" * 70)
    print()
    has_final = "total_final" in aggregate
    print(f"Languages processed:  {aggregate.get('language_count', 0)}")
    print(f"Post-edited rows:     {aggregate.get('total_questions', 0):,}")
    print(f"Total edited:         {aggregate.get('total_edited', 0):,}")
    print(f"Overall edit rate:    {aggregate.get('overall_edit_rate', 0):.1%}  (over post-edited rows)")
    print(f"All-models-failed:    {aggregate.get('total_all_models_failed', 0):,}")
    print(f"Total fallback_gemini:{aggregate.get('total_fallback_gemini', 0):,}")
    print(f"Overall fallback rate:{aggregate.get('overall_fallback_rate', 0):.1%}")
    if has_final:
        print()
        print(f"Benchmark rows (final):    {aggregate.get('total_final', 0):,}")
        print(f"Edit rate over benchmark:  {aggregate.get('overall_edit_rate_over_final', 0):.1%}")
        print(f"Kept as original (final):  {aggregate.get('total_kept_original', 0):,}")
        print(f"  └ normalize-skipped:     {aggregate.get('total_normalize_skipped', 0):,}  (never post-edited)")
    print()
    print("Per Language")
    print("-" * 78)
    if has_final:
        print(f"{'lang':<8} {'final':>8} {'p.edited':>9} {'edit%(f)':>9} "
              f"{'skipped':>9} {'fb %':>7} {'avg dist':>9}")
        for lang in sorted(by_language):
            s = by_language[lang]
            print(
                f"{lang:<8} "
                f"{s.get('total_final', s['total_questions']):>8,} "
                f"{s['total_questions']:>9,} "
                f"{s.get('edit_rate_over_final', s['edit_rate']):>8.1%} "
                f"{s.get('normalize_skipped', 0):>9,} "
                f"{s['fallback_rate']:>6.1%} "
                f"{s['edit_distance_avg']:>9.2f}"
            )
    else:
        print(f"{'lang':<8} {'total':>8} {'edited':>8} {'edit %':>9} {'fb %':>9} {'avg dist':>10}")
        for lang in sorted(by_language):
            s = by_language[lang]
            print(
                f"{lang:<8} "
                f"{s['total_questions']:>8,} "
                f"{s['questions_with_edits']:>8,} "
                f"{s['edit_rate']:>8.1%} "
                f"{s['fallback_rate']:>8.1%} "
                f"{s['edit_distance_avg']:>10.2f}"
            )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single input JSONL file to read instead of discovering per-cell "
             "results.jsonl under output/postedit/ (disables the final join).",
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
        help="Comma-separated language codes to include (default: all found "
             "under output/postedit/).",
    )
    parser.add_argument(
        "--final-root",
        type=str,
        default=str(DEFAULT_FINAL_ROOT),
        help="Released benchmark root to join against for the true row "
             f"universe (default: {DEFAULT_FINAL_ROOT}).",
    )
    parser.add_argument(
        "--no-final",
        action="store_true",
        help="Skip the join against --final-root; report only over the "
             "post-edited rows.",
    )
    return parser.parse_args()


def _discover_languages(out_root: Path) -> list[str]:
    """Language codes present under ``out_root`` (every dir except siblings
    like ``logs``/``exports`` that aren't a language)."""
    if not out_root.is_dir():
        return []
    skip = {"logs", "exports"}
    return sorted(
        p.name for p in out_root.iterdir()
        if p.is_dir() and p.name not in skip
    )


def gather_records(
    out_root: Path,
    final_root: Optional[Path],
    languages: list[str],
    explicit_input: Optional[str],
) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, dict[str, int]]]:
    """Load post-edit records and (optionally) join against the final benchmark.

    Returns ``(by_language_records, final_totals, normalize_skipped_by_kg)``.
    When ``explicit_input`` is set or ``final_root`` is unavailable, the two
    final-derived maps are empty and the records come from native discovery
    under ``out_root``.
    """
    by_language: dict[str, list[dict]] = defaultdict(list)
    final_totals: dict[str, int] = {}
    skipped_by_kg: dict[str, dict[str, int]] = {}

    if explicit_input:
        for record in load_jsonl(Path(explicit_input)):
            by_language[record.get("language", "unknown")].append(record)
        return by_language, final_totals, skipped_by_kg

    if final_root is not None and final_root.is_dir():
        # Drive off the final benchmark: it is the true row universe, and each
        # final cell maps 1:1 to its post-edit results.jsonl (post-edit qids ⊆
        # final qids). Reads each post-edit file exactly once.
        for lang in languages:
            total = 0
            per_kg: Counter = Counter()
            for cell in iter_cells(final_root, lang):
                final_n = sum(1 for _ in iter_records(cell))
                total += final_n
                pe_path = output_path_for(out_root, cell)
                pe_n = 0
                if pe_path.is_file():
                    for record in load_jsonl(pe_path):
                        by_language[lang].append(record)
                        pe_n += 1
                per_kg[cell.kg] += final_n - pe_n
            final_totals[lang] = total
            skipped_by_kg[lang] = dict(per_kg)
        return by_language, final_totals, skipped_by_kg

    # No final join: native discovery of post-edit results only.
    for path in iter_output_results(out_root, languages or None):
        for record in load_jsonl(path):
            by_language[record.get("language", "unknown")].append(record)
    return by_language, final_totals, skipped_by_kg


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_root = DEFAULT_OUTPUT_DIR
    final_root = None if args.no_final else Path(args.final_root)

    if args.languages:
        languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    else:
        languages = _discover_languages(out_root)

    if final_root is not None and not final_root.is_dir():
        print(f"Warning: --final-root {final_root} not found; "
              "reporting over post-edited rows only.")
        final_root = None

    print(f"Reading post-edit results for {len(languages)} language(s)"
          + (f", joining against {final_root}" if final_root else "") + " ...")

    by_language_records, final_totals, skipped_by_kg = gather_records(
        out_root, final_root, languages, args.input,
    )

    # --languages already scopes the final-join path; filter the input/native
    # paths here too so an explicit list always wins.
    if args.languages:
        wanted = set(languages)
        by_language_records = {
            k: v for k, v in by_language_records.items() if k in wanted
        }

    if not by_language_records:
        print("Error: no records to analyse.")
        return 1

    # Compute per-language stats (joined against final row counts when available)
    by_language_stats: dict[str, dict] = {}
    for lang in sorted(by_language_records):
        print(f"  Computing stats for {lang}...")
        by_language_stats[lang] = compute_language_stats(
            by_language_records[lang], total_final=final_totals.get(lang),
        )

    # Write per-language stats files.  Keys starting with "_" are internal
    # (e.g. `_per_tid_full`, used by compute_aggregate_stats) and are not
    # part of the public JSON contract.
    for lang, stats in by_language_stats.items():
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        path = lang_dir / f"statistics_{lang}.json"
        public_stats = {k: v for k, v in stats.items() if not k.startswith("_")}
        if lang in skipped_by_kg:
            public_stats["normalize_skipped_by_kg"] = skipped_by_kg[lang]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(public_stats, f, ensure_ascii=False, indent=2)
        print(f"  wrote {path}")

    # Aggregate stats
    aggregate = compute_aggregate_stats(by_language_stats)
    if skipped_by_kg:
        aggregate["normalize_skipped_by_language_kg"] = skipped_by_kg
    agg_path = output_dir / "statistics_aggregate.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"  wrote {agg_path}")

    print_console_summary(by_language_stats, aggregate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
