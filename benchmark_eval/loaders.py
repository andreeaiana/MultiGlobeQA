"""Load benchmark gold rows and prediction records from JSONL files.

Two entry points:

* :func:`load_gold_rows` — reads a benchmark JSONL (the same file
  the model section's runner loads from) and returns
  ``{question_id: GoldRow}``. ``answer_type`` is resolved as
  follows:

  1. **Registry-first**: when the row's ``tid`` is in the template
     registry (``benchmark_models/schemas/template_registry.json``),
     the registry's ``answer_type`` wins. The registry carries
     ``_TID_TO_CANONICAL_OVERRIDE`` already applied, so a TID whose
     row-level type tag is overloaded (one vocab tag covering several
     canonical types) or disagrees with the shape of its gold is
     canonicalised via the registry without the loader needing to
     re-consult the override map.
  2. **Fallback**: when the TID is not in the registry (typically a
     new TID added to the share before the registry was regenerated),
     ``benchmark_models.schemas.resolve_canonical_answer_type`` is
     called with ``tid`` so the override map *is* consulted at this
     stage. Without that, a brand-new override-eligible TID would
     score under the wrong canonical until the registry is rebuilt.

  ``answer_unit`` is derived per-row from ``ground_truth_type`` via
  ``DEFAULT_UNIT_BY_BENCHMARK_TYPE`` — not from the registry's
  ``answer_unit`` field, which is documented as analysis-rollup-only
  (see ``benchmark_models/schemas/template_registry.py:79-84``).
* :func:`load_predictions` — reads a ``predictions.jsonl`` produced
  by the model section and returns a list of
  :class:`benchmark_models.base.Prediction`.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from benchmark_eval.base import GoldRow
from benchmark_eval.types import default_unit_for
from benchmark_models.backends._error_class import classify_error_message
from benchmark_models.base import Prediction, parse_bool_or_none
from benchmark_models.io import discover_benchmark_shards, iter_jsonl_rows
from benchmark_models.schemas import (
    TemplateRegistry,
    load_template_registry,
    resolve_canonical_answer_type,
)

# Per-TID unit corrections for benchmark rows whose ``ground_truth_type``
# tag is dimensionally wrong. ``answer_unit`` is normally derived per-row
# from ``ground_truth_type`` via ``default_unit_for``; when that tag is a
# factual mislabel, the derived unit poisons scoring (unit_match forces
# EM=0, and the wrong numeric *kind* picks the wrong tolerance ladder).
#
# TID 48.1 ("How large is the area affected by {event} in {entity}?")
# ships ``ground_truth_type='distance_kilometers'`` but its gold is an
# AREA in km^2 (its sibling TID 47.1 correctly ships ``area_km2``). Left
# uncorrected, a physically correct km^2 answer is unwinnable: the gold
# demands 'km' so unit_match is always False. Mapping it to the area
# unit here restores winnability and selects the area tolerance bucket —
# identical handling to a correctly-tagged area row.
#
# TID 56.1 ("What is the perimeter of {entity}?") is the mirror-image
# defect: it ships a bare ``ground_truth_type='numeric'`` (no unit), so a
# km perimeter length is scored with the dimensionless *count* tolerance
# ladder and no km enforcement (a 'miles' answer would pass). Mapping it
# to ``distance_kilometers`` restores km-unit enforcement and the
# magnitude-adaptive distance tolerance. Keyed on the FTID (``tid`` field
# of the benchmark row). 48.1 is large-split only; 56.1 is present in the
# small split too.
_TID_UNIT_CORRECTION: dict[str, str] = {
    "48.1": "area_km2",
    "56.1": "distance_kilometers",
}


def load_gold_rows(
    benchmark_path: Path | str,
    *,
    registry: TemplateRegistry | None = None,
    include_invalid: bool = False,
    language: str | None = None,
) -> dict[str, GoldRow]:
    """Read a benchmark JSONL and return a dict keyed by ``question_id``.

    Dispatches every row on the per-row ``ground_truth_type`` field
    (canonicalized via :func:`benchmark_eval.types.canonicalize_answer_type`).
    Rows missing a ground-truth type are skipped with a warning.

    Args:
        benchmark_path: Path to the benchmark JSONL.
        registry: Optional :class:`TemplateRegistry` for per-TID
            ``answer_unit`` fallback; loaded lazily when omitted.
        include_invalid: When ``False`` (default) drops rows flagged
            with ``is_valid=False``. Set ``True`` to include them for
            diagnostic scoring of the validator's own accuracy.
        language: When given, only load shards for that language (the
            ``<kg>/<sfid>/<language>/`` layer). Omitting it loads every
            language's shards and collapses them last-wins — correct
            (scoring fields are language-invariant) but ~17x more I/O and
            it makes the duplicate-qid guard cry wolf. The eval scores one
            language per cell, so callers should pass it.
    """
    from collections import Counter

    reg = registry if registry is not None else load_template_registry()
    path = Path(benchmark_path)
    out: dict[str, GoldRow] = {}
    n_skipped_invalid = 0
    n_skipped_no_type = 0
    # Track duplicates so overlapping shards (e.g. a partial regen
    # concatenated alongside an old shard) don't silently corrupt the
    # gold set via last-wins assignment.
    n_duplicate_qids = 0
    # Count per-TID skips so a missing-TID warning surfaces the actual
    # TID set rather than just the aggregate — multilingual /
    # multimodal shares may carry new TIDs before the template
    # registry is regenerated, and loud per-TID counts force the
    # operator to notice rather than scoring on a silent subset.
    n_skipped_no_type_by_tid: Counter[str] = Counter()
    # ``discover_benchmark_shards`` handles all three accepted shapes
    # (single file, leaf ``benchmark.jsonl/`` shard dir, share-root
    # tree with the ``<kg>/<sfid>/<lang>/benchmark.jsonl/`` layout)
    # and filters out the sibling ``gold_triples.jsonl/`` shards that
    # have a different schema — a plain recursive ``iter_jsonl_rows``
    # would mix them. The README's ``--benchmark`` contract documents
    # the share-root case explicitly.
    shards = discover_benchmark_shards(path, language=language)
    def _iter_all_rows():
        for shard in shards:
            yield from iter_jsonl_rows(shard)
    for row in _iter_all_rows():
        tid = row.get("tid")
        if tid is None:
            logger.warning("Skipping benchmark row from {}: no tid.", path.name)
            continue
        qid = row.get("question_id")
        if qid is None:
            continue

        # is_valid filter — default-exclude rows the generator's
        # own validator rejected. Opt-in diagnostic path keeps
        # them for re-scoring analysis. Route through
        # parse_bool_or_none so a string-serialised ``"false"``
        # (which would be truthy under naive ``bool(...)``) is
        # handled the same as a real ``False``.
        if not include_invalid and parse_bool_or_none(row.get("is_valid")) is False:
            n_skipped_invalid += 1
            continue

        # Resolve canonical answer-type. Strategy:
        #
        # 1. If the TID is in the registry, prefer the registry's
        #    ``answer_type``. The registry encodes the per-TID
        #    canonical overrides for overloaded vocab tags
        #    (``categorical``, ``grid_cell_id``) *and* benchmark-side
        #    mislabels, where a row's declared type does not match the
        #    shape of its gold. Treating the registry as authoritative
        #    keeps the eval and model pipelines consistent.
        # 2. If the TID is unknown to the registry, fall back to the
        #    per-row ``ground_truth_type`` via the flat vocabulary
        #    (still useful during development / for new TIDs the
        #    registry hasn't been regenerated for yet).
        raw_gt_type = row.get("ground_truth_type")
        try:
            info = reg.get(tid)
            answer_type = info.answer_type
            sfid = info.sfid
        except KeyError:
            sfid = row.get("sfid", "")
            try:
                # Pass ``tid`` so the per-TID override map is
                # consulted at this fallback stage — a new override-
                # eligible TID added to the share before the registry
                # was regenerated would otherwise score under the
                # wrong canonical type.
                answer_type = resolve_canonical_answer_type(
                    raw_gt_type, fallback=None, tid=tid,
                )
            except KeyError:
                n_skipped_no_type += 1
                n_skipped_no_type_by_tid[tid] += 1
                continue

        # Unit: derived from the benchmark-type convention in
        # :data:`DEFAULT_UNIT_BY_BENCHMARK_TYPE`, which only assigns a
        # unit to types that actually carry one
        # (``distance_kilometers``, ``area_km2``, ``numeric_km``,
        # ``polar_coordinates``). For every other type ``answer_unit``
        # is ``None``, which makes ``unit_match`` in ``_score_numeric``
        # default to ``True`` so scoring compares numeric proximity
        # only — the model is neither rewarded nor punished for
        # declaring a unit on a unit-less answer.
        #
        # The template registry also has an ``answer_unit`` field on
        # :class:`TemplateInfo`, but it's documented as informational
        # only (see ``benchmark_models/schemas/template_registry.py``)
        # — for per-TID analysis rollups, not consulted by the
        # scorer. The authoritative unit is the per-row value below.
        # The asymmetry with ``answer_type`` (registry-authoritative)
        # is intentional: ``answer_type`` is a TID property;
        # ``answer_unit`` must reflect the row's actual
        # ``ground_truth_type`` label because a row's unit can only
        # be determined from its declared type.
        # Per-TID unit correction for dimensionally-mislabeled tags
        # (e.g. TID 48.1's area gold tagged distance_kilometers) wins
        # over the per-row default. ``default_unit_for`` maps the
        # (corrected) benchmark-type tag to its canonical unit so the
        # value is identical to a natively-correct row.
        corrected_gt_type = _TID_UNIT_CORRECTION.get(tid, raw_gt_type)
        answer_unit = default_unit_for(corrected_gt_type)

        is_true_premise = parse_bool_or_none(row.get("is_true_premise"))

        if qid in out:
            n_duplicate_qids += 1
        out[qid] = GoldRow(
            question_id=qid,
            tid=tid,
            sfid=sfid,
            answer_type=answer_type,
            answer_unit=answer_unit,
            ground_truth=row.get("ground_truth"),
            strata_cell=row.get("strata_cell"),
            language=row.get("language", "eng"),
            kg_source=row.get("kg_source"),
            is_true_premise=is_true_premise,
        )
    if n_skipped_invalid:
        logger.info(
            "Skipped {} is_valid=False rows from {} (pass --include-invalid to keep)",
            n_skipped_invalid,
            path.name,
        )
    if n_skipped_no_type:
        # Per-TID breakdown rather than aggregate-only — operator
        # sees exactly which TIDs are missing from the registry and
        # can regenerate it.
        per_tid = ", ".join(
            f"{tid}={count}"
            for tid, count in n_skipped_no_type_by_tid.most_common()
        )
        logger.warning(
            "Skipped {} rows with unresolvable ground_truth_type from {}; "
            "per-TID breakdown: [{}]. The template registry probably "
            "needs regenerating: "
            "``uv run python -m scripts.build_template_registry_from_benchmark``",
            n_skipped_no_type,
            path.name,
            per_tid,
        )
    if n_duplicate_qids:
        # Surface duplicate qids — usually indicates overlapping
        # shards (a partial re-generation concatenated onto the
        # original) or accidental cross-language mixing. Last-wins is
        # preserved for symmetry with the pairwise loader; the
        # warning makes the duplication visible.
        logger.warning(
            "load_gold_rows: {} duplicate question_id rows encountered "
            "in {}; last value wins. This usually indicates overlapping "
            "shards or a partial re-run concatenated onto an existing "
            "shard. Inspect the input layout if scoring looks off.",
            n_duplicate_qids,
            path,
        )
    # Always print the skipped counts, even when zero — otherwise a
    # benchmark with all-corrupt rows looks like a clean "Loaded N"
    # and the operator misses the failure.
    logger.info(
        "Loaded {} gold rows from {} ({} skipped: {} invalid + {} no-type)",
        len(out), path.name,
        n_skipped_invalid + n_skipped_no_type,
        n_skipped_invalid, n_skipped_no_type,
    )
    return out


def load_predictions(predictions_path: Path | str) -> list[Prediction]:
    """Read a predictions JSONL or directory of part shards.

    Accepts the same shapes as :func:`load_gold_rows`: a single
    JSONL file or a directory of Spark ``part-*.json`` shards. The
    chunked-parallel runner concatenates shards into one file before
    eval; programmatic callers can skip the concatenation by passing
    a directory directly.

    One :class:`benchmark_models.base.Prediction` per row. Malformed
    rows are skipped with a warning, mirroring
    :func:`benchmark_models.io.iter_jsonl_rows` semantics.
    """
    path = Path(predictions_path)
    out: list[Prediction] = []
    skipped = 0
    n_reclassified = 0
    for row in iter_jsonl_rows(path):
        try:
            pred = Prediction.model_validate(row)
        except Exception as exc:
            skipped += 1
            logger.warning(
                "Skipping malformed prediction in {}: {}",
                path.name,
                exc,
            )
            continue
        # Reclassify request-failure errors that the inference-time
        # classifier mislabelled. ``raw_completion`` preserves the provider
        # message, so we upgrade the class at load time (no re-inference).
        #
        # Two cases, both moving mass into the correct harness bucket:
        #   * ``None``/``unknown`` -> whatever the message says
        #     (``context_overflow`` or transport ``infra``).
        #   * ``code_gen`` -> ``context_overflow`` ONLY. An over-context
        #     400 wrapped by smolagents as ``AgentGenerationError`` /
        #     ``AgentMaxStepsError`` (both code-gen names) shadows the
        #     specific context-overflow message at inference time, so the
        #     class is corrected here. We upgrade only on a
        #     context-overflow marker so a genuine code error whose trace
        #     merely mentions "timeout" is NOT demoted to ``infra``.
        if pred.final_state == "error":
            hit = classify_error_message(str(pred.raw_completion or ""))
            if pred.error_class in (None, "unknown"):
                if hit is not None and hit != pred.error_class:
                    pred = pred.model_copy(update={"error_class": hit})
                    n_reclassified += 1
            elif pred.error_class == "code_gen" and hit == "context_overflow":
                pred = pred.model_copy(update={"error_class": "context_overflow"})
                n_reclassified += 1
        out.append(pred)
    # Always print the skipped count, even when zero — otherwise an
    # all-corrupt input file looks like a clean "Loaded 0 predictions"
    # and the operator misses the failure.
    logger.info(
        "Loaded {} predictions ({} skipped) from {}",
        len(out), skipped, path.name,
    )
    if n_reclassified:
        logger.info(
            "  reclassified {} substrate-error row(s) unknown -> harness "
            "(context_overflow/infra) from raw_completion markers",
            n_reclassified,
        )
    return out


__all__ = ["load_gold_rows", "load_predictions"]
