"""Runner — top-level run loop for benchmark_models.

Reads a benchmark JSONL file, constructs :class:`Example` instances,
calls a :class:`Model`'s ``predict`` / ``predict_batch``, and writes
:class:`Prediction` records to a JSONL output file. Supports:

* **Resume**: reads existing predictions from the output file at
  startup, builds a ``set[str]`` of completed ``question_id``s, and
  skips them in the loop. Each prediction line is flushed immediately
  so a crash mid-run loses at most one record.

* **Batching**: for tier 1 / tier 2, chunks examples into batches of
  ``batch_size`` and calls ``predict_batch`` (which backends like vLLM
  override for throughput). Tier 3 uses per-example ``predict`` because
  the agent loop is inherently sequential.

The runner does NOT own MLflow logging (that will be added in a later
commit); it focuses on the JSONL-in, JSONL-out contract.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from loguru import logger

from benchmark_models.base import Example, Model, Prediction
from benchmark_models.io import discover_benchmark_shards, iter_jsonl_rows

# Callback signature: (current_index, total, latest_prediction) -> None
ProgressCallback = Callable[[int, int, Prediction], None]


# Per-backend batch-size defaults. API backends fan a batch across a
# ThreadPoolExecutor; tier-3 backends run a per-example agent loop.
_DEFAULT_API_BATCH_SIZE = 16
_DEFAULT_TIER3_BATCH_SIZE = 1


def resolve_default_batch_size(config_or_model: object) -> int:
    """Pick a sensible batch size for a given variant config / model.

    Resolution rules:

    * Tier 3 (any backend): 1 — the smolagents agent loop is
      inherently per-example. Tier-3 concurrency is via
      ``max_concurrent`` in ``predict_batch``, not batch_size.
    * vLLM-served (tier 1 / 2): the variant's ``config.batch_size``
      (default 32 on ``VLLMHttpVariantConfig``). vLLM's continuous
      batching pays off at 32-64.
    * API backends (openai/google) tier 1 / 2:
      16. The ThreadPoolExecutor in each backend fans out the batch
      across N workers; 16 saturates API rate limits without
      pathological tail-latency.

    Accepts either a ``VariantConfig`` or a constructed ``Model``
    instance (uses duck typing on ``tier`` / ``backend`` / ``config``).
    """
    # Pull tier + backend whether we were given a config or a Model.
    tier = getattr(config_or_model, "tier", None)
    if tier is None:
        config = getattr(config_or_model, "config", None)
        tier = getattr(config, "tier", None) if config is not None else None
    backend = getattr(config_or_model, "backend", None)
    if backend is None:
        config = getattr(config_or_model, "config", None)
        backend = getattr(config, "backend", None) if config is not None else None

    # Tier 3 is always per-example regardless of backend.
    # NOTE: ``tier`` from a Model instance is "tier3" (literal); from
    # a VariantConfig it's "tool_agent" (the YAML key). Both forms
    # are recognised.
    if tier in ("tier3", "tool_agent"):
        return _DEFAULT_TIER3_BATCH_SIZE

    # vLLM-served variants honour the per-variant config.batch_size.
    if backend == "vllm_http":
        config = getattr(config_or_model, "config", config_or_model)
        return int(getattr(config, "batch_size", 32) or 32)

    # API backends.
    return _DEFAULT_API_BATCH_SIZE


def load_benchmark(
    path: Path | str,
    *,
    language: str | None = None,
) -> list[Example]:
    """Read a benchmark file/shard/share-root and return :class:`Example` instances.

    Args:
        path: Three accepted shapes:

            1. **Single file**: a ``.jsonl`` (or ``.json``) — read line
               by line.
            2. **Leaf shard directory**: a ``benchmark.jsonl/`` directory
               containing ``part-*.json`` files — read directly.
            3. **Share-root directory**: a higher-level root containing
               the nested ``<kg>/<sfid>/<lang>/benchmark.jsonl/`` layout
               (e.g. ``benchmark/final/large/``). Walked via
               :func:`discover_benchmark_shards`; every matching leaf is
               concatenated. Pass ``language`` to restrict the walk to
               one language.
        language: Optional language override. When provided:

            * Filters the share-root walk to ``<lang>``.
            * Sets the language field on every loaded :class:`Example`
              regardless of any per-row ``language`` field.
            * Cross-checks per-row language; logs one WARNING per
              mismatched language to catch wrong-directory mistakes.

    Returns:
        A list of :class:`Example`, one per JSONL row across all
        loaded shards.
    """
    path = Path(path)
    shards = discover_benchmark_shards(path, language=language)
    examples: list[Example] = []
    mismatch_observed: set[str] = set()
    for shard in shards:
        for row in iter_jsonl_rows(shard):
            if language is not None:
                row_lang = row.get("language")
                if (
                    row_lang is not None
                    and row_lang != language
                    and row_lang not in mismatch_observed
                ):
                    mismatch_observed.add(row_lang)
                    logger.warning(
                        "Row in {} carries language={!r} but --language="
                        "{!r} was passed; the override wins, but this "
                        "usually means the wrong directory was selected. "
                        "Verify the share layout matches --language.",
                        shard.name,
                        row_lang,
                        language,
                    )
            examples.append(
                Example.from_benchmark_row(
                    row, language=language, source_path=shard,
                ),
            )
    if len(shards) == 1:
        logger.info("Loaded {} examples from {}", len(examples), shards[0].name)
    else:
        logger.info(
            "Loaded {} examples from {} shard{} under {}",
            len(examples),
            len(shards),
            "" if len(shards) == 1 else "s",
            path,
        )
    return examples


def load_completed_ids(output_path: Path) -> set[str]:
    """Read an existing predictions JSONL and return completed question_ids.

    Used for resume: any question_id already in the output file is
    skipped in the current run.
    """
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                qid = data.get("question_id")
                # Use ``is not None`` so empty-string qids still
                # count as completed; resume should treat *any*
                # populated key as a finished row, not just truthy
                # ones.
                if qid is not None:
                    completed.add(qid)
            except json.JSONDecodeError:
                # Truncated last line from a prior crash — skip.
                continue
    if completed:
        logger.info(
            "Resuming: {} predictions already in {}", len(completed), output_path.name
        )
    return completed


def load_existing_snapshot_id(output_path: Path) -> str | None:
    """Read the most recent prediction's ``snapshot_id`` from an
    existing predictions JSONL, or return ``None`` when the file is
    missing/empty/malformed.

    Used by ``run()`` to warn loudly when the resume target was
    produced under a different snapshot (KG paths or Delta versions
    changed between runs). Resume is purely question_id-based, so
    without this check a regenerated benchmark with identical qids
    silently mixes stale predictions with fresh ones.
    """
    if not output_path.exists():
        return None
    # Read the LAST valid line, not the first — runs in progress may
    # have prepended early predictions before a partial regen, so the
    # most recent line is the most informative.
    last_snapshot_id: str | None = None
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = data.get("snapshot_id")
            if isinstance(sid, str):
                last_snapshot_id = sid
    return last_snapshot_id


def rewrite_dropping_errored(output_path: Path) -> tuple[int, int, Path | None]:
    """Rewrite ``output_path`` excluding rows with ``final_state == "error"``.

    Before rewriting, the current file is copied to a timestamped
    backup ``<name>.<YYYY-MM-DDTHH-MM-SS>.bak`` in the same directory.
    The rewrite itself is atomic: a temp file is written and then
    :func:`os.replace` swaps it into place.

    This is the mechanism behind the ``--retry-errors`` CLI flag: by
    removing errored predictions, the normal resume logic treats their
    ``question_id``s as unanswered and re-runs them on the next pass.
    Non-errored rows (answered / abstained / cap_hit) are preserved
    as-is.

    Args:
        output_path: Path to the predictions JSONL file. If it doesn't
            exist, nothing happens and ``(0, 0, None)`` is returned.

    Returns:
        ``(kept, dropped, backup_path)`` — counts of rows kept and
        dropped, plus the path to the backup (``None`` if no file
        existed).
    """
    if not output_path.exists():
        return 0, 0, None

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    backup_path = output_path.with_suffix(f"{output_path.suffix}.{timestamp}.bak")
    shutil.copy2(output_path, backup_path)
    logger.info("Backed up existing predictions to {}", backup_path.name)

    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    kept = 0
    dropped = 0
    with output_path.open() as inp, tmp_path.open("w") as out:
        for line in inp:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                # Preserve malformed lines untouched (unlikely — the
                # resumer would skip them anyway, but we don't want to
                # silently drop user data).
                out.write(line if line.endswith("\n") else line + "\n")
                kept += 1
                continue
            if data.get("final_state") == "error":
                dropped += 1
                continue
            out.write(line if line.endswith("\n") else line + "\n")
            kept += 1

    os.replace(tmp_path, output_path)
    logger.info(
        "Dropped {} errored predictions, kept {} non-errored (will re-run errored on resume)",
        dropped,
        kept,
    )
    return kept, dropped, backup_path


def run(
    model: Model,
    examples: list[Example],
    output_path: Path | str,
    *,
    batch_size: int | None = None,
    resume: bool = True,
    retry_errors: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> list[Prediction]:
    """Run a model on a list of examples and write predictions to disk.

    Args:
        model: The :class:`Model` to run. Already constructed by the
            registry factory.
        examples: The benchmark questions to run.
        output_path: Filesystem path to the predictions JSONL file.
            Created if it doesn't exist; appended to if it does
            (resume mode).
        batch_size: Number of examples per ``predict_batch`` call.
            When ``None`` (default), the value is resolved via
            :func:`resolve_default_batch_size` — vLLM honours its
            config; tier-1/2 API backends use 16 to activate
            ThreadPoolExecutor batching; tier-3 uses 1 (the agent
            loop is per-example). Explicit values override the
            resolver.
        resume: If ``True`` (default), skip examples whose
            ``question_id`` is already in the output file. When
            ``False``, the existing predictions file (if any) is
            **truncated** before the run begins, so the output file
            contains only this run's predictions. ``run.log`` is
            still append-only across runs.
        retry_errors: If ``True``, rewrite ``output_path`` before the
            main loop to drop any rows with ``final_state == "error"``
            (first copying the original to a timestamped ``.bak``).
            Combined with ``resume=True`` this re-runs just the
            errored questions while preserving all other predictions.
            Ignored when ``resume=False``.
        progress_callback: Optional callable invoked after each
            prediction with ``(current_index, total, prediction)``.
            The CLI uses this for per-question progress display.

    Returns:
        The list of :class:`Prediction` records produced in this run
        (excluding any that were skipped due to resume).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve batch_size from the model when the caller didn't pass an
    # explicit value, so ThreadPoolExecutor batching activates on API
    # backends by default.
    if batch_size is None:
        batch_size = resolve_default_batch_size(model)
        logger.info(
            "resolve_default_batch_size: model={} → batch_size={}",
            getattr(model, "model_id", type(model).__name__),
            batch_size,
        )

    # ``resume=False`` requested: clear any prior predictions file so
    # the output contains only this run's records. Without this, the
    # existing file gets *appended to* (the main loop opens in ``"a"``
    # mode), producing a mixed file with old + new predictions for the
    # same question_ids — observed in oracle_verbalized smoke runs and
    # required ``tail -N`` to disentangle. ``run.log`` keeps its
    # append-only behaviour so cross-run logging history is preserved.
    if not resume and output_path.exists():
        logger.info("resume=False: removing existing predictions file at {}", output_path)
        output_path.unlink()

    # Set up file logging alongside the predictions JSONL. Every run
    # gets a persistent log so warnings, errors, and timing are
    # recoverable after the fact.
    log_path = output_path.parent / "run.log"
    log_id = logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation=None,  # single file per run directory
        mode="a",  # append (resume-friendly)
    )
    logger.info("Run log: {}", log_path)

    try:
        # Retry errored rows: rewrite the JSONL dropping any final_state=error
        # predictions. The normal resume logic below then re-runs their ids.
        if retry_errors and resume:
            kept, dropped, _ = rewrite_dropping_errored(output_path)
            if dropped == 0:
                logger.info("No errored predictions to retry.")

        # Resume: load completed ids and filter examples.
        completed_ids: set[str] = set()
        if resume:
            completed_ids = load_completed_ids(output_path)

            # Warn loudly when the existing predictions were produced
            # under a different snapshot_id (KG paths or Delta versions
            # changed). Resume is purely question_id-based — without
            # this warning, a regenerated benchmark with identical qids
            # would silently mix stale predictions with fresh ones.
            # Warning-only by design: the user keeps agency to continue
            # if the change was intentional.
            existing_sid = load_existing_snapshot_id(output_path)
            current_sid = getattr(model, "snapshot_id", None)
            if (
                existing_sid is not None
                and current_sid is not None
                and existing_sid != current_sid
            ):
                logger.warning(
                    "RESUME SNAPSHOT MISMATCH: existing predictions in "
                    "{} were produced under snapshot_id={!r}; current "
                    "run is using snapshot_id={!r}. The two halves of "
                    "the output file will refer to different benchmark "
                    "states. Consider --no-resume if the benchmark or "
                    "KG content changed between runs; consider "
                    "snapshotting the run directory if the change was "
                    "intentional.",
                    output_path.name,
                    existing_sid,
                    current_sid,
                )

        todo = [ex for ex in examples if ex.question_id not in completed_ids]
        if len(todo) < len(examples):
            logger.info(
                "Skipping {} already-completed examples ({} remaining)",
                len(examples) - len(todo),
                len(todo),
            )
        if not todo:
            logger.info("All examples already completed — nothing to do.")
            return []

        predictions: list[Prediction] = []
        total = len(todo)

        with output_path.open("a") as out_f:
            # Process in batches.
            for i in range(0, len(todo), batch_size):
                batch = todo[i : i + batch_size]
                if batch_size > 1:
                    batch_preds = model.predict_batch(batch)
                else:
                    batch_preds = [model.predict(ex) for ex in batch]

                for pred in batch_preds:
                    line = pred.model_dump_json()
                    out_f.write(line + "\n")
                    out_f.flush()
                    predictions.append(pred)
                    if progress_callback is not None:
                        progress_callback(len(predictions), total, pred)

            logger.info(
                "Wrote {} predictions to {}",
                len(predictions),
                output_path.name,
            )

        return predictions
    finally:
        # Always detach the per-run file handler so repeated calls (e.g.,
        # in tests or multi-variant runner invocations, and any exception
        # / early return path) don't stack handlers.
        logger.remove(log_id)
