"""Resume helpers for the post-edit pipeline.

Three resume modes (see :data:`config.RESUME_MODES`):

* ``off`` — ignore any existing output file. The pipeline writes from
  scratch (caller-controlled; the helpers below are a no-op for
  ``off``).
* ``skip-done`` — read the existing output's ``question_id`` set; the
  pipeline filters incoming questions to those not already done and
  appends new records to the file.
* ``retry-failed`` — read the existing output, identify rows where any
  model returned ``ok=False`` *or* ``selected_model is None`` (no
  decision could be made). Re-run only those qids. Atomically rewrite
  the output: keep the good rows, replace the failed rows.

Output records are :class:`schema.EnsembleRecord` JSONL — one record
per line.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger


def load_existing_qids(path: Path) -> set[str]:
    """Return the set of ``question_id`` values already present in
    ``path``. Returns an empty set if the file doesn't exist."""
    qids: set[str] = set()
    if not path.is_file():
        return qids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            if qid:
                qids.add(qid)
    return qids


def _row_is_failed(row: dict) -> bool:
    """A row counts as failed if any model has ``ok=False`` or no
    ``selected_model`` was chosen."""
    if row.get("selected_model") is None:
        return True
    for mr in (row.get("model_outputs") or {}).values():
        if not mr.get("ok", False):
            return True
    return False


def split_done_failed(path: Path) -> tuple[set[str], set[str]]:
    """Return ``(done_qids, failed_qids)`` from an existing output.

    ``done_qids`` is the set of qids whose row exists in the file.
    ``failed_qids`` is the subset of those qids that need re-running
    under ``--resume retry-failed``. The relation always holds:
    ``failed_qids ⊆ done_qids``.
    """
    done_qids: set[str] = set()
    failed_qids: set[str] = set()
    if not path.is_file():
        return done_qids, failed_qids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("question_id")
            if not qid:
                continue
            done_qids.add(qid)
            if _row_is_failed(row):
                failed_qids.add(qid)
    return done_qids, failed_qids


def rewrite_dropping_qids(path: Path, qids_to_drop: set[str]) -> int:
    """Atomically rewrite ``path``, removing lines whose qid is in
    ``qids_to_drop``. Returns the number of lines kept.

    Used by ``retry-failed``: the failed rows are dropped here, then
    the pipeline re-runs them and appends fresh rows to the file.
    """
    if not qids_to_drop:
        return sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
    if not path.is_file():
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    n_kept = 0
    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                # Preserve unparseable lines verbatim — caller's call.
                dst.write(line)
                continue
            qid = row.get("question_id")
            if qid in qids_to_drop:
                continue
            dst.write(line if line.endswith("\n") else line + "\n")
            n_kept += 1
    os.replace(tmp, path)
    logger.info(
        "rewrite_dropping_qids: kept {} rows, dropped {} rows from {}",
        n_kept, len(qids_to_drop), path,
    )
    return n_kept
