"""Reproducibility hashes for an eval run.

The aggregate report carries two provenance hashes so that re-running
the scorer with a different alias store or a re-issued benchmark
share is detectable at a glance:

* ``alias_store_sha256`` — SHA256 of the alias-store JSON file
  contents. The alias store is small (~10 MB) so content-hashing is
  cheap (~50 ms) and gives a strong "did this file change" guarantee.

* ``benchmark_snapshot_sha256`` — SHA256 of a deterministic
  ``(relative_path, size_bytes)`` manifest for every JSONL shard
  under the benchmark root. Content-hashing the full benchmark would
  read tens of GB and take minutes; the manifest is structural only
  and captures any add / remove / size-change without reading row
  bytes. The tradeoff: an in-place edit that preserves byte size is
  invisible to this hash, which is rare in practice (rewriting a
  shard usually changes its compressed length).

Both helpers raise :class:`FileNotFoundError` for a missing input —
silently emitting a ``None`` hash would defeat the point of the
provenance field.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "compute_alias_store_sha256",
    "compute_benchmark_snapshot_sha256",
]


def compute_alias_store_sha256(alias_store_path: Path) -> str:
    """Return ``hex(SHA256)`` of the alias-store JSON file contents.

    Uses streaming reads so the implementation works for arbitrarily
    large alias stores without materializing the whole file in
    memory.
    """
    alias_store_path = Path(alias_store_path)
    if not alias_store_path.is_file():
        raise FileNotFoundError(
            f"alias store not found: {alias_store_path}"
        )
    h = hashlib.sha256()
    with alias_store_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_SHARD_CHUNK_BYTES = 1 << 20  # 1 MiB read buffer keeps memory bounded.


def _sha256_of_file(path: Path) -> str:
    """SHA256 of a file's contents, computed in 1 MiB chunks so
    multi-GB shards don't blow up memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_SHARD_CHUNK_BYTES)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_benchmark_snapshot_sha256(benchmark_root: Path) -> str:
    """Return ``hex(SHA256)`` of the
    ``(relpath, size, content_sha256)`` manifest for every JSONL
    shard under ``benchmark_root``.

    Each shard's full content is hashed in 1 MiB chunks and rolled
    into the manifest, so any byte change anywhere changes the
    snapshot id — including content edits that don't alter file size.

    The manifest is sorted by relative path so the output is
    deterministic across reruns regardless of OS / filesystem ordering.

    Captures every ``part-*.json`` file under any
    ``benchmark.jsonl/`` or ``gold_triples.jsonl/`` directory.
    Excludes ``_SUCCESS`` markers and any sibling parquet directories
    so the snapshot id reflects the JSON source-of-truth specifically.

    Performance: ~5-10s extra at run start on a ~2 GB / ~2000-shard
    benchmark. One-time cost; acceptable for reproducibility.
    """
    benchmark_root = Path(benchmark_root)
    if not benchmark_root.exists():
        raise FileNotFoundError(
            f"benchmark root not found: {benchmark_root}"
        )

    entries: list[tuple[str, int, str]] = []
    for shard in benchmark_root.rglob("part-*"):
        if not shard.is_file():
            continue
        # Restrict to JSONL shards (not parquet artefacts under a
        # sibling ``benchmark.parquet/`` tree).
        parent_name = shard.parent.name
        if not (parent_name.endswith(".jsonl") or "jsonl" in parent_name):
            continue
        rel = shard.relative_to(benchmark_root)
        entries.append(
            (str(rel), shard.stat().st_size, _sha256_of_file(shard))
        )

    entries.sort()
    h = hashlib.sha256()
    for rel, size, content_sha in entries:
        h.update(f"{rel}\t{size}\t{content_sha}\n".encode())
    return h.hexdigest()
