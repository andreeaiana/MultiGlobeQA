"""I/O helpers shared between the model section and the eval pipeline.

Exposes:

* :func:`iter_jsonl_rows` — yield decoded rows from either a single
  ``.jsonl`` file or a directory of Spark ``part-*.json`` /
  ``part-*.jsonl`` shards.
* :func:`discover_benchmark_shards` — walk the
  ``<root>/<kg>/<sfid>/<lang>/benchmark.jsonl/`` layout and return leaf
  shard dirs, optionally filtered by language.
* :func:`discover_gold_shards` — walk the
  ``<root>/<kg>/<sfid>/gold_triples.jsonl/`` layout and return leaf
  shard dirs.

The two ``discover_*`` helpers exist because a blind
``rglob("part-*.json")`` over ``benchmark/final/<size>/<kg>/<sfid>/`` would mix
``benchmark.jsonl/`` rows (one schema) and ``gold_triples.jsonl/`` rows
(a different schema). Caller-side row validation would then fail
opaquely. The helpers walk to the layout's leaf-shard granularity so
each call site reads only one schema at a time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from loguru import logger


def iter_jsonl_rows(
    path: Path | str,
    *,
    recursive: bool = False,
) -> Iterator[dict]:
    """Yield JSON-decoded rows from ``path``.

    ``path`` may be either:

    * A single ``.jsonl`` (or ``.json``) file — the file is read line
      by line, each non-empty line decoded as JSON and yielded.
    * A directory — by default, only direct children matching
      ``part-*.json`` and ``part-*.jsonl`` are read (the *leaf shard
      directory* convention — e.g.
      ``benchmark/<kg>/<sfid>/benchmark.jsonl/``). Pass
      ``recursive=True`` to walk the tree recursively (used by the
      template-registry exporter, which iterates the whole benchmark).
      Files are sorted lexicographically for deterministic ordering
      across runs.

    **Why leaf-only is the default**: the share layout
    ``benchmark/<kg>/<sfid>/`` contains *two* sibling shard
    directories (``benchmark.jsonl/`` and ``gold_triples.jsonl/``)
    that hold rows with **different schemas**. A recursive walk over
    the parent silently mixes them and the runner / eval loaders see
    rows that fail their pydantic validation. Leaf-only avoids that
    class of bug; opt-in recursion is for tools that explicitly want
    cross-shard concatenation (and validate per-row).

    Malformed JSON lines are skipped with a loguru warning rather than
    raising; this matches the behaviour of the previous file-only
    loaders. Empty lines are silently skipped.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is a directory containing no shard files
            (when ``recursive=False``, only direct children are
            considered; pass ``recursive=True`` to walk subdirectories).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.is_dir():
        if recursive:
            files = sorted(
                list(p.rglob("part-*.json")) + list(p.rglob("part-*.jsonl"))
            )
            if not files:
                raise ValueError(
                    f"No part-*.json or part-*.jsonl files under directory: {p} "
                    f"(recursive=True)"
                )
        else:
            files = sorted(
                list(p.glob("part-*.json")) + list(p.glob("part-*.jsonl"))
            )
            if not files:
                raise ValueError(
                    f"No part-*.json or part-*.jsonl files in directory: {p}. "
                    f"This iterator is leaf-only by default; pass "
                    f"recursive=True to walk subdirectories, or supply the "
                    f"leaf shard directory (e.g. "
                    f"benchmark/<kg>/<sfid>/benchmark.jsonl/)."
                )
    else:
        files = [p]

    for fp in files:
        with fp.open() as f:
            for lineno, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSON at {}:{}: {}",
                        fp.name,
                        lineno,
                        exc,
                    )
                    continue


def _is_leaf_shard_dir(p: Path) -> bool:
    """Return True if ``p`` is a directory holding ``part-*.json[l]`` shards."""
    if not p.is_dir():
        return False
    for entry in p.iterdir():
        if entry.is_file() and entry.name.startswith("part-") and entry.suffix in (
            ".json",
            ".jsonl",
        ):
            return True
    return False


def _reject_ambiguous_size_root(p: Path) -> None:
    """Refuse a walk root that holds both ``large/`` and ``small/`` trees.

    ``benchmark/final/`` contains two sibling benchmarks (``large/`` and
    ``small/``) that share the same ``question_id`` space; a recursive walk
    from that parent would glob both and silently merge them. Callers must
    name one size.
    """
    if (p / "large").is_dir() and (p / "small").is_dir():
        raise ValueError(
            f"{p} contains both 'large/' and 'small/' benchmark trees; "
            f"point the path at one size explicitly (e.g. {p / 'large'} "
            f"or {p / 'small'})."
        )


def discover_benchmark_shards(
    root: Path | str,
    *,
    language: str | None = None,
) -> list[Path]:
    """Walk ``root`` for ``benchmark.jsonl/`` shard directories.

    Expected layout (the post-edited benchmark share):

        <root>/<kg>/<sfid>/<lang>/benchmark.jsonl/part-*.json

    Args:
        root: Either a single file, a leaf ``benchmark.jsonl/`` shard
            directory, or a higher-level root containing the nested
            ``<kg>/<sfid>/<lang>/benchmark.jsonl/`` layout. The first
            two are returned as-is (single-shard mode); the third is
            walked.
        language: If given, only return shards whose parent directory
            name matches (i.e. ``<root>/<kg>/<sfid>/<language>/benchmark.jsonl``).
            If omitted, every language's shards are returned.

    Returns:
        A list of leaf paths suitable for passing to
        :func:`iter_jsonl_rows`. Sorted for deterministic ordering. The
        list may contain a single file or shard dir (when ``root`` was
        already a leaf) or many shard dirs (when ``root`` was a walked
        higher-level directory).

    Raises:
        FileNotFoundError: ``root`` does not exist.
        ValueError: ``root`` is a directory but the walk produced no
            ``benchmark.jsonl/`` leaves. The error message lists what
            the walker actually found so the user can diagnose mis-pointed
            paths quickly.
    """
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        return [p]
    # Require the leaf-shard short-circuit to match by directory name
    # too — otherwise a mis-pointed path at a sibling shard type (e.g.
    # ``.../gold_triples.jsonl/``) would silently return rows with the
    # wrong schema and the failure would surface much later as a
    # confusing field-mapping error.
    if p.name == "benchmark.jsonl" and _is_leaf_shard_dir(p):
        return [p]
    _reject_ambiguous_size_root(p)
    leaves: list[Path] = []
    for candidate in sorted(p.rglob("benchmark.jsonl")):
        if not _is_leaf_shard_dir(candidate):
            continue
        if language is not None and candidate.parent.name != language:
            continue
        leaves.append(candidate)
    if not leaves:
        observed = sorted({entry.name for entry in p.rglob("*") if entry.is_dir()})[:10]
        raise ValueError(
            f"No benchmark.jsonl/ leaf shard directories found under "
            f"{p}"
            + (f" (language filter={language!r})" if language else "")
            + f". Observed sample subdirs: {observed!r}. Expected layout "
            f"<root>/<kg>/<sfid>/<lang>/benchmark.jsonl/part-*.json."
        )
    return leaves


def discover_gold_shards(root: Path | str) -> list[Path]:
    """Walk ``root`` for ``gold_triples.jsonl/`` shard directories.

    Expected layout:

        <root>/<kg>/<sfid>/gold_triples.jsonl/part-*.json

    Args:
        root: Either a single file, a leaf ``gold_triples.jsonl/`` shard
            directory, or a higher-level root containing the nested
            ``<kg>/<sfid>/gold_triples.jsonl/`` layout.

    Returns:
        A list of leaf paths suitable for passing to
        :func:`iter_jsonl_rows`. Sorted for deterministic ordering.

    Raises:
        FileNotFoundError: ``root`` does not exist.
        ValueError: ``root`` is a directory but the walk produced no
            ``gold_triples.jsonl/`` leaves.
    """
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        return [p]
    # Mirror the name-check from ``discover_benchmark_shards``: a
    # mis-pointed path at a ``benchmark.jsonl/`` directory would
    # otherwise be accepted here and feed question rows where gold
    # triples are expected.
    if p.name == "gold_triples.jsonl" and _is_leaf_shard_dir(p):
        return [p]
    _reject_ambiguous_size_root(p)
    leaves: list[Path] = []
    for candidate in sorted(p.rglob("gold_triples.jsonl")):
        if not _is_leaf_shard_dir(candidate):
            continue
        leaves.append(candidate)
    if not leaves:
        observed = sorted({entry.name for entry in p.rglob("*") if entry.is_dir()})[:10]
        raise ValueError(
            f"No gold_triples.jsonl/ leaf shard directories found under "
            f"{p}. Observed sample subdirs: {observed!r}. Expected layout "
            f"<root>/<kg>/<sfid>/gold_triples.jsonl/part-*.json."
        )
    return leaves
