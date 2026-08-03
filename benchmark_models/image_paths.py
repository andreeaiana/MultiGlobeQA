"""Image-path resolution for the multimodal slice.

The published dataset stores ``images[*].path`` as a short relative
segment (``images/<filename>``) interpreted as relative to
``<dataset_root>/multimodal/<kg_source>/``. ``<dataset_root>`` is the
directory whose immediate child is ``multimodal/`` — discovered at
load time by walking up from the JSONL shard's path.

This module also handles a second, prefixed form: any path containing
a ``multimodal/<kg>/`` segment is re-rooted against the actual dataset
root, the prefix in front of ``multimodal/`` is discarded, and the
``<kg>`` segment of the stored path is replaced with the row's
``kg_source``. Both forms therefore resolve to the same location under
a single loader contract.
"""

from __future__ import annotations

from pathlib import Path


def resolve_image_path(stored: str, source_path: Path, kg_source: str) -> str:
    """Resolve a stored image path to an absolute filesystem path.

    Two stored-path formats are accepted:

    1. **Clean form** (published convention): ``images/<filename>`` or
       any path that does not contain a ``multimodal/`` segment.
       Resolved as ``<dataset_root>/multimodal/<kg_source>/<stored>``.

    2. **Prefixed form**: any path containing a ``multimodal/<kg>/``
       segment behind some other prefix (e.g.
       ``<prefix>/multimodal/<kg>/images/<file>``). The segment from
       ``multimodal/`` onwards is extracted, its first component (the
       stored ``<kg>``) is replaced with the caller-supplied
       ``kg_source``, and the remainder is composed under the
       discovered ``<dataset_root>``.

    Args:
        stored: The ``path`` value as written on an ``images[*]`` entry.
        source_path: Path of the JSONL shard the row was loaded from
            (typically the leaf ``benchmark.jsonl/`` directory, or any
            descendant thereof). Used to discover ``<dataset_root>``.
        kg_source: The row's ``kg_source`` field — canonical KG name.
            Wins over any KG segment that appears in a prefixed
            stored path so the resolved location matches the row's
            identity rather than the stored prefix's claim.

    Returns:
        The resolved path as a string. Returns the original ``stored``
        value unchanged when ``source_path`` has no ``multimodal/``
        ancestor, when ``kg_source`` is empty, or when ``stored`` is
        an absolute path. The caller sees a normal ``FileNotFoundError``
        at read time if the resolved path doesn't exist.
    """
    if not stored or not kg_source:
        return stored

    stored_path = Path(stored)
    if stored_path.is_absolute():
        # Caller supplied an absolute path; trust it.
        return stored

    src = Path(source_path).resolve()
    dataset_root: Path | None = None
    for ancestor in (src, *src.parents):
        if ancestor.name == "multimodal":
            dataset_root = ancestor.parent
            break
    if dataset_root is None:
        return stored

    parts = stored_path.parts
    if "multimodal" in parts:
        # Prefixed form: skip everything up to and including the
        # ``<kg>`` segment that follows ``multimodal/``.
        idx = parts.index("multimodal")
        after = parts[idx + 1 :]
        if not after:
            return stored
        tail_parts = after[1:] if len(after) > 1 else after
        tail = Path(*tail_parts) if tail_parts else Path(after[0])
    else:
        tail = stored_path

    return str(dataset_root / "multimodal" / kg_source / tail)


__all__ = ["resolve_image_path"]
