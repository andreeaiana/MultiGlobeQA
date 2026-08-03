"""Walk the sampled multilingual benchmark layout.

The sampled tree is::

    benchmark/sampled_multilingual/
      <kg>/sfid_<x>/<lang>/benchmark.jsonl/part-*
      multimodal/<inner_kg>/<lang>/benchmark.jsonl/part-*

This module groups shards into *cells* — one (kg, sfid_dir, language)
unit. Each cell maps to one output file under
``output/postedit/<lang>/<kg>/<sfid_dir>/results.jsonl`` (or
``output/postedit/<lang>/multimodal/<inner_kg>/results.jsonl`` for the
multimodal subset).

The walker intentionally stays read-only over the input tree and does
no normalization — that's :mod:`pipelines.postedit.normalize`'s job.
Pipelines call :func:`iter_cells` to drive per-cell processing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass(slots=True, frozen=True)
class Cell:
    """One (kg, sfid_dir, language) leaf in the sampled tree.

    ``kg`` is the top-level KG bucket name. For the multimodal subtree
    ``kg == "multimodal"`` and ``sfid_dir`` carries the inner KG name
    (``osmh3kg`` or ``worldkg``); the resulting output paths therefore
    nest under ``multimodal/<inner_kg>/``.
    """
    kg: str
    sfid_dir: str       # "sfid_a" … "sfid_n", or inner-kg for multimodal
    language: str
    bench_dir: Path     # path to the ``benchmark.jsonl/`` *directory*

    @property
    def out_relpath(self) -> Path:
        """Relative path under ``output/postedit/<lang>/`` for this cell."""
        return Path(self.kg) / self.sfid_dir / "results.jsonl"


def iter_cells(
    benchmark_root: Path,
    language: str,
    *,
    kg: str | None = None,
    sfid: str | None = None,
) -> Iterator[Cell]:
    """Yield every cell in the sampled tree matching the filters.

    * ``kg`` — restrict to one of ``kwg`` / ``osmh3kg`` / ``worldkg`` /
      ``multimodal``. ``None`` walks all four.
    * ``sfid`` — restrict to one ``sfid_<x>`` directory under the kg.
      Has no effect for the multimodal bucket (which doesn't have
      sfids; its second level is the inner kg).
    """
    if not benchmark_root.is_dir():
        raise FileNotFoundError(f"benchmark_root not found: {benchmark_root}")

    kg_dirs = (
        [benchmark_root / kg] if kg
        else sorted(p for p in benchmark_root.iterdir() if p.is_dir())
    )

    for kg_dir in kg_dirs:
        if not kg_dir.is_dir():
            logger.warning("Skipping missing kg dir: {}", kg_dir)
            continue
        kg_name = kg_dir.name

        if kg_name == "multimodal":
            # multimodal / <inner_kg> / <lang> / benchmark.jsonl
            for inner in sorted(p for p in kg_dir.iterdir() if p.is_dir()):
                lang_dir = inner / language
                bench_dir = lang_dir / "benchmark.jsonl"
                if not bench_dir.is_dir():
                    continue
                yield Cell(
                    kg="multimodal",
                    sfid_dir=inner.name,
                    language=language,
                    bench_dir=bench_dir,
                )
            continue

        # Regular kg: <kg>/sfid_<x>/<lang>/benchmark.jsonl
        sfid_dirs = sorted(
            p for p in kg_dir.iterdir()
            if p.is_dir() and p.name.startswith("sfid_")
        )
        if sfid:
            sfid_dirs = [p for p in sfid_dirs if p.name == sfid]
        for sfid_dir in sfid_dirs:
            lang_dir = sfid_dir / language
            bench_dir = lang_dir / "benchmark.jsonl"
            if not bench_dir.is_dir():
                continue
            yield Cell(
                kg=kg_name,
                sfid_dir=sfid_dir.name,
                language=language,
                bench_dir=bench_dir,
            )


def iter_records(cell: Cell) -> Iterator[dict]:
    """Yield every JSON record from a cell's ``benchmark.jsonl/part-*`` shards."""
    for shard in sorted(cell.bench_dir.glob("part-*")):
        if not shard.is_file():
            continue
        with shard.open(encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                yield json.loads(line)


def output_path_for(out_root: Path, cell: Cell) -> Path:
    """Return the output JSONL path for a given cell."""
    return out_root / cell.language / cell.out_relpath


def iter_output_results(
    out_root: Path,
    languages: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield every per-cell ``results.jsonl`` the pipeline wrote under
    ``out_root/<lang>/``, covering all KGs *including* the
    ``multimodal/<inner_kg>/`` subtree.

    The pipeline writes each cell to
    ``out_root/<lang>/<kg>/<sfid_or_inner_kg>/results.jsonl`` — a uniform
    two-levels-under-the-language-dir layout for both regular KGs
    (``kwg/sfid_a/``) and multimodal (``multimodal/worldkg/``). A single
    ``*/*/results.jsonl`` glob therefore matches both and naturally skips
    stale top-level ``results_*.jsonl`` concatenations and the ``exports/``
    and ``logs/`` siblings.

    * ``languages`` — restrict to these language codes. ``None`` walks every
      language directory found under ``out_root``.
    """
    if not out_root.is_dir():
        return
    if languages is not None:
        lang_dirs = [out_root / lang for lang in languages]
    else:
        lang_dirs = sorted(p for p in out_root.iterdir() if p.is_dir())
    for lang_dir in lang_dirs:
        if lang_dir.is_dir():
            yield from sorted(lang_dir.glob("*/*/results.jsonl"))
