"""KG-path resolution helpers and shared Spark/Sedona JAR coordinates.

* ``SEDONA_PACKAGES`` — coordinates for the Sedona / Delta JARs,
  imported by ``TripleBridge.__init__`` so the JAR set stays in one
  place.
* ``_is_delta_table(path)`` — checks whether a directory has a
  ``_delta_log/`` subdirectory. Used by ``TripleBridge`` to load
  sidecar indexes only when present, and by tests.
* ``resolve_kg_path(kg_source, kg_root)`` — composes the KG snapshot
  path with precedence: ``--kg-root`` arg > ``BENCHMARK_KGS_ROOT``
  env > default ``benchmark/kgs/``. Called from ``cli.py`` to pass
  the resolved path into ``EntityService(TripleBridge(...))``.

Add new behaviour to
:mod:`benchmark_models.kg_tools.triple_bridge` (data layer) or
:mod:`benchmark_models.kg_tools.entity_service` (entity decoding)
rather than here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Sedona packages for Spark 4.x — must match what question_generator uses.
SEDONA_PACKAGES = (
    "org.apache.sedona:sedona-spark-shaded-4.0_2.13:1.8.1,"
    "org.datasyslab:geotools-wrapper:1.8.1-33.1,"
    "io.delta:delta-spark_2.13:4.1.0"
)

# Env var for the per-runner KG root. Production runs typically leave
# this at the default ``benchmark/kgs/`` and pass ``--kg-source`` instead.
BENCHMARK_KGS_ROOT_ENV = "BENCHMARK_KGS_ROOT"
DEFAULT_KGS_ROOT = "benchmark/kgs"


def _is_delta_table(path: str | Path) -> bool:
    """Return True when ``path`` contains a ``_delta_log`` subdirectory
    (the marker for a Delta Lake table). False for bare parquet
    directories or a single parquet file.
    """
    p = Path(path)
    return p.is_dir() and (p / "_delta_log").is_dir()


def resolve_kg_path(
    kg_source: str | None = None,
    kg_root: str | None = None,
) -> str | None:
    """Resolve ``kg_path`` from CLI args + env vars + defaults.

    Resolution order (highest precedence first):
    1. The ``kg_root`` arg (CLI ``--kg-root``).
    2. The ``BENCHMARK_KGS_ROOT`` env var.
    3. ``benchmark/kgs/`` (default).

    Then appends ``kg_source`` (CLI ``--kg-source``). Returns ``None``
    when ``kg_source`` is not provided — caller falls back to the
    single-KG env-var path or to a test fixture.
    """
    if kg_source is None:
        return None
    root = kg_root or os.environ.get(BENCHMARK_KGS_ROOT_ENV, DEFAULT_KGS_ROOT)
    return str(Path(root) / kg_source)


__all__ = [
    "BENCHMARK_KGS_ROOT_ENV",
    "DEFAULT_KGS_ROOT",
    "SEDONA_PACKAGES",
    "_is_delta_table",
    "resolve_kg_path",
]
