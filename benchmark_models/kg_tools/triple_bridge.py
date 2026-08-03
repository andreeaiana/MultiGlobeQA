"""Thin Spark wrapper that queries an RDF-triples Delta table natively.

The on-disk production tables under ``benchmark/kgs/<kg>/`` are pure
``(subject, predicate, object)`` triples (osmh3kg additionally has a
``date`` column); this bridge does not assume any denormalised
entity-summary table. The entity-shaped API the agent's tools want
is materialised on top by
:class:`benchmark_models.kg_tools.entity_service.EntityService`.

Four query primitives:

* :meth:`fetch_triples_for_subject` — all triples about a given
  subject. Backs entity-shaped reads (the 2-hop geometry resolution
  in :class:`~benchmark_models.kg_tools.entity_service.EntityService`).
* :meth:`fetch_triples_pointing_to` — all triples whose object is a
  given value; used to discover the entity that points at a
  geometry node (reverse hop) and for incoming-link queries.
* :meth:`search_object_substring` — fuzzy substring match in the
  ``object`` column, optionally restricted to a predicate set.
  Backs the ``find_entity(name)`` fallback path.
* :meth:`query_distinct_objects` — distinct objects across a
  predicate set (e.g. all rdf:type values → category list).

Sidecar indexes
---------------
The bridge auto-detects two optional Z-ordered Delta tables, when they
are present alongside the KG's triples table:

* ``<kg_path>/subject_index/`` — same triples, Z-ordered on
  ``subject``. Lets ``filter(subject == X)`` get parquet row-group
  min/max pruning even when X is one of tens of millions of distinct
  subjects. Without this, every subject query is a full scan.
* ``<kg_path>/name_index/`` — pre-filtered to label predicates,
  with a lowercased ``name_lower`` column. Backs
  ``search_object_substring`` so fuzzy name search reads from a
  small, indexed table rather than scanning the full triples table.

Both are optional: if absent, the bridge falls back to the raw
triples table (slower but correct).

Spark session config: Delta + Sedona packages (see
``SEDONA_PACKAGES``), 8 GiB driver heap, 8 GiB JVM direct-buffer
cap, shuffle 8 partitions, logging quieted. Sedona is loaded
because some downstream callers may want spatial functions on the
bridge's session; the bridge itself doesn't use Sedona ops.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

# Share the JAR-coordinate constant with ``executor_bridge`` rather
# than declaring a second copy, so both bridges pin exactly the same
# Spark dependency versions and cannot drift apart.
from benchmark_models.kg_tools.executor_bridge import (
    SEDONA_PACKAGES,
    _is_delta_table,
)
from benchmark_models.kg_tools.schemas import KGSchema


class TripleBridge:
    """Persistent Spark/Sedona session that queries a triples table.

    Lifecycle: constructed once at runner startup (~10-15 s cold,
    ~3 s warm with cached jars), shared across all tier-3 sessions in
    the run, closed via :meth:`close` at runner exit.
    """

    def __init__(
        self,
        kg_path: str | Path,
        schema: KGSchema,
        *,
        java_home: str | None = None,
    ) -> None:
        """Construct a triples-backed KG bridge.

        Args:
            kg_path: Path to the KG's Delta table root (e.g.
                ``"benchmark/kgs/kwg"``). The bridge expects the
                table at this path to carry ``(subject, predicate,
                object[, date])`` columns. Also probed for the
                optional sidecar indexes at ``<kg_path>/subject_index/``
                and ``<kg_path>/name_index/``.
            schema: Per-KG predicate vocabulary, see
                :mod:`benchmark_models.kg_tools.schemas`.
            java_home: ``JAVA_HOME`` override for environments where
                Java isn't on the standard locations. Inherits from
                the env if not set.
        """
        # JAVA_HOME bootstrap: explicit arg > existing env > probe a
        # couple of well-known JDK locations. Spark/Sedona require
        # ``JAVA_HOME`` to point at a JDK 17+ install; without it
        # Py4J crashes ~30s in with an opaque "java command not
        # found" trace.
        if java_home:
            os.environ["JAVA_HOME"] = java_home
        elif "JAVA_HOME" not in os.environ:
            for candidate in [
                "/usr/lib/jvm/java-21-openjdk",
                "/usr/lib/jvm/java-17-openjdk",
            ]:
                if os.path.isdir(candidate):
                    os.environ["JAVA_HOME"] = candidate
                    break

        from sedona.spark import SedonaContext

        logger.info("Starting Spark/Sedona session for TripleBridge...")
        # SedonaContext.builder().getOrCreate() returns a session with
        # Sedona's spatial UDFs already registered (via the SedonaSQL
        # extension auto-loaded from spark.jars.packages). A subsequent
        # ``SedonaContext.create(spark)`` call registers them a *second*
        # time, producing dozens of "replaced a previously registered
        # function" WARNs at startup. They're harmless (same UDF
        # definitions both times) but noisy — let the one auto-registration
        # stand.
        self.spark = (
            SedonaContext.builder()
            .appName(f"benchmark_models_kg_{schema.name}")
            .master("local[*]")
            .config("spark.jars.packages", SEDONA_PACKAGES)
            # Delta session extension + catalog. The delta-spark JAR
            # is already in SEDONA_PACKAGES, but Spark only registers
            # the Delta SQL path when these two configs are set —
            # without them, ``spark.read.format("delta")`` raises
            # DELTA_CONFIGURE_SPARK_SESSION_WITH_EXTENSION_AND_CATALOG.
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            # Memory sizing — both heap and JVM off-heap direct buffers
            # need headroom for a persistent agent session.
            #
            # ``spark.driver.memory=16g`` gives the JVM heap room for
            # Spark's Catalyst planner + Parquet column readers across
            # many small per-call queries.
            #
            # ``spark.driver.extraJavaOptions=-XX:MaxDirectMemorySize=16g``
            # caps the JVM's off-heap direct ByteBuffer pool at 16 GiB
            # (default would shadow ``-Xmx``, i.e. the heap value; we
            # set it explicitly so the cap doesn't silently track heap
            # changes).
            #
            # Why 16 GiB rather than less:
            #   * 4 GiB is exhausted by a tier-3a get_entity against
            #     kwg — ~107 parquet shards read in parallel via async
            #     NIO file channels, each holding a direct buffer until
            #     GC. Fatal at 4.28 GiB allocated.
            #   * 8 GiB is exhausted by a tier-3a-image run against
            #     osmh3kg multimodal — the slice's broader name /
            #     category searches scan more shards than kwg, and the
            #     agent's 10 KG-calls-per-question accumulates buffers
            #     across the persistent Spark session faster than GC
            #     reclaims them. Stacktrace: ``Thread-N
            #     OutOfMemoryError: Cannot reserve ~133MB direct buffer
            #     ... limit: 8589934592`` at
            #     ``SimpleAsynchronousFileChannelImpl``.
            # On a 128 GiB SLURM allocation, 16 GiB heap + 16 GiB
            # direct = 32 GiB still leaves ~96 GiB for OS, Python,
            # vLLM client buffers, etc., so both caps have room to be
            # raised again if a larger KG / slice exhausts them too.
            .config("spark.driver.memory", "16g")
            .config("spark.driver.extraJavaOptions", "-XX:MaxDirectMemorySize=16g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("ERROR")
        logger.info("Spark/Sedona session ready")

        self.schema = schema
        self._kg_path = str(kg_path)

        # Resolve the three load paths: main triples + optional
        # subject_index + optional name_index. The main table backs
        # ``fetch_triples_pointing_to`` and ``query_distinct_objects``;
        # the subject index (when present) backs subject-keyed reads;
        # the name index (when present) backs substring search.
        kg_dir = Path(self._kg_path)
        subject_index_path = kg_dir / "subject_index"
        name_index_path = kg_dir / "name_index"

        self._triples_df = self._load_delta(kg_dir, role="main triples")
        self._subject_df = (
            self._load_delta(subject_index_path, role="subject_index")
            if _is_delta_table(subject_index_path)
            else self._triples_df
        )
        if not _is_delta_table(subject_index_path):
            logger.warning(
                "No subject_index/ sidecar at {} for {}; subject-keyed reads "
                "fall back to the full triples table (slow at production scale).",
                subject_index_path, schema.name,
            )
        self._name_df = (
            self._load_delta(name_index_path, role="name_index")
            if _is_delta_table(name_index_path)
            else None
        )
        if self._name_df is None:
            logger.info(
                "No name_index/ sidecar at {}; find_entity(name) falls "
                "back to substring scan over the full triples table.",
                name_index_path,
            )

    def _load_delta(self, path: Path | str, *, role: str) -> Any:
        """Load a Delta table at ``path``. Logs the table cardinality
        as a sanity check that the load succeeded."""
        logger.info("Loading {} from {}", role, path)
        df = self.spark.read.format("delta").load(str(path))
        # Don't call .count() during construction — for the production
        # KGs that's tens of millions of rows × multiple loads. Just
        # log the columns; if a downstream filter fires
        # UNRESOLVED_COLUMN, the schema is what we report.
        logger.info("{} loaded; columns: {}", role, df.columns)
        return df

    # ------------------------------------------------------------------
    # Query primitives
    # ------------------------------------------------------------------

    def fetch_triples_for_subject(
        self,
        subject: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """All triples whose subject equals ``subject``.

        Reads from the subject_index sidecar when available so the
        equality filter gets Z-ordered row-group pruning. Returns
        plain Python dicts (not Spark Row objects) so callers don't
        carry a Spark dependency.
        """
        from pyspark.sql import functions as F

        df = self._subject_df.filter(F.col("subject") == F.lit(subject))
        if limit is not None:
            df = df.limit(limit)
        return [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
            }
            for r in df.collect()
        ]

    def fetch_triples_pointing_to(
        self,
        object_value: str,
        *,
        predicates: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """All triples whose object equals ``object_value``.

        Used by reverse-hop queries — e.g. given a geometry node id,
        find the entity that links to it via
        ``schema.geometry_link_predicates``. When ``predicates`` is
        provided, restricts to that predicate set; otherwise scans
        all predicates.
        """
        from pyspark.sql import functions as F

        df = self._triples_df.filter(F.col("object") == F.lit(object_value))
        if predicates is not None:
            preds = tuple(predicates)
            if preds:
                df = df.filter(F.col("predicate").isin(*preds))
        if limit is not None:
            df = df.limit(limit)
        return [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
            }
            for r in df.collect()
        ]

    def search_object_substring(
        self,
        predicates: Iterable[str],
        substring: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Substring match in ``object`` restricted to a predicate set.

        Backs the ``find_entity(name)`` fuzzy path. Reads from the
        name_index sidecar when available (small pre-filtered Delta
        with a lowercased ``name_lower`` column). Falls back to a
        full-scan substring filter on the triples table when the
        sidecar is absent — slow at production scale.
        """
        from pyspark.sql import functions as F

        preds = tuple(predicates)
        needle = str(substring).lower()

        if self._name_df is not None:
            df = self._name_df.filter(F.col("name_lower").contains(needle))
            if preds:
                df = df.filter(F.col("predicate").isin(*preds))
            df = df.limit(limit)
            # The name_index Delta carries (subject, predicate, object,
            # name_lower) columns, so the original-cased ``object``
            # column is always present — access it directly rather
            # than via the dict-style ``.get(default)`` which Spark Row
            # doesn't support (raises PySparkAttributeError: Attribute
            # 'get' is not supported).
            return [
                {
                    "subject": r["subject"],
                    "predicate": r["predicate"],
                    "object": r["object"],
                }
                for r in df.collect()
            ]

        # Fallback: scan the full triples table. Predicate filter first
        # for column pruning, then case-insensitive contains on object.
        df = self._triples_df
        if preds:
            df = df.filter(F.col("predicate").isin(*preds))
        df = df.filter(F.lower(F.col("object")).contains(needle)).limit(limit)
        return [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
            }
            for r in df.collect()
        ]

    def query_distinct_objects(
        self,
        predicates: Iterable[str],
        *,
        limit: int = 200,
    ) -> list[str]:
        """Distinct ``object`` values across the given predicate set,
        sorted lexicographically for stable output. Backs
        ``list_categories`` (distinct types) and similar discovery
        queries.
        """
        from pyspark.sql import functions as F

        preds = tuple(predicates)
        df = self._triples_df
        if preds:
            df = df.filter(F.col("predicate").isin(*preds))
        rows = (
            df.select(F.col("object"))
            .filter(F.col("object").isNotNull())
            .distinct()
            .orderBy("object")
            .limit(limit)
            .collect()
        )
        return [r["object"] for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle + diagnostics
    # ------------------------------------------------------------------

    @property
    def kg_path(self) -> str:
        """Useful for diagnostics and for tests asserting routing."""
        return self._kg_path

    def get_delta_versions(self) -> dict[str, int]:
        """Return the Delta-table version for the main triples table
        and each sidecar index (subject_index, name_index) when
        present. Used by
        :func:`benchmark_models.snapshot.upgrade_to_versioned` to
        upgrade the paths-only snapshot id to its versioned form so
        re-runs against a different Delta version produce a different
        snapshot id.

        Keys in the returned dict are the absolute filesystem paths
        of the three tables — sorted by
        :func:`benchmark_models.snapshot.build_versioned_snapshot_id`
        for deterministic hashing.

        Uses the ``DeltaTable.forPath`` Python API rather than
        the SQL ``DESCRIBE HISTORY`` form. The SQL form requires the
        path to resolve through Spark's catalog,
        which refuses relative paths and produces
        ``TABLE_OR_VIEW_NOT_FOUND`` for any path that isn't a registered
        table. The Python API takes a literal filesystem path, so it
        works on the bare Delta directories the bridge loads.

        Returns an empty dict when every probe fails — the snapshot
        then stays paths-only with a logger warning per missed table.
        """
        from delta.tables import DeltaTable

        kg_dir = Path(self._kg_path)
        candidates: list[tuple[str, Path]] = [
            ("main", kg_dir),
            ("subject_index", kg_dir / "subject_index"),
            ("name_index", kg_dir / "name_index"),
        ]
        versions: dict[str, int] = {}
        for label, path in candidates:
            if not _is_delta_table(path):
                # Sidecars are optional — log at DEBUG so missing-index
                # runs don't spam WARN. The main table missing IS a
                # problem, but _is_delta_table on the constructor's
                # kg_path would have failed there first.
                logger.debug(
                    "get_delta_versions: skipping {} ({}); not a Delta table",
                    label, path,
                )
                continue
            abs_path = str(path.resolve())
            try:
                row = (
                    DeltaTable.forPath(self.spark, abs_path)
                    .history(1)
                    .select("version")
                    .collect()
                )
                if row:
                    versions[abs_path] = int(row[0]["version"])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "get_delta_versions: failed to read Delta history "
                    "for {} at {} ({}). Snapshot id will omit this table.",
                    label, abs_path, exc,
                )
        return versions

    def close(self) -> None:
        """Stop the Spark session and release resources.

        Tolerates a JVM that's already dead — if the driver crashed
        mid-run (heap OOM, direct-buffer exhaustion, SLURM OOM-killer
        on the SLURM step), the py4j gateway socket is closed and
        ``spark.stop()`` raises ``Py4JNetworkError``/
        ``ConnectionRefusedError``. By the time ``close()`` runs, the
        predictions JSONL is already written; we don't want the cleanup
        path to mask the upstream issue with a fresh traceback, so
        catch and log instead. The warning fires once and points the
        operator at the real failure mode (something killed the JVM).
        """
        if hasattr(self, "spark") and self.spark is not None:
            try:
                self.spark.stop()
                logger.info(
                    "Spark session stopped (TripleBridge for {})",
                    self.schema.name,
                )
            except Exception as exc:  # noqa: BLE001 — must not propagate from cleanup
                logger.warning(
                    "Spark session was already dead at close() ({}). "
                    "Predictions were written before this; investigate "
                    "the JVM lifecycle in run.log / SLURM stderr.",
                    exc,
                )


__all__ = ["TripleBridge"]
