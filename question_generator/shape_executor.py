"""Execute one (TID, KG) shape end-to-end on a Spark session.

Steps per call:

1. Build per-pool filtered temp views from the KG's entity Delta table.
2. Register the KG's triples Delta table as a temp view.
3. Render placeholders in the main SQL (pools, sampling, seed, caps, join).
4. Execute main SQL → benchmark rows.
5. Register benchmark rows as ``_gt_result`` temp view.
6. Render placeholders in the gold SQL (pool_all views, triples, predicates).
7. Execute gold SQL → per-entity triples.
8. Join gold triples back to benchmark rows by the declared pool_map keys
   (e.g. entity_a_id + entity_b_id), concatenating the per-entity triple
   arrays, and serialise to a JSON string for storage.

The module is deliberately Spark-oriented — pool views are Spark temp views,
joins happen in SQL or via DataFrame ops. If a DuckDB port lands later,
most of this file is mechanical.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pyspark.sql.functions as F
from loguru import logger
from pyspark.sql import DataFrame, SparkSession

from question_generator.shape_loader import RunSpec, TidMeta, TidShape


def _mix_seed(seed: int, *parts: str) -> int:
    """Deterministic md5-fold of parts into seed (PYTHONHASHSEED-independent).

    Returns a positive 31-bit int
    — Spark's rand(seed) is happy with that range.
    """
    h = hashlib.md5("|".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _sample_countries_for_cell(
    strata_df: DataFrame,
    entities_df: DataFrame,
    pool_filters: list[str],
    stratum: str,
    cell_seed: int,
    countries_per_cell: int,
) -> list[str]:
    """Pick ``countries_per_cell`` countries that have ≥ 1 candidate entity
    in **every** pool for this cell.

    Only countries whose intersection across all ``pool_filters`` is
    non-empty for this stratum cell are sampled. Drawing uniformly from
    ``strata.country_iso3.distinct()`` within the stratum instead lets tiny
    countries with zero matching entities win the lottery (e.g. HIC_high
    drawing LVA + RUS yields 11 strata rows and 0 POIs). Asymmetric pools
    (e.g. POI x Region) are handled by intersecting the per-pool country
    sets: a country is eligible only if both pools have entities there.

    Deterministic under ``cell_seed``: same inputs always yield the same
    pick. Returning an empty list means no country satisfies all pool
    filters in this stratum — caller should skip the cell.
    """
    strata_in_cell = strata_df.filter(F.col("strata_cell") == stratum).select(
        "entity_id", "country_iso3"
    )

    candidates: DataFrame | None = None
    for pf in pool_filters:
        nodes = entities_df.filter(pf) if pf and pf.strip() else entities_df
        per_pool_countries = (
            nodes.join(
                strata_in_cell, nodes["id"] == strata_in_cell["entity_id"], "inner"
            )
            .select("country_iso3")
            .distinct()
        )
        candidates = (
            per_pool_countries
            if candidates is None
            else candidates.join(per_pool_countries, "country_iso3", "inner")
        )

    if candidates is None:
        return []

    return [
        r["country_iso3"]
        for r in (
            candidates.orderBy(F.rand(cell_seed))
            .limit(countries_per_cell)
            .collect()
        )
    ]


# --------------------------------------------------------------------------- #
# KG path resolution
# --------------------------------------------------------------------------- #


def _import_kg_paths():
    """Resolve the KG path constants from ``question_generator.config``.

    Imported lazily so tests that don't need Spark or KG data can import this
    module without failing on missing config symbols.
    """
    from question_generator import config as _cfg
    return {
        "kwg": {
            "entities": _cfg.KWG_ENTITIES_PATH,
            "triples": _cfg.KWG_TRIPLES_PATH,
            "strata": _cfg.strata_path("kwg"),
        },
        "worldkg": {
            "entities": _cfg.WORLDKG_NODES_PATH,
            "triples": _cfg.WORLDKG_TRIPLES_PATH,
            "strata": _cfg.strata_path("worldkg"),
        },
        "osmh3kg": {
            "entities": _cfg.OSMH3KG_NODES_PATH,
            "triples": _cfg.OSMH3KG_TRIPLES_PATH,
            "strata": _cfg.strata_path("osmh3kg"),
        },
    }


@dataclass(frozen=True)
class KgPaths:
    """Delta table paths for one KG."""

    entities: str
    triples: str
    strata: str


def resolve_kg_paths(kg: str) -> KgPaths:
    """Return the entity + triples + strata Delta paths for ``kg``.

    Raises ``KeyError`` if ``kg`` is not one of the supported labels.
    """
    paths = _import_kg_paths()
    if kg not in paths:
        raise KeyError(
            f"unknown KG {kg!r}; supported: {sorted(paths)}"
        )
    return KgPaths(
        entities=paths[kg]["entities"],
        triples=paths[kg]["triples"],
        strata=paths[kg]["strata"],
    )


# --------------------------------------------------------------------------- #
# Per-KG cached context
# --------------------------------------------------------------------------- #


@dataclass
class KgContext:
    """Per-KG state held across the inner TID × stratum loop.

    ``strata_df`` is cached + materialized once per KG because every cell
    filters it by stratum; without the cache Spark re-scans the Delta
    source per join. ``entities_df`` is deliberately *not* cached — on
    osmh3kg it's 14.3M rows with large WKT strings, and caching it as
    Java objects uses ~50-80 GB heap.
    The ``triples_view`` is a registered Spark temp view name; the
    underlying Delta is not cached for the same memory reason.

    ``gold_triples_view`` is registered from a *persisted* Delta table —
    the same triples Delta pre-filtered to just the gold-relevant predicate
    IRIs, with a parquet-native bloom filter on ``subject``. The filter
    collapses a full-KG scan from ~9 B rows to ~300 M for osmh3kg / ~600 M
    for worldkg. The bloom filter lets per-cell joins (which probe with
    ~tens of entity URIs) skip the vast majority of files via per-file
    membership lookup. First run for a given (kg, prefix_map) writes this
    Delta; subsequent runs read it directly, so the gold-SQL warmup cost is
    paid once and persists across runs.
    """

    kg_label: str
    entities_df: DataFrame
    strata_df: DataFrame
    triples_view: str
    gold_triples_view: str
    paths: KgPaths


# Bumped whenever the gold-Delta materialization recipe changes (filter scope,
# bloom indexes, partitioning, etc). Mixed into the cache key below so a
# change here forces a fresh write to a new path while old paths stay readable.
_GOLD_MATERIALIZATION_VERSION = "v3-parquet-bloom-subject"


def _prefix_map_sha(kg_prefix: dict[str, list[str]]) -> str:
    """8-char sha of one KG's predicate IRI map + materialization version.

    Used as a cache key for the materialized gold-triples Delta — bumping any
    IRI list OR ``_GOLD_MATERIALIZATION_VERSION`` changes the sha and forces a
    re-materialization, while leaving other KGs' caches intact.
    """
    payload = {
        "prefix": kg_prefix,
        "version": _GOLD_MATERIALIZATION_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:8]


def _gold_triples_table_path(kg: str, sha: str) -> str:
    """Stable on-disk location for the pre-filtered, bloom-indexed gold-triples Delta.

    If ``MULTIGLOBEQA_GOLD_CACHE_DIR`` is set, the Delta lives directly under
    that directory (``$MULTIGLOBEQA_GOLD_CACHE_DIR/<kg>_triples_gold_<sha>``)
    instead of under ``data/<kg>/``. Used to stage the gold Delta on
    node-local SSD when the default ``data/`` lives on a slow network FS.
    """
    import os
    override = os.environ.get("MULTIGLOBEQA_GOLD_CACHE_DIR")
    if override:
        return str(Path(override) / f"{kg}_triples_gold_{sha}")
    from question_generator import config as _cfg
    return str(Path(_cfg.DATA_DIR) / kg / f"{kg}_triples_gold_{sha}")


def _gold_triples_table_exists(spark: SparkSession, dest_path: str) -> bool:
    """True iff a committed Delta table exists at ``dest_path``.

    Checks for the ``_delta_log`` directory rather than catching exceptions on
    read — a partial write leaves files but no log, so file-existence alone
    isn't sufficient.
    """
    return (Path(dest_path) / "_delta_log").is_dir()


def _materialize_gold_triples(
    spark: SparkSession,
    triples_df: DataFrame,
    all_iris: list[str],
    dest_path: str,
    *,
    kg: str | None = None,
) -> None:
    """One-time write: pre-filter ``triples_df`` to ``all_iris`` and write
    the result as a Delta with a parquet-native bloom filter on ``subject``.

    The gold SQL joins twice on ``t.subject`` (entity URIs + geo-node URIs)
    with a small broadcast probe set (~tens to hundreds of URIs per cell).
    A per-file bloom on ``subject`` lets the scan skip the vast majority of
    files via per-row-group membership lookup. Hash-bucket partitioning on
    ``subject`` costs ~6x more cell wall clock than the bloom, because
    Spark does not auto-derive a partition predicate from a broadcast-driven
    IN-list, so all partition files get scanned anyway.

    Also writes a sibling ``_predicates.json`` next to the Delta containing
    the cache-key inputs ({kg, version, predicates, written_at}). This makes
    orphan-cache cleanup grep-able: stale ``_gold_<sha>`` directories from
    older predicate sets or version bumps can be diffed against the live
    cache key without re-deriving the sha.
    """
    import datetime as _dt

    logger.info(
        "Materializing gold-triples Delta at {} (one-time cost; future runs read directly)",
        dest_path,
    )
    filtered = (
        triples_df
        .filter(F.col("predicate").isin(all_iris))
        .select("subject", "predicate", "object")
    )
    (
        filtered.write
        .format("delta")
        .option("parquet.bloom.filter.enabled#subject", "true")
        .option("parquet.bloom.filter.fpp#subject", "0.01")
        .mode("overwrite")
        .save(dest_path)
    )

    sidecar = Path(dest_path) / "_predicates.json"
    sidecar.write_text(
        json.dumps(
            {
                "kg": kg,
                "version": _GOLD_MATERIALIZATION_VERSION,
                "predicates": list(all_iris),
                "written_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    logger.info("Gold-triples Delta ready at {}", dest_path)


def load_kg_context(
    spark: SparkSession,
    kg: str,
    prefix_map: dict[str, dict[str, list[str]]] | None = None,
) -> KgContext:
    """Load entity/strata/triples handles for one KG.

    Caches + materializes ``strata_df`` so downstream filters don't rescan
    the Delta source. Registers ``_triples_<kg>`` (the raw triples view)
    and ``_triples_gold_<kg>`` (the same Delta table pre-filtered to just
    ``prefix_map[kg]`` IRIs) as temp views. Returns a ``KgContext`` the
    caller must pass to ``close_kg_context`` (typically in a ``finally``)
    to release the strata cache.

    ``prefix_map`` may be ``None`` for callers that don't use
    gold-triples SQL; in that case ``gold_triples_view`` falls back to the
    raw triples view.
    """
    paths = resolve_kg_paths(kg)

    # Explicit dedup keys: bare dropDuplicates() only collapses exact-row
    # duplicates, not rows that differ in auxiliary columns (load timestamps,
    # version flags, etc.) but share the same (entity_id, country, stratum).
    # Each such duplicate would otherwise become its own pool-view row, get a
    # separate ROW_NUMBER rank, and produce repeated entity_b values at
    # adjacent ranks.
    strata_df = (
        spark.read.format("delta")
        .load(paths.strata)
        .dropDuplicates(["entity_id", "country_iso3", "strata_cell"])
        .cache()
    )
    strata_rows = strata_df.count()  # materialize the cache

    entities_df = spark.read.format("delta").load(paths.entities)

    triples_view = f"_triples_{kg}"
    triples_df = spark.read.format("delta").load(paths.triples)
    triples_df.createOrReplaceTempView(triples_view)

    # Persisted gold-triples Delta. The first run for a (kg,
    # prefix_map) combo materializes the filtered subset to disk with a
    # bloom filter on ``subject``; subsequent runs skip straight to read.
    # Filtering lazily on a temp view instead makes the gold SQL
    # re-evaluate the 9 B-row predicate scan on every CTE join
    # (two scans per cell, no caching across cells).
    if prefix_map is not None and kg in prefix_map:
        # Union of every IRI declared for this KG across all roles. Roles
        # unused by the active gold SQL are harmless to include — the bloom
        # does the per-cell file-skipping, render_gold_sql narrows the IN()
        # to just the roles a given TID uses. Set + sort: drops duplicates
        # when one IRI maps to multiple roles, and gives a stable order so
        # materialization logs diff cleanly across runs.
        all_iris: list[str] = sorted(
            {iri for iris in prefix_map[kg].values() for iri in iris}
        )
        sha = _prefix_map_sha(prefix_map[kg])
        dest_path = _gold_triples_table_path(kg, sha)
        if not _gold_triples_table_exists(spark, dest_path):
            _materialize_gold_triples(
                spark, triples_df, all_iris, dest_path, kg=kg,
            )
        else:
            logger.info(
                "Reusing materialized gold-triples Delta at {} (sha={})",
                dest_path, sha,
            )
        gold_triples_view = f"_triples_gold_{kg}"
        spark.read.format("delta").load(dest_path).createOrReplaceTempView(
            gold_triples_view
        )
    else:
        gold_triples_view = triples_view

    logger.info(
        "Loaded KG context for {}: strata={} rows (cached), entities uncached, "
        "triples={}, gold_triples={}",
        kg, strata_rows, triples_view, gold_triples_view,
    )
    return KgContext(
        kg_label=kg,
        entities_df=entities_df,
        strata_df=strata_df,
        triples_view=triples_view,
        gold_triples_view=gold_triples_view,
        paths=paths,
    )


def close_kg_context(ctx: KgContext) -> None:
    """Release the strata cache held by ``ctx``. Best-effort; never raises."""
    try:
        ctx.strata_df.unpersist()
    except Exception as e:
        logger.debug("unpersist failed for {}: {} (non-fatal)", ctx.kg_label, e)


# --------------------------------------------------------------------------- #
# View registration
# --------------------------------------------------------------------------- #


def build_pool_view(
    spark: SparkSession,
    entities_path_or_df: "str | DataFrame",
    pool_filter: str,
    view_name: str,
    *,
    strata_path: str | None = None,
    strata_df: DataFrame | None = None,
    stratum: str | None = None,
) -> None:
    """Filter entities, inner-join strata, register as ``view_name``.

    ``entities_path_or_df`` may be either a Delta path (re-reads
    on every call) or a pre-loaded DataFrame (preferred; the caller holds the
    single read). Likewise ``strata_df`` is preferred over ``strata_path``
    for the same caching reason. The cached-DataFrame path reads once per
    KG and filters per cell.

    When strata is supplied the join drops entities that have no strata row
    and attaches ``country_iso3`` + ``strata_cell`` columns needed by
    diversity-partition SQL. If ``stratum`` is also set the strata side is
    pre-filtered to that cell so the join simultaneously narrows to it.

    Deliberately no ``orderBy + limit`` pool cap: on planet-scale entity
    tables the sort-and-take collapses to a single reducer on the full
    pool (~50K rows with multi-KB WKT strings each), turning pool-build
    into a serial bottleneck. AQE skewJoin + auto-broadcast
    (256MB threshold, set in ``init_spark``) handle downstream country
    skew on the main SQL's ``country_iso3`` equi-join instead.
    """
    if isinstance(entities_path_or_df, str):
        df = spark.read.format("delta").load(entities_path_or_df)
    else:
        df = entities_path_or_df

    if pool_filter.strip():
        df = df.filter(pool_filter)

    strata: DataFrame | None
    if strata_df is not None:
        strata = strata_df
    elif strata_path is not None:
        strata = (
            spark.read.format("delta").load(strata_path).dropDuplicates()
        )
    else:
        strata = None

    if strata is not None:
        if stratum is not None:
            strata = strata.filter(F.col("strata_cell") == stratum)
        strata = strata.select("entity_id", "country_iso3", "strata_cell")
        # Some KGs already carry country_iso3 on the entity row (osmh3kg) and
        # some don't (kwg). Drop it pre-join so the joined frame has exactly
        # one country_iso3 column — otherwise downstream `a.country_iso3` /
        # `b.country_iso3` references inside CTE windows resolve ambiguously
        # and Spark refuses to plan the query.
        if "country_iso3" in df.columns:
            df = df.drop("country_iso3")
        df = df.join(
            strata, df["id"] == strata["entity_id"], "inner"
        ).drop("entity_id")
    elif stratum is not None:
        raise ValueError("stratum supplied without strata; cannot filter")

    df.createOrReplaceTempView(view_name)


def register_triples_view(
    spark: SparkSession,
    triples_path: str,
    view_name: str,
) -> None:
    df = spark.read.format("delta").load(triples_path)
    df.createOrReplaceTempView(view_name)


# --------------------------------------------------------------------------- #
# Placeholder rendering
# --------------------------------------------------------------------------- #


def render_partition_clause(diversity_keys: list[str]) -> str:
    if not diversity_keys:
        raise ValueError("diversity_keys must be non-empty for row_number_hash")
    return f"PARTITION BY {', '.join(diversity_keys)}"


def _resolve_parameters(
    meta: TidMeta, kg: str, variant_id: str | None = None,
) -> dict[str, str | int | float]:
    """Merge shape-wide, per-variant, and per-KG parameter sources.

    Resolution order (later overlays earlier):
      1. ``meta.parameters`` — shape-wide defaults
      2. ``meta.variants[variant_id].parameters`` — per-variant overrides
         (skipped if variant_id is None or the variant is missing)
      3. ``meta.parameters_per_kg[kg]`` — per-KG overrides

    Per-KG wins over per-variant wins over shape-wide. The order
    reflects scope: per-KG is the deepest constraint (data-density),
    per-variant is the next (e.g., a specific category's display label),
    shape-wide is the loosest. Variants can introduce keys that
    shape-wide doesn't declare; the loader's per-KG validator accepts
    overrides on variant-introduced keys.

    Loader-side validation (TidMeta._per_kg_keys_subset_of_parameters)
    guarantees per-KG keys are a subset of the union of shape + variant
    keys, so a typo inside a per-KG override is caught at meta-load
    time, not at render.
    """
    resolved = dict(meta.parameters)
    if variant_id is not None and variant_id in meta.variants:
        resolved.update(meta.variants[variant_id].parameters)
    overrides = meta.parameters_per_kg.get(kg, {})
    resolved.update(overrides)
    return resolved


def render_main_sql(
    shape: TidShape,
    kg: str,
    run_spec: RunSpec,
    pool_view_names: dict[str, str],
) -> str:
    """Substitute every placeholder in ``shape.sql`` for a given (kg, run)."""
    sampling = shape.meta.sampling
    substitutions: dict[str, str] = {
        "{run_seed}": str(run_spec.run_seed),
        "{total_sample}": str(run_spec.total_sample),
    }
    # {join_condition} only substitutes when the meta declares one. Single-
    # pool TIDs leave it unset and their SQL must not reference the token
    # (the contract checker enforces both directions).
    if shape.meta.join_condition is not None:
        substitutions["{join_condition}"] = shape.meta.join_condition
    # Cross-join sampling placeholders: only substituted when the
    # corresponding sampling field is declared in meta.yaml. TIDs that
    # don't use the cross-join + bucket-rank structure leave these unset.
    # The contract checker enforces that any reference to a sampling-tied
    # placeholder in the SQL is matched by a declared field in meta.
    if sampling.diversity_keys is not None:
        substitutions["{diversity_partition}"] = render_partition_clause(
            sampling.diversity_keys
        )
    if sampling.per_bucket_cap is not None:
        substitutions["{per_bucket_cap}"] = str(sampling.per_bucket_cap)
    if sampling.pool_cap is not None:
        substitutions["{pool_cap}"] = str(sampling.pool_cap)
    # Pool placeholders driven by meta.pools — same source the contract
    # validates against, so adding pool_c to a TID's meta + SQL is a
    # one-place change. Missing-view-name is a programmer error (the
    # caller forgot to build a view for a declared pool); raise loudly.
    for pool_name in shape.meta.pools:
        if pool_name not in pool_view_names:
            raise KeyError(
                f"TID {shape.meta.tid}: meta.pools declares {pool_name!r} "
                f"but the executor was not given a view name for it"
            )
        substitutions[f"{{{pool_name}}}"] = pool_view_names[pool_name]

    # Free-form per-TID parameters from meta.yaml. Resolution layers:
    # shape-wide → per-variant → per-KG (later wins). Per-KG handles
    # data-density tuning (radius_km on kwg vs osmh3kg); per-variant
    # handles category-label substitution (sfid_i 72/73/75/76's
    # event_a_label, which a SQL CASE would otherwise have to encode).
    rendered_params = _resolve_parameters(shape.meta, kg, shape.variant_id)
    for key, value in rendered_params.items():
        substitutions[f"{{{key}}}"] = str(value)

    sql = shape.sql
    for token, value in substitutions.items():
        sql = sql.replace(token, value)
    return sql


def render_gold_sql(
    shape: TidShape,
    kg: str,
    pool_view_names: dict[str, str],
    triples_view: str,
    prefix_map: dict[str, dict[str, list[str]]],
) -> str:
    """Substitute every placeholder in ``shape.gold_sql`` for a given (kg, run)."""
    kg_roles = prefix_map[kg]

    all_iris: list[str] = []
    for role in shape.meta.gold_triples.predicates:
        all_iris.extend(kg_roles[role])
    predicate_iris = ", ".join(f"'{iri}'" for iri in all_iris)

    geom_link_iri = kg_roles["geometry_link"][0]
    geom_wkt_iri = kg_roles["geometry_wkt"][0]

    substitutions: dict[str, str] = {
        "{triples_view}": triples_view,
        "{predicate_iris}": predicate_iris,
        "{pred_geometry_link}": geom_link_iri,
        "{pred_geometry_wkt}": geom_wkt_iri,
    }
    # Convention: gold SQL references each pool as {<pool_name>_all} to
    # make the "I want the unfiltered pool view here" intent explicit
    # (vs the main SQL's {<pool_name>}).
    for pool_name in shape.meta.pools:
        if pool_name not in pool_view_names:
            raise KeyError(
                f"TID {shape.meta.tid}: meta.pools declares {pool_name!r} "
                f"but the executor was not given a view name for it"
            )
        substitutions[f"{{{pool_name}_all}}"] = pool_view_names[pool_name]

    # Free-form per-TID parameters — same resolution as render_main_sql
    # (shape-wide → per-variant → per-KG). Gold and main share the same
    # {key} substitutions so e.g. {radius_km} or {event_a_label} stays
    # consistent between the main aggregate and the gold's URI fan-out.
    rendered_params = _resolve_parameters(shape.meta, kg, shape.variant_id)
    for key, value in rendered_params.items():
        substitutions[f"{{{key}}}"] = str(value)

    sql = shape.gold_sql
    for token, value in substitutions.items():
        sql = sql.replace(token, value)
    return sql


# --------------------------------------------------------------------------- #
# Gold-triple → question-id join (used by exporter, kept here so the
# entity-role label convention lives next to the executor it pairs with).
# --------------------------------------------------------------------------- #


def attach_question_ids_to_gold(
    bench_df: DataFrame,
    gold_df: DataFrame,
    pool_map: dict[str, str],
) -> DataFrame:
    """Join entity-keyed gold triples to surviving benchmark rows.

    The gold SQL emits ``(entity_id, subject, predicate, object)``. The
    benchmark df at this point has ``question_id`` (uuid, fixed by the
    exporter's cache) plus the role columns ``entity_a_id`` / ``entity_b_id``.
    For each role in ``pool_map`` we join gold by ``entity_id == role_col``
    and emit a flat row, then union the per-role results.

    Output schema: ``(question_id, entity_role, subject, predicate, object)``.
    """
    if not pool_map:
        raise ValueError("pool_map is empty; nothing to attach")

    pieces: list[DataFrame] = []
    for role_col, _pool_name in pool_map.items():
        role_label = (
            role_col.removeprefix("entity_").removesuffix("_id")
            if role_col.startswith("entity_") and role_col.endswith("_id")
            else role_col
        )
        per_role = bench_df.select(
            "question_id", F.col(role_col).alias("entity_id")
        ).join(gold_df, on="entity_id", how="inner").select(
            "question_id",
            F.lit(role_label).alias("entity_role"),
            "subject",
            "predicate",
            "object",
        )
        pieces.append(per_role)

    flat = pieces[0]
    for p in pieces[1:]:
        flat = flat.unionByName(p)
    return flat


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #


def execute_tid_kg(
    spark: SparkSession,
    shape: TidShape,
    kg: str,
    run_spec: RunSpec,
    prefix_map: dict[str, dict[str, list[str]]],
    *,
    stratum: str | None = None,
    kg_paths: KgPaths | None = None,
    kg_context: KgContext | None = None,
) -> tuple[DataFrame, DataFrame]:
    """Execute one (TID, KG [, stratum]) combination end-to-end.

    Returns a 2-tuple ``(main_df, gold_df)``:

    * ``main_df`` — one row per benchmark candidate, with the TID's canonical
      ``result_columns`` plus ``tid``, ``kg_source``, ``strata_cell``,
      ``ground_truth_type``, ``sfid`` constants. No ``question_id`` yet
      (assigned later by the exporter, post-dedup).
    * ``gold_df`` — flat per-triple rows with columns
      ``(entity_id, subject, predicate, object)``. Keyed by entity, not by
      question_id; the exporter joins this to surviving questions via
      ``entity_a_id`` / ``entity_b_id``.

    When ``stratum`` is set, pool views are narrowed to entities with that
    ``strata_cell`` value before the main SQL runs. ``run_spec.total_sample``
    therefore behaves as a *per-cell* cap: calling once per stratum
    produces up to ``total_sample`` rows from each cell independently.

    Args:
        stratum: strata-cell label (e.g. ``"HIC_high"``) to restrict both
            pool views to. ``None`` means pool across all strata.
        kg_paths: override for entity + triples paths (tests supply an
            in-memory fixture here). Defaults to resolving from the
            ``question_generator.config`` module.
    """
    if kg not in shape.applicable_kgs:
        raise ValueError(
            f"TID {shape.meta.tid} variant {shape.variant_id!r} does not "
            f"declare {kg!r} as applicable; "
            f"declared: {sorted(shape.applicable_kgs)}"
        )
    # Prefer the cached context when available; otherwise fall back to
    # lazy per-call Delta reads so tests and one-off callers still work.
    if kg_context is not None:
        entities_src: str | DataFrame = kg_context.entities_df
        strata_src_df: DataFrame | None = kg_context.strata_df
        strata_src_path: str | None = None
        triples_view = kg_context.triples_view
        gold_triples_view = kg_context.gold_triples_view
    else:
        if kg_paths is None:
            kg_paths = resolve_kg_paths(kg)
        entities_src = kg_paths.entities
        strata_src_df = None
        strata_src_path = kg_paths.strata
        triples_view = f"_triples_{kg}"
        register_triples_view(spark, kg_paths.triples, triples_view)
        # Fallback path (tests / one-off callers): no pre-filtered gold view,
        # so the gold SQL joins against the raw triples view. Slow but
        # correct; the KG-context path is what production runs use.
        gold_triples_view = triples_view

    # 1a. Country sub-sample: pick N deterministic country_iso3 values per
    #     (stratum, tid, variant, run_seed) restricted to countries that have
    #     candidate entities in EVERY pool for this cell. Bounds cross-join
    #     volume and prevents tiny-country-with-no-entities draws, which
    #     yield empty cells. Only applies when we have
    #     a stratum + a strata DF; pool-across-all mode does no country
    #     narrowing.
    narrowed_strata = strata_src_df
    if stratum is not None and strata_src_df is not None:
        # Resolve entities_src to a DataFrame for the country sampler. In
        # the kg_context path it already is one; in the fallback path
        # (tests / one-off callers) it's a Delta path string.
        entities_df_for_sampling = (
            entities_src
            if not isinstance(entities_src, str)
            else spark.read.format("delta").load(entities_src)
        )
        pool_filters = [
            pcfg.filter for pcfg in shape.applicable_kgs[kg].values()
        ]
        cell_seed = _mix_seed(
            run_spec.run_seed, stratum, shape.meta.tid, shape.variant_id
        )
        selected_countries = _sample_countries_for_cell(
            strata_src_df,
            entities_df_for_sampling,
            pool_filters,
            stratum,
            cell_seed,
            run_spec.countries_per_cell,
        )
        if not selected_countries:
            logger.warning(
                "Empty stratum {} for TID {} KG {}: pool will be empty",
                stratum, shape.meta.tid, kg,
            )
        else:
            logger.debug(
                "Sampled {} countries for {} cell {}: {}",
                len(selected_countries), kg, stratum, selected_countries,
            )
        # Narrow even when empty — an isin over [] yields zero rows, which
        # propagates naturally through the pool → main-SQL → gold chain.
        narrowed_strata = strata_src_df.filter(
            F.col("country_iso3").isin(selected_countries)
        )

    # 1b. Build pool views. Each pool gets a unique view name namespaced by
    #     kg and stratum so concurrent executions in the same SparkSession
    #     don't collide on global temp-view names.
    view_suffix = kg if stratum is None else f"{kg}_{stratum}"
    pool_view_names: dict[str, str] = {}
    # Namespace view names with variant_id so POI and Region variants for
    # the same TID+KG+stratum don't collide on the same SparkSession.
    for pool_name, pool_cfg in shape.applicable_kgs[kg].items():
        view_name = (
            f"_shape_{shape.meta.tid.replace('.', '_')}"
            f"_{shape.variant_id}_{view_suffix}_{pool_name}"
        )
        # Dispatch on PoolConfig.source. The default "entities" source uses
        # the cached per-KG entities_df (joined to strata). Alternate
        # sources (currently only "osmh3kg_roads_aggregated") read from a
        # pre-registered Spark temp view. Those views already carry
        # country_iso3 and use synthesized MD5 ids that don't match the
        # strata Delta's entity_id, so the strata join is suppressed.
        if pool_cfg.source == "entities":
            build_pool_view(
                spark, entities_src, pool_cfg.filter, view_name,
                strata_path=strata_src_path,
                strata_df=narrowed_strata,
                stratum=stratum,
            )
        else:
            alt_df = spark.table(pool_cfg.source)
            if narrowed_strata is not None:
                # Country sub-sample still applies — narrowed_strata holds
                # the selected country set for the cell. Filter the
                # aggregated view to that country set so the cell budget
                # is honoured.
                country_keys = (
                    narrowed_strata.select("country_iso3").distinct()
                )
                alt_df = alt_df.join(country_keys, on="country_iso3", how="inner")
            build_pool_view(
                spark, alt_df, pool_cfg.filter, view_name,
                strata_path=None,
                strata_df=None,
                stratum=None,
            )
        pool_view_names[pool_name] = view_name

    # 3. Render and execute main SQL. Checkpoint (eager) physically writes
    #    the tiny result to the checkpoint dir, truncating the logical plan
    #    so downstream reads (gold SQL via temp view, exporter staging write)
    #    skip replaying pool builds + cross-joins + ST_DistanceSphere. This
    #    is stronger than .cache(): cache memory accumulates across cells
    #    (driver OOMs → 7-min GC pauses → heartbeat timeout), while
    #    checkpoint files are garbage-collected by Spark's own cleanup.
    main_sql = render_main_sql(shape, kg, run_spec, pool_view_names)
    main_df = spark.sql(main_sql)
    if spark.sparkContext._jsc.sc().getCheckpointDir().isDefined():
        main_df = main_df.checkpoint(eager=True)
    else:
        main_df = main_df.cache()

    # 3b. Add canonical constants. Deliberately no question_id assignment
    #     here — uuid() is non-deterministic, and a count() doesn't force
    #     materialization (Catalyst prunes the column for SELECT COUNT(*)),
    #     so caching wouldn't lock IDs and any second downstream reference
    #     would re-roll uuids and re-execute the upstream DAG. The exporter
    #     assigns question_id once after dedup; gold rows are joined to
    #     it via entity_a_id / entity_b_id post-hoc.
    main_df = (
        main_df.withColumn("tid", F.lit(shape.meta.tid))
        .withColumn("kg_source", F.lit(kg))
        .withColumn("ground_truth_type", F.lit(shape.meta.ground_truth_type))
        .withColumn("sfid", F.lit(shape.meta.sfid))
        .withColumn(
            "strata_cell",
            F.lit(stratum) if stratum is not None else F.lit(None).cast("string"),
        )
    )

    # 4. Register benchmark rows as _gt_result for the gold SQL to read.
    gt_result_view = (
        f"_gt_result_{shape.meta.tid.replace('.', '_')}"
        f"_{shape.variant_id}_{view_suffix}"
    )
    main_df.createOrReplaceTempView(gt_result_view)

    # 5. Render and execute gold SQL with _gt_result swapped in. Output is
    #    flat (entity_id, subject, predicate, object) — no GROUP BY. A
    #    collect_list aggregate here would be a single-task reducer that
    #    stalls per-cell wall clock; emitting flat rows lets every join
    #    downstream parallelise normally. Output is keyed by entity_id;
    #    the exporter joins it to question_id via entity_a_id / entity_b_id.
    gold_sql = render_gold_sql(
        shape, kg, pool_view_names, gold_triples_view, prefix_map,
    )
    gold_sql = gold_sql.replace("_gt_result", gt_result_view)
    gold_df = spark.sql(gold_sql)

    return main_df, gold_df
