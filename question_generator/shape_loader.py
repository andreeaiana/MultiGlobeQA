"""Loader for per-TID SQL shape files under ``question_generator/sql_shapes/``.

A *shape* is three sibling files for one TID:

- ``<tid>.sql``        — main question-generation SQL
- ``<tid>.gold.sql``   — gold-triples extraction SQL (reads from ``_gt_result``)
- ``<tid>.meta.yaml``  — declarative metadata (applicable KGs, pools, sampling,
  gold-triple config, defaults)

This module parses those files plus a top-level ``runs/*.yaml`` run spec and
the shared ``config/kg_prefixes.yaml`` into typed objects (Pydantic) that the
executor and contract-check consume.

The loader never touches Spark and never reads entity data; it only parses
filesystem configuration.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Meta models (one TidMeta per <tid>.meta.yaml)
# --------------------------------------------------------------------------- #


class PoolConfig(BaseModel):
    """Filter spec for one pool within one KG.

    ``source`` selects which Spark dataset the pool reads from:

    - ``"entities"`` (default): the per-KG entities Delta (``entities_df``
      from ``KgContext``). Strata join is applied as usual.
    - ``"osmh3kg_roads_aggregated"``: the per-session aggregated road view
      registered by ``road_setup.register_road_aggregation_views``. One row
      per ``(country_iso3, name, node_type)`` named road with a merged
      geometry. The view already carries ``country_iso3`` and uses
      synthesized MD5 ``id`` values that are NOT present in the strata
      Delta, so the executor SKIPS the strata join for this source.
      osmh3kg-only (kwg/worldkg have no equivalent view).
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["entities", "osmh3kg_roads_aggregated"] = "entities"
    filter: str


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["row_number_hash"] = "row_number_hash"
    # All three knobs are optional. They drive the {diversity_partition}
    # / {per_bucket_cap} / {pool_cap} placeholders, which only appear in SQLs
    # that use the cross-join + bucket-rank sampling structure. TIDs using
    # rank-equality joins (or other structures) leave these unset and rely
    # on join_condition + meta.parameters for their per-TID knobs.
    diversity_keys: list[str] | None = Field(None, min_length=1)
    per_bucket_cap: int | None = Field(None, ge=1)
    pool_cap: int | None = Field(None, ge=1)


class GoldTriplesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pool_map: dict[str, str]
    predicates: list[str] = Field(..., min_length=1)


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_sample: int = Field(..., ge=1)
    run_seed: int


class VariantConfig(BaseModel):
    """One pool-pairing variant of a TID.

    A single TID file can declare multiple variants (e.g. POI × POI vs.
    Region × Region for TID 1.1). The shared blocks (SQL, sampling, gold
    config, defaults) stay outside ``variants`` so they can't drift between
    variants of the same TID.

    ``parameters`` holds per-variant overrides that layer on top of the
    shape-wide ``TidMeta.parameters``. Use case: when sibling variants of
    one ftid need different SQL-substituted values that aren't expressible
    via the variant's ``applicable_kgs[*].pool_X.filter`` alone.

    Concrete example: sfid_i variants each substitute a different display
    label into ``{{event_A}}`` via the SQL projection
    ``'{event_a_label}' AS event_a_name``. With per-variant
    ``parameters: { event_a_label: ... }`` every variant declares its own
    literal and the SQL just substitutes ``{event_a_label}`` directly.
    The alternative, one shape-wide SQL ``CASE node_type WHEN ... THEN
    ... END``, has to be edited every time a category is added.

    Resolution order at render time:
      1. start with ``TidMeta.parameters`` (shape-wide defaults)
      2. overlay ``VariantConfig.parameters`` (per-variant)
      3. overlay ``TidMeta.parameters_per_kg[kg]`` (per-KG)

    Per-KG overrides win because they reflect the deepest constraint
    (kwg's data sparsity). Variant overrides win against shape-wide
    defaults because the variant is the more specific scope.
    """

    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    applicable_kgs: dict[str, dict[str, PoolConfig]]
    parameters: dict[str, str | int | float] = Field(default_factory=dict)


class TidMeta(BaseModel):
    """Parsed contents of ``<tid>.meta.yaml``.

    Mirrors the YAML 1:1. Any unknown top-level key is rejected so typos
    can't silently disable a setting.
    """

    model_config = ConfigDict(extra="forbid")

    tid: str
    tid_family: int
    sfid: str
    ssfid: int
    spatial_function: str
    ground_truth_type: str
    answer_unit: str | None = None

    result_columns: list[str] = Field(..., min_length=1)
    pools: list[str] = Field(..., min_length=1)
    variants: dict[str, VariantConfig] = Field(..., min_length=1)
    # Optional: present for binary-relation TIDs (cross-join + filter), absent
    # for single-pool unary TIDs that have no join. The contract checker only
    # requires {join_condition} as a placeholder when this field is set.
    # min_length=1 rejects the empty-string footgun: "" passes the not-None
    # check and substitutes into the SQL as an empty ON clause, producing
    # invalid SQL Spark rejects with no useful diagnostic.
    join_condition: str | None = Field(None, min_length=1)
    sampling: SamplingConfig
    gold_triples: GoldTriplesConfig
    defaults: DefaultsConfig

    # Free-form SQL substitutions. Each key maps to a {key} placeholder in
    # either main or gold SQL. Validated by shape_contract.check_shape:
    # every key listed here MUST appear in at least one SQL; every
    # {placeholder} in either SQL must be either built-in or declared here;
    # keys may not collide with built-in token names. Defaults to {}, so a
    # meta.yaml that declares no substitutions is valid.
    parameters: dict[str, str | int | float] = Field(default_factory=dict)

    # Per-KG parameter overrides. Outer key is a KG name (osmh3kg, worldkg,
    # kwg); inner dict overrides specific entries from `parameters` when
    # rendering for that KG. Other keys fall through to `parameters`.
    # Use case: dense KGs (osmh3kg, worldkg) want a tight radius_km (5)
    # for tractable joins, while sparse KGs (kwg) need a wider radius (50)
    # to find any neighbours. Without this, a shape would have to choose
    # one or hardcode different `radius_km` per ftid family.
    #
    # Validated by the loader (model_validator below): every key inside
    # any per-KG dict MUST exist in `parameters` — otherwise a typo would
    # silently fail to override anything. KG names themselves are not
    # validated against variant.applicable_kgs; an override for a KG that
    # no variant uses is harmless dead code, e.g. when staging a future
    # variant.
    parameters_per_kg: dict[str, dict[str, str | int | float]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _per_kg_keys_subset_of_parameters(self) -> TidMeta:
        # parameters_per_kg keys must be a subset of the union of
        # shape-wide parameters AND any variant-level parameters. Variants
        # can introduce new keys (per-variant scope) — a per-KG override
        # of a variant-introduced key is valid; only typos in unrelated
        # keys are caught here.
        all_param_keys = set(self.parameters)
        for variant in self.variants.values():
            all_param_keys |= set(variant.parameters)
        for kg, overrides in self.parameters_per_kg.items():
            unknown = set(overrides) - all_param_keys
            if unknown:
                raise ValueError(
                    f"TID {self.tid}: parameters_per_kg[{kg!r}] declares "
                    f"key(s) {sorted(unknown)} that are not in parameters "
                    "(neither shape-wide nor variant-scope). Per-KG "
                    "overrides may only override keys already declared."
                )
        return self


class TidShape(BaseModel):
    """A fully-loaded TID with one variant resolved.

    ``sql`` and ``gold_sql`` are left untouched as strings so the contract
    check can inspect them verbatim and the executor can render placeholders.
    ``variant_id`` records which pool pairing this shape is scoped to;
    consumers read ``shape.applicable_kgs`` to get the variant's KG map.
    """

    model_config = ConfigDict(extra="forbid")

    meta: TidMeta
    variant_id: str
    sql: str
    gold_sql: str
    shape_dir: Path

    @property
    def applicable_kgs(self) -> dict[str, dict[str, PoolConfig]]:
        return self.meta.variants[self.variant_id].applicable_kgs

    @property
    def ref(self) -> str:
        """Run-spec reference form, ``"<tid>:<variant>"``."""
        return f"{self.meta.tid}:{self.variant_id}"


# --------------------------------------------------------------------------- #
# Run-spec model (top-level runs/<name>.yaml)
# --------------------------------------------------------------------------- #


class RunSpec(BaseModel):
    """Declarative run configuration: which TIDs run against which KGs + knobs."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    tids: list[str] = Field(..., min_length=1)
    kgs: list[str] = Field(..., min_length=1)
    strata: list[str] = Field(default_factory=list)
    run_seed: int
    total_sample: int = Field(..., ge=1)
    output_dir: str
    # Countries sampled per (stratum, tid, run_seed) before the entity×strata
    # join. Bounds the cross-join volume so ROW_NUMBER OVER (PARTITION BY
    # country_iso3, …) can't hit a 5M-row partition when one country (DEU,
    # USA) concentrates most of the entities. A run still yields
    # 2 × per_bucket_cap rows, so smoke targets are still reached.
    countries_per_cell: int = Field(2, ge=1)
    # Per-TID overrides (optional). Each entry may override any subset of
    # ``sampling.per_bucket_cap``, ``defaults.total_sample``, etc.
    overrides: dict[str, dict] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Public loaders
# --------------------------------------------------------------------------- #


def load_run_spec(path: str | Path) -> RunSpec:
    """Parse a run-spec YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return RunSpec.model_validate(data)


def parse_tid_ref(ref: str) -> tuple[str, str | None]:
    """Split a run-spec reference into ``(tid, variant_id)``.

    Accepts either ``"1.1"`` or ``"1.1:poi"``. A bare TID defers variant
    selection to ``load_tid_shape``, which auto-picks the single variant
    when only one is declared.
    """
    if ":" in ref:
        tid, variant = ref.split(":", 1)
        return tid, variant
    return ref, None


def load_tid_shape(
    tid: str,
    shapes_root: str | Path,
    variant_id: str | None = None,
) -> TidShape:
    """Locate and parse the three files for one TID, resolving one variant.

    The loader walks ``shapes_root/*/<tid>.meta.yaml`` (sfid-agnostic) so
    callers don't need to know which ``sfid_*`` subdirectory hosts a TID.

    ``variant_id`` picks a pool-pairing variant. When ``None``, the meta
    must declare exactly one variant; otherwise the caller must name one.

    Raises ``FileNotFoundError`` if any of the three files is missing, or
    ``KeyError`` when the requested variant is not declared.
    """
    shapes_root = Path(shapes_root)
    matches = list(shapes_root.glob(f"*/{tid}.meta.yaml"))
    if not matches:
        raise FileNotFoundError(
            f"No meta file found for TID {tid!r} under {shapes_root}. "
            f"Expected {shapes_root}/<sfid>/{tid}.meta.yaml"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"TID {tid!r} resolves to multiple meta files: "
            f"{[str(m) for m in matches]}"
        )
    meta_path = matches[0]
    shape_dir = meta_path.parent
    sql_path = shape_dir / f"{tid}.sql"
    gold_path = shape_dir / f"{tid}.gold.sql"
    for p in (sql_path, gold_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing SQL file for TID {tid!r}: {p}")

    with open(meta_path) as f:
        meta_raw = yaml.safe_load(f)
    meta = TidMeta.model_validate(meta_raw)
    if meta.tid != tid:
        raise ValueError(
            f"Meta file {meta_path} declares tid={meta.tid!r} but was loaded "
            f"for tid={tid!r}"
        )

    if variant_id is None:
        if len(meta.variants) == 1:
            variant_id = next(iter(meta.variants))
        else:
            raise ValueError(
                f"TID {tid!r} declares {len(meta.variants)} variants "
                f"({sorted(meta.variants)}); caller must supply variant_id "
                f"(e.g. via run-spec ref '{tid}:<variant>')"
            )
    if variant_id not in meta.variants:
        raise KeyError(
            f"TID {tid!r}: variant {variant_id!r} not declared; "
            f"available: {sorted(meta.variants)}"
        )

    return TidShape(
        meta=meta,
        variant_id=variant_id,
        sql=sql_path.read_text(),
        gold_sql=gold_path.read_text(),
        shape_dir=shape_dir,
    )


def compute_shape_sha(shape: TidShape) -> str:
    """12-char SHA for resume-invalidation, scoped to the active variant.

    Hash covers the raw ``<tid>.sql`` text, the raw ``<tid>.gold.sql`` text,
    the shared meta fields (everything except ``variants``), and the
    serialized block of the selected variant. Editing variant B therefore
    does *not* invalidate cells staged under variant A.
    """
    import json

    meta_dump = shape.meta.model_dump()
    scoped = {k: v for k, v in meta_dump.items() if k != "variants"}
    scoped["variant_id"] = shape.variant_id
    scoped["variant"] = meta_dump["variants"][shape.variant_id]

    payload = (
        shape.sql.encode()
        + b"\0"
        + shape.gold_sql.encode()
        + b"\0"
        + json.dumps(scoped, sort_keys=True, default=str).encode()
    )
    return hashlib.sha256(payload).hexdigest()[:12]


def compute_run_sha(
    run_seed: int,
    total_sample: int,
    per_bucket_cap: int,
    countries_per_cell: int = 2,
    prefix_map: dict | None = None,
) -> str:
    """8-char SHA of run-level sampling knobs + predicate prefix map.

    Separate from the shape hash so flipping ``total_sample`` doesn't require
    editing the shape YAML to invalidate cells. ``countries_per_cell`` is
    included so altering the country-subsample cap also re-runs every cell.
    ``prefix_map`` is hashed here because it's injected into the gold SQL at
    render time — a fix to an incorrect predicate IRI must invalidate all
    cells or stale gold triples would survive via marker-skip.
    """
    import json

    base = f"{run_seed}|{total_sample}|{per_bucket_cap}|{countries_per_cell}"
    if prefix_map is not None:
        prefix_blob = json.dumps(prefix_map, sort_keys=True, default=str)
        base = f"{base}|prefix={prefix_blob}"
    return hashlib.sha256(base.encode()).hexdigest()[:8]


def load_prefix_map(
    path: str | Path = "config/kg_prefixes.yaml",
) -> dict[str, dict[str, list[str]]]:
    """Load the per-KG predicate-role → IRI-list map.

    Returns ``{kg: {role: [iri, ...]}}``. The gold SQL uses this to resolve
    ``{predicate_iris}``, ``{pred_geometry_link}``, ``{pred_geometry_wkt}``.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected mapping at top level")
    for kg, roles in raw.items():
        if not isinstance(roles, dict):
            raise ValueError(f"{path}:{kg}: expected mapping of role -> iri list")
        for role, iris in roles.items():
            if not (isinstance(iris, list) and iris and all(
                isinstance(i, str) for i in iris
            )):
                raise ValueError(
                    f"{path}:{kg}.{role}: expected non-empty list of strings, "
                    f"got {iris!r}"
                )
    return raw
