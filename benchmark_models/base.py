"""Core abstractions for the benchmark_models package.

Defines:
    - ``EntityRef``: one entity slot's contents (id, name, optional category).
    - ``Example``: pydantic model wrapping each benchmark question.
    - ``Usage``, ``ToolCall``: small nested pydantic models used by ``Prediction``.
    - ``Prediction``: single flat pydantic model for a model's answer to one example.
    - ``Model``: abstract base class with ``predict()`` and ``predict_batch()``.

Design notes (see the plan file for full rationale):

* Single flat ``Prediction`` model (not three subclasses) — tier-specific
  extras are optional fields. Same shape for tier 1 / tier 2 / tier 3.
  Simpler to serialize as JSONL, simpler for the eval pipeline to read.

* ``usage`` and ``latency_ms`` are *always* present, even on tier 1.
  They power the cost / runtime analysis and reviewer questions about
  efficiency, and the cost of recording them is negligible.

* ``predict_batch()`` has a default implementation that loops over
  ``predict()``. Backends that benefit from batching (vLLM, async API)
  override it. Tier 3 (per-example agent loop) uses the default.

* ``Example`` represents entities as a ``dict[str, EntityRef]`` keyed
  by uppercase letter (``"A"``, ``"B"``, ``"C"``, ``"D"``, ...). Templates
  declare entities by letter (``{{entity_A}}``); the dict mirrors that
  language and handles 0, 1, 2, 3, 4, or any number of entities without
  schema changes.

* ``Example`` carries ``ground_truth_type`` (the per-row schema label
  used by every backend to pin the JSON response schema — see
  :attr:`Example.ground_truth_type` for the rationale). The gold
  *value* fields (``ground_truth``, ``is_valid``, ``validation_errors``,
  ``error``, ``is_true_premise``) are excluded. Schema-shape disclosure
  is intentional; gold-value leakage is structurally impossible. The
  eval pipeline reads gold values from the source JSONL by
  ``question_id`` join.

* ``Example.from_benchmark_row(row)`` is the canonical loader. It
  extracts entity slots, parses ``parameters`` from JSON if present,
  drops gold fields, promotes ``strata_cell``, and stashes unknown
  fields in ``raw`` (with a warning so schema drift is visible).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from benchmark_models.image_paths import resolve_image_path

# ----------------------------------------------------------------------------
# Type aliases
# ----------------------------------------------------------------------------

Tier = Literal["tier1", "tier2", "tier3"]
"""Tier identifiers used across the runner, registry, MLflow, and eval pipeline."""

FinalState = Literal[
    "answered",
    "abstained",
    "cap_hit",
    "error",
    "no_final_answer",
]
"""Termination state of a tier-3 agent run (and a structural mirror
for tier-1 / tier-2 predictions).

* ``answered`` — agent called ``final_answer({"answer": ..., "unit": ...})``
  with a parseable payload.
* ``abstained`` — agent called ``final_answer({"answer": None})``
  (or, for tier 1 / 2, the model returned ``answer=None`` before
  any cap).
* ``cap_hit`` — agent exhausted ``max_tool_calls`` without calling
  ``final_answer`` (tier 3 only).
* ``error`` — unrecoverable runtime error: exception during
  ``agent.run()``, transport failure after retries exhausted, or
  unknown TID raised by the runner / registry layer.
* ``no_final_answer`` — agent finished cleanly (under the cap, no
  exception) without ever registering a ``final_answer`` call. Most
  common cause: the model wrapped the call inside a function
  definition (``def solve(): final_answer(...); solve()``), and the
  smolagents executor only recognises top-level invocations. The
  recorded ``answer`` would otherwise contain the raw Python source
  of the last code step, which the eval pipeline cannot parse.
"""

ErrorClass = Literal[
    "infra",
    "context_overflow",
    "code_gen",
    "schema_violation",
    "config",
    "unknown",
]
"""Qualitative error category for ``final_state="error"`` predictions.

Populated by :func:`benchmark_models.backends._error_class.classify_error`
from the exception that caused the error, or set explicitly when the
runner / registry layer skips a question (``config``). ``None`` for
successful predictions and for tier-3 outcomes that are already
fully described by ``final_state`` alone (``answered``, ``abstained``,
``cap_hit``, ``no_final_answer``).

See :mod:`benchmark_models.backends._error_class` for the full
definition of each category.
"""


# ----------------------------------------------------------------------------
# EntityRef — one entity slot
# ----------------------------------------------------------------------------


class EntityRef(BaseModel):
    """One entity slot's contents.

    Templates declare entities by letter (``{{entity_A}}``,
    ``{{entity_B}}``, ...). Each slot usually has an ``id`` and a
    ``name``; some templates also constrain the entity by ``category``
    (from ``parameters["entity_<letter>_category"]``) or carry a
    ``subcategory`` (from the top-level ``entity_<letter>_subcategory``
    field) as extra semantic metadata.

    Category-only slots — where the benchmark row supplies
    ``entity_<letter>_subcategory`` but no ``_id`` / ``_name`` — also
    produce an ``EntityRef``. In that case ``id`` and ``name`` are
    empty strings and ``subcategory`` carries the slot's identity
    (e.g., "Pharmacy" for a *"How many Pharmacy are within 5 km of X?"*
    question).
    """

    id: str
    """Stable entity identifier as it appears in the source KG (e.g., an
    OSM node id like ``"2476384356"``). Empty string for category-only
    slots."""

    name: str
    """Human-readable display name of the entity (e.g., ``"Cafe Kanel"``).
    May contain quotes or special characters as stored in the benchmark.
    Empty string for category-only slots."""

    category: str | None = None
    """Category constraint from the template, if any (e.g.,
    ``"natural features"``, ``"points of interest"``). ``None`` when the
    template did not declare an ``entity_X_category`` constraint."""

    subcategory: str | None = None
    """Subcategory metadata from the benchmark row's top-level
    ``entity_<letter>_subcategory`` field (e.g., ``"Pharmacy"``).
    ``None`` when the row did not include that field for this slot.

    For category-only slots (``id`` and ``name`` both empty),
    ``subcategory`` is the only identifier — the slot represents a
    *type* ("any Pharmacy") rather than a specific instance."""


# ----------------------------------------------------------------------------
# Gold-field parsing helpers
# ----------------------------------------------------------------------------


def parse_bool_or_none(value: Any) -> bool | None:
    """Coerce a gold-field boolean that may be a real bool, a
    ``"true"`` / ``"false"`` string, or ``None``.

    The upstream benchmark generator (in ``benchmark_generation``)
    writes ``is_true_premise`` as the **string** ``"false"`` /
    ``"true"`` rather than a JSON boolean. A naive
    ``if row.get("is_true_premise"):`` then treats the string
    ``"false"`` as truthy, silently inverting premise-aware scoring.

    This helper is the single choke point for that cast. Call it
    whenever reading ``is_true_premise`` (or any similarly-encoded
    boolean) from the benchmark JSONL.

    Args:
        value: A real ``bool``, a case-insensitive ``"true"``/``"false"``
            string, ``None``, or a handful of common alternates
            (``"t"``/``"f"``, ``"yes"``/``"no"``, ``"1"``/``"0"``).

    Returns:
        The parsed boolean, or ``None`` if ``value`` was ``None`` or
        an empty string.

    Raises:
        ValueError: if the value is present but cannot be interpreted
            as a boolean. Fail-loud so data-entry regressions surface
            immediately rather than masquerading as False.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "":
            return None
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    raise ValueError(
        f"parse_bool_or_none: cannot interpret {value!r} as boolean. "
        f"Expected real bool, 'true'/'false' (any case), or None."
    )


# ----------------------------------------------------------------------------
# ImageContent — one image attached to a multimodal Example
# ----------------------------------------------------------------------------


class ImageContent(BaseModel):
    """One image attached to a multimodal example.

    The 2026-05-07 share onwards ships images by URL reference: the
    benchmark JSON carries the target ``path`` and ``mime_type`` per
    image, and the matching ``multimodal:image_url`` gold-triple
    points at the source URL. Image bytes are downloaded out-of-band
    by ``scripts/download_multimodal_images.py``, which writes each
    file to ``{question_id}_{slot}.{ext}`` under
    ``benchmark/final/<size>/multimodal/<kg>/images/``.

    The runner consumes ``ImageContent`` records when
    ``multimodal_mode == "image"``: each backend's adapter loads the
    file, base64-encodes it, and packs it into a provider-specific
    content block alongside the prompt text. In ``multimodal_mode == "name"``
    the image bytes are not loaded — the placeholder ``[IMAGE_<slot>]``
    in the question is substituted with the entity's display name
    instead, giving a text-only baseline against which the visual
    contribution can be measured.
    """

    slot: str
    """Uppercase letter matching the placeholder in the cleaned
    question text (``"A"`` for ``[IMAGE_A]``). Aligns with the entity
    slots on :class:`Example.entities` so a backend can resolve
    ``slot="A"`` to ``example.entities["A"]`` for name-mode rendering."""

    path: str
    """Filesystem path to the image.

    On disk the canonical (published) form is a short relative segment
    ``images/<qid>_<slot>.<ext>`` interpreted relative to
    ``<dataset_root>/multimodal/<kg_source>/``. At load time
    :meth:`Example.from_benchmark_row` resolves this against the source
    JSONL location via :func:`benchmark_models.image_paths.resolve_image_path`
    and writes back the absolute path, so by the time downstream code
    (backends, ``read_image_bytes``) sees an ``ImageContent`` the value
    is an absolute filesystem path. A legacy-format shim accepts the
    earlier ``benchmark/eng/multimodal/<kg>/images/<file>`` shape that
    the generator emitted before the published-dataset migration —
    see :func:`benchmark_models.image_paths.resolve_image_path` for
    the resolution rules. Backends open this lazily — the runner does
    not pre-load bytes."""

    mime_type: str
    """MIME type as written by the cleanup script
    (``"image/jpeg"`` / ``"image/png"`` / ``"image/webp"`` / ...).
    Each backend adapter passes this through to the provider so the
    upload is correctly tagged."""


# ----------------------------------------------------------------------------
# Example — input shape
# ----------------------------------------------------------------------------

# Fields the model section MUST NOT see — gold answers and validation
# diagnostics. ``Example.from_benchmark_row`` strips these; the eval
# pipeline reads them from the source JSONL directly via a
# ``question_id`` join.
#
# ``ground_truth_type`` is NOT in this set. It's a schema label
# describing the shape of the expected answer (e.g. ``"boolean"``,
# ``"coordinates"``), not the answer itself. Backends use the per-row
# value to build the correct JSON response schema for each question.
_GOLD_FIELDS: frozenset[str] = frozenset(
    {
        "ground_truth",
        "is_valid",
        "validation_errors",
        "error",
        # Premise-check flag. When set to a falsy value the question's
        # premise does not hold and the benchmark generator emits
        # ``ground_truth: None``. Exposing this to the model would be a
        # leak equivalent to handing over the answer, so it is stripped
        # alongside the other gold fields. The eval pipeline can still
        # read it from the source JSONL by question_id join.
        "is_true_premise",
    }
)

# Top-level row fields the loader handles explicitly. Any field outside
# this set (and outside the entity / parameters / gold-strip patterns)
# lands in ``Example.raw`` with a warning so schema drift is visible.
# Process-wide bookkeeping for the schema-drift warning. Populated
# the first time a given unknown field is observed; subsequent rows
# with the same field are silent. Reset between tests via the
# ``reset_schema_drift_warnings`` helper below for hygiene.
_WARNED_UNKNOWN_FIELDS: set[str] = set()


def reset_schema_drift_warnings() -> None:
    """Clear the warn-once set used by :meth:`Example.from_benchmark_row`.

    Tests that exercise schema-drift warnings should call this before
    the assertion phase so a previously-warned-about field still
    triggers a fresh warning.
    """
    _WARNED_UNKNOWN_FIELDS.clear()


_KNOWN_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "question_id",
        "tid",
        "sfid",
        "vid",
        "variant_stid",
        "question",
        "base_template",
        "ground_truth_type",
        "kg_source",
        "language",
        "strata_cell",
        "image_path",
        "images",
        "is_multimodal",
        "parameters",
    }
)

# Per-entity source-attribution fields shipped by the multimodal
# share generator. The loader does not consume them, but they're
# dataset provenance and the analysis layer wants them. Listed
# as suffixes so the entity-letter prefix stays generic ("entity_a_*",
# "entity_b_*", ...). Matched fields land in ``Example.raw`` with no
# warn-once noise.
_SILENT_RAW_SUFFIXES: frozenset[str] = frozenset(
    {
        "_image_source",
        "_image_url",
    }
)


def _is_silent_raw_field(key: str) -> bool:
    """Return ``True`` if ``key`` is a known per-entity attribution
    field that should land in ``raw`` without triggering the
    schema-drift warning."""
    if not key.startswith("entity_") and not key.startswith("event_"):
        return False
    return any(key.endswith(suffix) for suffix in _SILENT_RAW_SUFFIXES)


class Example(BaseModel):
    """One benchmark question, loaded from the benchmark JSONL.

    Use :meth:`from_benchmark_row` to construct from a parsed JSONL row;
    that classmethod handles entity-slot extraction, parameters parsing,
    gold-field stripping, and unknown-field bookkeeping in one place.

    Strictly excludes gold answer fields (``ground_truth``,
    ``is_valid``, ``validation_errors``, ``error``, ``is_true_premise``)
    so leakage to the model is structurally impossible.
    ``ground_truth_type`` is kept as schema metadata — it names the
    shape of the expected answer (``"boolean"``, ``"coordinates"``,
    ``"bounding_box"``) rather than the answer itself, and backends
    use it to build the per-question JSON response schema.
    """

    model_config = ConfigDict(extra="forbid")

    # ---- identity ----

    question_id: str
    """Stable UUID identifying this question across runs."""

    tid: str
    """Template ID (e.g., ``"1.1"``)."""

    sfid: str
    """Spatial Function ID (e.g., ``"A"``)."""

    vid: str | None = None
    """Variant ID (e.g. ``"1.1.2"``) uniquely identifying one
    template-variant pairing. ``None`` when the row doesn't supply it."""

    variant_stid: str | None = None
    """Sub-template ID within a variant (``"1"``, ``"2"``, …).
    Complements :attr:`vid`; together they disambiguate question shape
    when one TID spawns multiple variants."""

    # ---- content ----

    question: str
    """The natural-language question text in the target language."""

    base_template: str
    """The original template with ``{{placeholders}}``, before
    substitution. The model never sees this; it's kept for analysis,
    prompt-engineering decisions, and template-shape reporting."""

    ground_truth_type: str | None = None
    """Per-question schema label for the expected answer (e.g.
    ``"boolean"``, ``"coordinates"``, ``"bounding_box"``). Backends
    use this to pin the JSON response schema — this is an intentional
    form of schema assistance given to the model (answer *shape* is
    provided; answer *value* must still be produced). The scorer
    dispatches per-type matchers on the canonicalised form. Not the
    answer itself. ``None`` when the row doesn't supply it (backends
    fall back to the per-TID ``TemplateInfo.answer_type``).

    The free-form output ablation on workhorse models quantifies how
    much of the headline reflects this format assistance versus raw
    reasoning capability."""

    # ---- entities ----

    entities: dict[str, EntityRef] = Field(default_factory=dict)
    """Entity slots keyed by uppercase letter (``"A"``, ``"B"``,
    ``"C"``, ``"D"``, ...). Empty dict if the question references no
    entities. Iteration in alphabetic order via :meth:`all_entity_ids`
    or :meth:`all_entity_names`."""

    events: dict[str, EntityRef] = Field(default_factory=dict)
    """Event slots keyed by uppercase letter (``"A"``, ``"B"``, ...).
    Mirrors :attr:`entities` but for first-class temporal/event
    references (e.g. SFID I earthquakes). Loaded from
    ``event_<letter>_id`` / ``event_<letter>_name`` /
    ``event_<letter>_subcategory`` keys in the benchmark row.
    Empty dict for questions that don't reference events. Kept
    semantically distinct from :attr:`entities` so iterators that
    reason about spatial entities don't accidentally also touch
    events (and vice versa)."""

    # ---- non-entity placeholder substitutions ----

    parameters: dict[str, Any] = Field(default_factory=dict)
    """Non-entity placeholder substitutions, parsed from the JSONL
    ``parameters`` string at load time. Holds values like
    ``cardinal_direction``, ``time_period``, ``event_A``,
    ``attribute_A``, ``value``, ``rank_position``, ``aggregation_type``,
    ``grid_type``, ``h3_resolution``, plus per-entity metadata like
    ``entity_A_category``. Empty dict if the JSONL row had no
    ``parameters`` field (the case in some sample exports)."""

    # ---- provenance / metadata ----

    kg_source: str
    """Source knowledge graph (``"worldkg"``, ``"osmh3kg"``, ``"kwg"``)."""

    language: str = "eng"
    """ISO 639-3 language code (e.g. ``"eng"``, ``"deu"``,
    ``"zho_CN"``) for the question text. Defaults to ``"eng"``. The
    full multilingual benchmark will produce one JSONL tree per
    language under ``benchmark/<iso639-3>/<kg>/<sfid>/...``; the
    runner extracts and sets this field at load time."""

    strata_cell: str | None = None
    """Stratified-sampling tag (e.g., ``"HIC_high"``). Used by the eval
    pipeline for per-stratum breakdowns. ``None`` when the source JSONL
    does not carry the field."""

    image_path: str | None = None
    """Single-image path field. Populated only by shares that carry
    one image per row at the top level; current multimodal shares
    populate :attr:`images` instead and leave this ``None``."""

    images: list[ImageContent] = Field(default_factory=list)
    """Images attached to this example, in slot order
    (``"A"``, ``"B"``, ...). Empty for non-multimodal rows. The
    benchmark generator emits this list directly with ``slot``,
    ``path``, and ``mime_type`` per image; image bytes are
    downloaded out-of-band via
    ``scripts/download_multimodal_images.py``. The question text
    uses ``[IMAGE_<slot>]`` placeholders that align with each entry's
    :attr:`ImageContent.slot`. The runner routes images to the
    backend (image mode) or substitutes names into the placeholders
    (name mode); see ``--multimodal-mode``."""

    is_multimodal: bool = False
    """Schema flag from the multimodal share variant. ``True`` for
    rows under ``benchmark/final/<size>/multimodal/<kg>/...``. Independent of
    :attr:`images` length — a row could in principle be multimodal
    without images attached (none observed in current shares)."""

    # ---- escape hatch ----

    raw: dict[str, Any] = Field(default_factory=dict)
    """Any JSONL fields the loader did not recognize land here. Logged
    as a warning at load time so schema drift is visible. The model
    section's prompt rendering never reads from ``raw``; it exists so
    eval-pipeline / analysis code can pick up new fields without
    requiring a new ``Example`` schema."""

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------

    @property
    def entity_a(self) -> EntityRef | None:
        """Convenience accessor for entity slot A."""
        return self.entities.get("A")

    @property
    def entity_b(self) -> EntityRef | None:
        """Convenience accessor for entity slot B."""
        return self.entities.get("B")

    @property
    def entity_c(self) -> EntityRef | None:
        """Convenience accessor for entity slot C."""
        return self.entities.get("C")

    @property
    def entity_d(self) -> EntityRef | None:
        """Convenience accessor for entity slot D."""
        return self.entities.get("D")

    def has_entity(self, slot: str) -> bool:
        """Return ``True`` if entity slot ``slot`` is populated.

        ``slot`` is matched case-insensitively (``"a"``, ``"A"``, both work).
        """
        return slot.upper() in self.entities

    def all_entity_ids(self) -> list[str]:
        """Return entity IDs in slot order (A, B, C, D, ...).

        Skips empty slots (a question with only A and C produces a
        2-element list, not 3). Useful for tier-3 entity-linking code
        that needs to look every entity up in the KG.
        """
        return [self.entities[k].id for k in sorted(self.entities)]

    def all_entity_names(self) -> list[str]:
        """Return entity display names in slot order (A, B, C, D, ...)."""
        return [self.entities[k].name for k in sorted(self.entities)]

    # ------------------------------------------------------------------
    # canonical loader
    # ------------------------------------------------------------------

    @classmethod
    def from_benchmark_row(
        cls,
        row: dict[str, Any],
        *,
        language: str | None = None,
        source_path: Path | str | None = None,
    ) -> Example:
        """Construct an ``Example`` from a parsed benchmark JSONL row.

        Responsibilities:
            * Extract ``entity_<letter>_id`` / ``entity_<letter>_name``
              keys into the :attr:`entities` dict, keyed by uppercase
              letter. Handles any number of entities (A, B, C, D, E, ...).
            * Parse the ``parameters`` JSON string into a dict, defaulting
              to ``{}`` if absent. Look up ``entity_<letter>_category``
              entries and populate the corresponding ``EntityRef.category``.
            * Strip every gold-answer field (``ground_truth``,
              ``is_valid``, ``validation_errors``, ``error``,
              ``is_true_premise``) so it never reaches the model.
              ``ground_truth_type`` is preserved on the Example as
              schema metadata (see the class docstring).
            * Promote ``strata_cell`` to a typed field if present.
            * Set ``language`` from the ``language`` parameter if
              provided (the runner passes this when reading a
              language-tagged JSONL file). Falls back to the row's
              ``language`` field, then ``"eng"``.
            * Resolve ``images[*].path`` to absolute filesystem paths
              when ``source_path`` is provided, via
              :func:`benchmark_models.image_paths.resolve_image_path`.
              Without ``source_path`` the stored value is kept verbatim
              (downstream code will see whatever the JSONL contained).
            * Warn (via loguru) about any unrecognized top-level field,
              and stash it in :attr:`raw` so eval / analysis code can
              still reach it.

        Args:
            row: One parsed JSON object from a benchmark JSONL file.
            language: Optional language code override. The runner passes
                this when it knows the language from the filename
                (e.g., ``benchmark_de.jsonl`` → ``"de"``).
            source_path: Path of the JSONL shard the row was loaded
                from. The runner threads this through so image paths
                can be resolved against the actual on-disk layout
                rather than the literal value stored in the JSONL.
                ``None`` skips image resolution — appropriate for
                tests that construct synthetic rows.

        Returns:
            A validated :class:`Example` instance.

        Raises:
            pydantic.ValidationError: if the row is missing required
                identity fields (``question_id``, ``tid``, ``sfid``,
                ``question``, ``base_template``, ``kg_source``).
        """
        # Parse parameters first — both entity-category lookup and
        # general parameter exposure depend on it. Annotated as
        # ``object`` so the isinstance ladder narrows cleanly without
        # pyright incorrectly flagging the catch-all else as unreachable.
        params_raw: object = row.get("parameters")
        parameters: dict[str, Any]
        if params_raw is None:
            parameters = {}
        elif isinstance(params_raw, dict):
            parameters = params_raw
        elif isinstance(params_raw, str):
            try:
                parsed: object = json.loads(params_raw)
                if isinstance(parsed, dict):
                    parameters = parsed
                else:
                    logger.warning(
                        "Example.from_benchmark_row: parameters JSON for "
                        "question_id={} parsed to {}, expected dict; "
                        "ignoring",
                        row.get("question_id"),
                        type(parsed).__name__,
                    )
                    parameters = {}
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Example.from_benchmark_row: failed to parse "
                    "parameters JSON for question_id={}: {}; ignoring",
                    row.get("question_id"),
                    exc,
                )
                parameters = {}
        else:
            logger.warning(
                "Example.from_benchmark_row: unexpected parameters type "
                "{} for question_id={}; ignoring",
                type(params_raw).__name__,
                row.get("question_id"),
            )
            parameters = {}

        # Extract entity and event slots. Both follow the same key
        # shape — ``<prefix>_<letter>_{id,name,subcategory}`` — so we
        # walk the row once and bucket by prefix. ``entity_*`` lookups
        # also consult ``parameters`` for ``entity_<letter>_category``;
        # ``event_*`` does not (the benchmark does not currently emit
        # event categories).
        consumed: set[str] = set()
        slot_data: dict[str, dict[str, dict[str, Any]]] = {
            "entity_": {},
            "event_": {},
        }
        for key, value in row.items():
            for prefix in ("entity_", "event_"):
                if not key.startswith(prefix):
                    continue
                tail = key[len(prefix):]
                # tail is e.g. "a_id", "b_name", "A_category", or
                # "a_subcategory".
                if "_" not in tail:
                    break
                letter_part, attr = tail.rsplit("_", 1)
                # ``category`` is canonically read from ``parameters``
                # for entities, but transient share versions have
                # placed it at the top level. Accept both —
                # ``_build_refs`` prefers the top-level value when
                # present and falls back to ``parameters`` otherwise.
                # Anything outside the known set lands in ``raw`` with
                # a warning so genuine schema drift remains visible.
                if attr not in {"id", "name", "subcategory", "category"}:
                    break
                # m1: require a single-letter slot key. Multi-letter
                # keys (e.g., ``entity_aaa_id``) would otherwise pass
                # the ``.isalpha()`` check and land as slot "AAA",
                # which the templates and downstream code don't
                # expect. No live row uses multi-letter slots; the
                # tighter check makes a future regression visible.
                if len(letter_part) != 1 or not letter_part.isalpha():
                    break
                letter = letter_part.upper()
                slot_data[prefix].setdefault(letter, {})[attr] = value
                consumed.add(key)
                break

        def _build_refs(
            prefix: str,
            data: dict[str, dict[str, Any]],
        ) -> dict[str, EntityRef]:
            """Materialize ``EntityRef`` objects for one prefix bucket."""
            kind_label = "entity" if prefix == "entity_" else "event"
            refs: dict[str, EntityRef] = {}
            for letter, attrs in sorted(data.items()):
                ent_id = attrs.get("id")
                ent_name = attrs.get("name")
                subcategory = attrs.get("subcategory")
                has_id = ent_id is not None
                has_name = ent_name is not None

                if has_id != has_name:
                    # Partially-populated id/name pair — skip rather
                    # than guess. Logged so we notice. Subcategory
                    # alone doesn't change the id/name consistency
                    # rule. SFID I emits ``event_a_name`` without
                    # ``event_a_id`` for several TIDs — those land
                    # here as warnings; the runner can decide how to
                    # handle name-only events at tier-3 entity-link
                    # time.
                    logger.warning(
                        "Example.from_benchmark_row: {} slot {} for "
                        "question_id={} is missing id or name "
                        "(id={!r}, name={!r}); skipping slot",
                        kind_label,
                        letter,
                        row.get("question_id"),
                        ent_id,
                        ent_name,
                    )
                    continue
                if not has_id and not has_name and subcategory is None:
                    # Truly empty slot — nothing to record. Silent,
                    # same as before.
                    continue

                # Resolve category. Top-level ``<prefix><letter>_category``
                # wins when present; otherwise fall back to the
                # canonical ``parameters[entity_<letter>_category]``
                # source for entity slots. Events don't currently
                # carry a category field in either location.
                category: str | None = None
                top_level_cat = attrs.get("category")
                if top_level_cat is not None:
                    category = str(top_level_cat)
                elif prefix == "entity_":
                    cat = parameters.get(f"entity_{letter}_category")
                    if cat is not None:
                        category = str(cat)
                refs[letter] = EntityRef(
                    # Empty string for category-only slots — EntityRef.id
                    # / .name are documented as empty in that case.
                    id=str(ent_id) if ent_id is not None else "",
                    name=str(ent_name) if ent_name is not None else "",
                    category=category,
                    subcategory=(
                        str(subcategory) if subcategory is not None else None
                    ),
                )
            return refs

        entities = _build_refs("entity_", slot_data["entity_"])
        events = _build_refs("event_", slot_data["event_"])

        # Gold-answer fields are never copied out of ``row`` — the
        # ``raw_extra`` loop below skips them alongside the known /
        # entity-slot keys, so nothing from ``_GOLD_FIELDS`` leaks into
        # ``Example.raw``. No mutation of the caller's dict.

        # Pull out the known top-level fields. ``parameters`` is
        # already consumed via params_raw above; entity_*_id /
        # entity_*_name keys are tracked in `consumed`.
        question_id = row.get("question_id")
        tid = row.get("tid")
        sfid = row.get("sfid")
        vid = row.get("vid")
        variant_stid = row.get("variant_stid")
        question = row.get("question")
        base_template = row.get("base_template")
        ground_truth_type = row.get("ground_truth_type")
        kg_source = row.get("kg_source")
        strata_cell = row.get("strata_cell")
        image_path = row.get("image_path")
        # ``is_multimodal`` may arrive as a real bool or as the string
        # ``"true"``/``"false"``. ``bool("false")`` is truthy, so route
        # through ``parse_bool_or_none`` instead of a naive cast.
        is_multimodal = parse_bool_or_none(row.get("is_multimodal")) or False
        images_raw = row.get("images") or []
        images: list[ImageContent] = []
        kg_source_str = str(kg_source) if kg_source is not None else ""
        for entry in images_raw:
            if not isinstance(entry, dict):
                logger.warning(
                    "Example.from_benchmark_row: images entry on "
                    "question_id={} is {} (expected dict); skipping",
                    row.get("question_id"),
                    type(entry).__name__,
                )
                continue
            try:
                img = ImageContent(**entry)
                if source_path is not None and kg_source_str:
                    img = img.model_copy(
                        update={
                            "path": resolve_image_path(
                                img.path, Path(source_path), kg_source_str,
                            ),
                        },
                    )
                images.append(img)
            except Exception as exc:
                logger.warning(
                    "Example.from_benchmark_row: malformed images entry "
                    "on question_id={}: {}; skipping",
                    row.get("question_id"),
                    exc,
                )
        row_language = row.get("language")
        effective_language = language or row_language or "eng"

        # Anything else top-level lands in raw with a warning. Gold
        # fields are filtered here so they never reach ``Example.raw``
        # — this is the single strip site for gold data.
        raw_extra: dict[str, Any] = {}
        for key, value in row.items():
            if key in _KNOWN_TOP_LEVEL_FIELDS:
                continue
            if key in consumed:
                continue
            if key == "parameters":
                continue
            if key in _GOLD_FIELDS:
                continue
            raw_extra[key] = value
            # Per-entity attribution fields land in raw silently —
            # they're documented dataset provenance, not drift.
            if _is_silent_raw_field(key):
                continue
            # Warn-once per field name across the process so a single
            # schema-drifted column doesn't spam thousands of rows
            # with identical warnings. The qid of the FIRST row that
            # exhibited the field is logged as a pointer; subsequent
            # rows are silent.
            if key not in _WARNED_UNKNOWN_FIELDS:
                _WARNED_UNKNOWN_FIELDS.add(key)
                logger.warning(
                    "Example.from_benchmark_row: unrecognized field "
                    "{!r} (first seen on question_id={}); storing in "
                    "Example.raw. Subsequent rows with this field "
                    "will be silent.",
                    key,
                    question_id,
                )

        return cls(
            question_id=str(question_id) if question_id is not None else "",
            tid=str(tid) if tid is not None else "",
            sfid=str(sfid) if sfid is not None else "",
            vid=str(vid) if vid is not None else None,
            variant_stid=str(variant_stid) if variant_stid is not None else None,
            question=str(question) if question is not None else "",
            base_template=str(base_template) if base_template is not None else "",
            ground_truth_type=(
                str(ground_truth_type) if ground_truth_type is not None else None
            ),
            entities=entities,
            events=events,
            parameters=parameters,
            kg_source=str(kg_source) if kg_source is not None else "",
            language=effective_language,
            strata_cell=str(strata_cell) if strata_cell is not None else None,
            image_path=str(image_path) if image_path is not None else None,
            images=images,
            is_multimodal=is_multimodal,
            raw=raw_extra,
        )


# ----------------------------------------------------------------------------
# Nested models used by Prediction
# ----------------------------------------------------------------------------


class Usage(BaseModel):
    """Token usage and counts for one prediction.

    All fields are integers and default to 0 so backends that don't
    distinguish stages (e.g., open-weights vLLM tier 1) can populate
    only ``input_tokens`` / ``output_tokens`` and leave the rest at 0.
    """

    input_tokens: int = 0
    """Total prompt tokens consumed."""

    output_tokens: int = 0
    """Total completion tokens generated."""

    reasoning_tokens: int = 0
    """Tier-2 thinking tokens, where the provider distinguishes them."""

    cached_tokens: int = 0
    """Prompt-caching hit tokens, where the provider reports them.

    Source per backend:
    * OpenAI: ``response.usage.prompt_tokens_details.cached_tokens``
    * Google: ``response.usage_metadata.cached_content_token_count``
    * vLLM: not API-level; left at 0 (KV cache is internal)."""

    cache_creation_tokens: int = 0
    """Tokens written to a new cache entry on this call. Distinct
    from :attr:`cached_tokens` (which is read-from-cache savings):
    cache_creation is the upfront premium when a new prefix is
    seeded into the cache. No shipped backend reports it separately,
    so it stays at 0."""

    tool_call_count: int = 0
    """Tier-3 only: number of tool invocations made by the agent."""

    api_call_count: int = 0
    """Total round-trips to the model (1 for tier 1/2; tier 3 = number
    of agent steps; 0 for predictions that never reached the model
    such as registry-miss errors).

    m2: default is 0, not 1 — backends explicitly set to 1 (or the
    tier-3 step count) when a real call was made. The previous
    default-1 value silently inflated api_call_count on records
    where no API call happened (e.g., DummyModel non-error and the
    error-prediction helpers in every backend that explicitly
    overrode to 0). Defaulting to 0 makes the implicit contract
    explicit: only set non-zero when a call was actually made."""


class ToolCall(BaseModel):
    """One tool invocation from a tier-3 agent loop.

    Stored verbatim in :attr:`Prediction.tool_calls` so the eval pipeline
    can analyze tool-call patterns, error rates, and traces. Includes
    both successful results and errors.
    """

    name: str
    """The tool name the model invoked (e.g., ``"find_entity"``)."""

    args: dict[str, Any] = Field(default_factory=dict)
    """The tool's input arguments as parsed from the model's function call."""

    result: Any = None
    """The tool's return value, JSON-serializable. ``None`` if the call errored."""

    error: str | None = None
    """A human-readable error message if the tool call failed."""

    latency_ms: float = 0.0
    """Wall-clock time the tool dispatch took, in milliseconds."""


# ----------------------------------------------------------------------------
# Prediction — saved record schema
# ----------------------------------------------------------------------------


class Prediction(BaseModel):
    """One model's answer to one example, with full provenance.

    Single flat schema across all tiers; tier-specific fields are
    optional. Serialized as one JSON line per record into
    ``results/<experiment>/<variant>/predictions.jsonl``. The eval
    pipeline (separate branch) reads these records and applies per-type
    matchers and metric aggregations.
    """

    # Provenance — always present.
    question_id: str
    tid: str
    sfid: str
    """Spatial Function ID, denormalized from the template registry at
    write time. Lets the eval pipeline group by spatial function family
    (``"A"``, ``"B"``, ...) without joining against ``TemplateRegistry``."""
    answer_type: str
    """Canonical Answer Type from the template registry, denormalized
    here for eval-time grouping without re-joining against the registry."""
    model_id: str
    """The variant id from ``models.yaml`` (e.g., ``"gemini-3-flash-plain"``)."""
    tier: Tier
    language: str
    prompt_template_id: str
    """Identifier of the prompt template used (e.g., ``"tier1.canonical"``)."""
    seed: int
    snapshot_id: str
    """KG snapshot pinning identifier (Option A from the plan
    discussion). Recorded so re-runs can verify they hit the same KG
    state. Constructed by :mod:`benchmark_models.snapshot`:

    * Tier 1 / tier 2: a 12-char hash of the KG path env vars
      (``WORLDKG_*``, ``OSMH3KG_*``, ``KWG_*``). No Spark required.
    * Tier 3: the same paths hash, plus a 12-char hash of the live
      Delta table versions (read via the runner's ``EntityService``,
      which proxies to ``TripleBridge`` over the shared Spark
      session), separated by ``@`` — e.g.,
      ``"a1b2c3d4e5f6@9f8e7d6c5b4a"``.

    Code provenance (which ``benchmark_models`` / ``benchmark_generation``
    commit produced these predictions) is *not* on the prediction
    record — it lives in MLflow run-level params instead, since it's
    constant across all predictions in a run."""

    # Response — always present (answer can be ``None`` for refusal).
    answer: Any
    """The model's structured answer (boolean, number, string, list, ...)
    or ``None`` to indicate abstention."""
    unit: str | None = None
    """Free-form unit string for numeric answers (e.g., ``"km"``).
    Captures what the model declared, even if it differs from the
    requested unit. ``None`` for non-numeric answer types."""
    refusal: bool = False
    """Convenience flag derived from ``answer is None``. Recorded
    explicitly so the eval pipeline does not have to recompute it."""

    # Tier-2 extras.
    reasoning: str | None = None
    """The reasoning trace for prompted-CoT variants
    (``{"reasoning": ..., "answer": ...}``). For native-thinking models,
    the provider's reasoning channel content if available, otherwise None."""
    sc_samples: list[Any] | None = None
    """Self-consistency samples when ``self_consistency_n > 1``.
    Length equals N; the final ``answer`` is the majority vote across
    these samples."""

    # Tier-3 extras.
    tool_calls: list[ToolCall] | None = None
    """Full agent-loop trace, including failed calls. ``None`` for
    tier 1 / tier 2."""
    cap_hit: bool | None = None
    """``True`` if the agent loop terminated by hitting ``max_tool_calls``.
    ``None`` for tier 1 / tier 2."""
    final_state: FinalState | None = None
    """Tier-3 termination reason. ``None`` for tier 1 / tier 2."""
    error_class: ErrorClass | None = None
    """Qualitative category when ``final_state == "error"``: ``infra``
    (transport / API), ``code_gen`` (tier-3 CodeAgent Python errors),
    ``schema_violation`` (structured-output parse failure), ``config``
    (unknown tid, dynamic-type skip, coverage gap, backend mis-route),
    or ``unknown`` (classification gap — fix when observed). ``None``
    for non-error predictions."""

    # Oracle extras.
    triple_context: str | None = None
    """The per-question formatted triple context, populated only for
    oracle runs. Captures the string that was substituted into the
    prompt's ``$triple_context`` slot (for tier-1 oracle modes) or
    into the tier-3 oracle prompt. ``None`` for non-oracle runs.

    Without this field the oracle condition is not auditable after
    the fact: the formatter runs at render time on the gold triples
    map, and the template files alone do not let you reconstruct
    what the model actually saw per question. Storing the rendered
    triple context closes that gap at ~500 B per prediction."""

    # Always present.
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    """Total wall-clock time for this prediction in milliseconds, including
    all backend round-trips and tool dispatches."""
    raw_completion: str = ""
    """The full provider response (or vLLM completion text) as received,
    for debugging. Not used by the eval pipeline; useful when investigating
    individual failures."""


# ----------------------------------------------------------------------------
# Model — abstract base class
# ----------------------------------------------------------------------------


class Model(ABC):
    """Abstract base class for every model variant.

    Subclasses live in ``benchmark_models/backends/`` (as backend
    classes) and ``benchmark_models/tiers/`` (as tier mixins). The
    registry composes a backend with a tier to produce a concrete
    Model instance.

    Subclasses must implement :meth:`predict`. The default
    :meth:`predict_batch` loops over :meth:`predict`; backends that
    benefit from batching (vLLM, async API) override it.

    Lifecycle expectations:
        * The runner constructs each Model once per experiment-YAML invocation.
        * The runner calls ``predict_batch`` (or ``predict`` for tier 3)
          on the questions in scope, possibly in chunks of 32–64 examples.
        * The runner is responsible for persisting the returned ``Prediction``
          records to disk (JSONL).

    Subclasses should NOT touch disk, MLflow, or any process-global state
    inside ``predict``/``predict_batch``. The runner owns those concerns.
    """

    #: Variant identifier from ``models.yaml`` (e.g., ``"qwen35-a3b-plain"``).
    #: Subclasses or factory code set this on the instance.
    model_id: ClassVar[str]

    #: Tier this Model implements (``"tier1"``, ``"tier2"``, or ``"tier3"``).
    tier: ClassVar[Tier]

    @abstractmethod
    def predict(self, example: Example) -> Prediction:
        """Run one example through this model and return its prediction.

        Implementations must:
            * Set every required field on the returned ``Prediction``
              (provenance, ``answer``, ``usage``, ``latency_ms``, ``raw_completion``).
            * Set ``refusal=True`` and ``answer=None`` if the model declines.
            * Populate tier-specific extras (``reasoning``, ``tool_calls``,
              ``cap_hit``, ``final_state``) when applicable.
            * Never raise on per-example failures; instead, populate
              ``final_state="error"`` and an explanatory ``raw_completion``.
              The runner uses exceptions only for fatal infrastructure issues.
        """
        raise NotImplementedError

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        """Run a batch of examples and return their predictions in order.

        Default implementation loops over :meth:`predict`. Backends that
        benefit from batching (vLLM, async API gather) override this and
        return predictions in the same order as ``examples``.
        """
        return [self.predict(ex) for ex in examples]

    def close(self) -> None:  # noqa: B027
        """Release any held resources (vLLM engine, HTTP clients, etc.).

        Default is a no-op (``noqa: B027`` — subclasses are free to
        override but are not required to). Override only when a
        backend owns a heavyweight resource (e.g., a loaded vLLM
        engine) that should be freed at the end of a run.
        """
