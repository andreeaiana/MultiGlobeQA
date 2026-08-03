"""Oracle baselines — gold-triples-injected prompt variants.

The oracle baselines are *ceilings*, not models: same underlying model
weights as the base tier runs, but the prompt is augmented with the
question's supporting triples (retrieved perfectly). Four modes:

* ``structured`` — gold triples as a JSON array. "Structured-retrieval
  result" ceiling.
* ``raw`` — gold triples as N-Triples-like lines with full URIs.
  Isolates the RDF-syntax-reading cost vs the structured mode.
* ``verbalized`` — gold triples grouped by subject and rendered as
  natural-language prose ("Entity X is a cafe with geometry POINT
  (...)"). Plain-RAG ceiling; directly comparable to future tier-4
  text-RAG runs.
* ``t3`` — gold triples injected (structured form) into a tier-3
  prompt that grants the Python sandbox but NO KG/web tools. Compute-
  only ceiling: given perfect retrieval, can the model write correct
  spatial code?

Gold-triples format: the JSONL shards under
``benchmark/final/<size>/<kg>/<sfid>/gold_triples.jsonl/`` carry one row per
triple, with the schema:

    {"question_id": "...", "entity_role": "a"|"b"|"c"|...,
     "subject": "...", "predicate": "...", "object": "..."}

:func:`load_gold_triples` groups the flat rows into the
``{question_id: list[Triple]}`` shape every formatter consumes.

Object-format heterogeneity across KGs is a feature of the source data:

* osmh3kg: bare WKT, e.g. ``POLYGON ((130.985 30.62, ...))``.
* kwg: literal-quoted WKT, e.g. ``"POLYGON ((-161.0 58.99, ...))"``.
* worldkg: RDF typed-literal, e.g.
  ``"Point(7.034 61.256)"^^<http://www.opengis.net/ont/geosparql#wktLiteral>``.

:func:`_strip_typed_literal` normalises the three forms before the
WKT-vs-entity-id heuristic in :func:`format_raw`.

Key design decision (leakage): gold triples are NOT attached to
:class:`Example`. That keeps non-oracle code paths structurally
incapable of seeing them, mirroring the existing discipline that
:class:`Example` excludes gold-answer fields.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from loguru import logger

if TYPE_CHECKING:
    from benchmark_models.oracle_simplify import SimplifyStats

# Supported oracle modes. T1 modes substitute into ``tier1_oracle_*.txt``
# templates; ``t3`` is handled by the ``SmolagentsOracleModel`` class
# separately.
OracleMode = Literal["structured", "raw", "verbalized", "t3"]

T1_ORACLE_MODES: tuple[OracleMode, ...] = ("structured", "raw", "verbalized")


class Triple(TypedDict):
    """One RDF triple from a gold-triples row.

    Subject and object are either entity identifiers (KG-specific URIs
    or bare ids) or literals (type names, WKT geometry strings).
    Predicate is a full URI.

    ``entity_role`` is optional — populated by :func:`load_gold_triples`
    from the per-row ``entity_role`` field of the gold-triples share.
    Callers that build ``Triple`` dicts directly may omit it; the
    formatters do not require it.
    """

    subject: str
    predicate: str
    object: str
    entity_role: NotRequired[str]


# Short predicate suffixes used by the ``structured`` formatter to
# keep the JSON readable. Full URIs are recoverable via the ``PREFIXES``
# header printed with the JSON block.
PREDICATE_PREFIXES: dict[str, str] = {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.opengis.net/ont/geosparql#": "geo",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/2002/07/owl#": "owl",
}


@dataclass
class OracleContext:
    """Bundle of state needed to render oracle prompts.

    Held only by oracle-aware code (the ``PromptRenderer`` for T1 modes
    and :class:`SmolagentsOracleModel` for T3). Never attached to
    :class:`Example` or any other shared state.

    ``wkt_simplify_tolerance``, when set to a positive value, runs
    every gold triple's WKT object through Douglas-Peucker
    simplification at construction time (see
    :mod:`benchmark_models.oracle_simplify`). Applied uniformly across
    all backends regardless of context-window capacity so the
    cross-backend oracle comparison stays apples-to-apples. The
    ``simplify_stats`` dict (question_id → list of per-triple
    ``SimplifyStats``) is populated as a side effect of construction
    and is what the §Methodology footnote should quote when reporting
    actual reduction ratios.

    Default ``wkt_simplify_tolerance=0.001`` (~111 m at the equator)
    matches the runtime default in
    :class:`benchmark_models.kg_tools.entity_service.EntityService`
    so oracle and tier-3a/3c use-kg modes simplify uniformly without
    explicit configuration. It matters because raw ZCTA / country
    polygons can exceed the 32 K context window of open-weights
    backends and silently truncate; with the default on, even a
    32-K-context model serves oracle modes correctly out of the box.

    Override explicitly to ``0`` (or ``None``) to disable
    simplification and feed raw WKT to the model — useful for paper
    ablations that quantify what the simplifier costs in answer
    accuracy.
    """

    mode: OracleMode
    gold_triples_map: dict[str, list[Triple]]
    wkt_simplify_tolerance: float | None = 0.001
    simplify_stats: dict[str, list["SimplifyStats"]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.wkt_simplify_tolerance:
            return
        # Defer the import so importing OracleContext doesn't drag
        # shapely into the process unless simplification is active.
        from benchmark_models.oracle_simplify import simplify_triples

        new_map: dict[str, list[Triple]] = {}
        for qid, ts in self.gold_triples_map.items():
            new_ts, stats = simplify_triples(ts, self.wkt_simplify_tolerance)
            new_map[qid] = new_ts
            if stats:
                self.simplify_stats[qid] = stats

        n_triples = sum(len(s) for s in self.simplify_stats.values())
        chars_saved = sum(
            s.original_chars - s.simplified_chars
            for ss in self.simplify_stats.values() for s in ss
        )
        coords_saved = sum(
            s.n_coords_before - s.n_coords_after
            for ss in self.simplify_stats.values() for s in ss
        )
        logger.info(
            "OracleContext: simplified {} WKT geometries across {} "
            "questions at ε={}; saved {:,} chars and {:,} coords total",
            n_triples, len(self.simplify_stats),
            self.wkt_simplify_tolerance, chars_saved, coords_saved,
        )
        # ``gold_triples_map`` is a regular field (not frozen), so a
        # direct attribute assignment works — no ``object.__setattr__``
        # gymnastics needed.
        self.gold_triples_map = new_map


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


_GEO_PREFIX_RE = __import__("re").compile(r"^geo(\d+)$")


def _normalize_subject(subj: str) -> str:
    """Canonicalise an osmh3kg subject id.

    osmh3kg gold triples sometimes prefix bare numeric ids with
    ``geo`` (e.g. ``geo13422232``). The benchmark's
    ``entity_<letter>_id`` field stores the bare numeric form
    (``13422232``). Without normalisation, the same entity appears
    under two distinct ids in the triple list, breaking entity
    grouping in :func:`format_verbalized`.

    Strips a leading ``geo`` only when followed by digits — leaves
    ``geosparql`` URIs and other ``geo*``-prefixed strings untouched.
    """
    m = _GEO_PREFIX_RE.match(subj)
    if m is not None:
        return m.group(1)
    return subj


def load_gold_triples(path: Path | str) -> dict[str, list[Triple]]:
    """Load gold triples from a single file, leaf shard dir, or share root.

    Format: one row per triple with schema
    ``{question_id, entity_role, subject, predicate, object}``. Rows
    sharing a ``question_id`` are grouped into a list preserving file
    order.

    ``path`` may be any of:

    1. A single ``.jsonl`` file.
    2. A leaf ``gold_triples.jsonl/`` shard directory.
    3. A higher-level root containing the nested
       ``<kg>/<sfid>/gold_triples.jsonl/`` layout (e.g.
       ``benchmark/final/large/``). Walked via :func:`discover_gold_shards`.

    Subjects of the form ``geo<digits>`` are canonicalised to bare
    digits via :func:`_normalize_subject` so the same osmh3kg entity
    consistently uses one id across its triple set.

    Returns:
        A dict mapping ``question_id`` → list of :class:`Triple` dicts.
    """
    from collections import defaultdict

    from benchmark_models.io import discover_gold_shards, iter_jsonl_rows

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gold triples file not found: {path}. Either pass "
            f"--gold-triples or make sure the default path exists."
        )
    shards = discover_gold_shards(path)

    # Dedup at read time. Upstream materialization sometimes asserts
    # the same (subject, predicate, object) triple more than once per
    # entity — typically when a triple was emitted by multiple ingest
    # paths and unioned without dedup. The duplicates have no semantic
    # content (RDF triples are set-valued) but inflate oracle prompts
    # 2-3× per question; worst case enough to push an individual row
    # over Gemini's 1M-token request limit. Dedup happens here so the
    # on-disk gold artifact is preserved byte-for-byte while oracle
    # prompts get the intended set semantics.
    grouped: dict[str, list[Triple]] = defaultdict(list)
    seen_per_qid: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    n_rows = 0
    n_duplicates = 0
    for shard in shards:
        for row in iter_jsonl_rows(shard):
            qid = row.get("question_id")
            if qid is None:
                continue
            n_rows += 1
            triple: Triple = {
                "subject": _normalize_subject(str(row.get("subject", ""))),
                "predicate": str(row.get("predicate", "")),
                "object": str(row.get("object", "")),
            }
            role = row.get("entity_role")
            if role is not None:
                triple["entity_role"] = str(role)
            key = (
                str(role) if role is not None else "",
                triple["subject"],
                triple["predicate"],
                triple["object"],
            )
            if key in seen_per_qid[qid]:
                n_duplicates += 1
                continue
            seen_per_qid[qid].add(key)
            grouped[qid].append(triple)

    n_kept = n_rows - n_duplicates
    if len(shards) == 1:
        logger.info(
            "Loaded {:,} gold triples for {:,} questions from {} "
            "({:,} duplicate rows skipped)",
            n_kept, len(grouped), shards[0].name, n_duplicates,
        )
    else:
        logger.info(
            "Loaded {:,} gold triples for {:,} questions from {} "
            "shards under {} ({:,} duplicate rows skipped)",
            n_kept, len(grouped), len(shards), path, n_duplicates,
        )
    return dict(grouped)


def assert_coverage(
    gold_triples_map: dict[str, list[Triple]],
    question_ids: list[str],
) -> None:
    """Fail loud if any ``question_id`` is missing from the map.

    Oracle runs are a deliberate scientific artifact — silent fallback
    to a non-oracle prompt for missing questions would muddy results.
    This check runs at the start of an oracle run and aborts before
    the first prediction if the input data is incomplete.
    """
    missing = [qid for qid in question_ids if qid not in gold_triples_map]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise LookupError(
            f"Oracle run cannot start: {len(missing)}/{len(question_ids)} "
            f"question_ids missing from the gold triples map: "
            f"[{preview}]{more}. Either extend the gold triples file or "
            f"restrict the question set."
        )


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _short_predicate(uri: str) -> str:
    """Return a short prefixed form of a predicate URI, or the URI itself."""
    for full, prefix in PREDICATE_PREFIXES.items():
        if uri.startswith(full):
            return f"{prefix}:{uri[len(full):]}"
    return uri


def _predicate_local_name(uri: str) -> str:
    """Extract the local name (after ``#`` or last ``/``) from a predicate URI.

    Used by :func:`format_verbalized` to produce readable prose even
    for predicates outside the standard prefix set. ``http://foo/bar#baz``
    → ``"baz"``; ``http://foo/bar/baz`` → ``"baz"``; bare names pass through.
    """
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    if "/" in uri:
        return uri.rsplit("/", 1)[-1]
    return uri


def format_structured(triples: list[Triple]) -> str:
    """Render triples as a JSON array with short predicate suffixes.

    The returned string is the full ``$triple_context`` slot contents,
    including the ``PREFIXES:`` header so the short names are
    unambiguous. Returns an empty-array block (still valid) for an
    empty triple list — aborting here would mask upstream bugs.
    """
    short = [
        {
            "subject": t["subject"],
            "predicate": _short_predicate(t["predicate"]),
            "object": t["object"],
        }
        for t in triples
    ]
    prefix_lines = "\n".join(
        f"  {prefix}: {uri}"
        for uri, prefix in PREDICATE_PREFIXES.items()
    )
    body = json.dumps(short, indent=2, ensure_ascii=False)
    return f"PREFIXES:\n{prefix_lines}\n\nTRIPLES:\n{body}"


_WKT_PREFIXES: tuple[str, ...] = (
    "POINT", "LINESTRING", "POLYGON", "MULTI", "GEOMETRYCOLLECTION",
)


def _strip_typed_literal(s: str) -> str:
    """Strip RDF typed-literal annotation and surrounding quotes from ``s``.

    Three input forms produced by the upstream KGs:

    * ``POLYGON ((...))`` — bare WKT (osmh3kg). Returned verbatim.
    * ``"POLYGON ((...))"`` — quote-wrapped WKT (kwg). Outer quotes
      stripped.
    * ``"Point(...)"^^<http://www.opengis.net/ont/geosparql#wktLiteral>``
      — RDF typed-literal (worldkg). Annotation and quotes stripped.

    The helper is conservative: it only strips when the input looks
    like one of the three patterns, leaving anything else untouched.
    """
    if not isinstance(s, str) or not s:
        return s
    # Typed-literal: '"<lex>"^^<...>' → '<lex>'
    if s.startswith('"') and "^^" in s:
        try:
            close = s.rindex('"')
            if close > 0:
                return s[1:close]
        except ValueError:
            pass
    # Plain quote-wrap: '"<lex>"' → '<lex>'
    if (
        len(s) >= 2
        and s.startswith('"')
        and s.endswith('"')
        and "^^" not in s
    ):
        return s[1:-1]
    return s


def _looks_like_wkt(obj_raw: str) -> bool:
    """Return True if ``obj_raw`` appears to be a WKT geometry literal.

    Strips typed-literal / quote wrapping first so the prefix check
    works across all three observed object formats.
    """
    if not obj_raw:
        return False
    inner = _strip_typed_literal(obj_raw).strip()
    return inner.upper().startswith(_WKT_PREFIXES)


def format_raw(triples: list[Triple]) -> str:
    """Render triples as N-Triples-like lines with full URIs.

    Subjects and non-literal objects are wrapped in angle brackets.
    Literal objects are JSON-quoted so whitespace and special
    characters survive intact. Each line ends with ``.`` as per the
    N-Triples convention (we don't validate strict N-Triples; this is
    a *readable* variant, not a parseable one).

    Object-format heterogeneity is handled by :func:`_looks_like_wkt`,
    which strips RDF typed-literal annotations and quote wrappers
    before the prefix check, so all three observed forms (bare,
    quote-wrapped, typed-literal) classify identically.
    """
    lines: list[str] = []
    for t in triples:
        subj = f"<{t['subject']}>"
        pred = f"<{t['predicate']}>"
        obj_raw = t["object"]
        # Heuristic: treat the object as an entity reference (angle
        # brackets) if it has no whitespace, no parens, doesn't look
        # like WKT, and appears as a subject elsewhere in the triple
        # set. The subject-presence check (``_is_entity_ref``) is the
        # final arbiter; a literal that happens to look like an id
        # (e.g. ``"cafe"``) is not an entity if no triple has it as
        # a subject.
        looks_like_id = (
            obj_raw
            and not _looks_like_wkt(obj_raw)
            and " " not in obj_raw
            and "(" not in obj_raw
            and "\n" not in obj_raw
        )
        if looks_like_id and _is_entity_ref(obj_raw, triples):
            obj = f"<{obj_raw}>"
        else:
            obj = json.dumps(obj_raw, ensure_ascii=False)
        lines.append(f"{subj} {pred} {obj} .")
    return "\n".join(lines)


def _is_entity_ref(candidate: str, triples: list[Triple]) -> bool:
    """Return True if ``candidate`` appears as a subject anywhere in ``triples``.

    Used by :func:`format_raw` to decide whether an object should be
    rendered as an entity reference (``<id>``) or a literal (``"..."``).
    Without this check, ``"cafe"`` (an rdf:type literal) would get
    wrapped in angle brackets and look like an entity.
    """
    return any(t["subject"] == candidate for t in triples)


def _is_aswkt(pred: str) -> bool:
    """True if ``pred`` is an ``asWKT`` predicate (any namespace)."""
    return pred.endswith("#asWKT") or pred.endswith("/asWKT")


def format_verbalized(triples: list[Triple]) -> str:
    """Render triples as natural-language prose, aggregated per entity.

    Triples are grouped by subject. For each subject, emit one sentence
    that combines its ``rdf:type`` (if present) and its geometry (via
    the two-hop ``geo:hasGeometry`` → ``geo:asWKT`` chain or the
    equivalent ``wkg:spatialObject`` → ``geo:asWKT`` chain).

    The chain is detected **dynamically** rather than by hardcoded
    predicate name: any triple whose object is the subject of an
    ``asWKT`` triple is treated as a geometry-chain link, regardless of
    the linking predicate's name. This means new schemas (OSMH3KG,
    custom KG dialects) work without ontology-specific code paths.

    Any remaining triples that don't fit the chain shape fall back to a
    predicate-level template (``Entity X has <pred-suffix> Y.``), so
    unusual schemas still render losslessly.

    Matching only ``hasGeometry`` as the chain predicate would silently
    drop the geometries of WorldKG entities, which link via
    ``spatialObject``, and bias tier-1 ``oracle_verbalized`` baselines
    downward on coordinate-dependent WorldKG questions.
    """
    # Group triples by subject while preserving first-seen order for
    # deterministic output.
    groups: dict[str, list[Triple]] = defaultdict(list)
    order: list[str] = []
    for t in triples:
        if t["subject"] not in groups:
            order.append(t["subject"])
        groups[t["subject"]].append(t)

    # Index geometry subjects (subjects of asWKT triples) → WKT literal.
    wkt_by_geo_id: dict[str, str] = {
        t["subject"]: t["object"] for t in triples if _is_aswkt(t["predicate"])
    }

    # Reverse-index the entity → geom chain by *object*, not by
    # predicate name. Any triple whose object is in ``wkt_by_geo_id`` is
    # a "this entity has this geometry" chain link. Last-write-wins on
    # the rare case where one entity has multiple geometry-chain links.
    chain_target: dict[str, str] = {
        t["subject"]: t["object"] for t in triples if t["object"] in wkt_by_geo_id
    }

    sentences: list[str] = []

    for subj in order:
        # Skip pure geometry subjects: those whose only role is to
        # carry an ``asWKT`` triple (their WKT folds into the owning
        # entity's sentence below).
        if subj in wkt_by_geo_id and all(
            _is_aswkt(t["predicate"]) for t in groups[subj]
        ):
            continue

        entity_triples = groups[subj]
        type_value: str | None = None
        chain_geom_id = chain_target.get(subj)
        wkt: str | None = (
            wkt_by_geo_id.get(chain_geom_id) if chain_geom_id else None
        )
        other_preds: list[Triple] = []

        for t in entity_triples:
            pred = t["predicate"]
            if pred.endswith("#type") or pred.endswith("/type"):
                type_value = t["object"]
            elif _is_aswkt(pred):
                # Direct ``asWKT`` on the entity itself (no chain).
                # Treat it as the entity's own geometry.
                if wkt is None:
                    wkt = t["object"]
            elif chain_geom_id is not None and t["object"] == chain_geom_id:
                # Chain predicate triple (any name) — consume it; the
                # WKT is rendered below as "with geometry X".
                continue
            else:
                other_preds.append(t)

        # Build the entity sentence from whichever parts are present.
        if type_value is not None and wkt is not None:
            sentences.append(
                f"Entity {subj} is a {type_value} with geometry {wkt}."
            )
        elif type_value is not None:
            sentences.append(f"Entity {subj} is a {type_value}.")
        elif wkt is not None:
            sentences.append(f"Entity {subj} has geometry {wkt}.")

        # Render any unusual predicates as one-off sentences.
        for t in other_preds:
            pred_short = _predicate_local_name(t["predicate"])
            sentences.append(
                f"Entity {subj} has {pred_short} {t['object']}."
            )

    return "\n".join(sentences)


def format_triples(triples: list[Triple], mode: OracleMode) -> str:
    """Dispatch to the right formatter. ``mode='t3'`` uses ``structured``."""
    if mode in ("structured", "t3"):
        return format_structured(triples)
    if mode == "raw":
        return format_raw(triples)
    if mode == "verbalized":
        return format_verbalized(triples)
    raise ValueError(f"Unknown oracle mode: {mode!r}")


__all__ = [
    "OracleMode",
    "OracleContext",
    "Triple",
    "T1_ORACLE_MODES",
    "assert_coverage",
    "format_raw",
    "format_structured",
    "format_triples",
    "format_verbalized",
    "load_gold_triples",
]
