"""Pre-execution contract check for TID shape files.

Run once per shape before the executor touches it. Fails loudly if:

1. The main SQL's outer-level ``SELECT`` projection list does not equal the
   meta's ``result_columns`` (same names, same order).
2. Either SQL file is missing a required placeholder (``{pool_a}``,
   ``{diversity_partition}``, etc.).
3. A predicate role declared in ``meta.gold_triples.predicates`` has no
   IRI list entry under ``prefix_map[kg]`` for some ``kg`` in
   ``shape.applicable_kgs`` (the resolved variant's KG map).

These checks replace the old exporter-side "coalesce from a 20-alias
whitelist" which hid defects until post-run inspection.
"""
from __future__ import annotations

import re

from question_generator.shape_loader import TidMeta, TidShape


class ShapeContractError(ValueError):
    """Raised when a TID shape violates its declared contract."""


# --------------------------------------------------------------------------- #
# Placeholder requirements
# --------------------------------------------------------------------------- #

# {join_condition} is required ONLY when the meta declares one (i.e. multi-
# pool / binary-relation TIDs). Single-pool TIDs may set
# meta.join_condition=None and omit the placeholder. The conditional check
# happens inside check_shape; this constant is the unconditionally-required
# set.
_MAIN_SQL_REQUIRED = (
    "{total_sample}",
)

_GOLD_SQL_REQUIRED = (
    "{triples_view}",
    "{predicate_iris}",
    "{pred_geometry_link}",
    "{pred_geometry_wkt}",
)

# Sampling-tied placeholders: optional in the SQL, but if a SQL references
# any of these the corresponding sampling field MUST be declared in meta.
# Validated per-shape inside check_shape (section 5).
_SAMPLING_TOKEN_TO_FIELD = {
    "diversity_partition": "diversity_keys",
    "per_bucket_cap": "per_bucket_cap",
    "pool_cap": "pool_cap",
}


# --------------------------------------------------------------------------- #
# SELECT-projection extractor
# --------------------------------------------------------------------------- #


_COMMENT_LINE = re.compile(r"--[^\n]*")
_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_comments(sql: str) -> str:
    sql = _COMMENT_BLOCK.sub(" ", sql)
    sql = _COMMENT_LINE.sub(" ", sql)
    return sql


def _find_top_level_select_slice(sql: str) -> tuple[int, int]:
    """Return (select_end, from_start) for the last top-level ``SELECT ... FROM``.

    "Top-level" means paren-depth 0 and not inside a string literal. The
    outer SELECT is the last SELECT / FROM pair at depth 0 in the file —
    everything before it is CTE / subquery scaffolding.
    """
    depth = 0
    in_single = False
    in_double = False
    # Collect every (keyword_name, start, end) at depth 0.
    tokens: list[tuple[str, int, int]] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if in_single:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            i += 1
            continue
        if depth == 0 and ch.isalpha():
            # Read keyword.
            j = i
            while j < n and (sql[j].isalnum() or sql[j] == "_"):
                j += 1
            word = sql[i:j].upper()
            if word in ("SELECT", "FROM"):
                tokens.append((word, i, j))
            i = j
            continue
        i += 1

    # Find the last top-level SELECT followed by a top-level FROM.
    selects = [(s, e) for (w, s, e) in tokens if w == "SELECT"]
    froms = [(s, e) for (w, s, e) in tokens if w == "FROM"]
    if not selects or not froms:
        raise ShapeContractError(
            "main SQL has no top-level SELECT...FROM pair"
        )
    select_start, select_end = selects[-1]
    # The FROM we want is the FIRST FROM after the last SELECT at depth 0.
    matching_from = next(
        ((fs, fe) for (fs, fe) in froms if fs > select_end), None
    )
    if matching_from is None:
        raise ShapeContractError(
            "main SQL has SELECT but no FROM at top level"
        )
    return select_end, matching_from[0]


def _split_top_level_commas(s: str) -> list[str]:
    """Split a comma-separated projection list, respecting parens and strings."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_single:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            buf.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


_AS_ALIAS = re.compile(r"\bAS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", re.IGNORECASE)
_BARE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _column_name(expr: str) -> str:
    """Return the output column name for a projection expression.

    Prefers the ``AS <name>`` alias; falls back to a bare identifier.
    Raises ``ShapeContractError`` if no name can be inferred (i.e. the
    projection is a complex expression with no alias).
    """
    expr = expr.strip().rstrip(";").strip()
    m = _AS_ALIAS.search(expr)
    if m:
        return m.group(1)
    if _BARE_NAME.match(expr):
        return expr
    raise ShapeContractError(
        f"cannot infer output column name for projection {expr!r} "
        "— every non-bare projection must use 'AS <name>'"
    )


def extract_result_columns(main_sql: str) -> list[str]:
    """Return the column names the outermost SELECT of ``main_sql`` produces."""
    stripped = _strip_comments(main_sql)
    select_end, from_start = _find_top_level_select_slice(stripped)
    projection_slice = stripped[select_end:from_start]
    parts = _split_top_level_commas(projection_slice)
    if not parts:
        raise ShapeContractError("main SQL outer SELECT has no projections")
    return [_column_name(p) for p in parts]


# --------------------------------------------------------------------------- #
# Built-in token set (per-shape)
# --------------------------------------------------------------------------- #

# Match any {identifier} placeholder so we can detect undeclared tokens.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Strategy- and pool-agnostic: tokens the executor always handles regardless
# of which TID it's rendering. Pool-specific tokens (`<pool>` for main SQL,
# `<pool>_all` for gold SQL) are added per-shape from meta.pools.
_NON_POOL_BUILTINS = frozenset({
    "diversity_partition", "run_seed", "per_bucket_cap", "pool_cap",
    "total_sample", "join_condition",
    "triples_view", "predicate_iris",
    "pred_geometry_link", "pred_geometry_wkt",
})


def _builtin_tokens(meta: TidMeta) -> frozenset[str]:
    """Built-in placeholder set for one TID, derived from meta.pools.

    Each declared pool contributes two tokens: ``<pool>`` (main SQL,
    filtered view) and ``<pool>_all`` (gold SQL, unfiltered view).
    """
    tokens: set[str] = set(_NON_POOL_BUILTINS)
    for pool_name in meta.pools:
        tokens.add(pool_name)
        tokens.add(f"{pool_name}_all")
    return frozenset(tokens)


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #


def check_shape(
    shape: TidShape,
    prefix_map: dict[str, dict[str, list[str]]],
) -> None:
    """Raise ``ShapeContractError`` if ``shape`` violates its declared contract."""
    # 1. Result columns ≡ outer SELECT projections.
    actual_cols = extract_result_columns(shape.sql)
    declared_cols = shape.meta.result_columns
    if actual_cols != declared_cols:
        raise ShapeContractError(
            f"TID {shape.meta.tid}: result_columns mismatch\n"
            f"  meta declared: {declared_cols}\n"
            f"  SQL produces:  {actual_cols}"
        )

    # 2. Required placeholders present. Both checks run on comment-stripped
    # SQL so mentions of placeholders in header comments (e.g.
    # "-- LIMIT {total_sample}" as a dead-code annotation, or
    # "no {join_condition}" in a single-pool TID's header) don't false-pass
    # OR false-fail the contract.
    stripped_sql = _strip_comments(shape.sql)
    stripped_gold_sql = _strip_comments(shape.gold_sql)
    for placeholder in _MAIN_SQL_REQUIRED:
        if placeholder not in stripped_sql:
            raise ShapeContractError(
                f"TID {shape.meta.tid}: main SQL missing required "
                f"placeholder {placeholder!r}"
            )
    # {join_condition} is required iff the meta declares one. Single-pool
    # TIDs (meta.join_condition=None) must NOT reference it; multi-pool
    # TIDs MUST reference it.
    if shape.meta.join_condition is not None:
        if "{join_condition}" not in stripped_sql:
            raise ShapeContractError(
                f"TID {shape.meta.tid}: meta.join_condition is declared but "
                f"main SQL does not reference {{join_condition}}. "
                f"Add the placeholder or set meta.join_condition: null."
            )
    else:
        if "{join_condition}" in stripped_sql:
            raise ShapeContractError(
                f"TID {shape.meta.tid}: main SQL references {{join_condition}} "
                f"but meta.join_condition is null. Either declare it in meta "
                f"or remove the placeholder from the SQL."
            )
    for placeholder in _GOLD_SQL_REQUIRED:
        if placeholder not in stripped_gold_sql:
            raise ShapeContractError(
                f"TID {shape.meta.tid}: gold SQL missing required "
                f"placeholder {placeholder!r}"
            )

    # 3. Every predicate role resolvable for every applicable KG.
    for kg in shape.applicable_kgs:
        if kg not in prefix_map:
            raise ShapeContractError(
                f"TID {shape.meta.tid}: KG {kg!r} declared applicable but "
                f"absent from prefix map (available: {sorted(prefix_map)})"
            )
        kg_roles = prefix_map[kg]
        for role in shape.meta.gold_triples.predicates:
            if role not in kg_roles:
                raise ShapeContractError(
                    f"TID {shape.meta.tid}: predicate role {role!r} "
                    f"has no IRI list under prefix_map[{kg!r}] "
                    f"(have: {sorted(kg_roles)})"
                )

    # 4. Parameters: bidirectional validation.
    #    - Every key in meta.parameters OR any variant.parameters MUST
    #      appear as a {key} in either SQL.
    #    - Every {placeholder} in either SQL must be either built-in or
    #      declared in meta.parameters or some variant.parameters.
    #    - Parameter keys may not collide with built-in token names
    #      (otherwise the executor's built-in substitution would silently
    #      mask the meta value).
    #
    # Variant-scope parameters (VariantConfig.parameters added at commit
    # 099f687) are valid placeholder declarations because the executor's
    # _resolve_parameters overlays them onto the shape-wide dict at
    # render time. A SQL using `{event_a_label}` only declared at
    # variant scope is valid as long as EVERY variant declares the key
    # (or shape-wide does). We require the strict union here — if a
    # placeholder is referenced but only some variants declare it, the
    # missing variant would render with the literal `{key}` left in the
    # SQL — caught by the "declared but never referenced" check below
    # only if no variant declares it.
    builtin_tokens = _builtin_tokens(shape.meta)
    variant_keys = set()
    for variant in shape.meta.variants.values():
        variant_keys |= set(variant.parameters)
    declared_keys = set(shape.meta.parameters) | variant_keys

    collisions = declared_keys & builtin_tokens
    if collisions:
        raise ShapeContractError(
            f"TID {shape.meta.tid}: meta.parameters keys collide with "
            f"built-in tokens: {sorted(collisions)}. Rename them."
        )

    # Strip comments before scanning so Jinja-style template tokens like
    # {{entity_A}} in SQL header comments don't trip the regex. The inner
    # {entity_A} of {{entity_A}} would otherwise be flagged as an undeclared
    # placeholder, even though it's just commentary about the question
    # template (substituted by the exporter, not by render_main_sql).
    #
    # Also scan meta.join_condition: it's a free-form string the executor
    # substitutes into the SQL via {join_condition}, so any {key} it
    # references is legitimately a parameter usage. Without this, a TID
    # whose only use of a parameter is inside join_condition (e.g. a
    # radius-filter parameter that lives in a spatial-join predicate)
    # would falsely fail "declared but never referenced".
    #
    # _strip_comments is intentionally NOT applied to join_condition: it's
    # a YAML scalar string with no SQL `--` or `/* */` syntax. Adding the
    # call would be a no-op but suggest a symmetry that doesn't exist
    # (the SQL fields are multi-line bodies; join_condition is a one-line
    # predicate fragment).
    join_condition_text = shape.meta.join_condition or ""
    referenced = (
        set(_PLACEHOLDER_RE.findall(_strip_comments(shape.sql)))
        | set(_PLACEHOLDER_RE.findall(_strip_comments(shape.gold_sql)))
        | set(_PLACEHOLDER_RE.findall(join_condition_text))
    )
    undeclared = referenced - builtin_tokens - declared_keys
    if undeclared:
        raise ShapeContractError(
            f"TID {shape.meta.tid}: SQL references undeclared placeholders: "
            f"{sorted(undeclared)}. Add them to meta.parameters or remove "
            f"them from the SQL. (Built-ins for this shape: "
            f"{sorted(builtin_tokens)})"
        )
    unused = declared_keys - referenced
    if unused:
        raise ShapeContractError(
            f"TID {shape.meta.tid}: meta.parameters declared but never "
            f"referenced in SQL: {sorted(unused)}. Remove them from the "
            f"meta or use them."
        )

    # 5. Sampling-token consistency: if SQL references {diversity_partition},
    #    {per_bucket_cap}, or {pool_cap}, the corresponding sampling field
    #    must be set. This catches the "SQL uses bucket-rank sampling but
    #    meta forgot to declare the cap" mismatch class. Pool-token presence
    #    is a separate check below — pool tokens declared in meta.pools
    #    don't have to be referenced (TID may use only pool_a in main SQL),
    #    but a referenced pool MUST be declared.
    for token, field_name in _SAMPLING_TOKEN_TO_FIELD.items():
        if f"{{{token}}}" in shape.sql and getattr(
            shape.meta.sampling, field_name
        ) is None:
            raise ShapeContractError(
                f"TID {shape.meta.tid}: main SQL references {{{token}}} but "
                f"sampling.{field_name} is not declared in meta.yaml. "
                f"Add it under sampling: or remove {{{token}}} from the SQL."
            )

    # 6. Pool-token presence: every pool referenced as {<name>} in main SQL
    #    or {<name>_all} in gold SQL must be declared in meta.pools.
    #    Caught by the bidirectional undeclared-placeholders check (step 4)
    #    via _builtin_tokens, so no additional logic needed here — this
    #    comment exists so future readers know the property is enforced.
