"""Master prompt templates and renderer for benchmark_models.

Three master prompt templates live in this package as ``.txt`` files:

* ``tier1.txt`` — plain LLM (parametric-knowledge floor).
* ``tier2.txt`` — reasoning-enhanced LLM (prompted CoT for variants
  without native thinking; native-thinking variants ignore the
  ``reasoning`` slot in the response and put their reasoning in the
  provider's reasoning channel).
* ``tier3.txt`` — tool-augmented agentic LLM with KG access. Includes
  a ``$tools_section`` slot that the agent loop fills in with the
  tool catalogue for the current ablation cell, and a
  ``$max_tool_calls`` slot for the cap.

All three templates are English-only by design (per the multilingual
strategy decision: English wrapper + native-language question text).
The question text is substituted into the ``$question`` slot in
whatever language the example carries.

Substitution uses :class:`string.Template` rather than ``str.format``
or Jinja so the ``{}`` characters in the embedded JSON examples
(``{"answer": null}``) don't need escaping. Slot names are dollar-
prefixed: ``$question``, ``$answer_type_description``, ``$unit_clause``,
``$tools_section``, ``$max_tool_calls``.

The renderer (:func:`render_prompt`) is the only entry point. It:

1. Loads the master template for the requested tier (cached).
2. Looks up the per-question :class:`AnswerTypeSpec` from the type
   registry, using the template's ``answer_type``.
3. Computes the natural-language description and unit clause from
   the type spec and the template's per-template unit override.
4. Substitutes all slots and returns the rendered prompt.

Dynamic-type templates (whose ``answer_type`` is ``None``) raise
:class:`PromptRenderingError` — the runner is supposed to emit a
strict-error :class:`Prediction` for these and never reach the prompt
renderer. The error message points the caller at the runner's
strict-error path.

The ``.txt`` files are kept as separate template files (rather than
inlined Python constants) because:

* **Inspection**: collaborators and reviewers can read the exact prompt
  the model sees by opening a single ``.txt`` file, without reading
  Python code.
* **Reproducibility**: the ``.txt`` file is a versioned artifact that
  corresponds 1:1 to what was sent to the model.
* **Ablation-friendly**: prompt variants (e.g., ``tier1_force_answer.txt``
  for the force-answer ablation) are added as files, not code edits.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from string import Template

from benchmark_models.base import Example, Tier
from benchmark_models.schemas import (
    TYPE_REGISTRY,
    AnswerTypeSpec,
    TemplateInfo,
)

# Default tier-3 tool-call cap. Matches Smolagents' library default
# (CodeAgent.max_steps=6) and the typical 5–8 step range used in the
# ReAct / ToolBench / agent-eval literature for QA-style tasks.
# Spatial-QA templates have Retrieval Hop Complexity ≤ 3, so 6 was
# Smolagents' upstream default and the original development-run value;
# we raised the cap to 10 to match the project plan's documented
# headline-experiments budget after observing qwen35-27b-tier3b hit
# 80 % cap_hit at 6 (it tries reasonable web-search + geocoding
# sequences but doesn't converge in 6 steps). 10 brings the open-weights
# 27B class closer to apples-to-apples comparison with the closed-source
# tier-3b column. The prompt's ``$max_tool_calls`` slot is substituted
# from the same value the agent loop sees, so prompt + agent never
# disagree.
DEFAULT_MAX_TOOL_CALLS: int = 10


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class PromptRenderingError(RuntimeError):
    """Raised when the prompt cannot be rendered for an example.

    Typically indicates an unknown ``answer_type`` that the registry
    loader should have caught at startup, or a mis-use of the
    ``tools_section`` / ``max_tool_calls`` slots for the given tier.
    The prompt renderer is not the right place to apply fallback
    logic — callers must intercept these upstream.
    """


# ----------------------------------------------------------------------------
# Template loading
# ----------------------------------------------------------------------------


_TIER_TO_TEMPLATE_FILE: dict[str, str] = {
    "tier1": "tier1.txt",
    "tier2": "tier2.txt",
    "tier3a": "tier3a.txt",
    "tier3b": "tier3b.txt",
    "tier3c": "tier3c.txt",
    "tier1_oracle_structured": "tier1_oracle_structured.txt",
    "tier1_oracle_raw": "tier1_oracle_raw.txt",
    "tier1_oracle_verbalized": "tier1_oracle_verbalized.txt",
    "tier3_oracle": "tier3_oracle.txt",
}

# Tier keys that include a ``$triple_context`` slot. Used by
# :func:`render_prompt` to validate that the caller supplied the slot
# argument when using these templates and not when using others.
_ORACLE_TIER_KEYS: frozenset[str] = frozenset({
    "tier1_oracle_structured",
    "tier1_oracle_raw",
    "tier1_oracle_verbalized",
    "tier3_oracle",
})


@cache
def _load_template(tier_key: str) -> Template:
    """Load and parse a master template file as a :class:`string.Template`.

    Accepts both base tier keys (``"tier1"``, ``"tier2"``, ``"tier3"``)
    and sub-tier keys (``"tier3a"``, ``"tier3b"``, ``"tier3c"``).
    Cached forever; the master templates are static at process startup.
    """
    filename = _TIER_TO_TEMPLATE_FILE[tier_key]
    resource = files("benchmark_models.prompts").joinpath(filename)
    with resource.open("r", encoding="utf-8") as f:
        return Template(f.read())


def reload_templates() -> None:
    """Clear the template cache. Test-only escape hatch."""
    _load_template.cache_clear()


# ----------------------------------------------------------------------------
# Slot computation
# ----------------------------------------------------------------------------


def _resolve_unit(
    template_info: TemplateInfo, type_spec: AnswerTypeSpec
) -> str | None:
    """Pick the unit string for an example.

    The per-template ``answer_unit`` from Excel takes precedence (e.g.,
    ``"kilometres"`` for distance templates, ``"kilometres squared"``
    for area). If the template doesn't override, fall back to the type
    spec's ``default_unit``. If the type doesn't want a unit field at
    all (booleans, entities, predefined values, ...), return ``None``.
    """
    if not type_spec.wants_unit_field:
        return None
    if template_info.answer_unit:
        return template_info.answer_unit
    return type_spec.default_unit


def _build_unit_clause(unit: str | None) -> str:
    """Build the ``$unit_clause`` slot string.

    Returns ``", in <unit>"`` for numeric types with a unit, or empty
    string for non-numeric types. The leading comma is part of the
    clause so the tier-1/2/3 templates read naturally either way:

    * Non-numeric: ``"Expected answer: a single entity name"``
    * Numeric:     ``"Expected answer: a numeric value, in kilometres"``
    """
    if not unit:
        return ""
    return f", in {unit}"


# Distance units recognised by :func:`_build_distance_clause`.
# Mirrors the ``distance`` group in
# ``benchmark_eval/scoring_spec.json`` (``unit_conversion.groups.distance``).
# Kept inline here so the prompt module has no dependency on
# ``benchmark_eval``; the two should be audited together if the spec
# gains new distance units. Comparison is case-insensitive and tolerates
# trailing ``s`` (kilometres → km).
_DISTANCE_UNITS: frozenset[str] = frozenset({
    "km", "kilometre", "kilometer",
    "m", "metre", "meter",
    "mi", "mile",
    "ft", "foot", "feet",
    "nmi", "nautical mile",
})


def _build_distance_clause(unit: str | None) -> str:
    """Build the ``$distance_clause`` slot string.

    For distance-related questions (km, m, mi, ft, nmi — see
    :data:`_DISTANCE_UNITS`), returns the geodesic-distance
    specification. Empty for all other question types (area, boolean,
    entity, count, etc.) — area units like ``km^2`` are excluded by
    construction because ``"km^2"`` doesn't match any entry in the
    distance set.

    The clause appears on its own line in the prompt, so it includes
    a leading newline when non-empty.
    """
    if not unit:
        return ""
    u = unit.lower().strip()
    # Strip trailing 's' once (kilometres → kilometre, miles → mile).
    if u.endswith("s") and u[:-1] in _DISTANCE_UNITS:
        u = u[:-1]
    # Area / unit-with-exponent disqualifier — e.g. ``km^2``, ``m²``,
    # ``kilometres squared``. Cheap pre-filter before the set lookup.
    if "squared" in u or "²" in u or "^2" in u:
        return ""
    if u not in _DISTANCE_UNITS:
        return ""
    return (
        "\nAll distances should be computed as geodesic distances "
        "along the Earth's surface, measured between the closest "
        "points of the two geometries."
    )


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def render_prompt(
    tier: Tier,
    example: Example,
    template_info: TemplateInfo,
    *,
    tools_section: str | None = None,
    max_tool_calls: int | None = None,
    prompt_variant: str | None = None,
    triple_context: str | None = None,
) -> str:
    """Render the master prompt for a given (tier, example) pair.

    Args:
        tier: Which master template to load (``"tier1"``, ``"tier2"``,
            or ``"tier3"``). Can be overridden by ``prompt_variant``.
        example: The benchmark question to render. ``example.question``
            is substituted verbatim into the ``$question`` slot.
        template_info: The template registry entry for ``example.tid``.
            Used to look up the answer type and unit override.
        tools_section: For tier 3 only, the rendered tool catalogue
            section that goes into the ``$tools_section`` slot. Must
            be ``None`` for tier 1 / tier 2 (the renderer raises if
            given). Must be non-``None`` for tier 3 (same).
        max_tool_calls: For tier 3 only, the cap that the agent loop
            will enforce. Substituted into the ``$max_tool_calls``
            slot so the prompt always describes the cap actually in
            force. Defaults to :data:`DEFAULT_MAX_TOOL_CALLS` (6)
            when ``None`` for tier 3. Must be ``None`` for tier 1 /
            tier 2 (the renderer raises if given).

    Returns:
        The fully-substituted prompt string, ready to send to a backend.

    Raises:
        PromptRenderingError: if the answer type is unknown, or if
            ``tools_section`` / ``max_tool_calls`` is misused for the
            tier.
    """
    if template_info.answer_type not in TYPE_REGISTRY:
        raise PromptRenderingError(
            f"Cannot render prompt for tid={example.tid}: unknown "
            f"answer_type {template_info.answer_type!r}. The "
            f"TemplateRegistry validator should have caught this at "
            f"load time."
        )

    tier_key_for_slots = prompt_variant or tier
    is_oracle = tier_key_for_slots in _ORACLE_TIER_KEYS
    needs_tools_section = tier == "tier3" and tier_key_for_slots != "tier3_oracle"

    if needs_tools_section:
        # ``tools_section`` is now required: tier3a / tier3b / tier3c
        # templates reference the ``$tools_section`` slot, which is
        # filled by :func:`benchmark_models.tier3.render_tools_section`
        # from the actual loaded tool list. Passing ``None`` would
        # leave the prompt with an empty tools description and the
        # model would have no way to know what's available.
        if tools_section is None:
            raise PromptRenderingError(
                f"render_prompt(tier='tier3', prompt_variant="
                f"{tier_key_for_slots!r}) requires a tools_section. "
                f"Use benchmark_models.tier3.render_tools_section() "
                f"to build it from the live tool list."
            )
        effective_cap = (
            max_tool_calls if max_tool_calls is not None else DEFAULT_MAX_TOOL_CALLS
        )
        if effective_cap < 1:
            raise PromptRenderingError(
                f"render_prompt(tier='tier3', ..., max_tool_calls="
                f"{effective_cap}) is invalid; the cap must be >= 1."
            )
    elif tier_key_for_slots == "tier3_oracle":
        # Tier-3 oracle has no retrieval tools but still uses the
        # Python-sandbox step cap. No tools_section slot in the
        # template — the cap is substituted the same way.
        if tools_section is not None:
            raise PromptRenderingError(
                f"render_prompt(prompt_variant='tier3_oracle', ...) does "
                f"not accept a tools_section argument; got {tools_section!r}."
            )
        effective_cap = (
            max_tool_calls if max_tool_calls is not None else DEFAULT_MAX_TOOL_CALLS
        )
        if effective_cap < 1:
            raise PromptRenderingError(
                f"render_prompt(prompt_variant='tier3_oracle', ..., "
                f"max_tool_calls={effective_cap}) is invalid; the cap "
                f"must be >= 1."
            )
    else:
        if tools_section is not None:
            raise PromptRenderingError(
                f"render_prompt(tier={tier!r}, ...) does not accept a "
                f"tools_section argument; got {tools_section!r}."
            )
        if max_tool_calls is not None:
            raise PromptRenderingError(
                f"render_prompt(tier={tier!r}, ...) does not accept a "
                f"max_tool_calls argument; got {max_tool_calls!r}."
            )
        effective_cap = 0  # unused

    if is_oracle:
        if triple_context is None:
            raise PromptRenderingError(
                f"render_prompt(prompt_variant={tier_key_for_slots!r}, "
                f"...) requires a triple_context argument; got None."
            )
    else:
        if triple_context is not None:
            raise PromptRenderingError(
                f"render_prompt(tier={tier!r}, prompt_variant="
                f"{prompt_variant!r}) does not accept a triple_context "
                f"argument; oracle mode is not active for this template."
            )

    type_spec = TYPE_REGISTRY[template_info.answer_type]
    unit = _resolve_unit(template_info, type_spec)
    unit_clause = _build_unit_clause(unit)
    distance_clause = _build_distance_clause(unit)

    # Use prompt_variant if given (e.g., "tier3b" for sub-tier-specific
    # prompt), otherwise fall back to the base tier key.
    tier_key = prompt_variant or tier
    template = _load_template(tier_key)
    substitutions: dict[str, str] = {
        "question": example.question,
        "answer_type_description": type_spec.nl_description,
        "unit_clause": unit_clause,
        "distance_clause": distance_clause,
    }
    if needs_tools_section:
        substitutions["tools_section"] = tools_section or ""
        substitutions["max_tool_calls"] = str(effective_cap)
    if tier_key_for_slots == "tier3_oracle":
        substitutions["max_tool_calls"] = str(effective_cap)
    if is_oracle:
        substitutions["triple_context"] = triple_context or ""
    return template.substitute(substitutions)


# ----------------------------------------------------------------------------
# PromptRenderer adapter
# ----------------------------------------------------------------------------


class PromptRenderer:
    """Backend-facing adapter around :func:`render_prompt`.

    Each backend constructs its prompts through a ``PromptRenderer``
    rather than calling :func:`render_prompt` directly. The adapter
    encapsulates the cross-cutting concerns that would otherwise need
    per-backend branches — today that's oracle mode (tier-1 only);
    future ablations like prompt-variant sensitivity and force-answer
    will plug in here too.

    For a non-oracle tier-1 or tier-2 run, the renderer delegates to
    :func:`render_prompt` with the base tier key. For an oracle tier-1
    run, the renderer looks up the example's gold triples in the
    :class:`OracleContext` map, formats them with the selected mode,
    and renders the oracle-variant template.

    Tier-3 runs (non-oracle) do NOT go through this adapter — the
    ``SmolagentsModel`` / ``SmolagentsOracleModel`` classes own their
    own prompt construction because they also need to render the
    tools section at runtime. This adapter only ever returns tier-1 /
    tier-2 prompts and their matching ``prompt_template_id``.

    The adapter is leakage-safe: the :class:`OracleContext` (and
    therefore the gold triples map) is only passed to
    :class:`PromptRenderer` instances constructed for oracle runs.
    Non-oracle instances hold no reference to any gold data and
    cannot access it.
    """

    def __init__(
        self,
        tier: Tier,
        *,
        oracle_context: object | None = None,
    ) -> None:
        """Construct a renderer for the given tier.

        Args:
            tier: Which base master template to render for non-oracle
                mode (``"tier1"`` or ``"tier2"``).
            oracle_context: An :class:`OracleContext` (or ``None``). Must
                be ``None`` for tier 2 — oracle is only supported for
                tier 1 in this adapter (tier-3 oracle uses a separate
                model class). Typed as ``object`` to avoid a circular
                import from :mod:`benchmark_models.oracle`.
        """
        from benchmark_models.oracle import T1_ORACLE_MODES, OracleContext

        if tier not in ("tier1", "tier2"):
            raise ValueError(
                f"PromptRenderer only supports tier1 / tier2 (got {tier!r}). "
                f"Tier 3 uses SmolagentsModel / SmolagentsOracleModel "
                f"directly."
            )
        if oracle_context is not None:
            if not isinstance(oracle_context, OracleContext):
                raise TypeError(
                    f"oracle_context must be an OracleContext, got "
                    f"{type(oracle_context).__name__}"
                )
            if oracle_context.mode not in T1_ORACLE_MODES:
                raise ValueError(
                    f"PromptRenderer only supports T1 oracle modes "
                    f"{T1_ORACLE_MODES!r} (got {oracle_context.mode!r}). "
                    f"For T3-oracle use SmolagentsOracleModel."
                )
            if tier != "tier1":
                raise ValueError(
                    f"Oracle mode is only supported for tier1 in the "
                    f"PromptRenderer (got tier={tier!r})."
                )
        self._tier: Tier = tier
        self._oracle_context = oracle_context

    @property
    def oracle_mode(self) -> str | None:
        """The active oracle mode, or ``None`` if this is a plain renderer."""
        if self._oracle_context is None:
            return None
        return self._oracle_context.mode  # type: ignore[attr-defined]

    def render(
        self, example: Example, template_info: TemplateInfo
    ) -> tuple[str, str, str | None]:
        """Render the prompt for ``example``.

        Returns:
            A tuple of ``(prompt_text, prompt_template_id,
            triple_context)``. ``prompt_template_id`` records which
            template variant was used (e.g. ``"tier1.canonical"``,
            ``"tier1.oracle.raw"``). ``triple_context`` is the
            formatted gold-triples string injected into the prompt
            for oracle runs, or ``None`` for non-oracle runs. Both
            are stored on the resulting :class:`Prediction` so the
            oracle condition is auditable after the fact without
            re-running the renderer.
        """
        if self._oracle_context is None:
            prompt = render_prompt(self._tier, example, template_info)
            return prompt, f"{self._tier}.canonical", None

        # Oracle path — look up triples, format, render oracle template.
        from benchmark_models.oracle import format_triples

        ctx = self._oracle_context  # type: ignore[assignment]
        mode = ctx.mode  # type: ignore[attr-defined]
        gold_map = ctx.gold_triples_map  # type: ignore[attr-defined]

        if example.question_id not in gold_map:
            # Coverage is asserted at run start, so this should never
            # happen — but fail loud rather than silently rendering a
            # non-oracle prompt if it does.
            raise LookupError(
                f"Oracle render: no gold triples for question_id="
                f"{example.question_id!r}. Coverage assertion missed "
                f"this — check assert_coverage at run start."
            )
        triples = gold_map[example.question_id]
        triple_context = format_triples(triples, mode=mode)
        prompt_variant = f"tier1_oracle_{mode}"
        prompt = render_prompt(
            self._tier,
            example,
            template_info,
            prompt_variant=prompt_variant,
            triple_context=triple_context,
        )
        return prompt, f"tier1.oracle.{mode}", triple_context


__all__ = [
    "DEFAULT_MAX_TOOL_CALLS",
    "PromptRenderer",
    "PromptRenderingError",
    "reload_templates",
    "render_prompt",
]
