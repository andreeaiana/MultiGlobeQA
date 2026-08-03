"""Variant registry — model factory for benchmark_models.

Loads ``models.yaml`` via :func:`benchmark_models.config.load_models_catalogue`,
and constructs :class:`Model` instances from variant configs via
:func:`create_model`. The runner calls :func:`create_model` once per
variant at startup; the returned Model is used for the entire run.

The factory dispatches on the variant config's type (the discriminated
union from :mod:`benchmark_models.config`) and constructs the right
backend-specific Model subclass.

Public API:

* :func:`create_model` — construct a Model from a variant name + config.
* :func:`load_and_create` — shortcut: load the catalogue, look up the
  variant by name, create the model. Convenience for CLI one-offs.
* :func:`list_variants` — list all variant names in the bundled catalogue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_models.base import Model
from benchmark_models.config import (
    GoogleVariantConfig,
    MultimodalMode,
    OpenAIVariantConfig,
    VariantConfig,
    VLLMHttpVariantConfig,
    load_models_catalogue,
)
from benchmark_models.schemas import TemplateRegistry, load_template_registry


def create_model(
    variant_name: str,
    config: VariantConfig,
    *,
    registry: TemplateRegistry | None = None,
    api_key: str | None = None,
    snapshot_id: str = "unset",
    kg_backend: object | None = None,
    oracle_context: object | None = None,
    max_steps_override: int | None = None,
    max_tokens_override: int | None = None,
    max_concurrent_override: int | None = None,
    web_search_timeout_override: float | None = None,
    multimodal_mode_override: MultimodalMode | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Model:
    """Construct a :class:`Model` from a variant name and config.

    Dispatches on the config type to the right backend-specific Model
    subclass. The returned Model is ready to call ``predict()`` /
    ``predict_batch()`` immediately.

    Args:
        variant_name: The key from ``models.yaml`` (e.g.,
            ``"gemini-3-flash-plain"``).
        config: The validated :data:`VariantConfig` for this variant.
        registry: Optional :class:`TemplateRegistry`. If ``None``,
            loads the bundled registry.
        api_key: Optional API key override. Passed through to the
            backend constructor. If ``None``, the backend reads from
            the appropriate environment variable (``GOOGLE_API_KEY``,
            ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
        snapshot_id: KG snapshot identifier recorded on every
            :class:`Prediction`. Defaults to ``"unset"`` for smoke
            tests; the runner computes the real value at startup.
        max_steps_override: Optional override for the tier-3 agent's
            ``max_steps`` (a.k.a. ``max_tool_calls``). When ``None``,
            uses the variant's ``max_tool_calls`` from ``models.yaml``,
            which defaults to :data:`DEFAULT_MAX_TOOL_CALLS`. Ignored
            for tier 1 / tier 2.
        max_tokens_override: Optional override for the tier-3 LLM
            per-step ``max_tokens``. When ``None``, uses the variant's
            ``max_tokens`` from ``models.yaml`` (if set), otherwise
            the per-sub-tier default. Ignored for tier 1 / tier 2.
        max_concurrent_override: Optional override for the tier-3
            ``predict_batch`` worker count. When ``None``, uses the
            variant's ``max_concurrent`` from ``models.yaml`` (if
            set), otherwise the default. Ignored for tier 1 / tier 2.
        web_search_timeout_override: Optional override for the
            tier-3b / 3c web-search per-request timeout (seconds).
            ``None`` falls back to the variant config or the default.
            Ignored for tier 1 / tier 2 / tier 3a.
        config_overrides: Optional dict of base-config field overrides
            (e.g. ``{"temperature": 0.7, "self_consistency_n": 3}``).
            Applied via ``config.model_copy(update=...)`` BEFORE the
            rest of the factory, so the new values are pydantic-validated
            (e.g. SC>1 with temperature=0 is rejected here) and visible
            to every downstream consumer. Use for CLI sampling-knob
            overrides (``--temperature``, ``--top-p``, ``--seed``,
            ``--max-tokens`` on tier 1/2, ``--self-consistency-n``).
        oracle_context: Optional :class:`OracleContext`. When set:
            - T1 variants get a :class:`PromptRenderer` constructed
              with the context; non-oracle runs build a plain renderer.
            - T3 variants (``tier == "tool_agent"``) with oracle mode
              ``"t3"`` construct a :class:`SmolagentsOracleModel`
              with the gold triples map instead of the normal
              :class:`SmolagentsModel`.
            - T2 does not support oracle mode.

    Returns:
        A :class:`Model` instance.

    Raises:
        NotImplementedError: if the backend is not yet implemented.
        ValueError: if oracle_context is set for an incompatible tier.
    """
    from benchmark_models.oracle import T1_ORACLE_MODES, OracleContext
    from benchmark_models.prompts import PromptRenderer

    reg = registry if registry is not None else load_template_registry()

    # Apply CLI sampling-knob overrides on top of the YAML-defined
    # variant config. Round-trip through ``model_dump`` +
    # ``model_validate`` re-runs the pydantic validators so combinations
    # like ``self_consistency_n=3`` with ``temperature=0`` are rejected
    # here rather than silently producing N identical SC samples
    # downstream. (``model_copy(update=...)`` skips validators by
    # default, so we can't use it here.)
    if config_overrides:
        merged = config.model_dump()
        merged.update(config_overrides)
        config = type(config).model_validate(merged)

    # Validate the oracle context against the tier it's being applied to.
    if oracle_context is not None:
        if not isinstance(oracle_context, OracleContext):
            raise TypeError(
                f"oracle_context must be an OracleContext, got "
                f"{type(oracle_context).__name__}"
            )
        mode = oracle_context.mode
        if config.tier == "reasoning":
            raise ValueError(
                f"Oracle mode {mode!r} is not supported for tier 2 "
                f"(variant {variant_name!r}). Use tier 1 or tier 3."
            )
        if config.tier == "plain" and mode not in T1_ORACLE_MODES:
            raise ValueError(
                f"Tier-1 variant {variant_name!r} requires a T1 oracle "
                f"mode {T1_ORACLE_MODES!r}, got {mode!r}."
            )
        if config.tier == "tool_agent" and mode != "t3":
            raise ValueError(
                f"Tier-3 variant {variant_name!r} requires oracle mode "
                f"'t3', got {mode!r}."
            )

    # Tier 3 (tool_agent): oracle mode -> SmolagentsOracleModel;
    # otherwise the normal SmolagentsModel.
    if config.tier == "tool_agent":
        # Determine the provider model ID from the config.
        if hasattr(config, "model_id"):
            provider_model_id = config.model_id
        elif hasattr(config, "model_path"):
            provider_model_id = config.model_path
        else:
            raise ValueError(
                f"Cannot determine model ID for tier-3 variant "
                f"{variant_name!r}: config has neither model_id nor "
                f"model_path."
            )

        # Resolve tier-3 knobs in priority order:
        #   CLI override → models.yaml variant config → built-in default
        # ``getattr`` lets variants that predate the new fields fall
        # through cleanly without YAML migration.
        max_steps = (
            max_steps_override
            if max_steps_override is not None
            else config.max_tool_calls
        )
        max_tokens = (
            max_tokens_override
            if max_tokens_override is not None
            else getattr(config, "tier3_max_tokens", None)
        )
        max_concurrent = (
            max_concurrent_override
            if max_concurrent_override is not None
            else getattr(config, "max_concurrent", None)
        )
        web_search_timeout = (
            web_search_timeout_override
            if web_search_timeout_override is not None
            else getattr(config, "web_search_timeout_seconds", None)
        )

        # G: resolve multimodal mode for tier-3 (same priority as
        # tiers 1+2 below). Tier-3 only honours "image" / "name" /
        # "off"; non-multimodal variants leave it "off".
        tier3_multimodal_mode: MultimodalMode = (
            multimodal_mode_override
            if multimodal_mode_override is not None
            else getattr(config, "multimodal_mode", "off")
        )

        if oracle_context is not None:
            from benchmark_models.tier3 import SmolagentsOracleModel

            kwargs: dict[str, Any] = dict(
                provider=config.backend,
                provider_model_id=provider_model_id,
                gold_triples_map=oracle_context.gold_triples_map,
                api_key=api_key,
                registry=reg,
                snapshot_id=snapshot_id,
                max_steps=max_steps,
                seed=config.seed,
            )
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if max_concurrent is not None:
                kwargs["max_concurrent"] = max_concurrent
            kwargs["multimodal_mode"] = tier3_multimodal_mode
            return SmolagentsOracleModel(variant_name, **kwargs)

        from benchmark_models.tier3 import SmolagentsModel

        kwargs = dict(
            provider=config.backend,
            provider_model_id=provider_model_id,
            sub_tier=config.sub_tier or "3a",
            api_key=api_key,
            registry=reg,
            snapshot_id=snapshot_id,
            max_steps=max_steps,
            seed=config.seed,
            kg_backend=kg_backend,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if max_concurrent is not None:
            kwargs["max_concurrent"] = max_concurrent
        if web_search_timeout is not None:
            kwargs["web_search_timeout_seconds"] = web_search_timeout
        kwargs["multimodal_mode"] = tier3_multimodal_mode
        return SmolagentsModel(variant_name, **kwargs)

    # Tiers 1 + 2: construct a PromptRenderer (oracle-aware for T1 oracle
    # runs; plain otherwise) and pass it to the backend.
    tier_literal = "tier1" if config.tier == "plain" else "tier2"
    renderer = PromptRenderer(
        tier=tier_literal,  # type: ignore[arg-type]
        oracle_context=oracle_context if config.tier == "plain" else None,
    )

    # Resolve the multimodal mode in priority order:
    #   CLI override → models.yaml variant config → default 'off'.
    multimodal_mode: MultimodalMode = (
        multimodal_mode_override
        if multimodal_mode_override is not None
        else config.multimodal_mode
    )

    if isinstance(config, GoogleVariantConfig):
        from benchmark_models.backends.google import GoogleModel

        return GoogleModel(
            variant_name,
            config,
            registry=reg,
            api_key=api_key,
            snapshot_id=snapshot_id,
            prompt_renderer=renderer,
            multimodal_mode=multimodal_mode,
        )

    if isinstance(config, OpenAIVariantConfig):
        from benchmark_models.backends.openai import OpenAIModel

        return OpenAIModel(
            variant_name,
            config,
            registry=reg,
            api_key=api_key,
            snapshot_id=snapshot_id,
            prompt_renderer=renderer,
            multimodal_mode=multimodal_mode,
        )

    if isinstance(config, VLLMHttpVariantConfig):
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        return VLLMHttpModel(
            variant_name,
            config,
            registry=reg,
            snapshot_id=snapshot_id,
            prompt_renderer=renderer,
            multimodal_mode=multimodal_mode,
        )

    raise NotImplementedError(
        f"Unknown config type {type(config).__name__} for variant "
        f"{variant_name!r}."
    )


def load_and_create(
    variant_name: str,
    *,
    models_yaml_path: Path | str | None = None,
    registry: TemplateRegistry | None = None,
    api_key: str | None = None,
    snapshot_id: str = "unset",
) -> Model:
    """Shortcut: load the catalogue, look up the variant, create the model.

    Convenience for CLI one-offs and smoke tests where the caller
    doesn't want to manage the catalogue object.
    """
    catalogue = load_models_catalogue(models_yaml_path)
    config = catalogue.get(variant_name)
    return create_model(
        variant_name,
        config,
        registry=registry,
        api_key=api_key,
        snapshot_id=snapshot_id,
    )


def list_variants(
    models_yaml_path: Path | str | None = None,
) -> list[str]:
    """Return all variant names from the catalogue."""
    catalogue = load_models_catalogue(models_yaml_path)
    return catalogue.list_variants()
