"""Per-experiment and per-variant configuration for benchmark_models.

Two configuration layers, each a pydantic-validated YAML file:

1. **Variant catalogue** (``benchmark_models/models.yaml``) — the full
   slate of model variants. Each entry maps a variant name (e.g.,
   ``"gpt-5.4-plain"``) to a :class:`VariantConfig` that declares
   which backend, which model id / weights, which tier, and per-variant
   runtime knobs (decoding params, batch size, thinking flags, etc.).

   Variant configs are a *discriminated union* on the ``backend`` field:
   :class:`OpenAIVariantConfig`, :class:`GoogleVariantConfig`,
   :class:`VLLMHttpVariantConfig`. Each
   subclass only allows the fields relevant to that backend, and
   ``extra="forbid"`` catches typos at load time.

2. **Experiment file** (``experiments/*.yaml``) — declares which
   variants, languages, tiers, and seeds to run in one invocation.
   Per-variant overrides let the experiment YAML tweak runtime knobs
   (batch_size, max_tokens, etc.) without editing ``models.yaml``.

Both are loaded via :func:`load_models_catalogue` and
:func:`load_experiment_config`, which validate against the pydantic
schemas and fail loudly on bad data.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmark_models.base import Tier
from benchmark_models.prompts import DEFAULT_MAX_TOOL_CALLS

# ----------------------------------------------------------------------------
# Variant configs (models.yaml entries)
# ----------------------------------------------------------------------------

# Tier values as used in models.yaml (different from the Prediction
# tier literal — "plain"/"reasoning"/"tool_agent" vs "tier1"/"tier2"/
# "tier3"). The runner maps between them.
TierName = Literal["plain", "reasoning", "tool_agent"]

TIER_NAME_TO_TIER: dict[TierName, Tier] = {
    "plain": "tier1",
    "reasoning": "tier2",
    "tool_agent": "tier3",
}

MultimodalMode = Literal["off", "image", "name"]
"""How a variant handles ``Example.images`` for multimodal questions.

``"off"`` — non-multimodal run. The variant should not be paired with
multimodal benchmark shards; pair-up validation lives at runner /
CLI level.

``"image"`` — image bytes are uploaded to the provider as a content
block alongside the prompt text. Requires a vision-capable backend;
the slate's API providers all support this and the open-weights
variants we serve via vLLM (Qwen 3.5, Gemma 4) are documented as
multimodal.

``"name"`` — text-only baseline. Each ``[IMAGE_<slot>]`` placeholder
in the question is replaced with the corresponding entity's
``name`` (and ``subcategory`` / ``category`` when present) before
the prompt reaches the backend. Lets us isolate the visual
contribution: the gap between an ``image`` and a ``name`` run on
otherwise identical config is the headline ablation."""


class VariantConfigBase(BaseModel):
    """Common fields shared by all backend-specific variant configs.

    Subclasses add backend-specific fields (``model_id``, ``model_path``,
    thinking toggles, vLLM engine params, etc.) and set ``backend`` to
    a :class:`Literal` for discriminated-union dispatch.
    """

    model_config = ConfigDict(extra="forbid")

    # Backend discriminator — overridden as a Literal in each subclass.
    backend: str

    # Tier selection.
    tier: TierName

    # Decoding parameters.
    temperature: float = 0.0
    """Sampling temperature. 0.0 for deterministic (tier 1 / tier 3);
    > 0.0 needed for self-consistency sampling in tier 2."""
    max_tokens: int = 512
    """Maximum output tokens per model call."""
    top_p: float = 1.0
    """Nucleus sampling parameter."""
    seed: int = 42
    """Random seed for reproducibility. Recorded in every Prediction."""

    # Tier-2 reasoning params (universal — any backend may use them).
    self_consistency_n: int = 1
    """Number of CoT samples for self-consistency voting. 1 = no SC
    (just greedy CoT or native thinking). 3 = SC@3 per the plan."""

    # Tier-3 tool params (universal — any backend may use them).
    sub_tier: Literal["3a", "3b", "3c"] | None = None
    """Sub-tier for tier 3: ``"3a"`` (KG + Python), ``"3b"``
    (Web + Python), ``"3c"`` (KG + Web + Python). ``None`` for
    tiers 1 and 2."""
    tools: list[str] | None = None
    """Tool names to expose for tier 3. ``None`` = all available.
    Explicit list for tool-ablation cells (e.g., ``["find_entity"]``
    for the lookup-only ablation)."""
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    """Per-question tool-call cap for tier 3. Defaults to
    :data:`DEFAULT_MAX_TOOL_CALLS` (10, matching the project plan's
    documented headline-run cap; raised from Smolagents' upstream 6
    after observing 80 % cap_hit rate on qwen35-27b-tier3b at 6).
    Experiment YAML can override per-variant; CLI ``--max-steps``
    overrides at run time."""

    # Tier-3 tuning knobs (only consulted when tier='tool_agent';
    # rejected by the validator for other tiers). Renamed so they
    # don't shadow the tier-1/tier-2 ``max_tokens`` knob above —
    # tier-3 uses *per-step* token caps with sub-tier-specific
    # defaults that have nothing to do with the base 512.
    tier3_max_tokens: int | None = None
    """Per-step LLM ``max_tokens`` for tier-3 calls. ``None`` falls
    back to the per-sub-tier default in
    :data:`benchmark_models.tier3._MAX_TOKENS_BY_SUB_TIER`. Override
    here for variants whose answers reliably overflow / underflow
    those defaults; CLI ``--max-tokens`` overrides at run time."""
    max_concurrent: int | None = None
    """Number of concurrent agent sessions in tier-3
    ``predict_batch``. ``None`` falls back to
    :data:`benchmark_models.tier3._DEFAULT_MAX_CONCURRENT`. Override
    when provider rate limits dictate a different fan-out; CLI
    ``--max-concurrent`` overrides at run time."""
    web_search_timeout_seconds: float | None = None
    """Per-request timeout for the tier-3b / 3c ``WebSearchTool``.
    ``None`` falls back to
    :data:`benchmark_models.tier3._DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS`.
    CLI ``--web-search-timeout`` overrides at run time."""

    # Multimodal mode. Stays ``"off"`` on regular-share variants;
    # image / name variants in models.yaml override per slate model.
    # CLI ``--multimodal-mode`` overrides at run time.
    multimodal_mode: MultimodalMode = "off"
    """How this variant treats ``Example.images``. ``"off"`` for the
    regular share; ``"image"`` to pack image bytes into a provider
    content block; ``"name"`` to substitute entity names into the
    ``[IMAGE_<slot>]`` placeholders for a text-only baseline. See
    :data:`MultimodalMode` for the full rationale."""

    @model_validator(mode="after")
    def _validate_tier_consistency(self) -> VariantConfigBase:
        """Sanity-check tier-specific fields."""
        if self.tier != "reasoning" and self.self_consistency_n > 1:
            raise ValueError(
                f"self_consistency_n={self.self_consistency_n} only makes "
                f"sense for tier='reasoning', but tier={self.tier!r}."
            )
        if self.self_consistency_n > 1 and self.temperature <= 0.0:
            # Google's gen_config exposes no ``seed`` parameter. With
            # ``temperature=0`` the SC samples are deterministic
            # duplicates of the same draw, ``sc_agreement`` saturates
            # at 1.0, and the headline EM matches a single-sample
            # greedy run at N× the cost. OpenAI's seed is best-effort.
            # Refusing the config here makes the failure mode loud
            # rather than silently inflating cost without changing EM.
            raise ValueError(
                f"self_consistency_n={self.self_consistency_n} requires "
                f"temperature > 0 (SC samples need stochasticity; the "
                f"Google API exposes no seed parameter, so "
                f"deterministic-temperature SC produces N identical "
                f"samples). Got temperature={self.temperature}."
            )
        if self.tier != "tool_agent" and self.tools is not None:
            raise ValueError(
                f"tools={self.tools!r} only makes sense for "
                f"tier='tool_agent', but tier={self.tier!r}."
            )
        if self.tier == "tool_agent" and self.sub_tier is None:
            raise ValueError(
                "sub_tier must be set for tier='tool_agent' "
                "(one of '3a', '3b', '3c')."
            )
        if self.tier != "tool_agent" and self.sub_tier is not None:
            raise ValueError(
                f"sub_tier={self.sub_tier!r} only makes sense for "
                f"tier='tool_agent', but tier={self.tier!r}."
            )
        # Tier-3 knobs only apply when tier='tool_agent'. Reject if
        # someone sets them on a tier-1/tier-2 variant — silently
        # ignoring them would hide config bugs.
        if self.tier != "tool_agent":
            for field, value in (
                ("tier3_max_tokens", self.tier3_max_tokens),
                ("max_concurrent", self.max_concurrent),
                ("web_search_timeout_seconds", self.web_search_timeout_seconds),
            ):
                if value is not None:
                    raise ValueError(
                        f"{field}={value!r} only makes sense for "
                        f"tier='tool_agent', but tier={self.tier!r}."
                    )
        return self


class OpenAIVariantConfig(VariantConfigBase):
    """Config for OpenAI API backends (GPT-5.4 family)."""

    backend: Literal["openai"]
    model_id: str
    """OpenAI model identifier (e.g., ``"gpt-5.4"``,
    ``"gpt-5.4-pro"``)."""
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "none"
    """Reasoning effort level. ``"none"`` (default) for tier 1 plain;
    ``"low"``/``"medium"``/``"high"``/``"xhigh"`` for tier 2 reasoning.
    Reasoning is integrated into the base gpt-5.4 model — there is no
    separate "thinking" model. Higher effort = more reasoning tokens
    = better answers but higher latency and cost."""


class GoogleVariantConfig(VariantConfigBase):
    """Config for Google API backends (Gemini 3 Flash)."""

    backend: Literal["google"]
    model_id: str
    """Google model identifier (e.g., ``"gemini-3-flash"``)."""
    thinking_level: Literal["minimal", "low", "medium", "high"] = "minimal"
    """Gemini thinking depth. ``"minimal"`` for tier 1 / tier 3;
    ``"high"`` for tier 2 reasoning."""
    enable_context_cache: bool = True
    """When True (default), Gemini's explicit context-caching API is
    used for prompt prefixes that exceed the 4096-token minimum.
    The backend splits the rendered prompt at ``Question:\\n``,
    calls ``client.caches.create(model=..., contents=[Part(prefix)],
    ttl=3600s)`` on the first request with a given
    ``sha256(prefix + model_id)`` key, and references
    ``cached_content=<name>`` on subsequent calls. Cached input
    bills at ~25% of the normal rate (~75% discount).

    Cache-expired errors (TTL elapsed mid-run) are detected at the
    call site, the registry entry is invalidated, and the next call
    re-creates. The whole path is thread-safe via a
    ``threading.Lock`` so concurrent ``predict_batch`` workers
    don't both seed the same prefix.

    Set ``False`` per-variant to disable for sensitivity analysis or
    for short-prefix runs where caching wouldn't help. The cached-
    token count is denormalised end-to-end (``cached_content_token_count``
    → ``Prediction.usage.cached_tokens`` → ``ScoreRecord.cached_tokens``
    → ``AggregateReport.total_cached_tokens``)."""


class VLLMHttpVariantConfig(VariantConfigBase):
    """Config for vLLM-served open-weights models talked to over HTTP.

    The SLURM template starts a per-cell ``vllm serve`` in the
    background, waits for ``/health``, then runs the smoke against
    ``base_url``. Client-side, this looks identical to the OpenAI
    Chat Completions API — the backend subclasses ``OpenAIModel`` and
    just overrides the client's ``base_url``.

    See ``benchmark_models/backends/vllm_http.py`` for the client; see
    the ``_VLLM_HTTP_*`` templates in
    ``scripts/benchmark_models/generate_slurm.py`` for the serve side.
    """

    backend: Literal["vllm_http"]
    model_path: str
    """HuggingFace model path (e.g., ``"Qwen/Qwen3.5-35B-A3B"``).
    Passed to ``vllm serve`` and used as the ``model`` field on
    Chat-Completions requests."""
    base_url: str | None = None
    """OpenAI-compat endpoint override. When ``None`` (default), the
    effective URL is derived from :attr:`vllm_port` —
    ``http://localhost:<vllm_port>/v1`` — keeping the backend client
    and the SLURM template's local serve in sync via a single source
    of truth. Set this to a non-local URL (e.g.
    ``http://remote-vllm:8001/v1``) only when you want the harness
    to talk to a serve outside the SLURM job. The effective value
    that the backend actually uses is exposed as
    :attr:`effective_base_url`."""
    vllm_port: int = 8000
    """Port the SLURM template binds ``vllm serve`` to. Also drives
    the default :attr:`effective_base_url` when ``base_url`` is left
    unset. Override here to launch on a non-default port — the
    backend URL follows automatically."""
    max_model_len: int = 32768
    """Passed to ``vllm serve --max-model-len``. Caps prompt + completion
    length; bound this to keep KV-cache memory predictable.

    32 K is Qwen3.5-27B's native context ceiling without rope_scaling
    and covers the typical tier-1/tier-2/tier-3b trajectory. Tier-3a
    use-kg agent loops grow per-step input by ~5–7 K (system prompt
    repeated + cumulative tool outputs), so the qwen35-27b-tier3a
    and qwen35-27b-tier3c variants set ``rope_scaling`` (YaRN
    factor=2.0) and ``max_model_len=65536`` to fit 7–10 step
    trajectories without context overflow.

    KV-cache cost on H100/H200 80 GB at ``max_num_seqs=64`` is roughly
    40 GB at 32 K (80 GB at 64 K with YaRN-extended variants); combined
    with ~14 GB Qwen 27B-FP8 weights, leaves comfortable headroom on
    H200's 141 GB. On H100's 80 GB, drop ``max_num_seqs`` to 32 when
    serving the YaRN-extended variants."""

    rope_scaling: dict[str, Any] | None = None
    """Optional rope-scaling config (JSON-encoded as
    ``--rope-scaling`` for ``vllm serve``). When set, vLLM extends the
    model's effective context window above its pretrained max.

    Standard YaRN config (matches Qwen3.5 model card)::

        rope_scaling:
          rope_type: yarn
          factor: 2.0
          original_max_position_embeddings: 32768

    Pair with ``max_model_len: 65536`` for the 2× extension YaRN
    enables. Perplexity degradation at factor=2.0 is well under 1% in
    the 28 K–60 K regime where tier-3a agent loops actually consume
    the budget; the trade-off is one paper footnote for
    multi-step-tool-use trajectories that would otherwise hit
    32 K-cap overflow.

    Defaults to ``None`` (model serves at its native context
    ceiling). Empty dicts are treated as ``None``."""
    max_num_seqs: int = 64
    """Passed to ``vllm serve --max-num-seqs``. Caps concurrent
    in-flight sequences per serve. KV-cache memory scales linearly
    with ``max_model_len * max_num_seqs``, so YaRN-extended variants
    (e.g. ``qwen35-27b-tier3a`` at 65 K) lower this to 16 to keep
    GPU+CPU KV cache within the SLURM ``--mem`` budget."""

    slurm_mem_gb: int = 80
    """SLURM ``--mem=<N>gb`` directive emitted in the generated
    vLLM-HTTP SLURM script header. Sized for the union of:

    * vLLM CPU-side allocations (~3-5 GiB scratch + KV-cache shadow
      that scales with ``max_model_len * max_num_seqs``)
    * Spark driver heap + direct buffer (up to 16 GiB; see
      ``benchmark_models.kg_tools.triple_bridge``)
    * Python harness + smolagents agent trace + loguru buffers (~3 GiB)

    The default 80 covers the native-32 K serving regime. YaRN-extended
    variants override to 128 to keep headroom under the larger KV
    cache (see ``_qwen35_27b_yarn_64k`` anchor in ``models.yaml``).

    API-backend variants don't read this field — their SLURM header
    template hardcodes ``--mem=32gb`` (no vLLM weight / KV cache
    overhead)."""
    dtype: str = "float16"
    """Weight dtype (``"float16"``, ``"bfloat16"``, ``"float8"``,
    ``"int4"``). Passed to ``vllm serve --dtype``."""
    quantization: str | None = None
    """Quantization scheme (``"awq"``, ``None``). Passed to
    ``vllm serve --quantization`` when set."""
    batch_size: int = 32
    """Number of examples to batch in ``predict_batch``. The serve does
    continuous batching internally; this controls client-side dispatch
    concurrency."""
    gpu_memory_utilization: float = 0.9
    """Fraction of GPU memory for the vLLM engine. Lower = safer;
    higher = more KV cache room."""
    thinking: bool = False
    """``True`` for models with a thinking toggle (e.g., Qwen3.5 with
    ``<think>...</think>`` blocks)."""

    @property
    def model_id(self) -> str:
        """OpenAI-shaped alias for :attr:`model_path`. The parent
        :class:`OpenAIModel` reads ``self.config.model_id`` when
        constructing requests; we expose ``model_path`` under that
        name so the parent's request-shaping code resolves correctly
        without a separate code path.
        """
        return self.model_path

    @property
    def reasoning_effort(self) -> Literal["none"]:
        """OpenAI-compatible no-op for vLLM.

        :class:`benchmark_models.backends.openai.OpenAIModel` reads
        ``self.config.reasoning_effort`` unconditionally
        (``openai.py:319``) to gate a 25k-token budget expansion for
        OpenAI reasoning models. The vLLM thinking toggle (Qwen 3+ /
        Gemma 4) flows through chat-completions request params
        instead — going through the OpenAI reasoning branch would
        over-allocate tokens past :attr:`max_model_len` and crash on
        the serve. Pinning to ``"none"`` keeps the parent on the
        plain-completion path; the ``thinking`` flag is consumed by
        the dedicated thinking-aware request builder.
        """
        return "none"

    @property
    def effective_base_url(self) -> str:
        """Resolved OpenAI-compat URL the backend actually talks to.

        Resolution order, highest first:

        1. Explicit :attr:`base_url` field — set in YAML when you
           want the harness to talk to a non-local serve (e.g.
           ``http://remote-vllm:8001/v1``). Always wins.
        2. ``VLLM_BASE_URL`` environment variable — set by the SLURM
           template to a per-job URL so two single-GPU jobs
           co-scheduled on the same multi-GPU node can each run a
           ``vllm serve`` on a job-derived port without racing for
           ``localhost:8000``.
        3. ``http://localhost:{vllm_port}/v1`` — the YAML
           :attr:`vllm_port` (default 8000), for interactive /
           local-dev runs where the user controls the port directly.

        The middle slot is the new one: without it, the runner
        client would always dial ``localhost:{vllm_port}`` regardless
        of which port the surrounding SLURM job actually started its
        serve on.
        """
        import os
        if self.base_url is not None:
            return self.base_url
        env_url = os.environ.get("VLLM_BASE_URL")
        if env_url:
            return env_url
        return f"http://localhost:{self.vllm_port}/v1"


# Discriminated union — pydantic reads the ``backend`` field to pick
# the right subclass. YAML stays flat (one dict per variant).
VariantConfig = Annotated[
    OpenAIVariantConfig
    | GoogleVariantConfig
    | VLLMHttpVariantConfig,
    Field(discriminator="backend"),
]
"""Union of all backend-specific variant configs, discriminated on
the ``backend`` field. Use this type in any function that accepts a
variant config without knowing which backend it is."""


# ----------------------------------------------------------------------------
# Models catalogue (models.yaml)
# ----------------------------------------------------------------------------


DEFAULT_MODELS_YAML = "models.yaml"


class ModelsCatalogue(BaseModel):
    """The full ``models.yaml`` file, parsed and validated.

    Top-level keys are variant names (e.g., ``"gpt-5.4-plain"``);
    values are :data:`VariantConfig` instances. The ``variants`` field
    is built programmatically from the YAML dict by injecting each
    variant name as metadata.
    """

    model_config = ConfigDict(extra="forbid")

    variants: dict[str, VariantConfig] = Field(default_factory=dict)
    """variant_name → :data:`VariantConfig` mapping."""

    def get(self, name: str) -> VariantConfig:
        """Look up a variant by name. Raises ``KeyError`` if absent."""
        try:
            return self.variants[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.variants.keys())[:10])
            raise KeyError(
                f"Unknown variant {name!r}. Available: [{available}, ...]"
            ) from exc

    def list_variants(self) -> list[str]:
        """Return all variant names in YAML order."""
        return list(self.variants.keys())

    def __contains__(self, name: object) -> bool:
        return name in self.variants

    def __len__(self) -> int:
        return len(self.variants)


def load_models_catalogue(path: Path | str | None = None) -> ModelsCatalogue:
    """Read and validate ``models.yaml``.

    Args:
        path: Filesystem path to the YAML file. When ``None``, loads
            the bundled ``models.yaml`` from the ``benchmark_models``
            package via :mod:`importlib.resources`.

    Returns:
        A validated :class:`ModelsCatalogue`.

    Raises:
        FileNotFoundError: if ``path`` is given but doesn't exist.
        yaml.YAMLError: on bad YAML syntax.
        pydantic.ValidationError: on schema violations.
    """
    if path is None:
        resource = files("benchmark_models").joinpath(DEFAULT_MODELS_YAML)
        with resource.open("r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        p = Path(path)
        with p.open() as f:
            raw = yaml.safe_load(f) or {}

    # The YAML is a flat dict: variant_name -> {backend, tier, ...}.
    # We validate each entry individually so error messages name the
    # offending variant.
    if not isinstance(raw, dict):
        raise ValueError(
            f"models.yaml should be a top-level YAML mapping, "
            f"got {type(raw).__name__}"
        )
    # m28: keys starting with ``_`` are treated as YAML-anchor
    # templates (not variants) and skipped. Lets ``models.yaml`` use
    # ``_qwen_defaults: &qwen`` style anchors to deduplicate shared
    # config without each anchor turning into a phantom variant.
    variants = {
        name: spec
        for name, spec in raw.items()
        if not name.startswith("_")
    }
    return ModelsCatalogue.model_validate({"variants": variants})


# ----------------------------------------------------------------------------
# Experiment config (experiments/*.yaml)
# ----------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """One experiment YAML file, parsed and validated.

    An experiment declares *which* variants to run, on *which* languages
    and tiers, with *which* seeds. Per-variant overrides let the
    experiment YAML tweak runtime knobs (batch_size, max_tokens, etc.)
    without editing ``models.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Free-form experiment name (goes into MLflow run name)."""

    languages: list[str]
    """ISO language codes to run (e.g., ``["en"]``,
    ``["en", "de", "zh-CN"]``)."""

    variants: list[str]
    """Variant names from ``models.yaml`` to include in this experiment."""

    seeds: list[int] = Field(default_factory=lambda: [42])
    """Random seeds. Each (variant, seed) pair becomes a separate MLflow
    child run. Default ``[42]`` for single-seed headline runs."""

    output_dir: str
    """Filesystem path for the results directory. Predictions are
    written to ``<output_dir>/<variant>/predictions.jsonl``."""

    question_subset: str | None = None
    """Optional path to a JSONL file containing a subset of question_ids
    to run. ``None`` = full benchmark."""

    overrides: dict[str, dict[str, Any]] | None = None
    """Per-variant field overrides. Each key is a variant name; the
    value is a dict of fields to merge into that variant's config.
    Example::

        overrides:
          qwen35-a3b-plain:
            batch_size: 128
            max_tokens: 1024
    """


def load_experiment_config(path: Path | str) -> ExperimentConfig:
    """Read and validate an experiment YAML file.

    Args:
        path: Filesystem path to the experiment YAML.

    Returns:
        A validated :class:`ExperimentConfig`.

    Raises:
        FileNotFoundError: if ``path`` doesn't exist.
        yaml.YAMLError: on bad YAML syntax.
        pydantic.ValidationError: on schema violations.
    """
    p = Path(path)
    with p.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(raw)
