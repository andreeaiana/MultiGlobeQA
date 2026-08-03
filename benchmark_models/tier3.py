"""Tier-3 model wrapper using Smolagents CodeAgent.

Wraps Smolagents' ``CodeAgent`` (or ``ToolCallingAgent`` for the
primitives ablation) to provide tool-augmented LLM inference. The
model writes Python code that calls KG tools and/or web tools,
executes it in a sandboxed environment, and iterates until it
produces a final answer.

Three sub-tier configurations:
- **3a** (KG + Python): ``find_entity``, ``find_entities_by_type``,
  ``get_entity``, ``list_categories`` + Python sandbox with
  shapely/geopy/pyproj/h3.
- **3b** (Web + Python): ``WebSearchTool`` + ``VisitWebpageTool`` +
  same Python sandbox. No KG access.
- **3c** (KG + Web + Python): all tools from 3a + 3b.

The model class (``SmolagentsModel``) handles:
- Constructing the right Smolagents model backend (``LiteLLMModel``,
  ``OpenAIModel``, ``VLLMModel``) from the variant config.
- Building the tool list for the configured sub-tier.
- Running the agent and capturing the execution trace into a
  ``Prediction`` record.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal

from loguru import logger

from benchmark_models.backends._error_class import classify_error
from benchmark_models.backends._retry import retry_with_backoff
from benchmark_models.base import (
    Example,
    Model,
    Prediction,
    Tier,
    ToolCall,
    Usage,
)
from benchmark_models.prompts import DEFAULT_MAX_TOOL_CALLS
from benchmark_models.schemas import TemplateRegistry, load_template_registry

# Sub-tier identifiers for tier 3.
SubTier = Literal["3a", "3b", "3c"]


# Per-sub-tier ``max_tokens`` for the tier-3 LLM call inside the
# Smolagents agent loop. Tier 3b's outputs are mostly thinking + retry
# prose (web results enter via *input*, not output), so the cap bounds
# essay growth without harming code generation. Tier 3a / 3c emit
# Python that uses the bridge's already-simplified geometry via
# ``get_entity``; the model rarely needs to write large WKT literals
# itself. 4096 is enough headroom for the largest agent code blocks
# observed in practice while leaving 28 K of input budget on a 32 K
# context window. At 8192 a single-token overflow is enough to make
# a 32 K-context model cap-hit on every tier-3a question.
# The oracle subclass uses tier-3a's value (4096) since its output
# behavior matches.
#
# Defaults — overridable per-variant via ``models.yaml`` (``max_tokens``
# field) and at run time via the ``--max-tokens`` CLI flag.
_MAX_TOKENS_BY_SUB_TIER: dict[str, int] = {
    "3a": 4096,
    "3b": 4096,
    "3c": 4096,
}

# Default web-search timeout (seconds). Stock smolagents
# ``WebSearchTool`` has no per-request timeout; DDG-lite rate-limits
# silently wedge an agent step indefinitely. 15s is the empirically
# observed point past which a request rarely succeeds. Override via
# ``--web-search-timeout`` CLI flag for tier-3b / tier-3c runs.
_DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS: float = 15.0

# Default tier-3 agent concurrency. ``predict_batch`` dispatches
# concurrent agent sessions across this many workers via
# ThreadPoolExecutor. Each agent session makes ~max_steps API calls,
# so per-question pressure on the provider rate-limit is roughly
# ``max_concurrent × max_steps``. Override via ``--max-concurrent``
# CLI flag for tier-3 runs.
_DEFAULT_MAX_CONCURRENT: int = 4


_BOOL_TRUE = frozenset({"true", "t", "yes", "y", "1"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n", "0"})


_MISSING_ANSWER = object()
"""Sentinel for ``_raw_parse``: the model returned a dict payload but
did not include an ``"answer"`` key.

Distinguishes two cases that look identical under ``.get("answer")``:

* The key is present and maps to ``None`` — the model explicitly
  abstained. Handled as ``final_state="abstained"``.
* The key is missing — the model emitted a dict with a different
  envelope (e.g., ``{"lat": ..., "lon": ...}``). Attempt type-aware
  recovery via :meth:`SmolagentsModel._recover_from_envelope_free`,
  then fall back to ``final_state="no_final_answer"``.

Using ``None`` as the sentinel would conflate the two cases; using a
string sentinel collides with legitimate string answers. Module-level
object() is unique and opaque."""


def _coerce_boolean(value: Any) -> bool | None:
    """Coerce a tier-3 answer into a real boolean or return ``None``.

    Tier-3 CodeAgent prompts sometimes instruct ``final_answer`` with
    ``"yes"`` / ``"no"`` string literals rather than real booleans
    (no JSON-schema enforcement inside the sandbox). Normalizing here
    keeps tier-3 boolean predictions comparable to tier-1 / tier-2
    predictions that arrive as real bools. ``None`` signals the
    value couldn't be interpreted as a boolean and the caller should
    treat the prediction as malformed.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Reject bare numerics: 0/1 are too easy to hit accidentally.
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _BOOL_TRUE:
            return True
        if s in _BOOL_FALSE:
            return False
    return None


_COORD_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*$"
)


def _coerce_coordinates(value: Any) -> str | None:
    """Coerce a coordinate answer into a canonical ``"lat, lon"`` string.

    Accepts: a list/tuple of two numerics, a dict with ``lat``/``lon``
    (or ``latitude``/``longitude``) keys, or a string containing two
    numbers separated by comma / semicolon / whitespace. Validates
    that latitude is in ``[-90, 90]`` and longitude in ``[-180, 180]``.
    Returns ``None`` on any failure — the caller treats that as a
    malformed prediction.
    """
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            lat, lon = float(value[0]), float(value[1])
        elif isinstance(value, dict):
            lat_key = next(
                (k for k in ("lat", "latitude") if k in value), None
            )
            lon_key = next(
                (k for k in ("lon", "long", "longitude", "lng") if k in value),
                None,
            )
            if lat_key is None or lon_key is None:
                return None
            lat, lon = float(value[lat_key]), float(value[lon_key])
        elif isinstance(value, str):
            m = _COORD_RE.match(value)
            if not m:
                return None
            lat, lon = float(m.group(1)), float(m.group(2))
        else:
            return None
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    return f"{lat}, {lon}"

# Libraries pre-imported in the CodeAgent's Python sandbox.
SANDBOX_AUTHORIZED_IMPORTS = [
    "shapely",
    "shapely.*",
    "geopy",
    "geopy.*",
    "pyproj",
    "pyproj.*",
    "h3",
    "math",
    "json",
    "re",
    "statistics",
    "collections",
]


def _build_smolagents_model(
    provider: str,
    model_id: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Construct the right Smolagents model backend.

    Maps our backend names (openai, google, vllm_http) to Smolagents
    model classes. Uses LiteLLM for the providers that don't have a
    native Smolagents model class (Google, vLLM).
    """
    if provider == "openai":
        from smolagents import OpenAIModel

        return OpenAIModel(
            model_id=model_id,
            api_key=api_key,
            **kwargs,
        )
    elif provider == "google":
        import os
        from smolagents import LiteLLMModel

        # Route through Vertex AI when GOOGLE_GENAI_USE_VERTEXAI is set,
        # mirroring the tier-1/2 google backend. LiteLLM picks up
        # ``GOOGLE_APPLICATION_CREDENTIALS`` automatically; pass
        # project + location explicitly via init kwargs so the routing
        # is independent of which Vertex env-var convention is active.
        # Otherwise (no Vertex flag), fall back to the public AI Studio
        # endpoint, which authenticates with GEMINI_API_KEY / GOOGLE_API_KEY.
        use_vertex = os.environ.get(
            "GOOGLE_GENAI_USE_VERTEXAI", ""
        ).lower() in ("true", "1", "yes")
        if use_vertex:
            litellm_kwargs = dict(kwargs)
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION")
            if project:
                litellm_kwargs.setdefault("vertex_project", project)
            if location:
                litellm_kwargs.setdefault("vertex_location", location)
            return LiteLLMModel(
                model_id=f"vertex_ai/{model_id}",
                **litellm_kwargs,
            )
        return LiteLLMModel(
            model_id=f"gemini/{model_id}",
            api_key=api_key,
            **kwargs,
        )
    elif provider == "vllm_http":
        # vLLM serve is OpenAI-compatible. Route Smolagents through
        # LiteLLM's OpenAI adapter against the per-cell ``vllm serve``
        # endpoint.
        # ``base_url`` is read from kwargs so the registry can plumb it
        # from the variant config; default matches the SLURM template's
        # in-job binding.
        import os
        from smolagents import LiteLLMModel

        base_url = kwargs.pop("base_url", None) or os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1",
        )
        # Honour the function's explicit ``api_key`` parameter first
        # (the standard signature used by every other branch), then
        # any value squirreled through kwargs (rare but possible from
        # the registry), then fall back to a dummy key — vLLM serve
        # doesn't require auth by default, but LiteLLM's request
        # signing refuses to construct without one.
        effective_api_key = (
            api_key or kwargs.pop("api_key", None) or "EMPTY"
        )
        return LiteLLMModel(
            model_id=f"openai/{model_id}",
            api_key=effective_api_key,
            api_base=base_url,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown provider for Smolagents: {provider!r}")


def _make_timeout_bounded_search_tool(timeout_seconds: float) -> Any:
    """Return a ``WebSearchTool`` subclass with a bounded request timeout.

    Defined as a factory so smolagents is imported lazily (the model
    section must not hard-require smolagents for non-tier-3 paths).
    """
    from smolagents import WebSearchTool

    class _TimeoutBounded(WebSearchTool):
        def search_duckduckgo(self, query: str) -> list:
            import requests

            response = requests.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            parser = self._create_duckduckgo_parser()
            parser.feed(response.text)
            return parser.results

    return _TimeoutBounded()


def _build_kg_tools(kg_backend: Any = None) -> list[Any]:
    """Construct KG tool wrappers (stateless; safe to share).

    Each returned tool holds only a reference to ``kg_backend`` and
    has no per-call mutable state — they may be reused across
    concurrent agent sessions without isolation concerns.
    """
    from benchmark_models.kg_tools.tools import (
        FindEntitiesByTypeTool,
        FindEntityTool,
        GetEntityTool,
        ListCategoriesTool,
    )

    return [
        FindEntityTool(backend=kg_backend),
        FindEntitiesByTypeTool(backend=kg_backend),
        GetEntityTool(backend=kg_backend),
        ListCategoriesTool(backend=kg_backend),
    ]


def _build_web_tools(
    web_search_timeout_seconds: float = _DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
) -> list[Any]:
    """Construct web-search tools (stateful; built per agent call).

    Stock smolagents' ``WebSearchTool`` is not documented as
    stateless — its parser and session may carry state that races
    when shared across threads. The caller (``_build_agent``)
    builds these per-call so concurrent ``predict_batch`` workers
    each get their own instances.

    Backend selection: when ``SERPER_API_KEY`` is set, use
    smolagents' ``GoogleSearchTool(provider="serper")`` instead of
    the default DDG-lite scraper. Serper returns Google results
    reliably (~200-500ms, no IP-blocking 403s that DDG-lite serves
    to datacenter ASNs). Falls back to DDG when the key is absent
    so the default code path is unchanged for local/dev runs.
    """
    import os
    from smolagents import VisitWebpageTool

    if os.environ.get("SERPER_API_KEY"):
        from smolagents import GoogleSearchTool
        return [
            GoogleSearchTool(provider="serper"),
            VisitWebpageTool(),
        ]
    return [
        _make_timeout_bounded_search_tool(web_search_timeout_seconds),
        VisitWebpageTool(),
    ]


# Pre-imported libraries description shared across sub-tiers. Single
# source so the prompt isn't lying about what's available in the
# sandbox; keep this in sync with ``SANDBOX_AUTHORIZED_IMPORTS``.
_PREIMPORTS_DESCRIPTION: str = (
    "Pre-imported libraries you can use in your code:\n"
    "- geopy (geopy.distance.geodesic for distance, "
    "geopy.geocoders.Nominatim for geocoding)\n"
    "- shapely (shapely.wkt.loads for parsing WKT geometry strings, "
    "spatial operations like contains, intersects, distance, area, centroid)\n"
    "- pyproj (coordinate projection for area/perimeter in metric units)\n"
    "- h3 (H3 spatial indexing)\n"
    "- math, json, re (standard library)"
)


def render_tools_section(tools: list[Any]) -> str:
    """Render the ``$tools_section`` prompt slot from a tool list.

    Walks each Smolagents-shaped tool (``.name``, ``.description``,
    ``.inputs``, ``.output_type``) and emits one bullet line per
    tool. ``final_answer`` is always appended; smolagents provides
    it as a built-in regardless of which other tools are loaded.
    The rendered section ends with the pre-imported libraries
    paragraph so prompt + agent agree on what's available without
    the prompt re-listing the imports outside the slot.

    When ``tools`` is empty (oracle mode), only ``final_answer`` and
    the pre-imports paragraph are emitted.
    """
    lines: list[str] = ["Available tools (call as Python functions):"]
    for tool in tools:
        name = getattr(tool, "name", "?")
        desc = (getattr(tool, "description", "") or "").strip().replace("\n", " ")
        inputs = getattr(tool, "inputs", {}) or {}
        sig_parts: list[str] = []
        for arg_name, arg_spec in inputs.items():
            arg_type = (arg_spec or {}).get("type", "Any")
            if (arg_spec or {}).get("nullable"):
                arg_type = f"{arg_type} | None"
            sig_parts.append(f"{arg_name}: {arg_type}")
        sig = ", ".join(sig_parts)
        out_type = getattr(tool, "output_type", "str") or "str"
        lines.append(f"- {name}({sig}) -> {out_type} — {desc}")
    lines.append("- final_answer(answer) — submit your final answer and stop")
    lines.append("")
    lines.append(_PREIMPORTS_DESCRIPTION)
    return "\n".join(lines)


def _build_tools(
    sub_tier: SubTier,
    kg_backend: Any = None,
    *,
    web_search_timeout_seconds: float = _DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
) -> list[Any]:
    """Build the tool list for a given sub-tier (convenience entry point).

    Equivalent to combining :func:`_build_kg_tools` and
    :func:`_build_web_tools`. For callers that don't need the
    per-call concurrency-isolation distinction (e.g. diagnostic /
    smoke scripts).
    """
    if sub_tier == "3a":
        return _build_kg_tools(kg_backend)
    elif sub_tier == "3b":
        return _build_web_tools(web_search_timeout_seconds)
    elif sub_tier == "3c":
        return _build_kg_tools(kg_backend) + _build_web_tools(
            web_search_timeout_seconds
        )
    else:
        raise ValueError(f"Unknown sub_tier: {sub_tier!r}")


def _clear_replayed_step_image(memory_step: Any, agent: Any) -> None:
    """Step callback: null ``observations_images`` on each ``ActionStep``.

    Why this exists — smolagents 1.24.0 auto-attaches the same
    ``images`` list passed to ``agent.run(prompt, images=...)`` to
    every ``ActionStep`` via ``observations_images`` (see
    ``smolagents/agents.py:573``). Combined with
    ``ActionStep.to_messages`` (``smolagents/memory.py:112-128``)
    which emits a user message carrying those images, the LLM input
    at Step N contains N copies of the same image: 1 from
    ``TaskStep.task_images`` plus (N-1) duplicates from prior
    ``ActionStep.observations_images``. vLLM served with
    ``--limit-mm-per-prompt '{"image":1}'`` rejects the second
    image and Step 2 onwards dies with
    ``BadRequestError: At most 1 image(s) may be provided in one prompt``.

    The fix: null ``observations_images`` on each ``ActionStep`` as
    soon as it's finalized (this callback fires from
    ``_finalize_step`` at ``smolagents/agents.py:622``). The static
    task image stays in ``TaskStep.task_images`` and is the only
    image-bearing message the model sees, on every step. Mirrors the
    cleanup-prior-step idiom in
    ``smolagents/vision_web_browser.py:save_screenshot``.

    Safe on all tier-3 variants: no-op when ``observations_images`` is
    already ``None`` (text-only runs, name-mode runs), no-op for
    non-``ActionStep`` types (``TaskStep`` / ``PlanningStep`` /
    ``FinalAnswerStep``). If a future smolagents version stops
    auto-attaching ``observations_images``, this callback becomes a
    no-op too — safe to keep.
    """
    from smolagents.memory import ActionStep
    del agent  # signature parity with other smolagents step callbacks
    if isinstance(memory_step, ActionStep) and memory_step.observations_images:
        memory_step.observations_images = None


class SmolagentsModel(Model):
    """Tier-3 :class:`Model` implementation using Smolagents CodeAgent.

    Wraps Smolagents' ``CodeAgent`` to provide tool-augmented inference
    with a Python sandbox. Supports all three sub-tiers (3a, 3b, 3c)
    via the ``sub_tier`` parameter.
    """

    model_id: str
    tier: Tier = "tier3"

    def __init__(
        self,
        variant_name: str,
        *,
        provider: str,
        provider_model_id: str,
        sub_tier: SubTier = "3a",
        api_key: str | None = None,
        registry: TemplateRegistry | None = None,
        snapshot_id: str = "unset",
        max_steps: int = DEFAULT_MAX_TOOL_CALLS,
        kg_backend: Any = None,
        seed: int = 42,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        max_tokens: int | None = None,
        web_search_timeout_seconds: float = _DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
        multimodal_mode: "MultimodalMode" = "off",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """Construct a tier-3 Smolagents model.

        Args:
            max_tokens: Per-step LLM output cap. ``None`` falls back
                to the per-sub-tier default in
                :data:`_MAX_TOKENS_BY_SUB_TIER`.
            max_concurrent: Number of concurrent agent sessions in
                :meth:`predict_batch`. Defaults to
                :data:`_DEFAULT_MAX_CONCURRENT`.
            web_search_timeout_seconds: Per-request timeout for the
                ``WebSearchTool`` (tier-3b / tier-3c only). Defaults
                to :data:`_DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS`.
            multimodal_mode: How the example's images reach the agent.
                ``"image"`` plumbs the example's images through to
                ``agent.run(images=...)``; ``"name"`` uses text
                substitution (no images attached). ``"off"`` (default)
                ignores images entirely — used for non-multimodal
                variants.
        """
        from benchmark_models.config import MultimodalMode  # local to avoid cycle

        if multimodal_mode not in ("off", "image", "name"):
            raise ValueError(
                f"Unknown multimodal_mode {multimodal_mode!r} — "
                f"must be 'off', 'image', or 'name'."
            )
        self.multimodal_mode: MultimodalMode = multimodal_mode
        self.model_id = variant_name
        self.tier = "tier3"
        self.sub_tier = sub_tier
        self.registry = registry if registry is not None else load_template_registry()
        self.snapshot_id = snapshot_id
        self.max_steps = max_steps
        self.seed = seed
        self._max_concurrent = max_concurrent
        self._web_search_timeout_seconds = web_search_timeout_seconds

        self._kg_backend = kg_backend

        # Build the ingredients for per-call agent construction.
        # Constructing one ``CodeAgent`` per :meth:`predict` call is
        # the thread-safety fix for concurrent batching: smolagents'
        # ``run()`` mutates ``self.memory`` / ``self.state`` /
        # ``self.python_executor`` on every invocation, so a shared
        # agent object races when ``predict_batch`` spawns workers.
        # ``CodeAgent.__init__`` is <10 ms (dict setup + system-prompt
        # string concat + a ``LocalPythonExecutor`` with no I/O), so
        # per-call construction is negligible against the 30–180 s
        # tier-3 question latency.
        # Tier-3 pins ``temperature=0`` regardless of the variant
        # config: CodeAgent format compliance is a deterministic skill
        # (emitting syntactically valid Python that calls tools by
        # name), not a creative one. Sampling at the variant's tier-1
        # temperature compounds parser failures on tier-3 runs. Tier 1
        # and tier 2 keep their per-variant temperatures from
        # models.yaml.
        #
        # ``max_tokens`` per LLM step. Smolagents agent memory appends
        # every prior step's full output into the next prompt, so
        # unbounded outputs let the context grow exponentially
        # (observed: 1.48M-token prompts on hard tier-3b questions,
        # 30K+ token essays per step). The value is sub-tier-specific:
        #
        # * Tier 3b (web + Python): 4096. Outputs are mostly
        #   thinking + retry prose; the cap bounds essay growth and
        #   matches the Anthropic / OpenAI tool-use guidance for code
        #   generation.
        # * Tier 3a / 3c (KG / KG+web + Python): 4096. With the
        #   bridge's get_entity simplifying WKT before handing it to
        #   the agent, the model rarely needs to emit large geometry
        #   literals itself. 4096 leaves 28 K of input budget on a
        #   32 K context; at 8192 a single-token overflow is enough
        #   to break tier-3a runs on a 32 K-context model.
        effective_max_tokens = (
            max_tokens
            if max_tokens is not None
            else _MAX_TOKENS_BY_SUB_TIER[sub_tier]
        )
        self._max_tokens = effective_max_tokens
        # Store the construction params and build the smolagents
        # model fresh in every ``_build_agent`` call rather than
        # reusing one instance across concurrent agent sessions.
        # Smolagents' model classes (LiteLLMModel,
        # OpenAIModel, VLLMModel) are not documented as stateless
        # for concurrent generate calls — they typically carry
        # mutable last_input_token_count / messages buffers that
        # race when threads share an instance. Per-call
        # construction is ~10 ms (HTTP client setup), negligible
        # against tier-3 question latency of 30-180 s.
        self._smol_model_kwargs: dict[str, Any] = {
            "provider": provider,
            "model_id": provider_model_id,
            "api_key": api_key,
            "temperature": 0,
            "max_tokens": effective_max_tokens,
        }
        # KG tools are lightweight wrappers over the shared
        # ``EntityService`` (which itself wraps a ``TripleBridge``
        # Spark session) and are stateless across calls — built once
        # and shared. Smolagents' ``WebSearchTool``, however, is
        # **not** documented as stateless: its parser / session may
        # carry state that races under concurrent agent sessions in
        # ``predict_batch``. We therefore retain the construction
        # parameters and build *only the web tools* fresh per
        # ``_build_agent`` call when the sub-tier needs them.
        self._authorized_imports = SANDBOX_AUTHORIZED_IMPORTS
        self._kg_backend_for_tools = kg_backend
        self._web_search_timeout_seconds = web_search_timeout_seconds
        self._kg_tools = _build_kg_tools(kg_backend) if sub_tier in ("3a", "3c") else []

        logger.info(
            "SmolagentsModel created: variant={}, sub_tier={}, "
            "provider={}, model_id={}, kg_tools={}",
            variant_name,
            sub_tier,
            provider,
            provider_model_id,
            [t.name for t in self._kg_tools],
        )

    def _get_per_call_tools(self) -> list[Any]:
        """Build the per-call tool list for this sub-tier.

        Extracted from :meth:`_build_agent` so the predict path can
        also pass the same list to :func:`render_tools_section` and
        keep the prompt + agent tool inventories in sync.
        """
        per_call_tools: list[Any] = list(self._kg_tools)
        if self.sub_tier in ("3b", "3c"):
            per_call_tools.extend(
                _build_web_tools(self._web_search_timeout_seconds)
            )
        return per_call_tools

    def _build_agent(self, tools: list[Any] | None = None) -> Any:
        """Construct a fresh :class:`~smolagents.CodeAgent` for one
        ``predict()`` call.

        Called per-question rather than once at ``__init__`` so that
        concurrent ``predict_batch`` workers each get their own
        isolated agent, memory, and executor — smolagents mutates
        these during ``run()``. The smolagents model class
        (``LiteLLMModel`` / ``OpenAIModel`` / ``VLLMModel``) is also
        constructed per-call, since the upstream classes are not
        documented as thread-safe for shared use. Web tools are
        also constructed per-call (stateful in stock smolagents); KG
        tools are stateless wrappers over a shared ``EntityService``
        and are reused.

        ``tools`` may be supplied by the caller to keep the agent's
        tool list in sync with the prompt's ``$tools_section``. When
        omitted, the parent's default per-sub-tier tool list is used.
        """
        from smolagents import CodeAgent

        smol_model = _build_smolagents_model(**self._smol_model_kwargs)
        per_call_tools = (
            tools if tools is not None else self._get_per_call_tools()
        )

        from benchmark_models.tier3_logging import LoguruAgentLogger

        return CodeAgent(
            tools=per_call_tools,
            model=smol_model,
            max_steps=self.max_steps,
            additional_authorized_imports=self._authorized_imports,
            # Smolagents 1.24.0 defaults code_block_tags to
            # ``("<code>", "</code>")``. Most LLMs (Gemini, Claude,
            # GPT) zero-shot-emit Python in ```python ... ``` fences,
            # so that default produces ~94% parse errors on tier-3b
            # smoke runs. ``"markdown"`` selects fence tags and
            # matches the model's natural output format.
            code_block_tags="markdown",
            # Bridge Smolagents' rich-console output into loguru so
            # per-run ``run.log`` captures tier-3 agent traces (step
            # banners, code blobs, tool results) the same way it
            # captures tier-1/2 warnings.
            logger=LoguruAgentLogger(),
            # Prevent the static task image from being replayed on
            # every ActionStep — see _clear_replayed_step_image
            # docstring for the full mechanism + smolagents source
            # references. Required for ``-image`` variants under
            # vLLM ``--limit-mm-per-prompt '{"image":1}'``; no-op
            # otherwise so it's safe to always register.
            step_callbacks=[_clear_replayed_step_image],
        )

    def predict(self, example: Example) -> Prediction:
        """Run one example through the Smolagents CodeAgent."""
        try:
            template_info = self.registry.get(example.tid)
        except KeyError as exc:
            return self._error_prediction(
                example, sfid="?", answer_type="?",
                message=f"unknown tid {example.tid!r}: {exc}",
            )

        assert template_info.answer_type is not None

        # Build the task prompt for the agent.
        from benchmark_models.prompts import render_prompt

        # Use sub-tier-specific prompt if available (tier3b has its own
        # prompt optimized for CodeAgent). Fall back to generic tier3.
        prompt_variant = f"tier{self.sub_tier}"  # "tier3a", "tier3b", "tier3c"

        # Build the per-call tool list ONCE so the prompt's
        # ``$tools_section`` and the agent's ``CodeAgent.tools`` stay
        # in sync. A future tool ablation that overrides the loaded
        # tool set propagates to the prompt automatically.
        per_call_tools = self._get_per_call_tools()
        tools_section = render_tools_section(per_call_tools)

        # Build a fresh agent for this call. Isolates memory / state /
        # executor from concurrent :meth:`predict_batch` workers.
        agent = self._build_agent(per_call_tools)

        prompt = render_prompt(
            "tier3",
            example,
            template_info,
            tools_section=tools_section,
            max_tool_calls=self.max_steps,
            prompt_variant=prompt_variant,
        )

        # When multimodal_mode == "image" and the row has images,
        # load them as PIL.Image objects and pass to agent.run(). The
        # underlying LiteLLM/smolagents bridge translates them to the
        # provider's content-block shape. For "name" mode the image
        # placeholders are substituted in the prompt via apply_name_mode
        # (rendered above by render_prompt); no images attached here.
        images = None
        if self.multimodal_mode == "image" and example.images:
            from benchmark_models.multimodal import load_pil_images

            images = load_pil_images(example)

        # Run the agent.
        t0 = time.perf_counter()
        try:
            result = retry_with_backoff(
                lambda: agent.run(prompt, images=images) if images else agent.run(prompt),
                provider="smolagents",
                question_id=example.question_id,
            )
        except Exception as exc:
            logger.warning(
                "SmolagentsModel: agent.run() failed for "
                "question_id={}: {}",
                example.question_id,
                exc,
            )
            return self._error_prediction(
                example,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                message=f"agent.run() failed: {exc}",
                error_class=classify_error(exc),
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract the execution trace from agent memory.
        tool_calls = self._extract_trace(agent)

        # Detect cap-hit: if the agent used all max_steps without
        # calling final_answer(), the result is whatever the last
        # step produced (often raw code or an intermediate value).
        hit_cap = self._detect_cap_hit(agent)

        if hit_cap:
            return Prediction(
                question_id=example.question_id,
                tid=example.tid,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                model_id=self.model_id,
                tier=self.tier,
                language=example.language,
                prompt_template_id=f"tier3.{self.sub_tier}",
                seed=self.seed,
                snapshot_id=self.snapshot_id,
                answer=None,
                unit=None,
                refusal=True,
                cap_hit=True,
                final_state="cap_hit",
                tool_calls=tool_calls if tool_calls else None,
                usage=self._extract_usage(agent),
                latency_ms=latency_ms,
                raw_completion=str(result)[:500],
            )

        # Detect "no final answer": agent finished cleanly without
        # ever registering a final_answer call. Common cause: model
        # wrapped the call inside a function definition. Without
        # this check the recorded answer would be the raw Python
        # source of the last step and the eval would mis-classify
        # it as a format error.
        if not self._has_proper_final_answer(agent):
            return Prediction(
                question_id=example.question_id,
                tid=example.tid,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                model_id=self.model_id,
                tier=self.tier,
                language=example.language,
                prompt_template_id=f"tier3.{self.sub_tier}",
                seed=self.seed,
                snapshot_id=self.snapshot_id,
                answer=None,
                unit=None,
                refusal=True,
                cap_hit=False,
                final_state="no_final_answer",
                tool_calls=tool_calls if tool_calls else None,
                usage=self._extract_usage(agent),
                latency_ms=latency_ms,
                raw_completion=str(result)[:500],
            )

        # Parse the result.
        answer, unit, refusal, malformed = self._parse_result(
            result, template_info
        )

        # Extract usage from the agent's logs.
        usage = self._extract_usage(agent)

        if malformed:
            final_state = "no_final_answer"
        elif refusal:
            final_state = "abstained"
        else:
            final_state = "answered"

        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=template_info.sfid,
            answer_type=template_info.answer_type,
            model_id=self.model_id,
            tier=self.tier,
            language=example.language,
            prompt_template_id=f"tier3.{self.sub_tier}",
            seed=self.seed,
            snapshot_id=self.snapshot_id,
            answer=answer,
            unit=unit,
            refusal=refusal,
            final_state=final_state,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            latency_ms=latency_ms,
            raw_completion=str(result),
        )

    def _parse_result(
        self, result: Any, template_info: Any
    ) -> tuple[Any, str | None, bool, bool]:
        """Parse the agent's final answer.

        Returns a 4-tuple ``(answer, unit, refusal, malformed)``:

        * ``refusal`` is True when the answer is ``None`` for any reason
          — an explicit abstention or an unparseable output.
        * ``malformed`` distinguishes the two cases. When True, the
          caller emits ``final_state="no_final_answer"`` (the model
          produced output but it didn't match the template's answer
          type). When False, the caller uses ``"abstained"``
          (explicit null) or ``"answered"`` (valid answer).

        Three-layer parse:

        1. :meth:`_raw_parse` unwraps the payload. When the model
           returns a dict that *has* an ``"answer"`` key, we use it.
           A dict without an ``"answer"`` key returns the sentinel
           :data:`_MISSING_ANSWER` so this method can distinguish
           *"model explicitly abstained"* (``answer`` key present but
           ``None``) from *"model missed the schema envelope"* (no
           ``answer`` key at all). Example: a Coordinates model that
           emits ``{"lat": ..., "lon": ...}`` is a schema miss, not
           an abstention — recovery below rescues it.
        2. :meth:`_recover_from_envelope_free` attempts type-aware
           recovery for known alternative shapes. For Coordinate
           templates we accept bare ``{"lat": ..., "lon": ...}``
           dicts; for Numeric we accept single-number dicts like
           ``{"value": 4.24}``. On no-match we return the sentinel
           so the caller treats the prediction as malformed.
        3. :meth:`_validate_against_type` does the final type check
           (booleans accept yes/no/true/false, numerics float-parse,
           H3 matches the 15-char hex pattern, coordinates parse as
           a lat/lon pair).
        """
        answer, unit = self._raw_parse(result)

        if answer is _MISSING_ANSWER:
            # Model returned a dict without an ``answer`` key. Try
            # type-aware recovery before flagging as malformed.
            recovered = self._recover_from_envelope_free(result, template_info)
            if recovered is _MISSING_ANSWER:
                return None, None, True, True
            answer = recovered

        if answer is None:
            # Explicit abstention: ``final_answer(None)`` or
            # ``final_answer({"answer": None})``.
            return None, unit, True, False

        validated, refused, malformed = self._validate_against_type(
            answer, template_info
        )
        if malformed:
            return None, None, True, True
        if refused:
            return None, unit, True, False
        # Numeric types without a unit inherit the template's default.
        if unit is None and template_info.answer_type and "Numeric" in template_info.answer_type:
            unit = template_info.answer_unit
        return validated, unit, False, False

    @staticmethod
    def _recover_from_envelope_free(
        payload: Any, template_info: Any
    ) -> Any:
        """Type-aware recovery from model-produced dicts that don't
        carry the canonical ``{"answer": ..., "unit": ...}`` envelope.

        Kept deliberately conservative: we only recover patterns that
        obviously match the declared answer type. When in doubt we
        return :data:`_MISSING_ANSWER` so the caller marks the
        prediction malformed — the honest classification when we
        can't confidently tell whether the model answered or just
        emitted a dict of intermediate state.

        Returns the recovered raw answer on success (still subject to
        :meth:`_validate_against_type`), or :data:`_MISSING_ANSWER`
        on no-match.
        """
        if not template_info:
            return _MISSING_ANSWER
        # Accept the same two shapes that ``_raw_parse`` accepts:
        # a raw dict, or a JSON-string that decodes to a dict.
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload.strip())
            except json.JSONDecodeError:
                return _MISSING_ANSWER
            if not isinstance(decoded, dict):
                return _MISSING_ANSWER
            payload = decoded
        if not isinstance(payload, dict):
            return _MISSING_ANSWER
        answer_type = template_info.answer_type or ""

        if "Coordinates" in answer_type:
            # Accept ``{"lat": X, "lon": Y}``, ``{"latitude": X,
            # "longitude": Y}``, and the short ``lng`` alias. The
            # subsequent ``_coerce_coordinates`` call validates the
            # numeric ranges.
            lat_keys = ("lat", "latitude")
            lon_keys = ("lon", "long", "lng", "longitude")
            if any(k in payload for k in lat_keys) and any(
                k in payload for k in lon_keys
            ):
                return payload

        if "Numeric" in answer_type:
            # Accept single-number payloads like ``{"value": 4.24}``
            # or ``{"distance_km": 4.24}`` / ``{"count": 99}``.
            # Strategy: prefer a known answer-bearing key when one
            # is present (``answer``, ``value``, ``distance_km``,
            # ``area_km2``, ``count``, ``result``); otherwise require
            # exactly one numeric field to avoid picking the wrong
            # number from a dict of intermediate state. The preferred
            # path lets payloads like
            # ``{"distance_km": 4.24, "confidence": 0.95}`` still
            # recover correctly — Claude in particular often appends
            # auxiliary numeric metadata.
            #
            # When no preferred key is present and the payload has
            # 2+ numeric fields, refuse to guess: only the
            # unambiguous ``len(numeric_fields) == 1`` case is
            # recovered. Failing fast here makes the
            # intermediate-state-dict failure mode visible in error
            # analysis instead of scoring an arbitrary number.
            preferred_keys = (
                "answer", "value", "distance_km", "area_km2",
                "count", "result", "answer_value",
            )
            for k in preferred_keys:
                if k in payload:
                    v = payload[k]
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        return v
            numeric_fields = [
                (k, v)
                for k, v in payload.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if len(numeric_fields) == 1:
                return numeric_fields[0][1]
            # 2+ ambiguous numeric fields with no preferred-key
            # match: refuse rather than guess. Prediction will be
            # marked malformed, which is the honest classification
            # when we can't confidently tell answer-from-state.

        # Booleans and Entity-hex intentionally do NOT recover:
        # a dict of any shape is unlikely to be a coherent boolean
        # or H3 index answer. Flag malformed and move on.
        return _MISSING_ANSWER

    @staticmethod
    def _raw_parse(result: Any) -> tuple[Any, str | None]:
        """First pass: unwrap smolagents' result into ``(answer, unit)``.

        The CodeAgent ultimately delivers whatever the model passed to
        ``final_answer(...)``. In practice this is either (a) a dict
        with ``answer`` / ``unit`` keys, (b) a JSON-string of the same
        shape, or (c) a raw scalar / bare string when the model took
        a shortcut. Later validation decides whether the unwrapped
        payload is type-correct.

        Returns :data:`_MISSING_ANSWER` in the ``answer`` slot when
        the model returned a dict without an ``"answer"`` key —
        distinguishes schema-miss from explicit abstention.
        """
        if result is None:
            return None, None
        if isinstance(result, dict):
            return (
                result.get("answer", _MISSING_ANSWER),
                result.get("unit"),
            )
        if isinstance(result, str):
            result_str = result.strip()
            try:
                data = json.loads(result_str)
                if isinstance(data, dict):
                    return (
                        data.get("answer", _MISSING_ANSWER),
                        data.get("unit"),
                    )
            except json.JSONDecodeError:
                pass
            return result_str, None
        return result, None

    # Hex-only H3 cell IDs, as used by the ``Entity hex string
    # (single)`` canonical. Decimal / signed-int forms are accepted via
    # ``_canonicalize_h3_hex`` for the H3 / S2 cell-index canonicals.
    _H3_RE = re.compile(r"^[0-9a-f]{14,16}$")

    # Closed enums for the predefined / cardinal / ternary types.
    _CARDINAL_VALUES: frozenset[str] = frozenset({
        "north", "south", "east", "west",
        "northeast", "northwest", "southeast", "southwest",
    })
    _TERNARY_VALUES: frozenset[str] = frozenset({"true", "false", "plausibly"})

    def _validate_against_type(
        self, answer: Any, template_info: Any
    ) -> tuple[Any, bool, bool]:
        """Validate ``answer`` against the template's declared answer type.

        Returns ``(validated, refused, malformed)``:

        * ``validated`` — the coerced answer on success, else ``None``.
        * ``refused`` — True only when the model *explicitly* returned
          a null (currently unreachable because ``_raw_parse`` catches
          that before we get here; retained for forward symmetry).
        * ``malformed`` — True when the answer is present but doesn't
          match the declared type. The caller converts this to
          ``final_state="no_final_answer"``.

        Dispatch covers every canonical answer type currently in the
        registry. Unknown types fall through to accept-as-is so a new
        type added downstream doesn't silently score as malformed
        before its tier-3 validator catches up.
        """
        answer_type = (template_info.answer_type or "") if template_info else ""

        if "Boolean" in answer_type:
            coerced = _coerce_boolean(answer)
            if coerced is None:
                return None, False, True
            return coerced, False, False

        if "Numeric" in answer_type:
            try:
                if isinstance(answer, bool):
                    # ``bool`` is a subclass of ``int``; reject True/False
                    # as numeric to avoid spurious 1.0 / 0.0 answers.
                    return None, False, True
                if isinstance(answer, (int, float)):
                    return float(answer), False, False
                if isinstance(answer, str):
                    return float(answer.strip()), False, False
                return None, False, True
            except (ValueError, TypeError):
                return None, False, True

        if answer_type == "Entity hex string (single)":
            # Hex-only H3 form. Decimal / signed-int forms belong
            # to the ``H3 cell index`` / ``S2 cell index``
            # canonicals below.
            if isinstance(answer, str) and self._H3_RE.match(answer.strip().lower()):
                return answer.strip().lower(), False, False
            return None, False, True

        if answer_type in {"H3 cell index", "S2 cell index"}:
            # Accept hex / decimal-uint64 / signed-int64-string forms;
            # the scorer's canonicalizer handles all three. Reject
            # bare small ints — H3 / S2 cells are 64-bit values, so a
            # decimal less than 2^40 is definitely not a real cell id
            # (the eval-side ``_canonicalize_h3_hex`` validator would
            # reject these too, but failing earlier here lets the
            # tier-3 trace surface the issue as a malformed answer).
            if isinstance(answer, (int, str)):
                s = str(answer).strip()
                if re.match(r"^-?\d+$", s):
                    try:
                        n = int(s)
                    except ValueError:
                        return None, False, True
                    if abs(n) < (1 << 40):
                        # Definitely not a 64-bit cell id.
                        return None, False, True
                    return s, False, False
                if self._H3_RE.match(s.lower()):
                    return s, False, False
            return None, False, True

        if answer_type == "Geohash string":
            if isinstance(answer, str) and re.match(
                r"^[0-9a-z]{4,12}$", answer.strip().lower()
            ):
                return answer.strip().lower(), False, False
            return None, False, True

        if answer_type.startswith("Composite string"):
            # ``H3(res=15): X | S2(lvl=20): Y`` — must contain the
            # pipe separator. Order-insensitive equality is the
            # scorer's job; tier-3 only validates shape.
            if isinstance(answer, str) and "|" in answer:
                return answer.strip(), False, False
            return None, False, True

        if answer_type == "Date":
            # ISO 8601 ``YYYY-MM-DD`` (or with time component).
            if isinstance(answer, str) and re.match(
                r"^\d{4}-\d{2}-\d{2}", answer.strip()
            ):
                return answer.strip(), False, False
            return None, False, True

        if answer_type.startswith("Ternary"):
            if isinstance(answer, str) and answer.strip().lower() in self._TERNARY_VALUES:
                return answer.strip().lower(), False, False
            if isinstance(answer, bool):
                return "true" if answer else "false", False, False
            return None, False, True

        if answer_type.startswith("Cardinal direction"):
            if isinstance(answer, str) and answer.strip().lower() in self._CARDINAL_VALUES:
                return answer.strip().lower(), False, False
            return None, False, True

        if answer_type == "Predefined value (left, right)":
            if isinstance(answer, str) and answer.strip().lower() in {"left", "right"}:
                return answer.strip().lower(), False, False
            return None, False, True

        if answer_type == "Predefined value (upstream, downstream)":
            if isinstance(answer, str) and answer.strip().lower() in {
                "upstream",
                "downstream",
            }:
                return answer.strip().lower(), False, False
            return None, False, True

        if answer_type.startswith("Set enumeration"):
            # Lists pass through; the scorer normalizes element by
            # element. Reject scalars so the agent's reply gets a
            # ``no_final_answer`` rather than a bogus single-element
            # set.
            if isinstance(answer, list):
                return answer, False, False
            return None, False, True

        if answer_type == "Coordinates (lat/long pair)":
            coerced = _coerce_coordinates(answer)
            if coerced is None:
                return None, False, True
            return coerced, False, False

        if answer_type.startswith("Polar coordinates"):
            # Dict ``{angle_degrees, distance_km}``, 2-list, or
            # ``"θ degrees, r km"`` string. Tier-3 only shape-checks;
            # the scorer's ``_coerce_polar`` handles canonicalization.
            if isinstance(answer, dict) and (
                "angle_degrees" in answer or "angle" in answer or "theta" in answer
            ):
                return answer, False, False
            if isinstance(answer, (list, tuple)) and len(answer) == 2:
                return answer, False, False
            if isinstance(answer, str) and "," in answer:
                return answer.strip(), False, False
            return None, False, True

        if answer_type.startswith("Bounding box"):
            if isinstance(answer, str) and answer.count(",") == 3:
                return answer.strip(), False, False
            if isinstance(answer, (list, tuple)) and len(answer) == 4:
                return answer, False, False
            if isinstance(answer, dict) and {
                "lat_min", "lon_min", "lat_max", "lon_max"
            }.issubset(answer.keys()):
                return answer, False, False
            return None, False, True

        # Unknown / free-form types: accept as-is. Keeping the
        # permissive branch last means future type additions do not
        # silently score as malformed.
        return answer, False, False

    def _detect_cap_hit(self, agent: Any) -> bool:
        """Check if the agent hit the step cap without calling final_answer.

        When the agent exhausts max_steps, it doesn't call final_answer —
        it just stops. The last step's is_final_answer will be False, and
        the result is whatever the last code block produced (often raw
        code or an intermediate value, not a proper answer).
        """
        try:
            steps = getattr(agent, "memory", None)
            if steps is None:
                return False
            action_steps = [
                s for s in steps.steps
                if getattr(s, "code_action", None) is not None
            ]
            if not action_steps:
                return False
            # If the last action step didn't call final_answer and
            # we used all available steps, it's a cap hit.
            last = action_steps[-1]
            if getattr(last, "is_final_answer", False):
                return False
            return len(action_steps) >= self.max_steps
        except Exception:
            return False

    def _has_proper_final_answer(self, agent: Any) -> bool:
        """Check whether *any* action step ever called ``final_answer()``.

        Distinguished from :meth:`_detect_cap_hit` which only fires
        when the agent exhausted its step budget. This catches the
        case where the agent finished cleanly under the cap but never
        registered a final-answer call — most often because the model
        wrote ``def solve(): final_answer(...); solve()``: the call
        happens at execution time inside a function frame, but the
        smolagents executor only flags ``is_final_answer=True`` on
        top-level invocations.

        Returns ``False`` if no action step has ``is_final_answer=True``,
        or if the agent has no action steps at all (silent failure).
        Returns ``True`` exactly when at least one action step
        registered ``final_answer``.
        """
        try:
            steps = getattr(agent, "memory", None)
            if steps is None:
                return False
            action_steps = [
                s for s in steps.steps
                if getattr(s, "code_action", None) is not None
            ]
            if not action_steps:
                return False
            return any(
                getattr(s, "is_final_answer", False) for s in action_steps
            )
        except Exception:
            return False

    def _extract_trace(self, agent: Any) -> list[ToolCall]:
        """Extract the execution trace from the agent's memory.

        Smolagents stores steps in ``agent.memory.steps`` — a list of
        ``ActionStep`` / ``TaskStep`` / ``PlanningStep`` objects.
        ``ActionStep`` has: ``code_action``, ``observations``,
        ``error``, ``is_final_answer``, ``token_usage``.
        """
        trace: list[ToolCall] = []
        try:
            steps = getattr(agent, "memory", None)
            if steps is None:
                return trace
            for step in steps.steps:
                # Only ActionStep has code_action; skip TaskStep/PlanningStep
                code = getattr(step, "code_action", None)
                if code is None:
                    continue
                observations = getattr(step, "observations", None) or ""
                error = getattr(step, "error", None)
                is_final = getattr(step, "is_final_answer", False)

                trace.append(ToolCall(
                    name="final_answer" if is_final else "code_step",
                    args={"code": str(code)[:2000]},
                    result=str(observations)[:2000],
                    error=str(error) if error else None,
                ))
        except Exception as exc:
            logger.warning("Failed to extract agent trace: {}", exc)
        return trace

    def _extract_usage(self, agent: Any) -> Usage:
        """Extract token usage from the agent's memory.

        Each ``ActionStep`` has a ``token_usage`` attribute with
        ``input_tokens`` and ``output_tokens``.

        ``tool_call_count`` is the count of action steps that did NOT
        terminate with ``final_answer``: i.e., real tool-call /
        compute steps. On a normal answered run the last step is the
        ``final_answer`` step and is excluded; on ``cap_hit`` /
        ``no_final_answer`` runs every action step was a real
        tool call and the count equals ``step_count``. A blanket
        ``max(0, step_count - 1)`` would undercount by one on
        exactly those failure cases, which is where the tool-use
        analysis most needs the count to be exact.
        """
        total_input = 0
        total_output = 0
        step_count = 0
        last_was_final = False
        try:
            steps = getattr(agent, "memory", None)
            if steps is None:
                return Usage()
            for step in steps.steps:
                # Only count ActionSteps (which have token_usage)
                tu = getattr(step, "token_usage", None)
                if tu is None:
                    continue
                step_count += 1
                total_input += getattr(tu, "input_tokens", 0) or 0
                total_output += getattr(tu, "output_tokens", 0) or 0
                last_was_final = bool(getattr(step, "is_final_answer", False))
        except Exception as exc:
            logger.warning("Failed to extract agent usage: {}", exc)
        tool_call_count = step_count - 1 if last_was_final else step_count
        return Usage(
            input_tokens=total_input,
            output_tokens=total_output,
            api_call_count=step_count,
            tool_call_count=max(0, tool_call_count),
        )

    def _error_prediction(
        self,
        example: Example,
        *,
        sfid: str,
        answer_type: str,
        message: str,
        error_class: str = "config",
        triple_context: str | None = None,
    ) -> Prediction:
        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=sfid,
            answer_type=answer_type,
            model_id=self.model_id,
            tier=self.tier,
            language=example.language,
            prompt_template_id="error",
            seed=self.seed,
            snapshot_id=self.snapshot_id,
            answer=None,
            refusal=True,
            final_state="error",
            error_class=error_class,  # type: ignore[arg-type]
            usage=Usage(api_call_count=0),
            latency_ms=0.0,
            raw_completion=f"SmolagentsModel error: {message}",
            triple_context=triple_context,
        )

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        """Run examples with bounded thread-pool concurrency.

        Lower concurrency than tier 1/2 backends (default 3 vs 8)
        because each tier-3 question makes multiple internal API calls
        via the agent loop. Too many parallel sessions would trigger
        rate limits.
        """
        from concurrent.futures import ThreadPoolExecutor

        if len(examples) <= 1:
            return [self.predict(ex) for ex in examples]

        max_workers = min(self._max_concurrent, len(examples))
        logger.info(
            "predict_batch (tier 3): {} examples with {} concurrent "
            "agent sessions",
            len(examples),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            predictions = list(pool.map(self.predict, examples))
        return predictions

    def close(self) -> None:
        """Release resources."""
        pass


class SmolagentsOracleModel(SmolagentsModel):
    """Tier-3 oracle variant: gold triples injected, no KG/web tools.

    The "compute-only ceiling" for spatial QA. The model receives the
    supporting triples directly in the prompt (so retrieval is perfect)
    and has only the Python sandbox — no ``find_entity`` / ``get_entity``
    / web-search tools. Measures whether the model can write correct
    spatial code given perfect data.

    Differences from :class:`SmolagentsModel`:

    * Constructed with an empty tool list (``tools=[]``). The CodeAgent
      can still run ``final_answer`` (smolagents provides that
      automatically) plus any sandbox library calls.
    * Renders the ``tier3_oracle`` prompt template with the gold
      triples formatted in structured form.
    * Requires a ``gold_triples_map`` at construction; predicting a
      question whose id is not in the map raises ``LookupError`` (the
      :func:`benchmark_models.oracle.assert_coverage` check should
      already have caught this before the first predict call).

    The ``gold_triples_map`` is passed at construction — not attached
    to :class:`Example` — so non-oracle code paths remain structurally
    unable to access it.
    """

    def __init__(
        self,
        variant_name: str,
        *,
        provider: str,
        provider_model_id: str,
        gold_triples_map: dict[str, Any],
        api_key: str | None = None,
        registry: TemplateRegistry | None = None,
        snapshot_id: str = "unset",
        max_steps: int = DEFAULT_MAX_TOOL_CALLS,
        seed: int = 42,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        max_tokens: int | None = None,
        multimodal_mode: "MultimodalMode" = "off",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """Construct a tier-3 oracle Smolagents model.

        Args:
            max_tokens: Per-step LLM output cap. ``None`` falls back
                to tier-3a's per-sub-tier default (4096) — gold WKT
                polygons are pre-simplified by ``OracleContext``'s
                ``wkt_simplify_tolerance``, so the agent rarely needs
                to re-emit large coordinate lists.
            max_concurrent: Concurrent agent sessions in
                :meth:`predict_batch`.
            multimodal_mode: Same as :class:`SmolagentsModel`. No
                multimodal oracle variants ship today but the field
                exists for symmetry.
        """
        from benchmark_models.config import MultimodalMode  # local to avoid cycle

        if multimodal_mode not in ("off", "image", "name"):
            raise ValueError(
                f"Unknown multimodal_mode {multimodal_mode!r} — "
                f"must be 'off', 'image', or 'name'."
            )
        self.multimodal_mode: MultimodalMode = multimodal_mode
        self.model_id = variant_name
        self.tier = "tier3"
        self.sub_tier = "oracle"  # not a real sub_tier literal; used only for logging
        self.registry = registry if registry is not None else load_template_registry()
        self.snapshot_id = snapshot_id
        self.max_steps = max_steps
        self.seed = seed
        self._max_concurrent = max_concurrent
        self._kg_backend = None  # no KG access in the oracle setting
        self._gold_triples_map = gold_triples_map

        # Tier-3 pins ``temperature=0`` (see SmolagentsModel for the
        # rationale). Oracle defaults ``max_tokens`` to the tier-3a
        # value (4096). The model receives WKT polygons via
        # OracleContext, which simplifies them up-front when
        # ``wkt_simplify_tolerance`` is set, so 4096 is enough headroom
        # for the typical shapely.distance / shapely.area call.
        effective_max_tokens = (
            max_tokens
            if max_tokens is not None
            else _MAX_TOKENS_BY_SUB_TIER["3a"]
        )
        self._max_tokens = effective_max_tokens
        # Per-call smolagents model construction. See the parent's
        # ``__init__`` for rationale.
        self._smol_model_kwargs: dict[str, Any] = {
            "provider": provider,
            "model_id": provider_model_id,
            "api_key": api_key,
            "temperature": 0,
            "max_tokens": effective_max_tokens,
        }
        # Oracle: no retrieval tools. The Python sandbox + smolagents'
        # built-in final_answer are the only capabilities. The
        # parent's ``_build_agent`` would otherwise try to construct
        # KG / web tools — we override below to bypass it.
        self._authorized_imports = SANDBOX_AUTHORIZED_IMPORTS
        self._kg_tools = []
        self._web_search_timeout_seconds = _DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS

        logger.info(
            "SmolagentsOracleModel created: variant={}, provider={}, "
            "model_id={}, gold_triples_size={}",
            variant_name,
            provider,
            provider_model_id,
            len(gold_triples_map),
        )

    def _build_agent(self, tools: list[Any] | None = None) -> Any:
        """Build a tools-empty CodeAgent for oracle inference.

        Overrides :meth:`SmolagentsModel._build_agent` because the
        parent's per-call factory would attempt to construct KG / web
        tools — incompatible with the oracle's "compute-only" design.
        The smolagents model is built per-call too, for the same
        thread-safety reason as in the parent.

        ``tools`` argument is accepted for parent-class signature
        parity but ignored: the oracle agent is always tools-empty
        by design.

        ``tool_call_count`` on resulting predictions reflects
        Python-only steps (no KG / web tools are wired in this
        configuration); in particular, every action step the model
        runs is a sandbox compute step, not a retrieval call.
        """
        from smolagents import CodeAgent

        from benchmark_models.tier3_logging import LoguruAgentLogger

        del tools  # signature parity only
        smol_model = _build_smolagents_model(**self._smol_model_kwargs)
        return CodeAgent(
            tools=[],
            model=smol_model,
            max_steps=self.max_steps,
            additional_authorized_imports=self._authorized_imports,
            code_block_tags="markdown",
            # See SmolagentsModel._build_code_agent for rationale.
            logger=LoguruAgentLogger(),
            # See _clear_replayed_step_image for rationale —
            # required for ``-image`` oracle variants.
            step_callbacks=[_clear_replayed_step_image],
        )

    def predict(self, example: Example) -> Prediction:
        """Run one example through the oracle agent.

        Overrides the base class to render the ``tier3_oracle`` prompt
        with gold triples injected rather than the sub-tier-specific
        KG/web prompts.
        """
        try:
            template_info = self.registry.get(example.tid)
        except KeyError as exc:
            return self._error_prediction(
                example, sfid="?", answer_type="?",
                message=f"unknown tid {example.tid!r}: {exc}",
            )

        assert template_info.answer_type is not None

        if example.question_id not in self._gold_triples_map:
            # assert_coverage should have caught this — fail loud.
            return self._error_prediction(
                example,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                message=(
                    f"T3-oracle: no gold triples for question_id="
                    f"{example.question_id!r}. assert_coverage missed "
                    f"this — check run-start validation."
                ),
            )

        from benchmark_models.oracle import format_triples
        from benchmark_models.prompts import render_prompt

        triples = self._gold_triples_map[example.question_id]
        # Compute ``triple_context`` *before* the agent.run() try block
        # so any failure path can preserve the audit trail of
        # "which gold triples did the model see?". Errors that occur
        # before this point (registry miss, missing gold) get
        # ``triple_context=None`` because there's nothing to record.
        triple_context = format_triples(triples, mode="t3")
        prompt = render_prompt(
            "tier3",
            example,
            template_info,
            prompt_variant="tier3_oracle",
            max_tool_calls=self.max_steps,
            triple_context=triple_context,
        )

        # Fresh agent per call — see ``SmolagentsModel.__init__`` for
        # why this matters under concurrent ``predict_batch``.
        agent = self._build_agent()

        # Same image plumbing as SmolagentsModel.predict. Oracle
        # variants have no multimodal entries in models.yaml, but the
        # code path is symmetric.
        images = None
        if self.multimodal_mode == "image" and example.images:
            from benchmark_models.multimodal import load_pil_images

            images = load_pil_images(example)

        t0 = time.perf_counter()
        try:
            result = retry_with_backoff(
                lambda: agent.run(prompt, images=images) if images else agent.run(prompt),
                provider="smolagents",
                question_id=example.question_id,
            )
        except Exception as exc:
            logger.warning(
                "SmolagentsOracleModel: agent.run() failed for "
                "question_id={}: {}",
                example.question_id,
                exc,
            )
            return self._error_prediction(
                example,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                message=f"agent.run() failed: {exc}",
                error_class=classify_error(exc),
                triple_context=triple_context,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        tool_calls = self._extract_trace(agent)
        hit_cap = self._detect_cap_hit(agent)

        if hit_cap:
            return Prediction(
                question_id=example.question_id,
                tid=example.tid,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                model_id=self.model_id,
                tier=self.tier,
                language=example.language,
                prompt_template_id="tier3.oracle",
                seed=self.seed,
                snapshot_id=self.snapshot_id,
                answer=None,
                unit=None,
                refusal=True,
                cap_hit=True,
                final_state="cap_hit",
                tool_calls=tool_calls if tool_calls else None,
                triple_context=triple_context,
                usage=self._extract_usage(agent),
                latency_ms=latency_ms,
                raw_completion=str(result)[:500],
            )

        if not self._has_proper_final_answer(agent):
            return Prediction(
                question_id=example.question_id,
                tid=example.tid,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                model_id=self.model_id,
                tier=self.tier,
                language=example.language,
                prompt_template_id="tier3.oracle",
                seed=self.seed,
                snapshot_id=self.snapshot_id,
                answer=None,
                unit=None,
                refusal=True,
                cap_hit=False,
                final_state="no_final_answer",
                tool_calls=tool_calls if tool_calls else None,
                triple_context=triple_context,
                usage=self._extract_usage(agent),
                latency_ms=latency_ms,
                raw_completion=str(result)[:500],
            )

        answer, unit, refusal, malformed = self._parse_result(
            result, template_info
        )
        usage = self._extract_usage(agent)

        if malformed:
            final_state = "no_final_answer"
        elif refusal:
            final_state = "abstained"
        else:
            final_state = "answered"

        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=template_info.sfid,
            answer_type=template_info.answer_type,
            model_id=self.model_id,
            tier=self.tier,
            language=example.language,
            prompt_template_id="tier3.oracle",
            seed=self.seed,
            snapshot_id=self.snapshot_id,
            answer=answer,
            unit=unit,
            refusal=refusal,
            final_state=final_state,
            tool_calls=tool_calls if tool_calls else None,
            triple_context=triple_context,
            usage=usage,
            latency_ms=latency_ms,
            raw_completion=str(result),
        )
