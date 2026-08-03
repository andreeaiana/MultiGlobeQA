"""vLLM-over-HTTP backend.

The SLURM template starts a per-cell ``vllm serve`` in the background,
waits for ``/health``, then runs the smoke against
``http://localhost:<port>/v1``. Client-side this looks identical to
the OpenAI Chat Completions API — we subclass :class:`OpenAIModel`
and just override the OpenAI client's ``base_url``.

Why HTTP rather than in-process: (1) uniform OpenAI-compat surface
across providers, (2) vLLM's continuous batching benefits more from
N concurrent HTTP requests than from one in-process caller doing
batched calls, and (3) Helix (RHEL 8, glibc 2.28) can't load any
in-tree vLLM wheel — the SLURM template wraps ``vllm serve`` in a
Singularity container that ships its own glibc. See
``scripts/benchmark_models/generate_slurm.py`` for the templates.

This backend does **not** start the serve itself; the SLURM script
does. Local development can point at any reachable ``vllm serve``
via ``base_url`` on the variant config.
"""

from __future__ import annotations

# Soft-import the openai SDK — same pattern as ``backends/openai.py``.
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised by env without extra
    OpenAI = None  # type: ignore[assignment,misc]

# httpx is used for the lightweight health-check probe (no SDK overhead).
try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

from benchmark_models.backends.openai import OpenAIModel
from benchmark_models.config import (
    MultimodalMode,
    VLLMHttpVariantConfig,
)
from benchmark_models.prompts import PromptRenderer
from benchmark_models.schemas import TemplateRegistry


class VLLMHttpModel(OpenAIModel):
    """vLLM-served model accessed via OpenAI-compatible HTTP.

    Subclass of :class:`OpenAIModel` — the chat-completions request /
    response shape and ``usage`` block parsing are identical. The only
    overrides are:

    * ``__init__`` — construct the OpenAI client with ``base_url``
      from the variant config. ``VLLMHttpVariantConfig.model_id`` is
      a derived property over ``model_path``, so the parent's
      request-shaping code (which reads ``self.config.model_id``)
      resolves to the HF repo path that ``vllm serve`` is hosting.
    * No JSON-mode prompt-token injection — vLLM does not require
      ``"json"`` to appear in the message body.
    """

    def __init__(
        self,
        variant_name: str,
        config: VLLMHttpVariantConfig,
        *,
        registry: TemplateRegistry | None = None,
        api_key: str | None = None,
        snapshot_id: str = "unset",
        max_concurrent: int = 8,
        prompt_renderer: PromptRenderer | None = None,
        multimodal_mode: MultimodalMode = "off",
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "The openai SDK is required for VLLMHttpModel. "
                "Install with: uv sync --group benchmark_models"
            )

        # vLLM serve doesn't require an API key by default; pass a
        # dummy when the user hasn't provided one. Some serves are
        # launched with --api-key, in which case the user can set
        # ``api_key`` explicitly on the constructor.
        effective_key = api_key or "EMPTY"

        # Build a synthetic OpenAIVariantConfig-shaped object for the
        # parent ``__init__`` (which expects ``config.model_id``,
        # ``config.temperature``, etc.). The vLLM config carries
        # ``model_path`` instead of ``model_id`` — alias here so the
        # parent's request-shaping code resolves correctly without a
        # second code path.
        super().__init__(
            variant_name=variant_name,
            config=config,  # type: ignore[arg-type]
            registry=registry,
            api_key=effective_key,
            snapshot_id=snapshot_id,
            max_concurrent=max_concurrent,
            prompt_renderer=prompt_renderer,
            multimodal_mode=multimodal_mode,
        )

        # Replace the default OpenAI client with one pointed at the
        # vLLM serve. ``effective_base_url`` resolves to
        # ``http://localhost:<vllm_port>/v1`` by default, or to the
        # explicit ``base_url`` override when set — single source of
        # truth shared with the SLURM template.
        self.client = OpenAI(
            api_key=effective_key, base_url=config.effective_base_url,
        )

    @classmethod
    def health_check(cls, base_url: str, timeout: float = 1.0) -> bool:
        """Return ``True`` when the vLLM serve at ``base_url`` answers
        a ``GET /v1/models`` request within ``timeout`` seconds.

        Used by callers that want to fail fast if the serve never
        came up. The SLURM template handles its own health-poll via
        ``curl``; this method is for in-process / interactive use.
        """
        if httpx is None:
            raise ImportError("httpx is required for health_check")
        try:
            response = httpx.get(
                f"{base_url.rstrip('/')}/models", timeout=timeout,
            )
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False


__all__ = ["VLLMHttpModel"]
