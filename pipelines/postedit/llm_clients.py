"""LLM client implementations: Gemini, DeepSeek, and vLLM (Qwen).

DeepSeek's API is OpenAI-compatible — we use the official ``openai``
SDK with ``base_url=https://api.deepseek.com`` and a DEEPSEEK_API_KEY
bearer token. ``deepseek-v4-flash`` is the production grammar/JSON
model; aliases ``deepseek-chat`` (non-thinking) and
``deepseek-reasoner`` (thinking) are deprecated 2026-07-24. Thinking
mode is OFF by default — we don't pass ``thinking`` or
``reasoning_effort`` so we avoid the gpt-5-mini-style output-token
explosion.
"""

import hashlib
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    GEMINI_MODEL,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_BASE,
    TEMPERATURE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
)


class LLMError(Exception):
    """Raised when an LLM call fails after retries."""

    def __init__(self, model_name: str, message: str):
        super().__init__(f"[{model_name}] {message}")
        self.model_name = model_name


@dataclass
class GenerationResult:
    raw_text: str
    prompt_tokens: int
    completion_tokens: int


class BaseLLMClient(ABC):
    """Abstract LLM client interface."""

    name: str
    model_id: str

    def __init__(self, name: str, model_id: str):
        self.name = name
        self.model_id = model_id

    def generate(self, system: str, user: str) -> GenerationResult:
        """Call the model with retry on transient errors."""
        last_error: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            try:
                start = time.time()
                result = self._call_api(system, user)
                # Empty or whitespace-only content is a documented
                # intermittent failure (DeepSeek JSON mode in particular,
                # also Gemini safety filters and vLLM token-budget
                # exhaustion). Treat it as a transient error so the
                # retry loop fires; otherwise downstream JSON parsing
                # marks the row ok=False without ever retrying.
                if not result.raw_text or not result.raw_text.strip():
                    raise LLMError(self.name, "empty content from API")
                latency_ms = int((time.time() - start) * 1000)
                logger.debug(
                    f"{self.name} OK in {latency_ms}ms "
                    f"(in={result.prompt_tokens}, out={result.completion_tokens})"
                )
                return result
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning(
                    f"{self.name} attempt {attempt + 1}/{MAX_RETRIES} failed: {e}"
                )
                if attempt < MAX_RETRIES - 1:
                    backoff = RETRY_BACKOFF_BASE ** attempt
                    logger.debug(f"{self.name} sleeping {backoff:.1f}s before retry")
                    time.sleep(backoff)
        logger.error(f"{self.name} failed after {MAX_RETRIES} attempts: {last_error}")
        raise LLMError(self.name, str(last_error))

    @abstractmethod
    def _call_api(self, system: str, user: str) -> GenerationResult:
        ...


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------
class GeminiClient(BaseLLMClient):
    """Gemini client using the google-genai SDK.

    Uses Gemini's *explicit context caching* to amortise the
    ~1700-token system prompt (which is byte-identical for every
    call within a (lang, kg) cell). The first call for a given
    system prompt creates a server-side cache via
    ``client.caches.create``; subsequent calls pass
    ``cached_content=cache.name`` and the cached portion is billed
    at $0.05/M instead of $0.50/M (10× cheaper). Storage cost is
    $1/M-tokens/h while the cache is alive — for our 1700-token
    prompt × 24h TTL × ≤17 distinct prompts that's ~$0.69 total,
    negligible against the saving.

    The cache registry is a class-level dict guarded by a lock so
    multi-voter ThreadPool calls share the same cache. On cache-
    create failure we negative-cache the entry so we don't retry-
    storm — fall back to per-call ``system_instruction`` exactly
    like the pre-caching code path.
    """

    # Class-level cache registry shared across all GeminiClient
    # instances in this process. Keyed on (model_id, sha256(system)[:16]);
    # value is the cache resource name, or None if creation failed
    # (negative cache to short-circuit retries).
    _cache_lock: threading.Lock = threading.Lock()
    _caches: dict[tuple[str, str], Optional[str]] = {}

    def __init__(self, api_key: Optional[str] = None, model_id: str = GEMINI_MODEL):
        super().__init__("gemini", model_id)
        from google import genai

        if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
            # Vertex mode: SDK reads GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION,
            # and credentials from GOOGLE_APPLICATION_CREDENTIALS / ADC.
            self._client = genai.Client()
        else:
            self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def _get_or_create_cache(self, system: str) -> Optional[str]:
        """Return the cache resource name for this system prompt, creating
        the cache on first use. Returns None on creation failure (caller
        falls back to per-call ``system_instruction``)."""
        prompt_hash = hashlib.sha256(system.encode()).hexdigest()[:16]
        cache_key = (self.model_id, prompt_hash)
        with self._cache_lock:
            if cache_key in self._caches:
                return self._caches[cache_key]
            try:
                from google.genai import types as genai_types

                cache = self._client.caches.create(
                    model=self.model_id,
                    config=genai_types.CreateCachedContentConfig(
                        system_instruction=system,
                        display_name=f"postedit-{prompt_hash}",
                        ttl="86400s",  # 24h; well over the longest SLURM job
                    ),
                )
                self._caches[cache_key] = cache.name
                logger.info(
                    "Gemini cache created: {} (prompt hash {})",
                    cache.name, prompt_hash,
                )
                return cache.name
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Gemini cache create failed ({}); falling back to "
                    "per-call system_instruction. Subsequent calls with "
                    "this prompt will reuse the same fallback.", e,
                )
                self._caches[cache_key] = None
                return None

    def _call_api(self, system: str, user: str) -> GenerationResult:
        from google.genai import types as genai_types

        # Disable thinking — this is a structured grammar task that doesn't
        # need reasoning. Disabling cuts output tokens dramatically and
        # therefore cost (output tokens billed at 6× input rate).
        config_kwargs: dict[str, Any] = {
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "response_mime_type": "application/json",
            "thinking_config": genai_types.ThinkingConfig(thinking_budget=0),
        }
        cached_name = self._get_or_create_cache(system)
        if cached_name:
            config_kwargs["cached_content"] = cached_name
        else:
            config_kwargs["system_instruction"] = system

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )

        text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        return GenerationResult(text, prompt_tokens, completion_tokens)


# ---------------------------------------------------------------------------
# DeepSeek (v4-flash)
# ---------------------------------------------------------------------------
class DeepSeekClient(BaseLLMClient):
    """DeepSeek client via the OpenAI SDK against api.deepseek.com.

    Uses the standard ``openai`` package with a custom ``base_url``.
    DeepSeek's API is fully Chat-Completions-compatible; the server
    returns identical response shapes (``choices[0].message.content``,
    ``usage.{prompt,completion}_tokens``) so the rest of the pipeline
    is unchanged.

    JSON mode requirements (per DeepSeek docs):
    * ``response_format={"type": "json_object"}``.
    * The system prompt **must** mention the word "json" and include
      a concrete example of the desired JSON shape — our
      :data:`prompts.SYSTEM_PROMPT` already does both.
    * Empty-content responses are a known intermittent failure mode;
      our :class:`BaseLLMClient` retry loop catches them.
    """

    def __init__(self, api_key: Optional[str] = None,
                 model_id: str = DEEPSEEK_MODEL,
                 base_url: str = DEEPSEEK_BASE_URL):
        super().__init__("deepseek", model_id)
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url,
            timeout=REQUEST_TIMEOUT,
        )

    def _call_api(self, system: str, user: str) -> GenerationResult:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return GenerationResult(text, prompt_tokens, completion_tokens)


# ---------------------------------------------------------------------------
# vLLM (Qwen)
# ---------------------------------------------------------------------------
class VLLMClient(BaseLLMClient):
    """vLLM client using direct HTTP calls (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        model_id: str = VLLM_MODEL,
        api_key: str = VLLM_API_KEY,
    ):
        super().__init__("qwen", model_id)
        import httpx

        self._base_url = base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(timeout=REQUEST_TIMEOUT, headers=self._headers)

    def health_check(self) -> bool:
        """Verify the vLLM endpoint is reachable."""
        try:
            r = self._client.get(f"{self._base_url}/v1/models")
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def _call_api(self, system: str, user: str) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
        }
        response = self._client.post(f"{self._base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return GenerationResult(text, prompt_tokens, completion_tokens)

    def close(self):
        self._client.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_clients(selected: list[str]) -> dict[str, BaseLLMClient]:
    """Construct only the requested LLM clients."""
    clients: dict[str, BaseLLMClient] = {}
    for name in selected:
        if name == "gemini":
            clients["gemini"] = GeminiClient()
        elif name == "deepseek":
            clients["deepseek"] = DeepSeekClient()
        elif name == "qwen":
            clients["qwen"] = VLLMClient()
        else:
            raise ValueError(f"unknown model: {name}")
    return clients
