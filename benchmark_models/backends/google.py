"""Google backend (Gemini 3 Flash).

Wraps the ``google-genai`` Python SDK to call Gemini models via the
Google AI API. Uses Gemini's ``response_json_schema`` for structured
output enforcement and ``thinking_config`` for the tier-2 reasoning
mode.

The backend handles all three tiers:
* **Tier 1 (plain)**: single ``generate_content`` call with
  ``thinking_level=minimal`` (or whatever the variant config says) and
  structured JSON output.
* **Tier 2 (reasoning)**: same call with ``thinking_level=high`` and
  optional ``include_thoughts=True`` to capture the reasoning trace.
  Self-consistency (``self_consistency_n > 1``) samples N completions
  at ``temperature > 0`` and takes the majority vote.
* **Tier 3 (tool_agent)**: uses Gemini's native function-calling. The
  agent loop in ``tiers/tool_agent.py`` orchestrates the multi-turn
  conversation; this backend provides the per-turn ``chat`` method.
  *(Tier-3 support is scaffolded but lands in a later commit.)*

API reference:
  - Structured output: https://ai.google.dev/gemini-api/docs/structured-output
  - Thinking: https://ai.google.dev/gemini-api/docs/thinking
  - Explicit context caching: https://ai.google.dev/gemini-api/docs/caching
  - SDK docs: https://googleapis.github.io/python-genai/

Explicit context caching: when
``GoogleVariantConfig.enable_context_cache`` is True (default) and
the rendered prompt prefix (everything before ``Question:\n``) is
≥ 4096 tokens, the backend calls ``client.caches.create(...)``
once per ``sha256(prefix + model_id)`` key and references
``cached_content=<name>`` on subsequent ``generate_content`` calls.
Cached prefix tokens bill at ~25% of the normal input rate.

Cache lifecycle: 1-hour TTL by default; cache-expired errors are
detected (substring-match on the SDK exception), the registry
entry is invalidated, and the call is retried once without the
cache reference. Distinct from generic retry so cache-expired
doesn't burn the ``_retry.py`` exponential-backoff budget.

The ``usage_metadata.cached_content_token_count`` (read at
``_call_and_parse`` below) is denormalised end-to-end into
``ScoreRecord.cached_tokens`` and
``AggregateReport.total_cached_tokens`` so post-cache costs are
visible in the eval pipeline.
"""

from __future__ import annotations

import json
import time
from typing import Any

from loguru import logger

# Soft-import the google-genai SDK. The module stays importable when the
# ``benchmark_models`` extra is not installed, so test collection does
# not block; ``GoogleModel.__init__`` raises a helpful ImportError if the
# backend is actually constructed without the SDK. The module is left
# at module scope so tests can patch it.
try:
    from google import genai
    from google.genai import types as _genai_types
except ImportError:  # pragma: no cover - exercised by env without extra
    genai = None  # type: ignore[assignment]
    _genai_types = None  # type: ignore[assignment]

from benchmark_models.backends._error_class import classify_error
from benchmark_models.backends._retry import retry_with_backoff
from benchmark_models.base import (
    Example,
    Model,
    Prediction,
    Tier,
    Usage,
)
from benchmark_models.config import TIER_NAME_TO_TIER, GoogleVariantConfig, MultimodalMode
from benchmark_models.multimodal import apply_name_mode, read_image_bytes
from benchmark_models.prompts import PromptRenderer
from benchmark_models.schemas import TemplateInfo, TemplateRegistry, load_template_registry


class GoogleModel(Model):
    """Gemini-backed :class:`Model` implementation.

    Constructed from a :class:`GoogleVariantConfig` (one entry in
    ``models.yaml``) and an optional template registry + API key.
    """

    def __init__(
        self,
        variant_name: str,
        config: GoogleVariantConfig,
        *,
        registry: TemplateRegistry | None = None,
        api_key: str | None = None,
        snapshot_id: str = "unset",
        max_concurrent: int = 8,
        prompt_renderer: PromptRenderer | None = None,
        multimodal_mode: MultimodalMode = "off",
    ) -> None:
        self.model_id = variant_name
        self.tier: Tier = TIER_NAME_TO_TIER[config.tier]
        self.config = config
        self.registry = registry if registry is not None else load_template_registry()
        self.snapshot_id = snapshot_id
        if genai is None:
            raise ImportError(
                "The google-genai SDK is required for GoogleModel. "
                "Install with: uv sync --group benchmark_models"
            )
        self.client = genai.Client(api_key=api_key)
        self._max_concurrent = max_concurrent
        self._renderer = prompt_renderer or PromptRenderer(tier=self.tier)
        self.multimodal_mode: MultimodalMode = multimodal_mode

        # Explicit-context-caching state.
        # ``_cache_registry`` maps ``sha256(prefix + model_id)`` to the
        # ``cached_content.name`` returned by ``client.caches.create()``.
        # ``_cache_lock`` guards the registry against concurrent
        # ``predict_batch`` workers seeding the same prefix simultaneously.
        # Gated by ``GoogleVariantConfig.enable_context_cache``.
        import threading

        self._cache_registry: dict[str, str] = {}
        self._cache_lock = threading.Lock()

    # Gemini's explicit-caching API requires a prefix of at least
    # 4096 tokens. Smaller prefixes are not cacheable; skip the create
    # call entirely for them so we don't pay for a count_tokens round
    # trip we can't act on.
    _CACHE_MIN_TOKENS = 4096
    # 1-hour TTL — matches a typical per-cell SLURM job duration; the
    # cache-expired error handler regenerates on TTL elapse.
    _CACHE_TTL_SECONDS = 3600
    # Marker that splits the rendered prompt into (cacheable prefix,
    # per-row suffix).
    _CACHE_PREFIX_MARKER = "Question:\n"

    def _resolve_cached_content(
        self, prompt: str
    ) -> str | None:
        """Return a ``cached_content.name`` for the prompt's stable
        prefix, or ``None`` when caching is disabled / the prefix is
        too short / the prompt has no marker.

        Resolution:
        1. Short-circuit if ``enable_context_cache=False``.
        2. Split the prompt at ``Question:\\n``; if no marker, no
           cacheable prefix → return ``None``.
        3. Look up the cache registry by ``sha256(prefix + model_id)``;
           reuse the cached name on hit.
        4. On miss: ``count_tokens(prefix)`` to gate the 4096-token
           minimum. If below, skip caching.
        5. ``client.caches.create(...)`` with TTL; store the returned
           ``cached_content.name`` in the registry.

        The lock guards step 3-5 so two concurrent workers don't both
        call ``caches.create`` for the same prefix.
        """
        import hashlib

        if not getattr(self.config, "enable_context_cache", False):
            return None
        idx = prompt.find(self._CACHE_PREFIX_MARKER)
        if idx <= 0:
            return None
        prefix = prompt[:idx]
        if not prefix.strip():
            return None

        cache_key = hashlib.sha256(
            (prefix + "\0" + self.config.model_id).encode("utf-8")
        ).hexdigest()

        with self._cache_lock:
            cached = self._cache_registry.get(cache_key)
            if cached is not None:
                return cached

            # Token-count gate. ``count_tokens`` is bills-as-0 so the
            # gate is free; the API call still has latency so do it
            # only on cache miss.
            try:
                token_response = self.client.models.count_tokens(
                    model=self.config.model_id,
                    contents=[
                        _genai_types.Part(text=prefix)  # type: ignore[attr-defined]
                    ],
                )
                raw_total = getattr(token_response, "total_tokens", 0)
                # Coerce to int — defensive against non-int return
                # types from older SDKs or test mocks.
                n_tokens = int(raw_total) if isinstance(raw_total, int) else 0
            except Exception as exc:
                logger.debug(
                    "GoogleModel: count_tokens failed ({}); skipping "
                    "context cache.", exc,
                )
                return None
            if n_tokens < self._CACHE_MIN_TOKENS:
                logger.debug(
                    "GoogleModel: context cache skipped — prefix only "
                    "{} tokens, < {} minimum.",
                    n_tokens, self._CACHE_MIN_TOKENS,
                )
                # Cache a sentinel so we don't re-count the same prefix.
                # Use empty string to indicate "below threshold; don't
                # try again".
                self._cache_registry[cache_key] = ""
                return None

            try:
                cached_content = self.client.caches.create(
                    model=self.config.model_id,
                    config=_genai_types.CreateCachedContentConfig(  # type: ignore[attr-defined]
                        contents=[
                            _genai_types.Part(text=prefix)  # type: ignore[attr-defined]
                        ],
                        ttl=f"{self._CACHE_TTL_SECONDS}s",
                    ),
                )
                name = cached_content.name
                if not isinstance(name, str):
                    return None
                self._cache_registry[cache_key] = name
                logger.info(
                    "GoogleModel: created context cache {} for prefix "
                    "({} tokens, TTL {}s)",
                    name, n_tokens, self._CACHE_TTL_SECONDS,
                )
                return name
            except Exception as exc:
                logger.warning(
                    "GoogleModel: caches.create failed ({}); skipping "
                    "context cache for this prompt.", exc,
                )
                return None

    def _invalidate_cache_for_prompt(self, prompt: str) -> None:
        """Clear the cache-name registry entry for ``prompt``'s prefix.

        Called by the call site when the API returns a cache-expired
        error: the next attempt at the same prefix re-creates the
        cache rather than referencing a dead name. Idempotent.
        """
        import hashlib

        idx = prompt.find(self._CACHE_PREFIX_MARKER)
        if idx <= 0:
            return
        prefix = prompt[:idx]
        cache_key = hashlib.sha256(
            (prefix + "\0" + self.config.model_id).encode("utf-8")
        ).hexdigest()
        with self._cache_lock:
            self._cache_registry.pop(cache_key, None)

    def _build_contents(self, prompt: str, example: Example) -> Any:
        """Return the ``contents`` arg for ``generate_content``.

        Three branches keyed on :attr:`multimodal_mode`:

        * ``"off"`` — pass the prompt string through unchanged. Same
          behaviour the backend has always had.
        * ``"name"`` — substitute ``[IMAGE_<slot>]`` placeholders with
          the entity's name; result stays a plain string.
        * ``"image"`` — return a list ``[<text>, <inline_data>, ...]``
          matching the google-genai SDK's multimodal content shape.
          Each :class:`ImageContent` becomes a ``Part`` with
          ``inline_data`` carrying the raw bytes and the declared
          mime type.
        """
        if self.multimodal_mode == "name":
            return apply_name_mode(prompt, example)
        if self.multimodal_mode == "image" and example.images:
            assert _genai_types is not None  # guarded by __init__ import check
            parts: list[Any] = [_genai_types.Part(text=prompt)]
            for image in example.images:
                parts.append(
                    _genai_types.Part(
                        inline_data=_genai_types.Blob(
                            mime_type=image.mime_type,
                            data=read_image_bytes(image),
                        )
                    )
                )
            return parts
        return prompt

    def predict(self, example: Example) -> Prediction:
        """Run one example through Gemini and return a Prediction."""
        try:
            template_info = self.registry.get(example.tid)
        except KeyError as exc:
            return self._error_prediction(
                example, sfid="?", answer_type="?",
                message=f"unknown tid {example.tid!r}: {exc}",
            )

        assert template_info.answer_type is not None

        if self.tier == "tier1":
            return self._predict_plain(example, template_info)
        elif self.tier == "tier2":
            return self._predict_reasoning(example, template_info)
        else:
            return self._error_prediction(
                example, sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                message=f"tier {self.tier} not yet implemented for GoogleModel",
            )

    # ------------------------------------------------------------------
    # Tier 1 — plain
    # ------------------------------------------------------------------

    def _predict_plain(
        self, example: Example, template_info: TemplateInfo
    ) -> Prediction:
        from benchmark_models.schemas import resolve_canonical_answer_type

        prompt, prompt_template_id, triple_context = self._renderer.render(
            example, template_info
        )
        answer_type = resolve_canonical_answer_type(
            example.ground_truth_type,
            template_info.answer_type,
            tid=example.tid,
        )
        schema = self._build_response_schema(template_info, answer_type=answer_type)
        return self._call_and_parse(
            example, template_info, prompt, schema,
            prompt_template_id=prompt_template_id,
            triple_context=triple_context,
        )

    # ------------------------------------------------------------------
    # Tier 2 — reasoning
    # ------------------------------------------------------------------

    def _predict_reasoning(
        self, example: Example, template_info: TemplateInfo
    ) -> Prediction:
        prompt, prompt_template_id, triple_context = self._renderer.render(
            example, template_info
        )
        # When Gemini thinking is on with include_thoughts=True
        # (``thinking_level`` medium / high), the reasoning trace is
        # emitted via the thinking-parts channel — the scorer populates
        # ``Prediction.reasoning`` from those parts. Requesting a
        # ``reasoning`` field in the structured-output schema on top
        # of that double-spends the output token budget: both the
        # thinking tokens AND the answer JSON compete for the same
        # max_output_tokens budget, and the reasoning field's
        # arbitrary length typically triggers mid-sentence truncation
        # of the JSON itself. When native thinking is not exposed
        # (level ``"low"`` or no thinking), we still need the
        # prompted-CoT reasoning field in the schema.
        from benchmark_models.schemas import resolve_canonical_answer_type

        include_reasoning = self.config.thinking_level not in ("medium", "high")
        answer_type = resolve_canonical_answer_type(
            example.ground_truth_type,
            template_info.answer_type,
            tid=example.tid,
        )
        schema = self._build_response_schema(
            template_info, include_reasoning=include_reasoning,
            answer_type=answer_type,
        )

        if self.config.self_consistency_n > 1:
            return self._predict_self_consistency(
                example, template_info, prompt, schema,
                triple_context=triple_context,
            )

        return self._call_and_parse(
            example, template_info, prompt, schema,
            prompt_template_id=prompt_template_id,
            triple_context=triple_context,
        )

    def _predict_self_consistency(
        self,
        example: Example,
        template_info: TemplateInfo,
        prompt: str,
        schema: dict,
        *,
        triple_context: str | None = None,
    ) -> Prediction:
        """Sample N reasoning chains and majority-vote the answer."""
        n = self.config.self_consistency_n
        samples: list[Any] = []
        total_usage = Usage(api_call_count=0)
        total_latency = 0.0
        raw_completions: list[str] = []

        # Gemini's gen_config doesn't expose a seed parameter, so SC
        # independence relies on temperature alone (config should
        # set ``temperature > 0`` for SC variants).
        for _ in range(n):
            pred = self._call_and_parse(
                example, template_info, prompt, schema,
                prompt_template_id="tier2.sc",
                temperature_override=self.config.temperature,
            )
            samples.append(pred.answer)
            total_usage.input_tokens += pred.usage.input_tokens
            total_usage.output_tokens += pred.usage.output_tokens
            total_usage.reasoning_tokens += pred.usage.reasoning_tokens
            total_usage.api_call_count += pred.usage.api_call_count
            total_latency += pred.latency_ms
            raw_completions.append(pred.raw_completion)

        # Stable JSON-key voting: dict-shaped answers (Coordinates,
        # Polar, BBox, set) round-trip to the same key regardless of
        # key order or float formatting. Falls back to ``str(...)``
        # for non-JSON-serializable shapes.
        from collections import Counter
        def _vote_key(a: Any) -> str:
            try:
                return json.dumps(a, sort_keys=True, default=str)
            except (TypeError, ValueError):
                return str(a)

        vote_candidates = [a for a in samples if a is not None]
        if vote_candidates:
            winner_key = Counter(_vote_key(a) for a in vote_candidates).most_common(1)[0][0]
            answer = next(a for a in vote_candidates if _vote_key(a) == winner_key)
        else:
            answer = None

        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=template_info.sfid,
            answer_type=template_info.answer_type,
            model_id=self.model_id,
            tier=self.tier,
            language=example.language,
            prompt_template_id="tier2.sc",
            seed=self.config.seed,
            snapshot_id=self.snapshot_id,
            answer=answer,
            unit=template_info.answer_unit,
            refusal=(answer is None),
            sc_samples=samples,
            triple_context=triple_context,
            usage=total_usage,
            latency_ms=total_latency,
            raw_completion="\n---SC---\n".join(raw_completions),
        )

    # ------------------------------------------------------------------
    # API call + response parsing (shared across tiers)
    # ------------------------------------------------------------------

    def _call_and_parse(
        self,
        example: Example,
        template_info: TemplateInfo,
        prompt: str,
        schema: dict,
        *,
        prompt_template_id: str,
        temperature_override: float | None = None,
        triple_context: str | None = None,
    ) -> Prediction:
        """Single generate_content call → Prediction."""
        temperature = (
            temperature_override
            if temperature_override is not None
            else self.config.temperature
        )

        # Thinking config. When thinking is enabled, Gemini counts
        # thinking tokens against max_output_tokens. The user's
        # max_tokens means "max answer tokens"; the backend adds
        # headroom for the thinking budget so the JSON answer doesn't
        # get truncated.
        thinking_level = self.config.thinking_level
        include_thoughts = thinking_level in ("medium", "high")
        if thinking_level in ("low", "medium", "high"):
            # Gemini charges thinking + answer tokens against the
            # same ``max_output_tokens``. The original 8192 budget
            # caused truncation when the schema included a
            # ``reasoning`` field — but we now drop that field for
            # native-thinking models (the reasoning trace comes from
            # the thinking channel instead). Without the reasoning
            # field the answer JSON is ~20 tokens, leaving ~8170 for
            # thinking — plenty. Keeping 8192 preserves
            # ``thinking_level="high"`` quality without the latency
            # regression that a 16384 budget caused (the model fills
            # whatever budget is available).
            effective_max_tokens = max(8192, self.config.max_tokens * 8)
        else:
            effective_max_tokens = self.config.max_tokens

        gen_config: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "temperature": temperature,
            "max_output_tokens": effective_max_tokens,
            "top_p": self.config.top_p,
        }

        gen_config["thinking_config"] = _genai_types.ThinkingConfig(
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        )

        # Attach the cached prompt prefix when available. The registry
        # lookup short-circuits when caching is disabled, when the
        # prefix has no marker, or when it's below the 4096-token
        # minimum.
        cached_content_name = self._resolve_cached_content(prompt)
        if cached_content_name:
            gen_config["cached_content"] = cached_content_name

        contents = self._build_contents(prompt, example)

        def _do_generate() -> Any:
            return self.client.models.generate_content(
                model=self.config.model_id,
                contents=contents,
                config=gen_config,
            )

        t0 = time.perf_counter()
        try:
            try:
                response = retry_with_backoff(
                    _do_generate,
                    provider="google",
                    question_id=example.question_id,
                )
            except Exception as inner:
                # Distinguish cache-expired (TTL elapsed mid-run) from
                # other failures so the cache-name registry can be
                # cleared and the next call re-creates rather than
                # referencing a dead name. The google-genai SDK
                # surfaces this as an InvalidArgument / 400-class
                # error mentioning "cached_content" or "cache" —
                # substring match because the exception class
                # structure varies by SDK version.
                if cached_content_name and any(
                    needle in str(inner).lower()
                    for needle in (
                        "cached_content",
                        "cache expired",
                        "cache not found",
                    )
                ):
                    logger.warning(
                        "GoogleModel: cached_content {} appears expired "
                        "({}); invalidating and retrying without cache.",
                        cached_content_name, inner,
                    )
                    self._invalidate_cache_for_prompt(prompt)
                    gen_config.pop("cached_content", None)
                    response = retry_with_backoff(
                        _do_generate,
                        provider="google",
                        question_id=example.question_id,
                    )
                else:
                    raise
        except Exception as exc:
            logger.warning(
                "GoogleModel: API call failed for question_id={}: {}",
                example.question_id,
                exc,
            )
            return self._error_prediction(
                example,
                sfid=template_info.sfid,
                answer_type=template_info.answer_type,
                message=f"API call failed: {exc}",
                error_class=classify_error(exc),
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Parse usage metadata. I: Gemini reports cache hits via
        # ``usage_metadata.cached_content_token_count``. Populated
        # when the request references a cached_content created via
        # the explicit context-caching API (Section H).
        meta = response.usage_metadata
        usage = Usage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            reasoning_tokens=getattr(meta, "thoughts_token_count", 0) or 0,
            cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
            api_call_count=1,
        )

        # Extract text and optional thinking from response parts.
        answer_text = ""
        reasoning_text = ""
        raw_parts: list[str] = []
        # When Gemini hits a safety filter, MAX_TOKENS, or RECITATION
        # finish_reason, the candidate has no content.parts — it's None,
        # not []. Iterating over it crashes the whole predict_batch.
        if response.candidates:
            content = response.candidates[0].content
            parts = (content.parts if content is not None else None) or []
            for part in parts:
                raw_parts.append(part.text or "")
                if getattr(part, "thought", False):
                    reasoning_text += part.text or ""
                else:
                    answer_text += part.text or ""
        raw_completion = "\n".join(raw_parts)

        # Parse the JSON answer.
        answer, unit, refusal = self._parse_json_answer(
            answer_text, template_info
        )

        return Prediction(
            question_id=example.question_id,
            tid=example.tid,
            sfid=template_info.sfid,
            answer_type=template_info.answer_type,
            model_id=self.model_id,
            tier=self.tier,
            language=example.language,
            prompt_template_id=prompt_template_id,
            seed=self.config.seed,
            snapshot_id=self.snapshot_id,
            answer=answer,
            unit=unit,
            refusal=refusal,
            reasoning=reasoning_text or None,
            triple_context=triple_context,
            usage=usage,
            latency_ms=latency_ms,
            raw_completion=raw_completion,
        )

    # ------------------------------------------------------------------
    # Schema construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response_schema(
        template_info: TemplateInfo,
        include_reasoning: bool = False,
        *,
        answer_type: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON schema for Gemini's ``response_json_schema``.

        The schema matches the structure the prompt tells the model to
        produce: ``{"answer": <typed>, "unit": <str>}`` for tier 1 /
        tier 3, or ``{"answer": ..., "reasoning": <str>, "unit": ...}``
        for tier 2.

        ``answer_type`` overrides ``template_info.answer_type`` when
        set; used by callers that resolved the per-row
        ``Example.ground_truth_type`` (authoritative per question).
        """
        from benchmark_models.schemas import TYPE_REGISTRY

        resolved = answer_type or template_info.answer_type
        assert resolved is not None
        type_spec = TYPE_REGISTRY[resolved]

        properties: dict[str, Any] = {
            "answer": type_spec.json_schema,
        }
        required = ["answer"]

        if type_spec.wants_unit_field:
            properties["unit"] = {"type": ["string", "null"]}

        if include_reasoning:
            properties["reasoning"] = {"type": "string"}
            required.append("reasoning")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    # ------------------------------------------------------------------
    # JSON response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_answer(
        text: str, template_info: TemplateInfo
    ) -> tuple[Any, str | None, bool]:
        """Parse the model's JSON response text.

        Returns ``(answer, unit, refusal)`` where ``refusal`` is
        ``True`` if the model produced ``answer: null``.
        """
        text = text.strip()
        if not text:
            return None, None, True
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "GoogleModel: failed to parse JSON from response for "
                "tid={}: {!r}",
                template_info.tid,
                text[:200],
            )
            return None, None, True

        if not isinstance(data, dict):
            return None, None, True

        answer = data.get("answer")
        unit = data.get("unit")
        refusal = answer is None
        return answer, unit, refusal

    # ------------------------------------------------------------------
    # Error helper
    # ------------------------------------------------------------------

    def _error_prediction(
        self,
        example: Example,
        *,
        sfid: str,
        answer_type: str,
        message: str,
        error_class: str = "config",
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
            seed=self.config.seed,
            snapshot_id=self.snapshot_id,
            answer=None,
            refusal=True,
            final_state="error",
            error_class=error_class,  # type: ignore[arg-type]
            usage=Usage(api_call_count=0),
            latency_ms=0.0,
            raw_completion=f"GoogleModel error: {message}",
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        """Run examples with bounded thread-pool concurrency.

        Uses :class:`concurrent.futures.ThreadPoolExecutor` with up to
        ``max_concurrent`` threads (default 8, capped to
        ``len(examples)`` so we don't spawn idle threads for small
        batches). The genai SDK is thread-safe for independent
        ``generate_content`` calls.

        Results are returned in the same order as ``examples``,
        regardless of which thread finishes first. Internally uses
        ``pool.map`` which preserves input order.
        """
        from concurrent.futures import ThreadPoolExecutor

        if len(examples) <= 1:
            return [self.predict(ex) for ex in examples]

        max_workers = min(self._max_concurrent, len(examples))
        logger.info(
            "predict_batch: {} examples with {} concurrent workers",
            len(examples),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            predictions = list(pool.map(self.predict, examples))
        return predictions

    def close(self) -> None:
        """Release the genai client (no-op; the client is stateless)."""
