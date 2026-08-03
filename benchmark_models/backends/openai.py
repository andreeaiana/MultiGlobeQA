"""OpenAI backend (GPT-5.4 family).

Wraps the ``openai`` Python SDK to call GPT models via the Chat
Completions API. Uses OpenAI's ``response_format`` with ``json_schema``
for structured output enforcement.

The backend handles tiers 1 and 2 on the same underlying model
(``gpt-5.4``); tier-2 reasoning is controlled by the
``reasoning_effort`` parameter, not by selecting a separate
"thinking" model.

* **Tier 1 (plain)**: single chat completion with
  ``reasoning_effort="none"``. Standard sampling parameters
  (``temperature``, ``top_p``, ``seed``) apply.
* **Tier 2 (reasoning)**: same model with
  ``reasoning_effort`` ∈ ``{low, medium, high, xhigh}``. The
  ``max_completion_tokens`` budget is increased to accommodate
  reasoning tokens that the model spends invisibly before the
  visible JSON answer. **Reasoning mode rejects** ``temperature``,
  ``top_p``, and ``seed`` — OpenAI's API forbids them on reasoning
  requests, and self-consistency relies on the per-sample
  ``seed_override`` instead. The backend omits these fields from
  the request body when ``reasoning_effort != "none"``.
* **Tier 3 (tool_agent)**: routes through smolagents'
  ``OpenAIModel`` adapter (see ``benchmark_models/tier3.py``); not
  invoked directly from this module.

API reference:
  - Structured output: https://developers.openai.com/api/docs/guides/structured-outputs
  - Reasoning models: https://developers.openai.com/api/docs/guides/reasoning
  - GPT-5.4 guide: https://developers.openai.com/api/docs/guides/latest-model
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

# Soft-import the openai SDK. The module stays importable when the
# ``benchmark_models`` extra is not installed, so test collection does
# not block; ``OpenAIModel.__init__`` raises a helpful ImportError if the
# backend is actually constructed without the SDK. The module is left
# at module scope so tests can patch it.
try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised by env without extra
    OpenAI = None  # type: ignore[assignment,misc]

from benchmark_models.backends._error_class import classify_error
from benchmark_models.backends._retry import retry_with_backoff
from benchmark_models.base import (
    Example,
    Model,
    Prediction,
    Tier,
    Usage,
)
from benchmark_models.config import (
    TIER_NAME_TO_TIER,
    MultimodalMode,
    OpenAIVariantConfig,
)
from benchmark_models.multimodal import apply_name_mode, encode_image_b64
from benchmark_models.prompts import PromptRenderer
from benchmark_models.schemas import TemplateInfo, TemplateRegistry, load_template_registry


class OpenAIModel(Model):
    """GPT-backed :class:`Model` implementation.

    Constructed from an :class:`OpenAIVariantConfig` (one entry in
    ``models.yaml``) and an optional template registry + API key.
    """

    def __init__(
        self,
        variant_name: str,
        config: OpenAIVariantConfig,
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
        self._max_concurrent = max_concurrent
        if OpenAI is None:
            raise ImportError(
                "The openai SDK is required for OpenAIModel. "
                "Install with: uv sync --group benchmark_models"
            )
        self.client = OpenAI(api_key=api_key)
        self._renderer = prompt_renderer or PromptRenderer(tier=self.tier)
        self.multimodal_mode: MultimodalMode = multimodal_mode

    def _build_message_content(
        self, prompt: str, example: Example
    ) -> str | list[dict[str, Any]]:
        """Return the ``content`` value for the user message.

        ``"off"`` and ``"name"`` return a plain string. ``"image"``
        returns the OpenAI Chat Completions multimodal block list with
        one ``text`` part and one ``image_url`` part per attached
        image; the URL is a data URI carrying the base64-encoded bytes
        inline."""
        if self.multimodal_mode == "name":
            return apply_name_mode(prompt, example)
        if self.multimodal_mode == "image" and example.images:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for image in example.images:
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{image.mime_type};base64,"
                                f"{encode_image_b64(image)}"
                            )
                        },
                    }
                )
            return blocks
        return prompt

    def predict(self, example: Example) -> Prediction:
        """Run one example through GPT and return a Prediction."""
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
                message=f"tier {self.tier} not yet implemented for OpenAIModel",
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
        from benchmark_models.schemas import resolve_canonical_answer_type

        prompt, prompt_template_id, triple_context = self._renderer.render(
            example, template_info
        )
        answer_type = resolve_canonical_answer_type(
            example.ground_truth_type,
            template_info.answer_type,
            tid=example.tid,
        )
        schema = self._build_response_schema(
            template_info, include_reasoning=True, answer_type=answer_type
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
        # Track full per-sample predictions so we can pull the
        # winner's reasoning string after the vote, rather than only
        # its answer. The earlier code stored just ``pred.answer``,
        # which silently dropped the CoT trace from every
        # SC-aggregated Prediction — observed empty
        # ``Prediction.reasoning`` on every qwen35-a3b-thinking-tier2
        # row even though the per-sample schemas required the field
        # and the per-sample Predictions had it populated.
        sample_preds: list[Prediction] = []
        total_usage = Usage(api_call_count=0)
        total_latency = 0.0
        raw_completions: list[str] = []

        for i in range(n):
            pred = self._call_and_parse(
                example, template_info, prompt, schema,
                prompt_template_id="tier2.sc",
                temperature_override=self.config.temperature,
                # Vary the seed per sample so each draw is genuinely
                # independent. Without this, OpenAI's seed honoring
                # produced identical samples across N attempts on
                # supported models, defeating the point of SC.
                seed_override=self.config.seed + i,
            )
            sample_preds.append(pred)
            total_usage.input_tokens += pred.usage.input_tokens
            total_usage.output_tokens += pred.usage.output_tokens
            total_usage.reasoning_tokens += pred.usage.reasoning_tokens
            total_usage.api_call_count += pred.usage.api_call_count
            total_latency += pred.latency_ms
            raw_completions.append(pred.raw_completion)

        samples: list[Any] = [p.answer for p in sample_preds]

        # Stable JSON-key voting: dict-shaped answers (Coordinates,
        # Polar, BBox, set) round-trip to the same key regardless of
        # key order or float formatting. Falls back to ``str(...)``
        # for non-JSON-serializable shapes.
        def _vote_key(a: Any) -> str:
            try:
                return json.dumps(a, sort_keys=True, default=str)
            except (TypeError, ValueError):
                return str(a)

        vote_candidates = [a for a in samples if a is not None]
        if vote_candidates:
            winner_key = Counter(_vote_key(a) for a in vote_candidates).most_common(1)[0][0]
            # First-matching sample wins on ties — same selection rule
            # as before; we additionally pull its reasoning to expose
            # the CoT trace that produced the winning answer on the
            # aggregated Prediction. Losing samples' reasonings are
            # not preserved structurally (only as text in
            # ``raw_completion``'s ``---SC---``-joined dump) since
            # ``Prediction`` doesn't have a parallel
            # ``sc_reasoning_samples`` field today; if scoring needs
            # per-sample CoT, that's a future addition.
            winner_idx = next(
                i for i, p in enumerate(sample_preds)
                if p.answer is not None and _vote_key(p.answer) == winner_key
            )
            answer = sample_preds[winner_idx].answer
            reasoning = sample_preds[winner_idx].reasoning
        else:
            answer = None
            reasoning = None

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
            reasoning=reasoning,
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
        seed_override: int | None = None,
    ) -> Prediction:
        """Single chat completion call → Prediction."""
        temperature = (
            temperature_override
            if temperature_override is not None
            else self.config.temperature
        )
        seed_value = (
            seed_override
            if seed_override is not None
            else self.config.seed
        )

        # Three regimes, picked from the (reasoning_effort, schema)
        # axes — the second branch is the one to think about, the
        # other two are the existing behaviour.
        #
        # 1. ``reasoning_effort != "none"`` (OpenAI reasoning models):
        #    the reasoning tokens are an
        #    internal channel that doesn't compete with the output
        #    JSON. GPT-5.4 reasoning can burn 5000–8000+ tokens on
        #    reasoning alone, so 25k is the minimum that keeps the
        #    answer JSON from being truncated.
        # 2. Prompted-CoT (reasoning_effort == "none" *and* the
        #    schema asks for a ``reasoning`` field — tier-2 on vLLM
        #    variants without a native thinking toggle, e.g.
        #    qwen35-27b-tier2): the reasoning lives *inside* the
        #    JSON output and competes with the rest of the response
        #    for the same token budget. The base 512 default is too
        #    tight — observed truncation mid-``reasoning`` string
        #    on qwen35-27b-tier2 smoke runs, producing a JSON parse
        #    failure and a forced refusal. 2048 covers a several-
        #    sentence reasoning paragraph + the answer envelope.
        # 3. Plain (no reasoning anywhere): keep ``config.max_tokens``
        #    untouched. Tier-1 stays at 512, tight-but-sufficient
        #    for an answer + unit pair.
        has_reasoning = self.config.reasoning_effort != "none"
        schema_requests_reasoning = "reasoning" in (
            schema.get("properties", {}) if isinstance(schema, dict) else {}
        )
        if has_reasoning:
            effective_max_tokens = max(25000, self.config.max_tokens * 16)
        elif schema_requests_reasoning:
            effective_max_tokens = max(2048, self.config.max_tokens * 4)
        else:
            effective_max_tokens = self.config.max_tokens

        message_content = self._build_message_content(prompt, example)
        messages = [{"role": "user", "content": message_content}]

        # Build the API call kwargs.
        api_kwargs: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "spatial_answer",
                    "schema": schema,
                    "strict": True,
                },
            },
            "max_completion_tokens": effective_max_tokens,
        }

        if has_reasoning:
            # Reasoning models don't support temperature, top_p, or
            # seed — they only accept the default (temperature=1). We
            # pass reasoning_effort instead and omit the sampling params.
            api_kwargs["reasoning_effort"] = self.config.reasoning_effort
        else:
            # Non-reasoning calls support full sampling control.
            api_kwargs["temperature"] = temperature
            api_kwargs["top_p"] = self.config.top_p
            api_kwargs["seed"] = seed_value

        t0 = time.perf_counter()
        try:
            response = retry_with_backoff(
                lambda: self.client.chat.completions.create(**api_kwargs),
                provider="openai",
                question_id=example.question_id,
            )
        except Exception as exc:
            logger.warning(
                "OpenAIModel: API call failed for question_id={}: {}",
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

        # Parse usage metadata.
        u = response.usage
        reasoning_tokens = 0
        if u and hasattr(u, "completion_tokens_details") and u.completion_tokens_details:
            reasoning_tokens = getattr(
                u.completion_tokens_details, "reasoning_tokens", 0
            ) or 0
        # Cached-token denormalization. OpenAI reports cache hits via
        # ``prompt_tokens_details.cached_tokens``. Subclasses override
        # ``_extract_cached_tokens`` when their API differs.
        cached_tokens = self._extract_cached_tokens(u)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0 if u else 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0 if u else 0,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            api_call_count=1,
        )

        # Extract text from response.
        raw_completion = ""
        if response.choices:
            msg = response.choices[0].message
            raw_completion = msg.content or ""

        # Parse the JSON answer. For OpenAI, the reasoning trace is in
        # the JSON response itself (the "reasoning" field of the
        # structured output), not in separate content blocks.
        answer, unit, reasoning_text, refusal = self._parse_json_answer(
            raw_completion, template_info
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
            reasoning=reasoning_text,
            triple_context=triple_context,
            usage=usage,
            latency_ms=latency_ms,
            raw_completion=raw_completion,
        )

    # ------------------------------------------------------------------
    # Schema construction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_cached_tokens(usage_obj: Any) -> int:
        """Pull cached-token count from an OpenAI ``usage`` block.

        Returns 0 when the SDK shape doesn't expose the field
        (older models / older SDK versions don't break — we just
        report no cache hits). Subclasses override this when their
        API reports cache hits under a different name.
        """
        if usage_obj is None:
            return 0
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is None:
            return 0
        return getattr(details, "cached_tokens", 0) or 0

    @staticmethod
    def _build_response_schema(
        template_info: TemplateInfo,
        include_reasoning: bool = False,
        *,
        answer_type: str | None = None,
    ) -> dict[str, Any]:
        """Build the JSON schema for OpenAI's structured output.

        OpenAI's ``json_schema`` strict mode requires ``additionalProperties: false``
        on all object schemas. The schema matches the structure the
        prompt tells the model to produce.

        ``answer_type`` overrides ``template_info.answer_type`` when
        set; used by callers that resolved the per-row
        ``Example.ground_truth_type``.
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
            required.append("unit")

        if include_reasoning:
            properties["reasoning"] = {"type": "string"}
            required.append("reasoning")

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    # ------------------------------------------------------------------
    # JSON response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_answer(
        text: str, template_info: TemplateInfo
    ) -> tuple[Any, str | None, str | None, bool]:
        """Parse the model's JSON response text.

        Returns ``(answer, unit, reasoning, refusal)``. For OpenAI,
        the reasoning trace is part of the structured JSON output
        (the ``"reasoning"`` key), not in separate content blocks.
        """
        text = text.strip()
        if not text:
            return None, None, None, True
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "OpenAIModel: failed to parse JSON for tid={}: {!r}",
                template_info.tid,
                text[:200],
            )
            return None, None, None, True

        if not isinstance(data, dict):
            return None, None, None, True

        answer = data.get("answer")
        unit = data.get("unit")
        reasoning = data.get("reasoning")
        refusal = answer is None
        return answer, unit, reasoning, refusal

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
            raw_completion=f"OpenAIModel error: {message}",
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, examples: list[Example]) -> list[Prediction]:
        """Run examples with bounded thread-pool concurrency."""
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
        """Release the OpenAI client (closes the HTTP session)."""
        self.client.close()
