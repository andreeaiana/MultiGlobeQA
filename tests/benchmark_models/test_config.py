"""Tests for ``benchmark_models.config``.

Covers the variant config discriminated union, the models catalogue
loader, and the experiment config loader. Focuses on:
- Each backend's config validates with correct fields.
- extra="forbid" catches typos.
- Discriminated union dispatches to the right subclass.
- Tier-consistency validators fire (e.g., self_consistency_n on non-reasoning).
- Models catalogue loads from YAML (synthetic + bundled).
- Experiment config validates all required fields.
- Per-variant overrides shape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from benchmark_models.config import (
    GoogleVariantConfig,
    ModelsCatalogue,
    OpenAIVariantConfig,
    VariantConfig,
    VLLMHttpVariantConfig,
    load_experiment_config,
    load_models_catalogue,
)

# ============================================================================
# Variant config — discriminated union
# ============================================================================


class TestOpenAIVariantConfig:
    def test_minimal(self) -> None:
        c = OpenAIVariantConfig(
            backend="openai", model_id="gpt-5.4", tier="plain"
        )
        assert c.backend == "openai"
        assert c.model_id == "gpt-5.4"
        assert c.reasoning_effort == "none"
        assert c.temperature == 0.0
        assert c.max_tokens == 512

    def test_reasoning_variant(self) -> None:
        c = OpenAIVariantConfig(
            backend="openai",
            model_id="gpt-5.4",
            tier="reasoning",
            reasoning_effort="high",
        )
        assert c.reasoning_effort == "high"

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="plain",
                model_path="oops",  # vLLM-only field
            )


class TestGoogleVariantConfig:
    def test_minimal(self) -> None:
        c = GoogleVariantConfig(
            backend="google", model_id="gemini-3-flash", tier="plain"
        )
        assert c.thinking_level == "minimal"

    def test_thinking_level_high(self) -> None:
        c = GoogleVariantConfig(
            backend="google",
            model_id="gemini-3-flash",
            tier="reasoning",
            thinking_level="high",
        )
        assert c.thinking_level == "high"

    def test_invalid_thinking_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GoogleVariantConfig(
                backend="google",
                model_id="gemini-3-flash",
                tier="plain",
                thinking_level="extreme",  # not in the Literal
            )


class TestVLLMHttpVariantConfig:
    def test_minimal(self) -> None:
        c = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/Qwen3.5-35B-A3B",
            tier="plain",
        )
        assert c.model_path == "Qwen/Qwen3.5-35B-A3B"
        assert c.dtype == "float16"
        assert c.quantization is None
        assert c.batch_size == 32
        assert c.gpu_memory_utilization == 0.9
        assert c.thinking is False

    def test_awq_quantization(self) -> None:
        c = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="some-org/quantized-model",
            tier="plain",
            quantization="awq",
            dtype="int4",
        )
        assert c.quantization == "awq"
        assert c.dtype == "int4"

    def test_effective_base_url_default(self, monkeypatch) -> None:
        """No explicit base_url, no VLLM_BASE_URL env → derive from
        :attr:`vllm_port` (the YAML default 8000 or per-variant
        override)."""
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        c = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/Qwen3.5-35B-A3B-FP8",
            tier="plain",
        )
        assert c.effective_base_url == "http://localhost:8000/v1"

    def test_effective_base_url_picks_up_env_var(self, monkeypatch) -> None:
        """When the SLURM template starts ``vllm serve`` on a
        $SLURM_JOB_ID-derived port and exports VLLM_BASE_URL, the
        backend client (which reads ``config.effective_base_url``)
        must follow — otherwise the smoke would still dial
        ``localhost:{vllm_port}`` regardless of which port the serve
        actually bound. Regression: pre-parametrization, two
        co-scheduled vLLM jobs on the same multi-GPU node both raced
        for port 8000; the second silently failed and the smoke kept
        talking to the first job's serve (wrong model loaded → 404)."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8742/v1")
        c = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/Qwen3.5-35B-A3B-FP8",
            tier="plain",
        )
        assert c.effective_base_url == "http://localhost:8742/v1"

    def test_explicit_base_url_wins_over_env_var(self, monkeypatch) -> None:
        """An explicit YAML ``base_url`` is for the remote-serve
        scenario (e.g., pointing the harness at a long-lived dev
        cluster) and must override any SLURM env-var the wrapper
        might have set."""
        monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8742/v1")
        c = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/Qwen3.5-35B-A3B-FP8",
            tier="plain",
            base_url="http://remote-vllm:8001/v1",
        )
        assert c.effective_base_url == "http://remote-vllm:8001/v1"

    def test_reasoning_effort_is_none(self) -> None:
        """:class:`benchmark_models.backends.openai.OpenAIModel` reads
        ``self.config.reasoning_effort`` unconditionally to decide
        whether to expand the token budget for OpenAI reasoning
        models. vLLM has no such concept (thinking is sent via
        chat-completions params instead), so the property is pinned
        to ``"none"`` to keep the parent on the plain-completion
        path. Regression: previously raised AttributeError on every
        tier-1/2 vLLM smoke."""
        c = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/Qwen3.5-35B-A3B",
            tier="plain",
        )
        assert c.reasoning_effort == "none"
        # Thinking-on vLLM variant must also stay "none" — the
        # parent's reasoning branch over-allocates tokens past
        # max_model_len, which would crash the serve.
        c_think = VLLMHttpVariantConfig(
            backend="vllm_http",
            model_path="Qwen/Qwen3.5-35B-A3B",
            tier="reasoning",
            thinking=True,
        )
        assert c_think.reasoning_effort == "none"


class TestDiscriminatedUnion:
    """Test that a raw dict is dispatched to the right subclass."""

    def _validate(self, data: dict) -> VariantConfig:
        # Pydantic discriminated union requires wrapping in a model
        # that has a field of type VariantConfig.
        from pydantic import TypeAdapter

        adapter = TypeAdapter(VariantConfig)
        return adapter.validate_python(data)

    def test_openai_dispatch(self) -> None:
        c = self._validate({"backend": "openai", "model_id": "gpt-5.4", "tier": "plain"})
        assert isinstance(c, OpenAIVariantConfig)

    def test_google_dispatch(self) -> None:
        c = self._validate(
            {"backend": "google", "model_id": "gemini-3-flash", "tier": "plain"}
        )
        assert isinstance(c, GoogleVariantConfig)

    def test_vllm_dispatch(self) -> None:
        c = self._validate(
            {"backend": "vllm_http", "model_path": "Qwen/Qwen3.5-35B-A3B", "tier": "plain"}
        )
        assert isinstance(c, VLLMHttpVariantConfig)

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._validate({"backend": "magic", "model_id": "x", "tier": "plain"})

    def test_cross_backend_fields_rejected(self) -> None:
        # model_path is vLLM-only; should fail on an openai config.
        with pytest.raises(ValidationError):
            self._validate(
                {
                    "backend": "openai",
                    "model_id": "gpt-5.4",
                    "tier": "plain",
                    "model_path": "/some/path",
                }
            )


# ============================================================================
# Tier-consistency validation
# ============================================================================


class TestTierConsistency:
    def test_self_consistency_on_non_reasoning_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="plain",
                self_consistency_n=3,
            )
        assert "self_consistency_n" in str(excinfo.value)

    def test_self_consistency_on_reasoning_ok(self) -> None:
        c = OpenAIVariantConfig(
            backend="openai",
            model_id="gpt-5.4",
            tier="reasoning",
            reasoning_effort="high",
            self_consistency_n=3,
            temperature=0.7,
        )
        assert c.self_consistency_n == 3

    def test_tools_on_non_tool_agent_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="plain",
                tools=["find_entity"],
            )
        assert "tools" in str(excinfo.value)

    def test_tools_on_tool_agent_ok(self) -> None:
        c = OpenAIVariantConfig(
            backend="openai",
            model_id="gpt-5.4",
            tier="tool_agent",
            sub_tier="3a",
            tools=["find_entity", "get_entity", "spatial_query"],
        )
        assert c.tools == ["find_entity", "get_entity", "spatial_query"]

    def test_tier3_max_tokens_on_non_tool_agent_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="plain",
                tier3_max_tokens=4096,
            )
        assert "tier3_max_tokens" in str(excinfo.value)

    def test_max_concurrent_on_non_tool_agent_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="reasoning",
                max_concurrent=8,
            )
        assert "max_concurrent" in str(excinfo.value)

    def test_web_search_timeout_on_non_tool_agent_raises(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="plain",
                web_search_timeout_seconds=30.0,
            )
        assert "web_search_timeout_seconds" in str(excinfo.value)

    def test_tier3_overrides_on_tool_agent_ok(self) -> None:
        c = OpenAIVariantConfig(
            backend="openai",
            model_id="gpt-5.4",
            tier="tool_agent",
            sub_tier="3b",
            tier3_max_tokens=4096,
            max_concurrent=2,
            web_search_timeout_seconds=20.0,
        )
        assert c.tier3_max_tokens == 4096
        assert c.max_concurrent == 2
        assert c.web_search_timeout_seconds == 20.0

    def test_sub_tier_required_for_tool_agent(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="tool_agent",
                # missing sub_tier
            )
        assert "sub_tier" in str(excinfo.value)

    def test_sub_tier_rejected_for_non_tool_agent(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            OpenAIVariantConfig(
                backend="openai",
                model_id="gpt-5.4",
                tier="plain",
                sub_tier="3a",
            )
        assert "sub_tier" in str(excinfo.value)


# ============================================================================
# Models catalogue
# ============================================================================


def _mini_catalogue_yaml() -> str:
    return yaml.dump(
        {
            "gpt-5.4-plain": {
                "backend": "openai",
                "model_id": "gpt-5.4",
                "tier": "plain",
            },
            "qwen35-a3b-plain": {
                "backend": "vllm_http",
                "model_path": "Qwen/Qwen3.5-35B-A3B",
                "tier": "plain",
                "dtype": "float16",
            },
        }
    )


class TestModelsCatalogue:
    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text(_mini_catalogue_yaml())
        cat = load_models_catalogue(path)
        assert len(cat) == 2
        assert "gpt-5.4-plain" in cat
        assert isinstance(cat.get("gpt-5.4-plain"), OpenAIVariantConfig)
        assert isinstance(cat.get("qwen35-a3b-plain"), VLLMHttpVariantConfig)

    def test_get_unknown_variant_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text(_mini_catalogue_yaml())
        cat = load_models_catalogue(path)
        with pytest.raises(KeyError) as excinfo:
            cat.get("nonexistent")
        assert "nonexistent" in str(excinfo.value)

    def test_list_variants(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text(_mini_catalogue_yaml())
        cat = load_models_catalogue(path)
        assert cat.list_variants() == ["gpt-5.4-plain", "qwen35-a3b-plain"]

    def test_empty_yaml_produces_empty_catalogue(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text("")
        cat = load_models_catalogue(path)
        assert len(cat) == 0

    def test_invalid_variant_in_yaml_fails(self, tmp_path: Path) -> None:
        path = tmp_path / "models.yaml"
        path.write_text(
            yaml.dump(
                {
                    "bad-variant": {
                        "backend": "openai",
                        # missing model_id
                        "tier": "plain",
                    }
                }
            )
        )
        with pytest.raises(ValidationError):
            load_models_catalogue(path)

    def test_load_bundled_models_yaml(self) -> None:
        cat = load_models_catalogue()
        assert isinstance(cat, ModelsCatalogue)
        # The bundled models.yaml carries the full slate.
        assert len(cat) >= 21
        # Spot-check a few known entries.
        assert "qwen35-a3b-plain" in cat
        assert isinstance(cat.get("qwen35-a3b-plain"), VLLMHttpVariantConfig)


# ============================================================================
# Experiment config
# ============================================================================


def _mini_experiment_yaml() -> str:
    return yaml.dump(
        {
            "name": "Smoke test",
            "languages": ["en"],
            "variants": ["gpt-5.4-plain"],
            "seeds": [42],
            "output_dir": "results/smoke",
        }
    )


class TestExperimentConfig:
    def test_load_minimal(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.yaml"
        path.write_text(_mini_experiment_yaml())
        cfg = load_experiment_config(path)
        assert cfg.name == "Smoke test"
        assert cfg.languages == ["en"]
        assert cfg.variants == ["gpt-5.4-plain"]
        assert cfg.seeds == [42]
        assert cfg.output_dir == "results/smoke"
        assert cfg.question_subset is None
        assert cfg.overrides is None

    def test_with_overrides(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.yaml"
        path.write_text(
            yaml.dump(
                {
                    "name": "Headline",
                    "languages": ["en", "de"],
                    "variants": ["gpt-5.4-plain", "qwen35-a3b-plain"],
                    "seeds": [42, 99],
                    "output_dir": "results/headline",
                    "overrides": {
                        "qwen35-a3b-plain": {
                            "batch_size": 128,
                            "max_tokens": 1024,
                        }
                    },
                }
            )
        )
        cfg = load_experiment_config(path)
        assert cfg.overrides is not None
        assert cfg.overrides["qwen35-a3b-plain"]["batch_size"] == 128

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.yaml"
        path.write_text(
            yaml.dump(
                {
                    "name": "Bad experiment",
                    # missing languages, variants, output_dir
                }
            )
        )
        with pytest.raises(ValidationError):
            load_experiment_config(path)

    def test_extra_field_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.yaml"
        data = yaml.safe_load(_mini_experiment_yaml())
        data["future_field"] = "oops"
        path.write_text(yaml.dump(data))
        with pytest.raises(ValidationError):
            load_experiment_config(path)

    def test_default_seeds(self, tmp_path: Path) -> None:
        path = tmp_path / "experiment.yaml"
        path.write_text(
            yaml.dump(
                {
                    "name": "No seeds specified",
                    "languages": ["en"],
                    "variants": ["gpt-5.4-plain"],
                    "output_dir": "results/default_seeds",
                }
            )
        )
        cfg = load_experiment_config(path)
        assert cfg.seeds == [42]
