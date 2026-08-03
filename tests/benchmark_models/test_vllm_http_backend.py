"""Tests for the vllm_http backend.

Mirrors test_deepseek_backend.py's pattern: the backend subclasses
:class:`OpenAIModel` so the test surface is small — construct via the
factory, assert the OpenAI client points at the right ``base_url``,
and that requests carry the right ``model`` (the HF repo path).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_config(**overrides):
    """Build a VLLMHttpVariantConfig with reasonable defaults.

    ``max_model_len`` deliberately uses the class default (16 K) so
    tests reflect production behaviour. Pass an explicit override
    only when a test needs a smaller window to exercise context-
    overflow paths.
    """
    from benchmark_models.config import VLLMHttpVariantConfig

    defaults = {
        "backend": "vllm_http",
        "tier": "plain",
        "model_path": "Qwen/Qwen3.5-35B-A3B",
        "vllm_port": 8000,
        "max_num_seqs": 64,
        "dtype": "float16",
        "batch_size": 32,
    }
    defaults.update(overrides)
    return VLLMHttpVariantConfig(**defaults)


class TestVLLMHttpVariantConfig:
    def test_defaults(self) -> None:
        config = _make_config()
        assert config.backend == "vllm_http"
        # ``base_url`` is None by default; the effective URL is
        # derived from ``vllm_port`` so port-only overrides stay
        # coherent with the SLURM template's local serve.
        assert config.base_url is None
        assert config.vllm_port == 8000
        assert config.effective_base_url == "http://localhost:8000/v1"
        # Class default sized to 32 K to cover the worst-case input
        # across every tier path in the catalogue with comfortable
        # margin: tier-3a/3c Step-6 with country polygons inlined
        # (~20–25 K accumulated context), tier-1 oracle-raw on dense
        # entities (~12 K), and any prompted-CoT tier-2 outputs. 32 K
        # is also Qwen3.5-27B's native ceiling without rope_scaling/YaRN.
        assert config.max_model_len == 32768
        assert config.model_path == "Qwen/Qwen3.5-35B-A3B"

    def test_model_id_aliases_model_path(self) -> None:
        """``model_id`` is a derived property over ``model_path`` so
        the parent OpenAIModel's request-shaping code resolves the
        right value without a separate dispatch path."""
        config = _make_config(model_path="Qwen/Qwen3.5-27B")
        assert config.model_id == "Qwen/Qwen3.5-27B"

    def test_port_override_propagates_to_effective_base_url(self) -> None:
        """Single source of truth: changing ``vllm_port`` must update
        the URL the backend client uses, without the user also having
        to set ``base_url``. Regression: previously these two fields
        could desync silently."""
        config = _make_config(vllm_port=9001)
        assert config.effective_base_url == "http://localhost:9001/v1"

    def test_explicit_base_url_wins_over_derived(self) -> None:
        """When ``base_url`` is set, it overrides the derivation so
        the harness can target a non-local vLLM serve."""
        config = _make_config(base_url="http://my-vllm-host:9000/v1")
        assert config.effective_base_url == "http://my-vllm-host:9000/v1"


class TestVLLMHttpModelConstruction:
    def test_constructor_points_openai_client_at_vllm_base_url(self) -> None:
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        with patch("benchmark_models.backends.vllm_http.OpenAI") as OpenAI_cls:
            VLLMHttpModel(
                "qwen35-a3b-plain",
                _make_config(),  # defaults: port=8000, base_url=None
            )
        # OpenAI gets constructed once on the parent's __init__ (which
        # we override to nothing useful here) and once on the
        # VLLMHttpModel branch where we explicitly point at the vLLM
        # base_url. The LAST call is the one we care about — and it
        # must carry the derived effective_base_url.
        last_call_kwargs = OpenAI_cls.call_args_list[-1].kwargs
        assert last_call_kwargs.get("base_url") == "http://localhost:8000/v1"

    def test_constructor_uses_port_override_in_effective_url(self) -> None:
        """Bumping ``vllm_port`` flows through to the OpenAI client's
        ``base_url`` without any explicit ``base_url`` setting."""
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        with patch("benchmark_models.backends.vllm_http.OpenAI") as OpenAI_cls:
            VLLMHttpModel(
                "qwen35-a3b-plain",
                _make_config(vllm_port=9001),
            )
        last_call_kwargs = OpenAI_cls.call_args_list[-1].kwargs
        assert last_call_kwargs.get("base_url") == "http://localhost:9001/v1"

    def test_constructor_accepts_explicit_api_key(self) -> None:
        """``--api-key`` from the CLI is honored — useful when the
        vLLM serve was launched with ``--api-key``."""
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        with patch("benchmark_models.backends.vllm_http.OpenAI") as OpenAI_cls:
            VLLMHttpModel(
                "qwen35-a3b-plain",
                _make_config(),
                api_key="custom-key",
            )
        last_call_kwargs = OpenAI_cls.call_args_list[-1].kwargs
        assert last_call_kwargs.get("api_key") == "custom-key"

    def test_constructor_supplies_dummy_api_key_when_missing(self) -> None:
        """vLLM serve doesn't require auth by default. The backend
        passes a dummy key when the caller didn't supply one — the
        OpenAI SDK refuses to construct without any key at all."""
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        with patch("benchmark_models.backends.vllm_http.OpenAI") as OpenAI_cls:
            VLLMHttpModel(
                "qwen35-a3b-plain",
                _make_config(),
            )
        last_call_kwargs = OpenAI_cls.call_args_list[-1].kwargs
        # Some non-empty string was passed (the SDK accepts "EMPTY"
        # by convention from local-LLM tutorials).
        assert last_call_kwargs.get("api_key")


class TestVLLMHttpFactoryDispatch:
    def test_registry_dispatches_vllm_http_variant_to_vllm_http_model(
        self,
    ) -> None:
        """``create_model`` resolves a ``vllm_http`` config into a
        ``VLLMHttpModel`` instance — not the old in-process one."""
        from benchmark_models.backends.vllm_http import VLLMHttpModel
        from benchmark_models.registry import create_model

        config = _make_config()
        with patch("benchmark_models.backends.vllm_http.OpenAI"):
            model = create_model("qwen35-a3b-plain", config)
        assert isinstance(model, VLLMHttpModel)


class TestVLLMHttpHealthCheck:
    def test_health_check_returns_true_on_200(self) -> None:
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        with patch("benchmark_models.backends.vllm_http.httpx") as httpx_mod:
            response = MagicMock()
            response.status_code = 200
            httpx_mod.get.return_value = response
            assert VLLMHttpModel.health_check("http://localhost:8000/v1")

    def test_health_check_returns_false_on_connection_error(self) -> None:
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        with patch("benchmark_models.backends.vllm_http.httpx") as httpx_mod:
            httpx_mod.get.side_effect = ConnectionError("refused")
            assert not VLLMHttpModel.health_check("http://localhost:8000/v1")

    def test_health_check_returns_false_on_non_200(self) -> None:
        from benchmark_models.backends.vllm_http import VLLMHttpModel

        with patch("benchmark_models.backends.vllm_http.httpx") as httpx_mod:
            response = MagicMock()
            response.status_code = 503
            httpx_mod.get.return_value = response
            assert not VLLMHttpModel.health_check("http://localhost:8000/v1")
