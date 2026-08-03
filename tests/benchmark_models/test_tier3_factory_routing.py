"""Tests for the tier-3 Smolagents factory's per-provider routing.

The factory must select the right LiteLLM model id + auth path for
each backend. DeepSeek and OpenAI compatibility is exercised in
``test_deepseek_backend.py``; this file pins the Google routing,
which has two distinct paths depending on ``GOOGLE_GENAI_USE_VERTEXAI``:

* Vertex AI (``vertex_ai/<model>`` + project/location kwargs, picks
  up ``GOOGLE_APPLICATION_CREDENTIALS`` automatically).
* Public AI Studio endpoint (``gemini/<model>`` + api_key).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestTier3SmolagentsFactoryRoutesGoogle:
    def test_google_with_vertex_flag_uses_vertex_ai_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``GOOGLE_GENAI_USE_VERTEXAI=true``, the factory must
        emit ``vertex_ai/<model>`` so LiteLLM routes to Vertex rather
        than the public Gemini endpoint."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="google",
                model_id="gemini-3-flash-preview",
            )
        assert litellm_class.called
        kwargs = litellm_class.call_args.kwargs
        assert kwargs["model_id"] == "vertex_ai/gemini-3-flash-preview"
        assert kwargs["vertex_project"] == "test-project"
        assert kwargs["vertex_location"] == "us-central1"
        # api_key isn't passed on the Vertex path — auth comes from
        # GOOGLE_APPLICATION_CREDENTIALS, which LiteLLM picks up itself.
        assert "api_key" not in kwargs

    def test_google_without_vertex_flag_uses_gemini_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default (no Vertex flag) → AI Studio endpoint, api_key path."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="google",
                model_id="gemini-3-flash-preview",
                api_key="aistudio-key",
            )
        assert litellm_class.called
        kwargs = litellm_class.call_args.kwargs
        assert kwargs["model_id"] == "gemini/gemini-3-flash-preview"
        assert kwargs["api_key"] == "aistudio-key"
        # No vertex_* kwargs on the AI Studio path.
        assert "vertex_project" not in kwargs
        assert "vertex_location" not in kwargs

    @pytest.mark.parametrize("flag_value", ["false", "0", "no", ""])
    def test_google_falsy_vertex_flag_uses_gemini_prefix(
        self, monkeypatch: pytest.MonkeyPatch, flag_value: str
    ) -> None:
        """Falsy strings for the flag stay on the AI Studio path."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", flag_value)
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="google",
                model_id="gemini-3-flash-preview",
                api_key="aistudio-key",
            )
        kwargs = litellm_class.call_args.kwargs
        assert kwargs["model_id"] == "gemini/gemini-3-flash-preview"

    @pytest.mark.parametrize("flag_value", ["true", "True", "TRUE", "1", "yes"])
    def test_google_truthy_vertex_flag_variants(
        self, monkeypatch: pytest.MonkeyPatch, flag_value: str
    ) -> None:
        """Truthy variants all route through Vertex."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", flag_value)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "l")
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="google",
                model_id="gemini-3-flash-preview",
            )
        assert (
            litellm_class.call_args.kwargs["model_id"]
            == "vertex_ai/gemini-3-flash-preview"
        )

    def test_google_vertex_without_project_or_location_omits_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When project/location env vars are absent, the factory omits
        the kwargs rather than passing ``None`` — LiteLLM falls back to
        its own env-var probing."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="google",
                model_id="gemini-3-flash-preview",
            )
        kwargs = litellm_class.call_args.kwargs
        assert kwargs["model_id"] == "vertex_ai/gemini-3-flash-preview"
        assert "vertex_project" not in kwargs
        assert "vertex_location" not in kwargs


class TestTier3SmolagentsFactoryRoutesVllmHttp:
    def test_vllm_http_uses_openai_prefix_and_api_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``vllm_http`` routes through LiteLLM with the OpenAI
        adapter and a custom ``api_base`` pointing at the local serve.
        Default base_url matches the SLURM template's in-job binding."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="vllm_http",
                model_id="Qwen/Qwen3.5-35B-A3B",
            )
        kwargs = litellm_class.call_args.kwargs
        assert kwargs["model_id"] == "openai/Qwen/Qwen3.5-35B-A3B"
        assert kwargs["api_base"] == "http://localhost:8000/v1"
        # vLLM serve doesn't require auth by default — backend supplies
        # a dummy key so LiteLLM's auth signing doesn't reject.
        assert kwargs["api_key"]

    def test_vllm_http_explicit_base_url_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit ``base_url`` kwarg on the factory call wins
        over both the default and the env var."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.setenv("VLLM_BASE_URL", "http://from-env:8000/v1")
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="vllm_http",
                model_id="Qwen/Qwen3.5-35B-A3B",
                base_url="http://explicit:9000/v1",
            )
        assert (
            litellm_class.call_args.kwargs["api_base"]
            == "http://explicit:9000/v1"
        )

    def test_vllm_http_env_var_overrides_default_when_no_explicit_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ``base_url`` isn't passed, the factory reads
        ``VLLM_BASE_URL`` from the env — useful for SLURM scripts that
        export the endpoint and run the harness without re-plumbing
        the config."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.setenv("VLLM_BASE_URL", "http://from-env:8000/v1")
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="vllm_http",
                model_id="Qwen/Qwen3.5-35B-A3B",
            )
        assert (
            litellm_class.call_args.kwargs["api_base"]
            == "http://from-env:8000/v1"
        )

    def test_vllm_http_explicit_api_key_param_is_honored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The factory's explicit ``api_key`` parameter must win over
        the dummy fallback. Regression: earlier the vllm_http branch
        only read kwargs.pop('api_key'), ignoring the standard
        positional/keyword arg used everywhere else in the dispatch."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="vllm_http",
                model_id="Qwen/Qwen3.5-35B-A3B",
                api_key="my-secret-token",
            )
        assert (
            litellm_class.call_args.kwargs["api_key"] == "my-secret-token"
        )

    def test_vllm_http_falls_back_to_dummy_key_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No explicit key → dummy 'EMPTY' so LiteLLM constructs cleanly
        against an unauthenticated ``vllm serve``."""
        from benchmark_models.tier3 import _build_smolagents_model

        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        with patch("smolagents.LiteLLMModel") as litellm_class:
            _build_smolagents_model(
                provider="vllm_http",
                model_id="Qwen/Qwen3.5-35B-A3B",
            )
        assert litellm_class.call_args.kwargs["api_key"] == "EMPTY"
