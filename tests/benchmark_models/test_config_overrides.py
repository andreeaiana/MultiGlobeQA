"""Tests for CLI sampling-knob overrides plumbed through create_model.

D11 added five CLI flags whose values land on the variant config via the
``config_overrides`` parameter to :func:`create_model`. The override
pipeline must:

* Apply the override on top of the YAML-defined config.
* Re-run pydantic validators so invalid combinations
  (e.g. ``self_consistency_n=3`` with ``temperature=0``) fail loud
  rather than silently producing N identical SC samples.
* Leave the original catalogue config untouched (immutability).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark_models.config import load_models_catalogue
from benchmark_models.registry import create_model

_CATALOGUE = load_models_catalogue()


class TestConfigOverrideApplication:
    def test_temperature_override_applies(self) -> None:
        # Pick a vLLM variant; vLLM doesn't need API keys so create_model
        # would proceed past validation but we only need it to apply the
        # override. We catch the inevitable construction step with a mock,
        # via a backend that uses VLLMHttpVariantConfig.
        # Easier: import the GoogleModel constructor and inspect what
        # config arrives. We don't actually construct the model — instead
        # we re-run the pure validation pipeline directly.
        config = _CATALOGUE.get("gemini-3-flash-plain")
        merged = config.model_dump()
        merged["temperature"] = 0.7
        new_config = type(config).model_validate(merged)
        assert new_config.temperature == 0.7
        assert new_config.top_p == config.top_p
        # Original catalogue config is untouched.
        assert config.temperature == 0.0

    def test_top_p_override(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        merged = config.model_dump()
        merged["top_p"] = 0.9
        new_config = type(config).model_validate(merged)
        assert new_config.top_p == 0.9

    def test_seed_override(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        merged = config.model_dump()
        merged["seed"] = 1234
        new_config = type(config).model_validate(merged)
        assert new_config.seed == 1234

    def test_max_tokens_override_tier1(self) -> None:
        # Tier 1/2 use base ``max_tokens`` (per-call response budget).
        config = _CATALOGUE.get("gemini-3-flash-plain")
        merged = config.model_dump()
        merged["max_tokens"] = 2048
        new_config = type(config).model_validate(merged)
        assert new_config.max_tokens == 2048

    def test_self_consistency_n_override_requires_temperature(self) -> None:
        # The tier-2 workhorse variant is now greedy (self-consistency
        # removed 2026-07-22, temperature reset to 0). Overriding SC to 5
        # validates only when a temperature > 0 is supplied alongside it.
        config = _CATALOGUE.get("qwen35-a3b-thinking-tier2")
        merged = config.model_dump()
        merged["self_consistency_n"] = 5
        merged["temperature"] = 0.7
        new_config = type(config).model_validate(merged)
        assert new_config.self_consistency_n == 5
        assert new_config.temperature == 0.7

    def test_self_consistency_n_with_temp_zero_is_rejected(self) -> None:
        # The validator rejects SC>1 + temp=0; this is the headline
        # invariant — guards against silent cost inflation.
        config = _CATALOGUE.get("gemini-3-flash-plain")  # tier 1, temp 0
        merged = config.model_dump()
        merged["self_consistency_n"] = 3
        # SC>1 only makes sense on tier 'reasoning', which is the FIRST
        # check that fires.
        with pytest.raises(ValidationError, match="self_consistency_n"):
            type(config).model_validate(merged)

    def test_self_consistency_n_tier2_with_temp_zero_is_rejected(self) -> None:
        # On a tier-2 variant, the SC>1+temp=0 check fires (not the
        # tier-mismatch check).
        config = _CATALOGUE.get("gemini-3-flash-thinking-tier2")  # tier 2
        merged = config.model_dump()
        merged["temperature"] = 0.0
        merged["self_consistency_n"] = 3
        with pytest.raises(ValidationError, match="temperature > 0"):
            type(config).model_validate(merged)


class TestCreateModelOverrides:
    """Round-trip the override through create_model and inspect the
    resulting Model's config attribute (or equivalent)."""

    def test_create_model_applies_override_dict(self) -> None:
        # Use the dummy variant path? The dummy isn't in the catalogue.
        # Use a vLLM config, but mock VLLMModel construction to avoid
        # loading actual weights. Simpler approach: assert on the
        # config produced when create_model is called by checking via
        # one of the API backends with a no-network construction.
        #
        # Actually create_model for api variants doesn't network-call
        # until predict. Let's call create_model with config_overrides
        # and inspect the returned model.config.
        config = _CATALOGUE.get("gemini-3-flash-plain")
        model = create_model(
            "gemini-3-flash-plain",
            config,
            api_key="dummy-key-for-construction",
            config_overrides={"temperature": 0.5, "seed": 99},
        )
        try:
            # GoogleModel stores config on the instance.
            assert model.config.temperature == 0.5  # type: ignore[attr-defined]
            assert model.config.seed == 99  # type: ignore[attr-defined]
        finally:
            model.close()

    def test_create_model_validates_invalid_combo(self) -> None:
        # Override that violates the validator should raise at
        # create_model time, before any backend construction.
        config = _CATALOGUE.get("gemini-3-flash-thinking-tier2")
        with pytest.raises(ValidationError):
            create_model(
                "gemini-3-flash-thinking-tier2",
                config,
                api_key="dummy",
                config_overrides={
                    "temperature": 0.0,
                    "self_consistency_n": 3,
                },
            )

    def test_create_model_no_overrides_keeps_original_config(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        original_temperature = config.temperature
        model = create_model(
            "gemini-3-flash-plain",
            config,
            api_key="dummy",
            config_overrides=None,
        )
        try:
            assert model.config.temperature == original_temperature  # type: ignore[attr-defined]
        finally:
            model.close()
