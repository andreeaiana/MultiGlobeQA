"""Tests for the pre-flight cost estimator.

The estimator is intentionally approximate — it multiplies per-tier
token rules of thumb by a price table and reports USD. The tests pin:

* Token counts scale linearly with ``n_rows``.
* Self-consistency multiplier applies to both input and output tokens.
* Oracle-mode bumps input only.
* The vLLM "free" entry produces $0 regardless of token volume.
* Unknown variant raises a clear error.

These are pure pricing-arithmetic tests; the estimator never touches
the network.
"""

from __future__ import annotations

import pytest

from benchmark_models.config import load_models_catalogue
from benchmark_models.cost_estimator import (
    _PROVIDER_PRICES,
    _TIER_BUDGETS,
    estimate_cost,
    format_estimate_table,
)

_CATALOGUE = load_models_catalogue()


class TestEstimateCost:
    def test_tier1_plain_scales_linearly(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        e1 = estimate_cost("gemini-3-flash-plain", config, 1000)
        e2 = estimate_cost("gemini-3-flash-plain", config, 10000)
        assert e2.input_tokens == 10 * e1.input_tokens
        assert e2.output_tokens == 10 * e1.output_tokens
        assert abs(e2.total_cost_usd - 10 * e1.total_cost_usd) < 1e-9

    def test_tier1_plain_token_counts(self) -> None:
        # 300 input + 60 output × 1000 rows = 300k in, 60k out.
        config = _CATALOGUE.get("gemini-3-flash-plain")
        e = estimate_cost("gemini-3-flash-plain", config, 1000)
        assert e.input_tokens == 300_000
        assert e.output_tokens == 60_000

    def test_tier2_greedy_token_counts(self) -> None:
        # qwen35-a3b-thinking-tier2 is greedy single-sample (self-consistency
        # removed — it was the only variant using it, which made
        # its T2 numbers a 3-sample majority vote while every other model's
        # T2 was single-sample). Tier-2 budget = 400 in / 500 out per row.
        config = _CATALOGUE.get("qwen35-a3b-thinking-tier2")
        e = estimate_cost("qwen35-a3b-thinking-tier2", config, 1000)
        assert e.self_consistency_n == 1
        assert e.input_tokens == 400_000
        assert e.output_tokens == 500_000

    def test_sc_multiplier_only_applies_to_tier_reasoning(self) -> None:
        # Tier-1 variants ignore self_consistency_n even when set
        # (the validator would reject SC>1 on tier 1 anyway, but the
        # estimator's safety belt is independent).
        config = _CATALOGUE.get("gemini-3-flash-plain")
        e = estimate_cost("gemini-3-flash-plain", config, 1000)
        assert e.self_consistency_n == 1

    def test_oracle_mode_bumps_input_not_output(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        base = estimate_cost("gemini-3-flash-plain", config, 1000)
        with_oracle = estimate_cost(
            "gemini-3-flash-plain", config, 1000, oracle_mode="structured",
        )
        # 1000 rows × 1500 extra input tokens per row = 1.5M extra input.
        assert with_oracle.input_tokens == base.input_tokens + 1_500_000
        # Output budget unchanged.
        assert with_oracle.output_tokens == base.output_tokens

    def test_vllm_variant_is_free(self) -> None:
        config = _CATALOGUE.get("qwen35-a3b-plain")
        e = estimate_cost("qwen35-a3b-plain", config, 100_000)
        assert e.input_cost_usd == 0.0
        assert e.output_cost_usd == 0.0
        assert e.total_cost_usd == 0.0

    def test_gemini_costing_matches_price_table(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        e = estimate_cost("gemini-3-flash-plain", config, 1000)
        pricing = _PROVIDER_PRICES["google"]
        expected_input = e.input_tokens / 1_000_000 * pricing.input_per_million
        expected_output = e.output_tokens / 1_000_000 * pricing.output_per_million
        assert abs(e.input_cost_usd - expected_input) < 1e-9
        assert abs(e.output_cost_usd - expected_output) < 1e-9

    def test_negative_n_rows_raises(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        with pytest.raises(ValueError, match="n_rows must be >= 0"):
            estimate_cost("gemini-3-flash-plain", config, -1)

    def test_zero_rows_is_zero_cost(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        e = estimate_cost("gemini-3-flash-plain", config, 0)
        assert e.input_tokens == 0
        assert e.output_tokens == 0
        assert e.total_cost_usd == 0.0


class TestTierBudgets:
    def test_tier3_input_dominates_tier1(self) -> None:
        # Sanity: agent loop has much larger input than plain Q&A.
        assert (
            _TIER_BUDGETS["tool_agent"].input_tokens_per_row
            > 5 * _TIER_BUDGETS["plain"].input_tokens_per_row
        )

    def test_tier2_output_dominates_tier1(self) -> None:
        # Reasoning trace makes tier-2 output much larger than tier-1.
        assert (
            _TIER_BUDGETS["reasoning"].output_tokens_per_row
            > 5 * _TIER_BUDGETS["plain"].output_tokens_per_row
        )


class TestFormatTable:
    def test_table_includes_totals_row(self) -> None:
        config = _CATALOGUE.get("gemini-3-flash-plain")
        e1 = estimate_cost("gemini-3-flash-plain", config, 1000)
        e2 = estimate_cost("gemini-3-flash-plain", config, 2000)
        table = format_estimate_table([e1, e2])
        assert "TOTAL" in table
        # 3000 rows × 300 input tokens = 900,000 input total.
        assert "900,000" in table
        # 3000 rows × 60 output tokens = 180,000 output total.
        assert "180,000" in table
