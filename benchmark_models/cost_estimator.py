"""Per-variant cost / token pre-flight estimator.

Reads a benchmark count + variant config and prints an approximate cost
estimate before the user spends real dollars on a headline run. The
estimator is **intentionally approximate** — per-row token counts are
tier-based rules of thumb, not measured. The point is to catch
order-of-magnitude surprises (e.g. "tier 3 SC@3 on 1M rows is $50k, not
$5k") before launch, not to predict the bill to the dollar.

Provider prices are baked in below in :data:`_PROVIDER_PRICES`. The
table needs **manual updating** as providers change their rate cards —
the file is annotated with the last-known-good date so a future
maintainer sees whether to trust it.

Last verified: 2026-04 (project plan reference prices). Prices are
quoted in USD per 1M tokens. Bump when the rate card changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmark_models.config import VariantConfig


# ---------------------------------------------------------------------------
# Provider price table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProviderPricing:
    """Per-1M-token USD prices for one provider."""

    input_per_million: float
    output_per_million: float
    notes: str = ""


# Caveat emptor: these are reference prices used for pre-flight
# estimation only. Real bills depend on rate-card changes, batch-API
# discounts, prompt caching, etc. Update this table when a provider
# changes its prices.
_PROVIDER_PRICES: dict[str, _ProviderPricing] = {
    "openai": _ProviderPricing(
        input_per_million=5.00,
        output_per_million=20.00,
        notes="GPT-5.4 standard rate card (estimate; verify before launch).",
    ),
    "google": _ProviderPricing(
        input_per_million=0.30,
        output_per_million=2.50,
        notes="Gemini 3 Flash standard rate card (estimate).",
    ),
    "vllm_http": _ProviderPricing(
        input_per_million=0.0,
        output_per_million=0.0,
        notes="Self-hosted; cost is GPU-hour, not API cost.",
    ),
}


# ---------------------------------------------------------------------------
# Per-tier token rules of thumb
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TierTokenBudget:
    """Approximate input / output tokens per question for one tier."""

    input_tokens_per_row: int
    output_tokens_per_row: int


# Empirical-ish rules of thumb. Tier 1 plain has small prompts +
# short JSON answers. Tier 1 oracle inflates input (gold triples
# injected). Tier 2 doubles output (reasoning trace). Tier 3 has
# large system prompts (tools) and multi-step agent loops with
# per-step LLM calls; the totals below assume ~4 steps average.
_TIER_BUDGETS: dict[str, _TierTokenBudget] = {
    "plain": _TierTokenBudget(input_tokens_per_row=300, output_tokens_per_row=60),
    "reasoning": _TierTokenBudget(input_tokens_per_row=400, output_tokens_per_row=500),
    "tool_agent": _TierTokenBudget(
        input_tokens_per_row=8000,   # 4 steps × ~2000 tokens of accumulated context
        output_tokens_per_row=1500,  # 4 steps × ~375 tokens of code / reasoning
    ),
}

# When ``--oracle-mode`` is set, the input prompt carries the gold
# triples (one row per question). Rough average: 1500 tokens of
# triple context per question.
_ORACLE_EXTRA_INPUT_TOKENS = 1500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEstimate:
    """Pre-flight cost estimate for one (variant, n_rows) pair."""

    variant: str
    provider: str
    tier: str
    n_rows: int
    self_consistency_n: int
    oracle_mode: str | None
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


def estimate_cost(
    variant: str,
    config: VariantConfig,
    n_rows: int,
    *,
    oracle_mode: str | None = None,
) -> CostEstimate:
    """Estimate the API cost of running ``variant`` over ``n_rows`` questions.

    The estimate uses per-tier token rules of thumb (no per-row token
    counting) and the baked-in price table. Output is intentionally
    approximate — accurate enough to catch order-of-magnitude surprises,
    not to predict the bill to the cent.

    Args:
        variant: Variant name (used for the report only).
        config: Validated variant config (carries backend, tier,
            self_consistency_n, etc.).
        n_rows: Number of benchmark questions the variant will see.
        oracle_mode: When set, adds the gold-triples prompt-bloat
            estimate (one shot per question).

    Returns:
        A :class:`CostEstimate` with token counts and USD figures.
    """
    if n_rows < 0:
        raise ValueError(f"n_rows must be >= 0, got {n_rows}")
    budget = _TIER_BUDGETS[config.tier]
    pricing = _PROVIDER_PRICES[config.backend]
    sc_n = config.self_consistency_n if config.tier == "reasoning" else 1
    extra_input = _ORACLE_EXTRA_INPUT_TOKENS if oracle_mode is not None else 0

    # SC@N runs the same prompt N times with stochastic decoding. Input
    # tokens are paid once per sample (without provider-side caching);
    # output tokens scale with N.
    input_per_row = (budget.input_tokens_per_row + extra_input) * sc_n
    output_per_row = budget.output_tokens_per_row * sc_n

    input_tokens = input_per_row * n_rows
    output_tokens = output_per_row * n_rows
    input_cost = input_tokens / 1_000_000 * pricing.input_per_million
    output_cost = output_tokens / 1_000_000 * pricing.output_per_million

    return CostEstimate(
        variant=variant,
        provider=config.backend,
        tier=config.tier,
        n_rows=n_rows,
        self_consistency_n=sc_n,
        oracle_mode=oracle_mode,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
    )


def format_estimate_table(estimates: list[CostEstimate]) -> str:
    """Render a list of :class:`CostEstimate` as a fixed-width table."""
    header = (
        f"{'variant':<40} {'provider':<10} {'tier':<11} {'n_rows':>8} "
        f"{'sc':>3} {'in_tok':>14} {'out_tok':>14} {'cost_usd':>10}"
    )
    rows = [header, "-" * len(header)]
    total_in = total_out = 0
    total_cost = 0.0
    for e in estimates:
        rows.append(
            f"{e.variant:<40} {e.provider:<10} {e.tier:<11} {e.n_rows:>8,d} "
            f"{e.self_consistency_n:>3d} {e.input_tokens:>14,d} "
            f"{e.output_tokens:>14,d} ${e.total_cost_usd:>9,.2f}"
        )
        total_in += e.input_tokens
        total_out += e.output_tokens
        total_cost += e.total_cost_usd
    rows.append("-" * len(header))
    rows.append(
        f"{'TOTAL':<40} {'':<10} {'':<11} {'':>8} {'':>3} "
        f"{total_in:>14,d} {total_out:>14,d} ${total_cost:>9,.2f}"
    )
    return "\n".join(rows)


__all__ = [
    "CostEstimate",
    "estimate_cost",
    "format_estimate_table",
]
