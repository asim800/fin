"""Capped gains and opportunity cost analysis for collar positions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from scipy.stats import norm

from src.structures import CollarPosition, CollarPricing


@dataclass
class UpsideScenario:
    """Result for a single upside scenario.

    Attributes:
        level_pct: Upside level as percentage (e.g., 0.10 for 10%).
        target_price: Price at this upside level.
        probability: Probability of reaching or exceeding this level.
        uncapped_gain: Gain without collar cap (per share).
        capped_gain: Gain with collar cap (per share).
        opportunity_cost: Foregone gain due to cap (per share).
        expected_opportunity_cost: Probability-weighted opportunity cost.
    """

    level_pct: float
    target_price: float
    probability: float
    uncapped_gain: float
    capped_gain: float
    opportunity_cost: float
    expected_opportunity_cost: float


@dataclass
class CappedGainsResult:
    """Complete capped gains analysis result.

    Attributes:
        collar: The collar position analyzed.
        pricing: Pricing information.
        upside_scenarios: List of upside scenario results.
        total_expected_opportunity_cost: Sum of probability-weighted costs.
        breakeven_upside: Upside percentage needed to break even.
        probability_model: Model used for probability calculation.
    """

    collar: CollarPosition
    pricing: CollarPricing
    upside_scenarios: list[UpsideScenario]
    total_expected_opportunity_cost: float
    breakeven_upside: float
    probability_model: Literal["lognormal", "empirical"]


class CappedGainsAnalyzer:
    """Analyzes opportunity cost of capped upside in collar positions.

    Calculates probability-weighted opportunity costs for discrete upside
    scenarios, showing the cost of capping gains via the short call.

    Args:
        volatility: Annualized volatility for probability calculations.
        risk_free_rate: Risk-free rate for forward price calculation.
        dividend_yield: Dividend yield (default 0.0).
    """

    # Default upside levels to analyze
    DEFAULT_UPSIDE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25]

    def __init__(
        self,
        volatility: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ) -> None:
        self.volatility = volatility
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    def analyze(
        self,
        collar: CollarPosition,
        pricing: CollarPricing,
        expiry_years: float,
        upside_levels: list[float] | None = None,
        probability_model: Literal["lognormal"] = "lognormal",
    ) -> CappedGainsResult:
        """Analyze capped gains across upside scenarios.

        Args:
            collar: The collar position to analyze.
            pricing: Collar pricing information.
            expiry_years: Time to expiry in years.
            upside_levels: List of upside percentages to analyze.
                          Defaults to [5%, 10%, 15%, 20%, 25%].
            probability_model: Probability model to use ("lognormal").

        Returns:
            CappedGainsResult with detailed scenario breakdown.
        """
        if upside_levels is None:
            upside_levels = self.DEFAULT_UPSIDE_LEVELS.copy()

        entry_price = collar.underlying_entry_price
        call_strike = collar.short_call.strike
        net_premium = pricing.net_premium

        # Max capped gain is at call strike (minus net premium)
        max_capped_gain = (call_strike - entry_price) - net_premium

        scenarios = []

        for level in upside_levels:
            target_price = entry_price * (1 + level)

            # Calculate probability of reaching this level
            prob = self._probability_above(
                entry_price, target_price, expiry_years, probability_model
            )

            # Uncapped gain (just stock appreciation minus premium)
            uncapped_gain = (target_price - entry_price) - net_premium

            # Capped gain (limited by call strike)
            if target_price <= call_strike:
                capped_gain = uncapped_gain
            else:
                capped_gain = max_capped_gain

            # Opportunity cost
            opportunity_cost = max(0, uncapped_gain - capped_gain)

            # Expected (probability-weighted) opportunity cost
            expected_cost = prob * opportunity_cost

            scenarios.append(
                UpsideScenario(
                    level_pct=level,
                    target_price=target_price,
                    probability=prob,
                    uncapped_gain=uncapped_gain,
                    capped_gain=capped_gain,
                    opportunity_cost=opportunity_cost,
                    expected_opportunity_cost=expected_cost,
                )
            )

        total_expected_cost = sum(s.expected_opportunity_cost for s in scenarios)
        breakeven = self.calculate_breakeven_upside(collar, pricing)

        return CappedGainsResult(
            collar=collar,
            pricing=pricing,
            upside_scenarios=scenarios,
            total_expected_opportunity_cost=total_expected_cost,
            breakeven_upside=breakeven,
            probability_model=probability_model,
        )

    def _probability_above(
        self,
        current_price: float,
        target_price: float,
        expiry_years: float,
        model: Literal["lognormal"],
    ) -> float:
        """Calculate probability of price exceeding target at expiry.

        Args:
            current_price: Current underlying price.
            target_price: Target price level.
            expiry_years: Time to expiry in years.
            model: Probability model to use.

        Returns:
            Probability (0 to 1) of exceeding target price.
        """
        if model == "lognormal":
            return self._lognormal_prob_above(current_price, target_price, expiry_years)
        else:
            raise ValueError(f"Unknown probability model: {model}")

    def _lognormal_prob_above(
        self,
        S0: float,
        K: float,
        T: float,
    ) -> float:
        """Calculate P(S_T > K) under lognormal model.

        Under risk-neutral measure (GBM):
        ln(S_T/S_0) ~ N((r-q-sigma^2/2)*T, sigma^2*T)

        P(S_T > K) = N(d2) where d2 is from BSM formula.
        """
        if T <= 0:
            return 1.0 if S0 > K else 0.0

        mu = self.risk_free_rate - self.dividend_yield
        sigma = self.volatility

        # d2 parameter (same as BSM)
        d2 = (math.log(S0 / K) + (mu - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

        return float(norm.cdf(d2))

    def to_dataframe(self, result: CappedGainsResult) -> pd.DataFrame:
        """Convert CappedGainsResult to DataFrame for display.

        Args:
            result: CappedGainsResult to convert.

        Returns:
            DataFrame with scenario analysis in tabular format.
        """
        rows = []
        for scenario in result.upside_scenarios:
            rows.append(
                {
                    "upside_level": f"{scenario.level_pct * 100:.0f}%",
                    "target_price": scenario.target_price,
                    "probability": scenario.probability,
                    "probability_pct": f"{scenario.probability * 100:.1f}%",
                    "uncapped_gain": scenario.uncapped_gain,
                    "capped_gain": scenario.capped_gain,
                    "opportunity_cost": scenario.opportunity_cost,
                    "expected_cost": scenario.expected_opportunity_cost,
                }
            )

        return pd.DataFrame(rows)

    def calculate_breakeven_upside(
        self,
        collar: CollarPosition,
        pricing: CollarPricing,
    ) -> float:
        """Calculate upside percentage where collar breaks even.

        This is the upside level needed to cover the net premium cost.

        Args:
            collar: The collar position.
            pricing: Collar pricing.

        Returns:
            Breakeven upside as percentage (e.g., 0.02 for 2%).
        """
        entry_price = collar.underlying_entry_price
        net_premium = pricing.net_premium

        # Breakeven when: (target - entry) - net_premium = 0
        # target = entry + net_premium
        breakeven_price = entry_price + net_premium
        breakeven_pct = (breakeven_price / entry_price) - 1.0

        return breakeven_pct
