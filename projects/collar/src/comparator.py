"""Collar comparison across different strike combinations."""

from __future__ import annotations

from datetime import date
from itertools import product
from typing import Type

import pandas as pd

from lib.pricing import BSMPricer, PricingEngine
from src.builder import CollarBuilder
from src.scenario import Scenario, ScenarioAnalyzer


class CollarComparator:
    """Compares collar positions across different strike combinations.

    Provides analysis of various put/call strike combinations to help
    identify optimal collar structures.

    Args:
        pricing_engine: Pricing engine class to use (default BSMPricer).
        risk_free_rate: Risk-free rate (default 0.05).
        dividend_yield: Dividend yield (default 0.0).
    """

    def __init__(
        self,
        pricing_engine: Type[PricingEngine] = BSMPricer,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ) -> None:
        self.pricing_engine = pricing_engine
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.builder = CollarBuilder(
            pricing_engine=pricing_engine,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        self.scenario_analyzer = ScenarioAnalyzer(pricing_engine=pricing_engine)

    def compare_strikes(
        self,
        symbol: str,
        spot: float,
        shares: int,
        put_strikes: list[float],
        call_strikes: list[float],
        expiry: date,
        volatility: float,
        scenarios: list[Scenario] | None = None,
        entry_date: date | None = None,
    ) -> pd.DataFrame:
        """Compare collar positions across strike combinations.

        Args:
            symbol: Underlying ticker symbol.
            spot: Current underlying price.
            shares: Number of shares.
            put_strikes: List of put strikes to test.
            call_strikes: List of call strikes to test.
            expiry: Option expiration date.
            volatility: Volatility assumption.
            scenarios: Optional list of scenarios for P&L analysis.
            entry_date: Entry date (default today).

        Returns:
            DataFrame with comparison results including pricing and Greeks.
        """
        if entry_date is None:
            entry_date = date.today()

        results = []

        for put_strike, call_strike in product(put_strikes, call_strikes):
            # Skip invalid combinations (put > call)
            if put_strike >= call_strike:
                continue

            position, pricing = self.builder.build_collar(
                symbol=symbol,
                spot=spot,
                shares=shares,
                put_strike=put_strike,
                call_strike=call_strike,
                expiry=expiry,
                volatility=volatility,
                entry_date=entry_date,
            )

            row = {
                "put_strike": put_strike,
                "put_strike_pct": put_strike / spot,
                "call_strike": call_strike,
                "call_strike_pct": call_strike / spot,
                "put_premium": pricing.put_price,
                "call_premium": pricing.call_price,
                "net_premium": pricing.net_premium,
                "net_premium_pct": pricing.net_premium_pct,
                "protection_level": position.protection_level,
                "upside_cap": position.upside_cap,
                "net_delta": pricing.net_delta,
                "net_gamma": pricing.net_gamma,
                "net_vega": pricing.net_vega,
                "net_theta": pricing.net_theta,
            }

            # Add scenario P&L if scenarios provided
            if scenarios:
                for scenario in scenarios:
                    result = self.scenario_analyzer.analyze_scenario(
                        collar=position,
                        initial_pricing=pricing,
                        scenario=scenario,
                        initial_vol=volatility,
                        risk_free_rate=self.risk_free_rate,
                        dividend_yield=self.dividend_yield,
                    )
                    # Sanitize scenario name for column
                    col_name = f"pnl_{scenario.name.lower().replace(' ', '_')}"
                    row[col_name] = result.total_pnl

            results.append(row)

        return pd.DataFrame(results)
