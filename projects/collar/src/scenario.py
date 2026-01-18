"""Scenario analysis for collar positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Type

import pandas as pd

from lib.pricing import BSMPricer, OptionGreeks, PricingEngine
from src.structures import CollarPosition, CollarPricing


@dataclass
class Scenario:
    """Represents a market scenario for analysis.

    Attributes:
        name: Descriptive name for the scenario.
        price_change_pct: Percentage change in underlying price.
        vol_change_pct: Percentage change in volatility (relative).
        days_elapsed: Number of days elapsed in the scenario.
    """

    name: str
    price_change_pct: float
    vol_change_pct: float = 0.0
    days_elapsed: int = 0


# Predefined market scenarios
NAMED_SCENARIOS: dict[str, Scenario] = {
    "mild_correction": Scenario(
        name="Mild Correction",
        price_change_pct=-0.10,
        vol_change_pct=0.20,
        days_elapsed=30,
    ),
    "sharp_correction": Scenario(
        name="Sharp Correction",
        price_change_pct=-0.20,
        vol_change_pct=0.50,
        days_elapsed=14,
    ),
    "crash": Scenario(
        name="Market Crash",
        price_change_pct=-0.35,
        vol_change_pct=1.00,
        days_elapsed=7,
    ),
    "gfc_2008": Scenario(
        name="GFC 2008",
        price_change_pct=-0.50,
        vol_change_pct=1.50,
        days_elapsed=90,
    ),
    "covid_2020": Scenario(
        name="COVID March 2020",
        price_change_pct=-0.34,
        vol_change_pct=2.00,
        days_elapsed=23,
    ),
    "rally": Scenario(
        name="Bull Rally",
        price_change_pct=0.15,
        vol_change_pct=-0.20,
        days_elapsed=60,
    ),
    "sideways": Scenario(
        name="Sideways",
        price_change_pct=0.00,
        vol_change_pct=0.10,
        days_elapsed=30,
    ),
}


@dataclass
class ScenarioResult:
    """Result of analyzing a collar under a scenario.

    Attributes:
        scenario: The input scenario.
        underlying_pnl: P&L from the underlying position.
        put_pnl: P&L from the put leg.
        call_pnl: P&L from the call leg.
        total_pnl: Total collar P&L.
        total_pnl_pct: Total P&L as percentage of initial position.
        unhedged_pnl: P&L if no hedge were in place.
        hedge_benefit: Benefit of hedge vs unhedged.
        scenario_greeks: Greeks at the scenario point.
        collar_binding: Which strike constrains the position.
    """

    scenario: Scenario
    underlying_pnl: float
    put_pnl: float
    call_pnl: float
    total_pnl: float
    total_pnl_pct: float
    unhedged_pnl: float
    hedge_benefit: float
    scenario_greeks: OptionGreeks
    collar_binding: Literal["put", "call", "neither"]


class ScenarioAnalyzer:
    """Analyzes collar positions under various market scenarios.

    Args:
        pricing_engine: Pricing engine class to use (default BSMPricer).
    """

    def __init__(
        self,
        pricing_engine: Type[PricingEngine] = BSMPricer,
    ) -> None:
        self.pricing_engine = pricing_engine

    def analyze_scenario(
        self,
        collar: CollarPosition,
        initial_pricing: CollarPricing,
        scenario: Scenario,
        initial_vol: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> ScenarioResult:
        """Analyze a collar position under a specific scenario.

        Args:
            collar: The collar position to analyze.
            initial_pricing: Initial pricing of the collar.
            scenario: The scenario to apply.
            initial_vol: Initial volatility assumption.
            risk_free_rate: Risk-free rate.
            dividend_yield: Dividend yield.

        Returns:
            ScenarioResult with P&L breakdown and scenario Greeks.
        """
        entry_price = collar.underlying_entry_price
        shares = collar.underlying_shares
        put_strike = collar.long_put.strike
        call_strike = collar.short_call.strike

        # Calculate new price and volatility
        new_price = entry_price * (1.0 + scenario.price_change_pct)
        new_vol = initial_vol * (1.0 + scenario.vol_change_pct)

        # Calculate remaining time to expiry
        original_days = (collar.long_put.expiry - collar.entry_date).days
        remaining_days = max(0, original_days - scenario.days_elapsed)
        remaining_years = remaining_days / 365.0

        # Underlying P&L
        underlying_pnl = (new_price - entry_price) * shares

        # Calculate new option values
        if remaining_years > 0:
            put_pricer = self.pricing_engine(
                spot=new_price,
                strike=put_strike,
                expiry_years=remaining_years,
                risk_free_rate=risk_free_rate,
                volatility=new_vol,
                dividend_yield=dividend_yield,
            )
            call_pricer = self.pricing_engine(
                spot=new_price,
                strike=call_strike,
                expiry_years=remaining_years,
                risk_free_rate=risk_free_rate,
                volatility=new_vol,
                dividend_yield=dividend_yield,
            )
            new_put_value = put_pricer.put_price()
            new_call_value = call_pricer.call_price()
            put_greeks = put_pricer.greeks("put")
            call_greeks = call_pricer.greeks("call")
        else:
            # At expiry, use intrinsic values
            new_put_value = max(0.0, put_strike - new_price)
            new_call_value = max(0.0, new_price - call_strike)
            put_greeks = OptionGreeks(
                delta=-1.0 if new_price < put_strike else 0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
            )
            call_greeks = OptionGreeks(
                delta=1.0 if new_price > call_strike else 0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
            )

        # Put P&L: long put, so gain if put value increases
        put_pnl = (new_put_value - initial_pricing.put_price) * shares

        # Call P&L: short call, so gain if call value decreases
        call_pnl = (initial_pricing.call_price - new_call_value) * shares

        # Total P&L
        total_pnl = underlying_pnl + put_pnl + call_pnl

        # Calculate as percentage of initial position value
        initial_position_value = entry_price * shares
        total_pnl_pct = total_pnl / initial_position_value

        # Unhedged P&L (just the underlying)
        unhedged_pnl = underlying_pnl

        # Hedge benefit
        hedge_benefit = total_pnl - unhedged_pnl

        # Determine which strike is binding
        if new_price <= put_strike:
            collar_binding: Literal["put", "call", "neither"] = "put"
        elif new_price >= call_strike:
            collar_binding = "call"
        else:
            collar_binding = "neither"

        # Net Greeks at scenario point
        scenario_greeks = OptionGreeks(
            delta=1.0 + put_greeks.delta - call_greeks.delta,
            gamma=put_greeks.gamma - call_greeks.gamma,
            vega=put_greeks.vega - call_greeks.vega,
            theta=-put_greeks.theta + call_greeks.theta,
        )

        return ScenarioResult(
            scenario=scenario,
            underlying_pnl=underlying_pnl,
            put_pnl=put_pnl,
            call_pnl=call_pnl,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            unhedged_pnl=unhedged_pnl,
            hedge_benefit=hedge_benefit,
            scenario_greeks=scenario_greeks,
            collar_binding=collar_binding,
        )

    def analyze_scenarios(
        self,
        collar: CollarPosition,
        initial_pricing: CollarPricing,
        scenarios: list[Scenario],
        initial_vol: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> pd.DataFrame:
        """Analyze a collar position under multiple scenarios.

        Args:
            collar: The collar position to analyze.
            initial_pricing: Initial pricing of the collar.
            scenarios: List of scenarios to analyze.
            initial_vol: Initial volatility assumption.
            risk_free_rate: Risk-free rate.
            dividend_yield: Dividend yield.

        Returns:
            DataFrame with scenario analysis results.
        """
        results = []

        for scenario in scenarios:
            result = self.analyze_scenario(
                collar=collar,
                initial_pricing=initial_pricing,
                scenario=scenario,
                initial_vol=initial_vol,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )

            results.append(
                {
                    "scenario_name": result.scenario.name,
                    "price_change": result.scenario.price_change_pct,
                    "vol_change": result.scenario.vol_change_pct,
                    "days_elapsed": result.scenario.days_elapsed,
                    "underlying_pnl": result.underlying_pnl,
                    "put_pnl": result.put_pnl,
                    "call_pnl": result.call_pnl,
                    "total_pnl": result.total_pnl,
                    "total_pnl_pct": result.total_pnl_pct,
                    "unhedged_pnl": result.unhedged_pnl,
                    "hedge_benefit": result.hedge_benefit,
                    "delta": result.scenario_greeks.delta,
                    "gamma": result.scenario_greeks.gamma,
                    "vega": result.scenario_greeks.vega,
                    "theta": result.scenario_greeks.theta,
                    "binding": result.collar_binding,
                }
            )

        return pd.DataFrame(results)
