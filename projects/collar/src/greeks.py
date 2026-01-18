"""Greeks analysis for collar positions."""

from __future__ import annotations

from typing import Type

import pandas as pd

from lib.pricing import BSMPricer, PricingEngine
from src.structures import CollarPosition


class GreeksAnalyzer:
    """Analyzes Greeks profiles for collar positions.

    Provides methods to generate delta, gamma, vega, and theta profiles
    across various underlying prices, volatilities, and time horizons.

    Args:
        pricing_engine: Pricing engine class to use (default BSMPricer).
    """

    def __init__(
        self,
        pricing_engine: Type[PricingEngine] = BSMPricer,
    ) -> None:
        self.pricing_engine = pricing_engine

    def delta_profile(
        self,
        collar: CollarPosition,
        spot: float,
        price_range_pct: tuple[float, float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        resolution: int = 50,
    ) -> pd.DataFrame:
        """Generate delta profile across price range.

        Args:
            collar: The collar position to analyze.
            spot: Current spot price.
            price_range_pct: Price range as (min_pct, max_pct) of spot.
            volatility: Volatility assumption.
            days_to_expiry: Days to expiration.
            risk_free_rate: Risk-free rate.
            dividend_yield: Dividend yield.
            resolution: Number of price points.

        Returns:
            DataFrame with price, price_pct, put_delta, call_delta, collar_delta.
        """
        expiry_years = days_to_expiry / 365.0
        put_strike = collar.long_put.strike
        call_strike = collar.short_call.strike

        min_price = spot * (1.0 + price_range_pct[0])
        max_price = spot * (1.0 + price_range_pct[1])
        prices = [
            min_price + (max_price - min_price) * i / (resolution - 1)
            for i in range(resolution)
        ]

        results = []
        for price in prices:
            put_pricer = self.pricing_engine(
                spot=price,
                strike=put_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )
            call_pricer = self.pricing_engine(
                spot=price,
                strike=call_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )

            put_delta = put_pricer.delta("put")
            call_delta = call_pricer.delta("call")
            collar_delta = 1.0 + put_delta - call_delta

            results.append(
                {
                    "price": price,
                    "price_pct": (price / spot - 1.0),
                    "put_delta": put_delta,
                    "call_delta": call_delta,
                    "collar_delta": collar_delta,
                }
            )

        return pd.DataFrame(results)

    def gamma_profile(
        self,
        collar: CollarPosition,
        spot: float,
        price_range_pct: tuple[float, float],
        volatility: float,
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        resolution: int = 50,
    ) -> pd.DataFrame:
        """Generate gamma profile across price range.

        Args:
            collar: The collar position to analyze.
            spot: Current spot price.
            price_range_pct: Price range as (min_pct, max_pct) of spot.
            volatility: Volatility assumption.
            days_to_expiry: Days to expiration.
            risk_free_rate: Risk-free rate.
            dividend_yield: Dividend yield.
            resolution: Number of price points.

        Returns:
            DataFrame with price, price_pct, put_gamma, call_gamma, collar_gamma.
        """
        expiry_years = days_to_expiry / 365.0
        put_strike = collar.long_put.strike
        call_strike = collar.short_call.strike

        min_price = spot * (1.0 + price_range_pct[0])
        max_price = spot * (1.0 + price_range_pct[1])
        prices = [
            min_price + (max_price - min_price) * i / (resolution - 1)
            for i in range(resolution)
        ]

        results = []
        for price in prices:
            put_pricer = self.pricing_engine(
                spot=price,
                strike=put_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )
            call_pricer = self.pricing_engine(
                spot=price,
                strike=call_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )

            put_gamma = put_pricer.gamma()
            call_gamma = call_pricer.gamma()
            collar_gamma = put_gamma - call_gamma

            results.append(
                {
                    "price": price,
                    "price_pct": (price / spot - 1.0),
                    "put_gamma": put_gamma,
                    "call_gamma": call_gamma,
                    "collar_gamma": collar_gamma,
                }
            )

        return pd.DataFrame(results)

    def vega_profile(
        self,
        collar: CollarPosition,
        spot: float,
        vol_range: tuple[float, float],
        days_to_expiry: int,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        resolution: int = 50,
    ) -> pd.DataFrame:
        """Generate vega profile across volatility range.

        Args:
            collar: The collar position to analyze.
            spot: Current spot price.
            vol_range: Volatility range as (min_vol, max_vol).
            days_to_expiry: Days to expiration.
            risk_free_rate: Risk-free rate.
            dividend_yield: Dividend yield.
            resolution: Number of volatility points.

        Returns:
            DataFrame with volatility, put_vega, call_vega, collar_vega.
        """
        expiry_years = days_to_expiry / 365.0
        put_strike = collar.long_put.strike
        call_strike = collar.short_call.strike

        vols = [
            vol_range[0] + (vol_range[1] - vol_range[0]) * i / (resolution - 1)
            for i in range(resolution)
        ]

        results = []
        for vol in vols:
            put_pricer = self.pricing_engine(
                spot=spot,
                strike=put_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=vol,
                dividend_yield=dividend_yield,
            )
            call_pricer = self.pricing_engine(
                spot=spot,
                strike=call_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=vol,
                dividend_yield=dividend_yield,
            )

            put_vega = put_pricer.vega()
            call_vega = call_pricer.vega()
            collar_vega = put_vega - call_vega

            results.append(
                {
                    "volatility": vol,
                    "put_vega": put_vega,
                    "call_vega": call_vega,
                    "collar_vega": collar_vega,
                }
            )

        return pd.DataFrame(results)

    def theta_decay(
        self,
        collar: CollarPosition,
        spot: float,
        volatility: float,
        days_range: range,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ) -> pd.DataFrame:
        """Generate theta decay profile over time.

        Args:
            collar: The collar position to analyze.
            spot: Current spot price.
            volatility: Volatility assumption.
            days_range: Range of days to expiry.
            risk_free_rate: Risk-free rate.
            dividend_yield: Dividend yield.

        Returns:
            DataFrame with days_to_expiry, put_theta, call_theta, collar_theta.
        """
        put_strike = collar.long_put.strike
        call_strike = collar.short_call.strike

        results = []
        for days in days_range:
            expiry_years = days / 365.0

            if expiry_years <= 0:
                continue

            put_pricer = self.pricing_engine(
                spot=spot,
                strike=put_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )
            call_pricer = self.pricing_engine(
                spot=spot,
                strike=call_strike,
                expiry_years=expiry_years,
                risk_free_rate=risk_free_rate,
                volatility=volatility,
                dividend_yield=dividend_yield,
            )

            put_theta = put_pricer.theta("put")
            call_theta = call_pricer.theta("call")
            collar_theta = -put_theta + call_theta

            results.append(
                {
                    "days_to_expiry": days,
                    "put_theta": put_theta,
                    "call_theta": call_theta,
                    "collar_theta": collar_theta,
                }
            )

        return pd.DataFrame(results)
