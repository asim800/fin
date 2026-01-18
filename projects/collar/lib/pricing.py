"""Black-Scholes-Merton option pricing library.

This module provides reusable option pricing functionality including:
- OptionGreeks dataclass for storing Greeks values
- BSMPricer class implementing the Black-Scholes-Merton model
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from scipy.stats import norm

if TYPE_CHECKING:
    from lib.volatility import VolatilityModel


@dataclass
class OptionGreeks:
    """Container for option Greeks.

    Attributes:
        delta: Price sensitivity per $1 move in underlying.
        gamma: Delta sensitivity per $1 move in underlying.
        vega: Price sensitivity per 1% volatility move.
        theta: Time decay per calendar day.
    """

    delta: float
    gamma: float
    vega: float
    theta: float


class PricingEngine(Protocol):
    """Interface for option pricing engines."""

    def call_price(self) -> float:
        """Calculate call option price."""
        ...

    def put_price(self) -> float:
        """Calculate put option price."""
        ...

    def delta(self, option_type: Literal["call", "put"]) -> float:
        """Calculate delta for the specified option type."""
        ...

    def gamma(self) -> float:
        """Calculate gamma (same for calls and puts)."""
        ...

    def vega(self) -> float:
        """Calculate vega (same for calls and puts)."""
        ...

    def theta(self, option_type: Literal["call", "put"]) -> float:
        """Calculate theta for the specified option type."""
        ...


class BSMPricer:
    """Black-Scholes-Merton option pricer.

    Implements the BSM model for European option pricing with dividend yield.

    Args:
        spot: Current underlying price.
        strike: Strike price.
        expiry_years: Time to expiry in years.
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%).
        volatility: Annual volatility (e.g., 0.20 for 20%). Optional if
            volatility_model is provided.
        dividend_yield: Annual dividend yield (default 0.0).
        volatility_model: Optional VolatilityModel for strike-dependent vol.
            If provided, volatility is computed from the model for this strike.

    Note:
        Either volatility OR volatility_model must be provided.
        If both are provided, volatility_model takes precedence.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        expiry_years: float,
        risk_free_rate: float,
        volatility: float | None = None,
        dividend_yield: float = 0.0,
        volatility_model: VolatilityModel | None = None,
    ) -> None:
        self.spot = spot
        self.strike = strike
        self.expiry_years = expiry_years
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

        # Compute forward price for volatility model
        forward = spot * math.exp((risk_free_rate - dividend_yield) * expiry_years)

        # Determine volatility from model or explicit parameter
        if volatility_model is not None:
            self.volatility = volatility_model.implied_vol(strike, forward, expiry_years)
        elif volatility is not None:
            self.volatility = volatility
        else:
            raise ValueError("Either volatility or volatility_model must be provided")

        # Pre-compute d1 and d2
        self._d1, self._d2 = self._compute_d1_d2()

    def _compute_d1_d2(self) -> tuple[float, float]:
        """Compute d1 and d2 parameters.

        d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        """
        if self.expiry_years <= 0:
            # At expiry, use intrinsic value logic
            return float("inf") if self.spot > self.strike else float("-inf"), 0.0

        sqrt_t = math.sqrt(self.expiry_years)
        d1 = (
            math.log(self.spot / self.strike)
            + (
                self.risk_free_rate
                - self.dividend_yield
                + 0.5 * self.volatility**2
            )
            * self.expiry_years
        ) / (self.volatility * sqrt_t)
        d2 = d1 - self.volatility * sqrt_t
        return d1, d2

    def call_price(self) -> float:
        """Calculate call option price.

        C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
        """
        if self.expiry_years <= 0:
            return max(0.0, self.spot - self.strike)

        discount_div = math.exp(-self.dividend_yield * self.expiry_years)
        discount_rf = math.exp(-self.risk_free_rate * self.expiry_years)

        return (
            self.spot * discount_div * norm.cdf(self._d1)
            - self.strike * discount_rf * norm.cdf(self._d2)
        )

    def put_price(self) -> float:
        """Calculate put option price.

        P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        """
        if self.expiry_years <= 0:
            return max(0.0, self.strike - self.spot)

        discount_div = math.exp(-self.dividend_yield * self.expiry_years)
        discount_rf = math.exp(-self.risk_free_rate * self.expiry_years)

        return (
            self.strike * discount_rf * norm.cdf(-self._d2)
            - self.spot * discount_div * norm.cdf(-self._d1)
        )

    def delta(self, option_type: Literal["call", "put"]) -> float:
        """Calculate delta.

        Call delta = e^(-qT) * N(d1)
        Put delta = -e^(-qT) * N(-d1)
        """
        if self.expiry_years <= 0:
            if option_type == "call":
                return 1.0 if self.spot > self.strike else 0.0
            else:
                return -1.0 if self.spot < self.strike else 0.0

        discount_div = math.exp(-self.dividend_yield * self.expiry_years)

        if option_type == "call":
            return discount_div * norm.cdf(self._d1)
        else:
            return -discount_div * norm.cdf(-self._d1)

    def gamma(self) -> float:
        """Calculate gamma (same for calls and puts).

        Gamma = e^(-qT) * n(d1) / (S * sigma * sqrt(T))
        """
        if self.expiry_years <= 0:
            return 0.0

        discount_div = math.exp(-self.dividend_yield * self.expiry_years)
        sqrt_t = math.sqrt(self.expiry_years)

        return (
            discount_div
            * norm.pdf(self._d1)
            / (self.spot * self.volatility * sqrt_t)
        )

    def vega(self) -> float:
        """Calculate vega per 1% volatility move (same for calls and puts).

        Vega = S * e^(-qT) * n(d1) * sqrt(T) / 100
        """
        if self.expiry_years <= 0:
            return 0.0

        discount_div = math.exp(-self.dividend_yield * self.expiry_years)
        sqrt_t = math.sqrt(self.expiry_years)

        return self.spot * discount_div * norm.pdf(self._d1) * sqrt_t / 100.0

    def theta(self, option_type: Literal["call", "put"]) -> float:
        """Calculate theta per calendar day.

        Returns theta as daily decay (negative for long positions).
        """
        if self.expiry_years <= 0:
            return 0.0

        discount_div = math.exp(-self.dividend_yield * self.expiry_years)
        discount_rf = math.exp(-self.risk_free_rate * self.expiry_years)
        sqrt_t = math.sqrt(self.expiry_years)

        # Common term: -(S * e^(-qT) * n(d1) * sigma) / (2 * sqrt(T))
        common = (
            -self.spot
            * discount_div
            * norm.pdf(self._d1)
            * self.volatility
            / (2.0 * sqrt_t)
        )

        if option_type == "call":
            theta_annual = (
                common
                + self.dividend_yield * self.spot * discount_div * norm.cdf(self._d1)
                - self.risk_free_rate * self.strike * discount_rf * norm.cdf(self._d2)
            )
        else:
            theta_annual = (
                common
                - self.dividend_yield * self.spot * discount_div * norm.cdf(-self._d1)
                + self.risk_free_rate * self.strike * discount_rf * norm.cdf(-self._d2)
            )

        # Convert to daily theta
        return theta_annual / 365.0

    def greeks(self, option_type: Literal["call", "put"]) -> OptionGreeks:
        """Calculate all Greeks for the specified option type.

        Args:
            option_type: Either "call" or "put".

        Returns:
            OptionGreeks dataclass with all Greek values.
        """
        return OptionGreeks(
            delta=self.delta(option_type),
            gamma=self.gamma(),
            vega=self.vega(),
            theta=self.theta(option_type),
        )
