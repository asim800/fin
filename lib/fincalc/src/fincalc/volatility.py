"""
Volatility models for option pricing.

Provides:
- VolatilityModel Protocol
- FlatVolatility (constant)
- SABRVolatility (stochastic volatility model)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Protocol


class VolatilityModel(Protocol):
    """Protocol for volatility models."""

    def implied_vol(
        self,
        strike: float,
        forward: float,
        expiry_years: float,
    ) -> float:
        """Calculate implied volatility for a given strike."""
        ...


@dataclass
class FlatVolatility:
    """Flat volatility model (constant across all strikes)."""

    volatility: float

    def implied_vol(
        self,
        strike: float,
        forward: float,
        expiry_years: float,
    ) -> float:
        """Return constant volatility regardless of strike."""
        return self.volatility


@dataclass
class SABRParameters:
    """
    SABR model parameters.

    Attributes:
        alpha: ATM volatility level
        beta: CEV exponent (typically 0.5 or 1.0)
        rho: Correlation between forward and vol (-1 to 1)
        nu: Vol-of-vol
    """

    alpha: float
    beta: float = 0.5
    rho: float = -0.35  # Negative for typical equity skew
    nu: float = 0.4

    def __post_init__(self) -> None:
        """Validate SABR parameters."""
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.nu < 0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")


class SABRVolatility:
    """
    SABR stochastic volatility model.

    Implements Hagan et al. (2002) approximation for implied volatility.
    """

    def __init__(self, params: SABRParameters) -> None:
        self.params = params

    def implied_vol(
        self,
        strike: float,
        forward: float,
        expiry_years: float,
    ) -> float:
        """Calculate SABR implied volatility using Hagan approximation."""
        if expiry_years <= 0:
            return self.params.alpha

        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu

        F = forward
        K = strike
        T = expiry_years

        # Handle ATM case separately
        if abs(F - K) < 1e-10:
            return self._atm_vol(F, T)

        return self._general_vol(F, K, T, alpha, beta, rho, nu)

    def _atm_vol(self, forward: float, expiry_years: float) -> float:
        """ATM volatility approximation."""
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu
        T = expiry_years
        F = forward

        F_mid = F ** (1 - beta)

        term1 = ((1 - beta) ** 2 / 24) * (alpha**2 / F ** (2 - 2 * beta))
        term2 = (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
        term3 = ((2 - 3 * rho**2) / 24) * nu**2

        return (alpha / F_mid) * (1 + (term1 + term2 + term3) * T)

    def _general_vol(
        self,
        F: float,
        K: float,
        T: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
    ) -> float:
        """General strike volatility using Hagan approximation."""
        if K <= 0 or F <= 0:
            return alpha

        log_FK = math.log(F / K)
        FK_mid = (F * K) ** ((1 - beta) / 2)

        # z parameter
        z = (nu / alpha) * FK_mid * log_FK

        # x(z) function
        if abs(z) < 1e-10:
            x_z = 1.0
        else:
            sqrt_term = math.sqrt(1 - 2 * rho * z + z**2)
            x_z = z / math.log((sqrt_term + z - rho) / (1 - rho))

        # Denominator corrections
        one_minus_beta = 1 - beta
        FK_beta = (F * K) ** (one_minus_beta / 2)

        denom_1 = 1 + (one_minus_beta**2 / 24) * log_FK**2
        denom_2 = 1 + (one_minus_beta**4 / 1920) * log_FK**4
        denom = FK_beta * denom_1 * denom_2

        # Numerator corrections
        num_1 = (one_minus_beta**2 / 24) * (alpha**2 / FK_beta**2)
        num_2 = (rho * beta * nu * alpha) / (4 * FK_beta)
        num_3 = ((2 - 3 * rho**2) / 24) * nu**2
        numer = 1 + (num_1 + num_2 + num_3) * T

        sigma = (alpha / denom) * x_z * numer
        return max(sigma, 0.001)

    @classmethod
    def from_atm_vol(
        cls,
        atm_vol: float,
        forward: float,
        expiry_years: float,
        beta: float = 0.5,
        rho: float = -0.35,
        nu: float = 0.4,
    ) -> SABRVolatility:
        """Create SABR model calibrated to ATM volatility."""
        alpha_init = atm_vol * forward ** (1 - beta)

        alpha = alpha_init
        for _ in range(20):
            params = SABRParameters(alpha=alpha, beta=beta, rho=rho, nu=nu)
            model = cls(params)
            vol = model.implied_vol(forward, forward, expiry_years)

            if abs(vol - atm_vol) < 1e-8:
                break

            d_alpha = alpha * 0.001
            params_up = SABRParameters(alpha=alpha + d_alpha, beta=beta, rho=rho, nu=nu)
            vol_up = cls(params_up).implied_vol(forward, forward, expiry_years)
            dvol_dalpha = (vol_up - vol) / d_alpha

            if abs(dvol_dalpha) > 1e-10:
                alpha = alpha - (vol - atm_vol) / dvol_dalpha
                alpha = max(alpha, 0.001)

        return cls(SABRParameters(alpha=alpha, beta=beta, rho=rho, nu=nu))


# Default SPY-like SABR parameters
DEFAULT_SPY_SABR = SABRParameters(
    alpha=0.20,
    beta=0.5,
    rho=-0.35,
    nu=0.40,
)
