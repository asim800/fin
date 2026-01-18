"""Volatility models for option pricing.

This module provides:
- VolatilityModel Protocol defining the interface
- FlatVolatility for backward compatibility
- SABRVolatility implementing the SABR stochastic volatility model
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


class VolatilityModel(Protocol):
    """Protocol for volatility models.

    Any volatility model must implement implied_vol() to return
    the implied volatility for a given strike.
    """

    def implied_vol(
        self,
        strike: float,
        forward: float,
        expiry_years: float,
    ) -> float:
        """Calculate implied volatility for a given strike.

        Args:
            strike: Option strike price.
            forward: Forward price of the underlying.
            expiry_years: Time to expiration in years.

        Returns:
            Implied volatility (annualized, e.g., 0.20 for 20%).
        """
        ...


@dataclass
class FlatVolatility:
    """Flat volatility model (constant across all strikes).

    Attributes:
        volatility: Constant volatility value.
    """

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
    """SABR model parameters.

    Attributes:
        alpha: ATM volatility level (initial vol of forward).
        beta: CEV exponent (typically 0.5 or 1.0 for equities).
        rho: Correlation between forward and volatility (-1 to 1).
              Negative values produce equity-like skew (puts more expensive).
        nu: Vol-of-vol (volatility of the volatility process).
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
    """SABR stochastic volatility model.

    Implements Hagan et al. (2002) approximation for implied volatility.

    The SABR model assumes:
    - dF = alpha * F^beta * dW1  (forward dynamics)
    - dalpha = nu * alpha * dW2  (vol dynamics)
    - dW1 * dW2 = rho * dt       (correlation)

    Args:
        params: SABRParameters containing alpha, beta, rho, nu.

    References:
        Hagan, P., et al. (2002). "Managing Smile Risk"
    """

    def __init__(self, params: SABRParameters) -> None:
        self.params = params

    def implied_vol(
        self,
        strike: float,
        forward: float,
        expiry_years: float,
    ) -> float:
        """Calculate SABR implied volatility using Hagan approximation.

        Args:
            strike: Option strike price.
            forward: Forward price of underlying.
            expiry_years: Time to expiration in years.

        Returns:
            Implied volatility at the given strike.
        """
        if expiry_years <= 0:
            return self.params.alpha

        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu

        F = forward
        K = strike
        T = expiry_years

        # Handle ATM case separately (avoid division by zero)
        if abs(F - K) < 1e-10:
            return self._atm_vol(F, T)

        # General SABR formula (Hagan et al. approximation)
        return self._general_vol(F, K, T, alpha, beta, rho, nu)

    def _atm_vol(self, forward: float, expiry_years: float) -> float:
        """ATM volatility approximation."""
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu
        T = expiry_years
        F = forward

        # ATM formula: sigma_ATM = alpha / F^(1-beta) * [1 + correction_terms * T]
        F_mid = F ** (1 - beta)

        term1 = ((1 - beta) ** 2 / 24) * (alpha**2 / F ** (2 - 2 * beta))
        term2 = (rho * beta * nu * alpha) / (4 * F ** (1 - beta))
        term3 = ((2 - 3 * rho**2) / 24) * nu**2

        sigma_atm = (alpha / F_mid) * (1 + (term1 + term2 + term3) * T)
        return sigma_atm

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
        # Avoid log(0) issues
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
        return max(sigma, 0.001)  # Floor at 0.1% to avoid numerical issues

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
        """Create SABR model calibrated to ATM volatility.

        Solves for alpha such that ATM implied vol equals target.

        Args:
            atm_vol: Target ATM volatility.
            forward: Forward price.
            expiry_years: Time to expiry.
            beta: CEV exponent.
            rho: Correlation.
            nu: Vol-of-vol.

        Returns:
            SABRVolatility instance with calibrated alpha.
        """
        # Initial guess: alpha ≈ atm_vol * F^(1-beta)
        alpha_init = atm_vol * forward ** (1 - beta)

        # Simple Newton iteration to solve for alpha
        alpha = alpha_init
        for _ in range(20):
            params = SABRParameters(alpha=alpha, beta=beta, rho=rho, nu=nu)
            model = cls(params)
            vol = model.implied_vol(forward, forward, expiry_years)

            if abs(vol - atm_vol) < 1e-8:
                break

            # Numerical derivative
            d_alpha = alpha * 0.001
            params_up = SABRParameters(alpha=alpha + d_alpha, beta=beta, rho=rho, nu=nu)
            vol_up = cls(params_up).implied_vol(forward, forward, expiry_years)
            dvol_dalpha = (vol_up - vol) / d_alpha

            if abs(dvol_dalpha) > 1e-10:
                alpha = alpha - (vol - atm_vol) / dvol_dalpha
                alpha = max(alpha, 0.001)  # Keep positive

        return cls(SABRParameters(alpha=alpha, beta=beta, rho=rho, nu=nu))


# Default SPY-like SABR parameters (typical equity skew)
DEFAULT_SPY_SABR = SABRParameters(
    alpha=0.20,  # ~20% ATM vol
    beta=0.5,  # Square root CEV
    rho=-0.35,  # Negative skew (puts more expensive)
    nu=0.40,  # Moderate vol-of-vol
)
