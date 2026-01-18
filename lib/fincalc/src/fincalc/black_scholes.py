"""
Black-Scholes option pricing and Greeks calculations.
"""

import math
import logging
from typing import Tuple, Dict
from scipy.stats import norm
from scipy.optimize import minimize_scalar


class BlackScholesCalculator:
    """Black-Scholes-Merton option pricing model."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def price(self, S: float, K: float, T: float, r: float,
              sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price.

        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float: Option price
        """
        if T <= 0:
            # Expired option
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        if sigma <= 0:
            self.logger.warning("Volatility must be positive")
            return 0.0

        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return max(price, 0.0)

        except Exception as e:
            self.logger.error(f"Error calculating BS price: {e}")
            return 0.0

    def price_both(self, S: float, K: float, T: float, r: float,
                   sigma: float) -> Tuple[float, float]:
        """
        Calculate both call and put prices.

        Returns:
        --------
        Tuple[float, float]: (call_price, put_price)
        """
        call = self.price(S, K, T, r, sigma, 'call')
        put = self.price(S, K, T, r, sigma, 'put')
        return call, put

    def implied_volatility(self, market_price: float, S: float, K: float,
                           T: float, r: float, option_type: str = 'call') -> float:
        """
        Calculate implied volatility using scipy optimization.

        Parameters:
        -----------
        market_price : float
            Market price of option
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (years)
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'

        Returns:
        --------
        float: Implied volatility
        """
        if market_price <= 0 or T <= 0:
            return 0.0

        def objective(vol):
            try:
                theoretical = self.price(S, K, T, r, vol, option_type)
                return abs(theoretical - market_price)
            except:
                return float('inf')

        try:
            result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
            return result.x if result.success else 0.0
        except Exception as e:
            self.logger.error(f"Error in IV calculation: {e}")
            return 0.0

    def greeks(self, S: float, K: float, T: float, r: float,
               sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option Greeks.

        Parameters:
        -----------
        S, K, T, r, sigma : float
            Standard BSM parameters
        option_type : str
            'call' or 'put'

        Returns:
        --------
        dict: {'delta', 'gamma', 'theta', 'vega', 'rho'}
        """
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        try:
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)

            sqrt_T = math.sqrt(T)
            exp_rT = math.exp(-r * T)
            phi_d1 = norm.pdf(d1)
            Phi_d1 = norm.cdf(d1)
            Phi_d2 = norm.cdf(d2)

            # Gamma and Vega are same for calls and puts
            gamma = phi_d1 / (S * sigma * sqrt_T)
            vega = S * phi_d1 * sqrt_T / 100  # Per 1% vol change

            if option_type.lower() == 'call':
                delta = Phi_d1
                theta = (-S * phi_d1 * sigma / (2 * sqrt_T) -
                         r * K * exp_rT * Phi_d2) / 365
                rho = K * T * exp_rT * Phi_d2 / 100
            else:
                delta = Phi_d1 - 1
                theta = (-S * phi_d1 * sigma / (2 * sqrt_T) +
                         r * K * exp_rT * norm.cdf(-d2)) / 365
                rho = -K * T * exp_rT * norm.cdf(-d2) / 100

            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }

        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}


# Convenience functions
def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = 'call') -> float:
    """Convenience function for BSM pricing."""
    calc = BlackScholesCalculator()
    return calc.price(S, K, T, r, sigma, option_type)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call') -> Dict[str, float]:
    """Convenience function for Greeks."""
    calc = BlackScholesCalculator()
    return calc.greeks(S, K, T, r, sigma, option_type)
