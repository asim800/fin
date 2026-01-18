"""
fincalc - Financial calculations: pricing, metrics, and risk.

This package provides:
- Black-Scholes option pricing
- SABR and other volatility models
- Option Greeks calculations
- Performance metrics (Sharpe, Sortino, Calmar)
- Risk metrics (VaR, CVaR, drawdown)
- Covariance estimation methods
"""

from fincalc.black_scholes import BlackScholesCalculator
from fincalc.volatility import (
    VolatilityModel,
    FlatVolatility,
    SABRVolatility,
    SABRParameters,
    DEFAULT_SPY_SABR,
)
from fincalc.performance import (
    annualized_return,
    annualized_standard_deviation,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    gain_to_pain_ratio,
)
from fincalc.risk import (
    calculate_var,
    calculate_cvar,
    calculate_downside_deviation,
    calculate_ulcer_index,
    calculate_tail_ratio,
    calculate_omega_ratio,
)
from fincalc.covariance import (
    CovarianceEstimator,
)

__all__ = [
    # Black-Scholes
    'BlackScholesCalculator',
    # Volatility
    'VolatilityModel',
    'FlatVolatility',
    'SABRVolatility',
    'SABRParameters',
    'DEFAULT_SPY_SABR',
    # Performance
    'annualized_return',
    'annualized_standard_deviation',
    'max_drawdown',
    'sharpe_ratio',
    'sortino_ratio',
    'calmar_ratio',
    'gain_to_pain_ratio',
    # Risk
    'calculate_var',
    'calculate_cvar',
    'calculate_downside_deviation',
    'calculate_ulcer_index',
    'calculate_tail_ratio',
    'calculate_omega_ratio',
    # Covariance
    'CovarianceEstimator',
]
