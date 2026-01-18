"""
Return calculations and resampling utilities.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    """
    Calculate returns from price data.

    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with assets as columns
    method : str
        'simple' for percentage returns or 'log' for log returns

    Returns:
    --------
    pd.DataFrame with returns
    """
    if method == 'simple':
        returns = prices.pct_change().dropna()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'")

    return returns


def resample_returns(returns: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """
    Resample returns to a different frequency using geometric compounding.

    Parameters:
    -----------
    returns : pd.DataFrame
        Daily (or other frequency) returns
    target_freq : str
        Target frequency ('W' for weekly, 'M' for monthly, 'Y' for yearly)

    Returns:
    --------
    pd.DataFrame with resampled returns
    """
    # Compound returns for each period
    resampled = (1 + returns).resample(target_freq).prod() - 1
    return resampled


def compound_returns(returns: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate compound return from a series of returns.

    Parameters:
    -----------
    returns : pd.DataFrame, pd.Series, or np.ndarray
        Period returns

    Returns:
    --------
    Compound return(s)
    """
    if isinstance(returns, pd.DataFrame):
        return (1 + returns).prod() - 1
    elif isinstance(returns, pd.Series):
        return (1 + returns).prod() - 1
    else:
        return np.prod(1 + returns) - 1


def annualize_returns(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Annualize period returns.

    Parameters:
    -----------
    returns : pd.DataFrame
        Period returns
    periods_per_year : int
        Number of periods per year (252 for daily, 52 for weekly, 12 for monthly)

    Returns:
    --------
    pd.DataFrame with annualized returns
    """
    compound = (1 + returns).prod()
    n_periods = len(returns)
    years = n_periods / periods_per_year
    return compound ** (1 / years) - 1


def sample_returns_bootstrap(returns: pd.DataFrame, num_samples: int,
                             seed: Optional[int] = None) -> pd.DataFrame:
    """
    Bootstrap sample returns (resample with replacement).

    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    num_samples : int
        Number of samples to draw
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame with sampled returns
    """
    if seed is not None:
        np.random.seed(seed)

    indices = np.random.choice(len(returns), size=num_samples, replace=True)
    return returns.iloc[indices].reset_index(drop=True)


def sample_returns_parametric(mean: np.ndarray, cov: np.ndarray,
                              num_samples: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample returns from multivariate normal distribution.

    Parameters:
    -----------
    mean : np.ndarray
        Mean returns vector
    cov : np.ndarray
        Covariance matrix
    num_samples : int
        Number of samples to draw
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    np.ndarray with sampled returns (num_samples, num_assets)
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.multivariate_normal(mean, cov, size=num_samples)
