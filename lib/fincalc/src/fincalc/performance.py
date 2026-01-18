"""
Performance metrics for portfolio analysis.

Provides Sharpe, Sortino, Calmar, and other risk-adjusted return metrics.
"""

import pandas as pd
import numpy as np
from typing import Union


def annualized_return(x: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Compute Annualized Return.

    Parameters:
    -----------
    x : pd.DataFrame or pd.Series
        Portfolio value series

    Returns:
    --------
    pd.DataFrame with Portfolio and Annualized Return columns
    """
    gross_return = x.iloc[-1] / x.iloc[0]
    days = len(x)
    years = days / 252
    ann_return = gross_return ** (1/years) - 1

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': ann_return.index,
            'Annualized Return': ann_return.values
        })
    else:
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Annualized Return': [ann_return]
        })
    return df


def annualized_standard_deviation(x: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Compute Annualized Standard Deviation.

    Parameters:
    -----------
    x : pd.DataFrame or pd.Series
        Portfolio value series

    Returns:
    --------
    pd.DataFrame with Portfolio and Standard Deviation columns
    """
    returns = x.pct_change().dropna()
    std = returns.std() * np.sqrt(252)

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': std.index,
            'Standard Deviation': std.values
        })
    else:
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Standard Deviation': [std]
        })
    return df


def max_drawdown(x: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate Maximum Peak to Trough Loss.

    Parameters:
    -----------
    x : pd.DataFrame or pd.Series
        Portfolio value series

    Returns:
    --------
    pd.DataFrame with Portfolio and Max Drawdown columns
    """
    roll_max = x.expanding().max()
    daily_drawdown = x/roll_max - 1.0
    max_daily_drawdown = daily_drawdown.expanding().min()

    if isinstance(x, pd.DataFrame):
        max_dd_values = max_daily_drawdown.min()
        max_dd = pd.DataFrame({
            'Portfolio': max_dd_values.index,
            'Max Drawdown': max_dd_values.values
        })
    else:
        max_dd_value = max_daily_drawdown.min()
        max_dd = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Max Drawdown': [max_dd_value]
        })
    return max_dd


def gain_to_pain_ratio(x: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate Schwager's Gain to Pain Ratio.

    Parameters:
    -----------
    x : pd.DataFrame or pd.Series
        Portfolio value series

    Returns:
    --------
    pd.DataFrame with Portfolio and Gain to Pain Ratio columns
    """
    returns = x.pct_change().dropna()
    positive_returns = returns[returns >= 0].sum()
    negative_returns = abs(returns[returns < 0].sum())
    gain_to_pain = positive_returns / negative_returns

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': gain_to_pain.index,
            'Gain to Pain Ratio': gain_to_pain.values
        })
    else:
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Gain to Pain Ratio': [gain_to_pain]
        })
    return df


def calmar_ratio(x: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Calculate Calmar Ratio (Annualized Return / Max Drawdown).

    Parameters:
    -----------
    x : pd.DataFrame or pd.Series
        Portfolio value series

    Returns:
    --------
    pd.DataFrame with Portfolio and Calmar Ratio columns
    """
    ann_ret = annualized_return(x)
    max_dd = max_drawdown(x)
    calmar_values = ann_ret['Annualized Return'].values / (-max_dd['Max Drawdown'].values)

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': x.columns,
            'Calmar Ratio': calmar_values
        })
    else:
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Calmar Ratio': [calmar_values[0]]
        })
    return df


def sharpe_ratio(x: Union[pd.DataFrame, pd.Series], rf: float = 0) -> pd.DataFrame:
    """
    Calculate Sharpe Ratio.

    Parameters:
    -----------
    x : pd.DataFrame or pd.Series
        Portfolio value series
    rf : float
        Risk-free rate (annualized)

    Returns:
    --------
    pd.DataFrame with Portfolio and Sharpe Ratio columns
    """
    returns = annualized_return(x)
    std = annualized_standard_deviation(x)
    data = returns.merge(std, on='Portfolio')

    sharpe_col = f'Sharpe Ratio (RF = {rf})'
    data[sharpe_col] = (data['Annualized Return'] - float(rf)) / data['Standard Deviation']

    return data[['Portfolio', sharpe_col]]


def sortino_ratio(x: Union[pd.DataFrame, pd.Series], rf: float = 0) -> pd.DataFrame:
    """
    Calculate Sortino Ratio (using downside deviation).

    Parameters:
    -----------
    x : pd.DataFrame or pd.Series
        Portfolio value series
    rf : float
        Risk-free rate (annualized)

    Returns:
    --------
    pd.DataFrame with Portfolio and Sortino Ratio columns
    """
    returns = annualized_return(x)
    rf_daily = rf / 252
    returns_data = x.pct_change().dropna()

    downside_returns = returns_data[returns_data < rf_daily]
    downside_std = downside_returns.std() * np.sqrt(252)

    if isinstance(x, pd.DataFrame):
        df = pd.DataFrame({
            'Portfolio': downside_std.index,
            'Downside Standard Deviation': downside_std.values
        })
    else:
        df = pd.DataFrame({
            'Portfolio': [x.name or 'Portfolio'],
            'Downside Standard Deviation': [downside_std]
        })

    data = returns.merge(df, on='Portfolio')
    sortino_col = f'Sortino Ratio (RF = {rf})'
    data[sortino_col] = (data['Annualized Return'] - float(rf)) / data['Downside Standard Deviation']

    return data[['Portfolio', sortino_col]]
