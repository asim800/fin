"""
findata - Financial data fetching, caching, and persistence.

This package provides utilities for:
- Fetching market data from Yahoo Finance
- Caching data with hash-based versioning
- Persistence (pickle, CSV, JSON)
- Return calculations and resampling
"""

from findata.fetcher import FinDataFetcher
from findata.cache import CacheManager
from findata.persistence import save_pickle, load_pickle, save_csv, load_csv
from findata.returns import calculate_returns, resample_returns, compound_returns

__all__ = [
    'FinDataFetcher',
    'CacheManager',
    'save_pickle',
    'load_pickle',
    'save_csv',
    'load_csv',
    'calculate_returns',
    'resample_returns',
    'compound_returns',
]
