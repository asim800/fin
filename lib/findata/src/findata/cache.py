"""
Caching utilities with hash-based versioning.

Provides efficient caching for financial data with automatic
cache invalidation based on ticker lists and date ranges.
"""

import os
import pickle
import hashlib
import logging
from typing import List, Optional, Any
from datetime import datetime


class CacheManager:
    """
    Cache manager with hash-based versioning.

    Handles caching of financial data with automatic invalidation
    when ticker lists or date ranges change.
    """

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize cache manager.

        Parameters:
        -----------
        cache_dir : str
            Directory for cache files
        """
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(cache_dir, exist_ok=True)

    def _generate_hash(self, items: List[str]) -> str:
        """
        Generate a short hash of a list for cache versioning.

        Parameters:
        -----------
        items : List[str]
            List of items to hash

        Returns:
        --------
        str: Short hash (first 8 characters)
        """
        sorted_items = sorted(items)
        item_string = ','.join(sorted_items)
        hash_object = hashlib.md5(item_string.encode())
        return hash_object.hexdigest()[:8]

    def get_cache_path(self, prefix: str, start_date: str, end_date: str,
                       tickers: Optional[List[str]] = None) -> str:
        """
        Generate cache file path based on parameters.

        Parameters:
        -----------
        prefix : str
            Cache file prefix (e.g., 'price_data')
        start_date : str
            Start date string
        end_date : str
            End date string
        tickers : List[str], optional
            List of tickers for hash generation

        Returns:
        --------
        str: Full cache file path
        """
        if tickers:
            ticker_hash = self._generate_hash(tickers)
            filename = f"{prefix}_{start_date}_{end_date}_{ticker_hash}.pkl"
        else:
            filename = f"{prefix}_{start_date}_{end_date}.pkl"

        return os.path.join(self.cache_dir, filename)

    def load(self, path: str) -> Optional[Any]:
        """
        Load data from cache.

        Parameters:
        -----------
        path : str
            Cache file path

        Returns:
        --------
        Cached data or None if not found
        """
        if not os.path.exists(path):
            self.logger.debug(f"Cache miss: {path}")
            return None

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"Cache hit: {path}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def save(self, data: Any, path: str) -> bool:
        """
        Save data to cache.

        Parameters:
        -----------
        data : Any
            Data to cache
        path : str
            Cache file path

        Returns:
        --------
        bool: True if successful
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Cached: {path}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
            return False

    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache files.

        Parameters:
        -----------
        pattern : str, optional
            Only delete files matching pattern

        Returns:
        --------
        int: Number of files deleted
        """
        deleted = 0
        for filename in os.listdir(self.cache_dir):
            if pattern is None or pattern in filename:
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(filepath)
                    deleted += 1
                except OSError:
                    pass
        self.logger.info(f"Cleared {deleted} cache files")
        return deleted

    def list_files(self) -> List[str]:
        """List all cache files."""
        return [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
