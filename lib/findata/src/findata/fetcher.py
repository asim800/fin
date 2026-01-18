"""
Yahoo Finance data fetching utilities.

Provides a unified interface for downloading market data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import List, Optional, Dict
from datetime import datetime


class FinDataFetcher:
    """
    Financial data fetcher for Yahoo Finance.

    Handles ticker data downloads with error handling and logging.
    """

    def __init__(self, start_date: str, end_date: str):
        """
        Initialize fetcher with date range.

        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        """
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)

    def fetch_ticker(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single ticker from Yahoo Finance.

        Parameters:
        -----------
        ticker : str
            Ticker symbol

        Returns:
        --------
        pd.DataFrame with Close and Volume columns or None if failed
        """
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                interval='1d',
                auto_adjust=True,
                progress=False
            )[['Close', 'Volume']]
            self.logger.info(f"Downloaded {len(data)} days of data for {ticker}")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to download data for {ticker}: {e}")
            return None

    def fetch_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch data for multiple tickers.

        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols

        Returns:
        --------
        pd.DataFrame with MultiIndex columns (Ticker, Metric)
        """
        columns = pd.MultiIndex.from_tuples([], names=['Ticker', 'Metric'])
        all_data = pd.DataFrame(columns=columns)

        for ticker in tickers:
            ticker_data = self.fetch_ticker(ticker)
            if ticker_data is not None:
                for col in ['Close', 'Volume']:
                    all_data[(ticker, col)] = ticker_data[col]

        self.logger.info(f"Fetched data for {len(all_data.columns)//2} tickers")
        return all_data

    def fetch_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch just close prices for multiple tickers.

        Parameters:
        -----------
        tickers : List[str]
            List of ticker symbols

        Returns:
        --------
        pd.DataFrame with tickers as columns
        """
        all_data = self.fetch_tickers(tickers)
        if all_data.empty:
            return pd.DataFrame()

        # Extract close prices
        close_prices = all_data.xs('Close', level=1, axis=1)
        close_prices.columns = close_prices.columns.get_level_values(0)
        return close_prices


def load_tickers(ticker_file: str) -> pd.DataFrame:
    """
    Load ticker symbols and weights from CSV file.

    Parameters:
    -----------
    ticker_file : str
        Path to ticker file (format: Symbol,Weight with headers)

    Returns:
    --------
    pd.DataFrame with columns ['Symbol', 'Weight']
    """
    try:
        # Read CSV with headers
        tickers_df = pd.read_csv(ticker_file, skipinitialspace=True)

        # Clean column names
        tickers_df.columns = tickers_df.columns.str.strip()

        # Standardize column names
        if 'ticker' in tickers_df.columns:
            tickers_df = tickers_df.rename(columns={'ticker': 'Symbol'})
        if 'weights' in tickers_df.columns:
            tickers_df = tickers_df.rename(columns={'weights': 'Weight'})

        # Remove empty rows and ensure weights are numeric
        tickers_df = tickers_df.dropna()
        tickers_df['Weight'] = pd.to_numeric(tickers_df['Weight'], errors='coerce')
        tickers_df = tickers_df.dropna()

        # Normalize weights to sum to 1
        tickers_df['Weight'] = tickers_df['Weight'] / tickers_df['Weight'].sum()

        logging.info(f"Loaded {len(tickers_df)} tickers from {ticker_file}")
        return tickers_df

    except Exception as e:
        logging.error(f"Failed to load ticker file {ticker_file}: {e}")
        raise
