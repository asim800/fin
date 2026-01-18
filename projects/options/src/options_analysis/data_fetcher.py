"""Data fetching module for stock quotes and option chains."""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import numpy as np


class DataFetcher:
    """Fetches stock quotes and option chain data from Yahoo Finance."""
    
    def __init__(self, request_delay: float = 1.0, timeout: int = 30):
        """
        Initialize DataFetcher.
        
        Args:
            request_delay: Delay between requests in seconds
            timeout: Request timeout in seconds
        """
        self.request_delay = request_delay
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
    def get_quote(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get current quote for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with quote data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            
            if hist.empty:
                self.logger.warning(f"No price data for {ticker}")
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            quote_data = {
                'symbol': ticker,
                'price': float(current_price),
                'volume': int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0,
                'timestamp': datetime.now(),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield')
            }
            
            self.logger.info(f"Retrieved quote for {ticker}: ${current_price:.2f}")
            return quote_data
            
        except Exception as e:
            self.logger.error(f"Error fetching quote for {ticker}: {e}")
            return None
    
    def get_quotes_batch(self, tickers: List[str]) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Get quotes for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Tuple of (price dictionary, price DataFrame)
        """
        prices = {}
        price_data = []
        
        for i, ticker in enumerate(tickers):
            quote = self.get_quote(ticker)
            
            if quote:
                prices[ticker] = quote['price']
                price_data.append(quote)
                
            # Add delay between requests
            if i < len(tickers) - 1:
                time.sleep(self.request_delay)
        
        # Create DataFrame from collected data
        if price_data:
            df = pd.DataFrame(price_data)
            df.set_index('symbol', inplace=True)
        else:
            df = pd.DataFrame()
            
        return prices, df
    
    def get_option_chain(self, ticker: str, expiry_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get option chain for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            expiry_date: Specific expiry date (optional)
            
        Returns:
            Dictionary with option chain data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get available expiry dates
            expiry_dates = stock.options
            
            if not expiry_dates:
                self.logger.warning(f"No option data available for {ticker}")
                return None
            
            option_data = {}
            
            # If no specific expiry requested, get all available
            if expiry_date is None:
                target_dates = expiry_dates
            else:
                target_dates = [expiry_date] if expiry_date in expiry_dates else []
            
            for exp_date in target_dates:
                try:
                    # Get option chain for this expiry
                    opt_chain = stock.option_chain(exp_date)
                    
                    # Process calls and puts
                    calls_df = self._process_option_data(opt_chain.calls, 'call')
                    puts_df = self._process_option_data(opt_chain.puts, 'put')
                    
                    # Format expiry date to match R format (e.g., "Feb.21.2025")
                    formatted_date = self._format_expiry_date(exp_date)
                    
                    option_data[formatted_date] = {
                        'calls': calls_df,
                        'puts': puts_df,
                        'expiry_date': exp_date
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error processing expiry {exp_date} for {ticker}: {e}")
                    continue
            
            if option_data:
                self.logger.info(f"Retrieved option chains for {ticker}: {len(option_data)} expiries")
                return option_data
            else:
                self.logger.warning(f"No valid option data for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching option chain for {ticker}: {e}")
            return None
    
    def _process_option_data(self, options_df: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """
        Process raw option data to match R format.
        
        Args:
            options_df: Raw options DataFrame from yfinance
            option_type: 'call' or 'put'
            
        Returns:
            Processed DataFrame with standardized columns
        """
        if options_df.empty:
            return pd.DataFrame()
        
        # Map columns to match R format
        # Start with a copy of the original data
        processed_df = options_df.copy()
        
        # Set contract symbol as index (equivalent to row names in R)
        processed_df.set_index('contractSymbol', inplace=True)
        
        # Map columns to R equivalents
        column_mapping = {
            'strike': 'Strike',
            'lastPrice': 'Last',
            'change': 'Chg',
            'bid': 'Bid',
            'ask': 'Ask',
            'volume': 'Vol',
            'openInterest': 'OI',
            'lastTradeDate': 'LastTradeTime',
            'impliedVolatility': 'IV',
            'inTheMoney': 'ITM'
        }
        
        # Rename columns to match R format
        rename_dict = {}
        for yf_col, r_col in column_mapping.items():
            if yf_col in processed_df.columns:
                rename_dict[yf_col] = r_col
        
        processed_df.rename(columns=rename_dict, inplace=True)
        
        # Keep only the columns we need
        needed_columns = list(column_mapping.values())
        existing_columns = [col for col in needed_columns if col in processed_df.columns]
        processed_df = processed_df[existing_columns]
        
        # Add missing columns with NaN values
        for r_col in needed_columns:
            if r_col not in processed_df.columns:
                processed_df[r_col] = np.nan
        
        # Set ITM to boolean for non-null values
        if 'ITM' in processed_df.columns:
            processed_df['ITM'] = processed_df['ITM'].fillna(False)
            
        # Convert LastTradeTime to proper timezone
        if 'LastTradeTime' in processed_df.columns:
            processed_df['LastTradeTime'] = pd.to_datetime(processed_df['LastTradeTime'])
        
        return processed_df
    
    def _format_expiry_date(self, date_str: str) -> str:
        """
        Format expiry date to match R format (e.g., "Feb.21.2025").
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            Formatted date string
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj.strftime('%b.%d.%Y')
        except Exception:
            return date_str
    
    def get_options_batch(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Get option chains for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary with option chain data for each ticker
        """
        option_chains = {}
        
        for i, ticker in enumerate(tickers):
            self.logger.info(f"Retrieving option chains for {ticker}")
            
            try:
                option_data = self.get_option_chain(ticker)
                if option_data:
                    option_chains[ticker] = option_data
                else:
                    self.logger.warning(f"No option data for {ticker}, skipping")
                    
            except Exception as e:
                self.logger.error(f"Problem with ticker {ticker}, skipping: {e}")
                continue
            
            # Add delay between requests
            if i < len(tickers) - 1:
                time.sleep(self.request_delay)
        
        return option_chains