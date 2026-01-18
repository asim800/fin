"""Option chain processing and transformation module."""

import logging
import os, re, math, pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import ipdb

import warnings
# warnings.simplefilter('error', RuntimeWarning)

class OptionChainProcessor:
    """Processes and transforms option chain data."""
    
    def __init__(self, data_folder: str = None):
        """Initialize OptionChainProcessor."""
        self.logger = logging.getLogger(__name__)
        
        # Use config's data folder if none provided
        if data_folder is None:
            from .config import Config
            config = Config()
            data_folder = config.data_folder
            
        self.data_folder = Path(data_folder)
        
        # Create data folder if it doesn't exist
        if not self.data_folder.exists():
            self.data_folder.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created data folder: {self.data_folder}")
        else:
            self.logger.debug(f"Using existing data folder: {self.data_folder}")
    
    def extract_contract_identifiers(self, option_chains: Dict[str, Any],
                                    prices: Dict[str, float] = None,
                                    current_time: datetime = None) -> Dict[str, Any]:
        """
        Extract contract identifiers from option symbols for matching calls and puts.
        This replicates the R logic for matching options by strike and expiry.

        Args:
            option_chains: Dictionary of option chain data by ticker
            prices: Dictionary of current prices by ticker (optional, for elasticity calculation)
            current_time: Current datetime (optional, for time to expiry calculation)

        Returns:
            Dictionary with processed contract identifiers
        """
        processed_chains = {}

        # Use current time if not provided
        if current_time is None:
            current_time = datetime.now()

        for ticker, ticker_data in option_chains.items():
            processed_chains[ticker] = {}

            # Get underlying price for this ticker
            underlying_price = prices.get(ticker) if prices else None

            for expiry_date, expiry_data in ticker_data.items():
                calls_df = expiry_data.get('calls', pd.DataFrame())
                puts_df = expiry_data.get('puts', pd.DataFrame())

                if calls_df.empty or puts_df.empty:
                    self.logger.warning(f"Missing calls or puts data for {ticker} {expiry_date}")
                    continue

                # Extract identifiers from contract symbols
                calls_identifiers = self._extract_identifiers(calls_df.index, 'C')
                puts_identifiers = self._extract_identifiers(puts_df.index, 'P')

                # Match calls and puts by identifier (with elasticity calculation if price available)
                matched_calls, matched_puts, pcOI, pcVol, pcIV, call_elasticity, put_elasticity = self._match_options(
                    calls_df, puts_df, calls_identifiers, puts_identifiers,
                    underlying_price=underlying_price,
                    expiry_date_str=expiry_date,
                    current_time=current_time
                )

                processed_chains[ticker][expiry_date] = {
                    'calls': matched_calls,
                    'puts': matched_puts,
                    'original_calls': calls_df,
                    'original_puts': puts_df,
                    'putcallOI': pcOI,
                    'putcallVol': pcVol,
                    'putcallIV': pcIV,
                    'call_elasticity': call_elasticity,
                    'put_elasticity': put_elasticity,
                    'expiry_date': expiry_data.get('expiry_date')
                }
        
        # Save processed chains to pickle file with date stamp
        self._save_processed_chains(processed_chains)

        return processed_chains

    def create_option_tables(self, processed_chains: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Create pivot tables for options with expiry dates as rows and strike prices as columns.

        Args:
            processed_chains: Dictionary of processed option chain data

        Returns:
            Dictionary with structure:
            {
                'TICKER': {
                    'call_price_table': DataFrame,
                    'call_iv_table': DataFrame,
                    'call_volume_table': DataFrame,
                    'call_oi_table': DataFrame,
                    'call_elasticity_table': DataFrame,
                    'put_price_table': DataFrame,
                    'put_iv_table': DataFrame,
                    'put_volume_table': DataFrame,
                    'put_oi_table': DataFrame,
                    'put_elasticity_table': DataFrame
                }
            }
        """
        tables_by_ticker = {}

        for ticker, ticker_data in processed_chains.items():
            try:
                # Collect data across all expiries for this ticker
                call_data = []
                put_data = []

                for expiry_date, expiry_data in ticker_data.items():
                    # Get calls data
                    calls_df = expiry_data.get('calls', pd.DataFrame())
                    call_elasticity_df = expiry_data.get('call_elasticity', pd.DataFrame())

                    if not calls_df.empty and 'Strike' in calls_df.columns:
                        for idx, row in calls_df.iterrows():
                            call_record = {
                                'expiry': expiry_date,
                                'strike': row['Strike'],
                                'price': row.get('Last', np.nan),
                                'bid': row.get('Bid', np.nan),
                                'ask': row.get('Ask', np.nan),
                                'iv': row.get('IV', np.nan),
                                'volume': row.get('Vol', np.nan),
                                'oi': row.get('OI', np.nan),
                            }

                            # Add elasticity and estimated delta if available
                            if not call_elasticity_df.empty and idx in call_elasticity_df.index:
                                call_record['elasticity'] = call_elasticity_df.loc[idx, 'call_elasticity']
                                if 'call_estimated_delta' in call_elasticity_df.columns:
                                    call_record['estimated_delta'] = call_elasticity_df.loc[idx, 'call_estimated_delta']
                                else:
                                    call_record['estimated_delta'] = np.nan
                            else:
                                call_record['elasticity'] = np.nan
                                call_record['estimated_delta'] = np.nan

                            call_data.append(call_record)

                    # Get puts data
                    puts_df = expiry_data.get('puts', pd.DataFrame())
                    put_elasticity_df = expiry_data.get('put_elasticity', pd.DataFrame())

                    if not puts_df.empty and 'Strike' in puts_df.columns:
                        for idx, row in puts_df.iterrows():
                            put_record = {
                                'expiry': expiry_date,
                                'strike': row['Strike'],
                                'price': row.get('Last', np.nan),
                                'bid': row.get('Bid', np.nan),
                                'ask': row.get('Ask', np.nan),
                                'iv': row.get('IV', np.nan),
                                'volume': row.get('Vol', np.nan),
                                'oi': row.get('OI', np.nan),
                            }

                            # Add elasticity and estimated delta if available
                            if not put_elasticity_df.empty and idx in put_elasticity_df.index:
                                put_record['elasticity'] = put_elasticity_df.loc[idx, 'put_elasticity']
                                if 'put_estimated_delta' in put_elasticity_df.columns:
                                    put_record['estimated_delta'] = put_elasticity_df.loc[idx, 'put_estimated_delta']
                                else:
                                    put_record['estimated_delta'] = np.nan
                            else:
                                put_record['elasticity'] = np.nan
                                put_record['estimated_delta'] = np.nan

                            put_data.append(put_record)

                if not call_data and not put_data:
                    self.logger.warning(f"No data to create tables for {ticker}")
                    continue

                # Create pivot tables for calls
                ticker_tables = {}

                if call_data:
                    call_df = pd.DataFrame(call_data)

                    # Create pivot tables with expiry as index and strike as columns
                    ticker_tables['call_price_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='price', aggfunc='first'
                    )
                    ticker_tables['call_bid_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='bid', aggfunc='first'
                    )
                    ticker_tables['call_ask_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='ask', aggfunc='first'
                    )
                    ticker_tables['call_iv_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='iv', aggfunc='first'
                    )
                    ticker_tables['call_volume_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='volume', aggfunc='first'
                    )
                    ticker_tables['call_oi_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='oi', aggfunc='first'
                    )
                    ticker_tables['call_elasticity_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='elasticity', aggfunc='first'
                    )
                    ticker_tables['call_estimated_delta_table'] = call_df.pivot_table(
                        index='expiry', columns='strike', values='estimated_delta', aggfunc='first'
                    )

                # Create pivot tables for puts
                if put_data:
                    put_df = pd.DataFrame(put_data)

                    ticker_tables['put_price_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='price', aggfunc='first'
                    )
                    ticker_tables['put_bid_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='bid', aggfunc='first'
                    )
                    ticker_tables['put_ask_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='ask', aggfunc='first'
                    )
                    ticker_tables['put_iv_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='iv', aggfunc='first'
                    )
                    ticker_tables['put_volume_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='volume', aggfunc='first'
                    )
                    ticker_tables['put_oi_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='oi', aggfunc='first'
                    )
                    ticker_tables['put_elasticity_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='elasticity', aggfunc='first'
                    )
                    ticker_tables['put_estimated_delta_table'] = put_df.pivot_table(
                        index='expiry', columns='strike', values='estimated_delta', aggfunc='first'
                    )

                tables_by_ticker[ticker] = ticker_tables
                self.logger.info(f"Created {len(ticker_tables)} pivot tables for {ticker}")

            except Exception as e:
                self.logger.error(f"Error creating tables for {ticker}: {e}")
                continue

        return tables_by_ticker

    def _save_processed_chains(self, processed_chains: Dict[str, Any], config: 'Config' = None) -> None:
        """
        Save processed chains to pickle file with date stamp.
        
        Args:
            processed_chains: Dictionary of processed option chain data
            config: Configuration object for path management
        """
        try:
            # Import Config here to avoid circular imports
            from .config import Config
            
            # Use provided config or create default
            if config is None:
                config = Config()
            
            # Use config's method to get the file path
            filepath = config.get_processed_chains_file_path()
            
            # Save to pickle file
            with open(filepath, 'wb') as f:
                pickle.dump(processed_chains, f)
            
            
        except Exception as e:
            self.logger.error(f"Error saving processed chains: {e}")
    
    def _extract_identifiers(self, contract_symbols: pd.Index, option_type: str) -> List[str]:
        """
        Extract identifiers from option contract symbols.
        
        This replicates the R regex logic:
        - For puts: sub('^(\\w{3}..*?)P(\\d+)', '\\1\\2', x)
        - For calls: sub('^(\\w{3}..*?)C(\\d+)', '\\1\\2', x)
        
        Args:
            contract_symbols: Index of contract symbols
            option_type: 'C' for calls, 'P' for puts
            
        Returns:
            List of extracted identifiers
        """
        identifiers = []
        
        # Pattern to match: ticker + date + option_type + strike
        # e.g., AAPL240119C00150000 -> AAPL240119150000
        pattern = rf'^(\w{{3}}.*?){option_type}(\d+)'
        
        for symbol in contract_symbols:
            try:
                match = re.match(pattern, symbol)
                if match:
                    # Combine groups 1 and 2 (everything before option type + strike)
                    identifier = match.group(1) + match.group(2)
                    identifiers.append(identifier)
                else:
                    # Fallback: use original symbol
                    identifiers.append(symbol)
            except Exception as e:
                self.logger.warning(f"Error processing symbol {symbol}: {e}")
                identifiers.append(symbol)
        
        return identifiers
    
    def _match_options(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                      calls_identifiers: List[str], puts_identifiers: List[str],
                      underlying_price: float = None, expiry_date_str: str = None,
                      current_time: datetime = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Match calls and puts by their identifiers and calculate elasticity.

        Args:
            calls_df: DataFrame of call options
            puts_df: DataFrame of put options
            calls_identifiers: List of call identifiers
            puts_identifiers: List of put identifiers
            underlying_price: Current price of underlying asset (optional, for elasticity)
            expiry_date_str: Expiry date string (optional, for delta calculation)
            current_time: Current datetime (optional, for time to expiry calculation)

        Returns:
            Tuple of (matched_calls_df, matched_puts_df, pcOI_df, pcVol_df, pcIV_df, call_elasticity_df, put_elasticity_df)
        """
        # Create identifier to index mapping
        calls_id_map = dict(zip(calls_identifiers, calls_df.index))
        puts_id_map = dict(zip(puts_identifiers, puts_df.index))
        
        # Find common identifiers
        common_ids = set(calls_identifiers) & set(puts_identifiers)
        
        if not common_ids:
            self.logger.warning("No matching call/put pairs found")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Extract matching rows
        matched_call_indices = [calls_id_map[id_] for id_ in common_ids if id_ in calls_id_map]
        matched_put_indices = [puts_id_map[id_] for id_ in common_ids if id_ in puts_id_map]
        
        matched_calls = calls_df.loc[matched_call_indices].copy()
        matched_puts = puts_df.loc[matched_put_indices].copy()
        
        try:
            # Calculate ratios as 1D arrays with proper handling of zeros
            with np.errstate(divide='ignore', invalid='ignore'):
                oi_ratio = np.divide(matched_puts['OI'].values.astype(float), 
                                   matched_calls['OI'].values.astype(float),
                                   out=np.full(matched_puts['OI'].values.shape, np.nan, dtype=float),
                                   where=matched_calls['OI'].values != 0)
            pcOI = pd.DataFrame({'putcallOI': oi_ratio}, index=matched_calls.index)
        except Exception as e:
            self.logger.warning(f'Error calculating OI ratios: {e}')
            pcOI = pd.DataFrame({'putcallOI': np.nan}, index=matched_calls.index)
            
        # Calculate other ratios as 1D arrays with proper handling of zeros
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio = np.divide(matched_puts['Vol'].values.astype(float), 
                                matched_calls['Vol'].values.astype(float),
                                out=np.full(matched_puts['Vol'].values.shape, np.nan, dtype=float),
                                where=matched_calls['Vol'].values != 0)
        pcVol = pd.DataFrame({'putcallVol': vol_ratio}, index=matched_calls.index)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            iv_ratio = np.divide(matched_puts['IV'].values.astype(float), 
                               matched_calls['IV'].values.astype(float),
                               out=np.full(matched_puts['IV'].values.shape, np.nan, dtype=float),
                               where=matched_calls['IV'].values != 0)
        pcIV = pd.DataFrame({'putcallIV': iv_ratio}, index=matched_calls.index)

        # Sort by strike price for consistent ordering
        if 'Strike' in matched_calls.columns:
            matched_calls = matched_calls.sort_values('Strike')
            matched_puts = matched_puts.sort_values('Strike')

        # Calculate elasticity if underlying price is provided
        call_elasticity_df = pd.DataFrame()
        put_elasticity_df = pd.DataFrame()

        if underlying_price is not None:
            try:
                # Calculate time to expiry if expiry date is provided
                T = 0.25  # Default to 3 months if not provided
                if expiry_date_str and current_time:
                    try:
                        expiry_dt = datetime.strptime(expiry_date_str, '%b.%d.%Y')
                        days_to_go = self._calculate_business_days(current_time, expiry_dt)
                        T = max(days_to_go / 252.0, 0.001)  # Convert to years, minimum 1 day
                    except Exception as e:
                        self.logger.warning(f"Error calculating time to expiry: {e}, using default T=0.25")

                # Calculate call elasticity
                call_elasticities = []
                call_estimated_deltas = []  # Track calculated deltas
                for _, row in matched_calls.iterrows():
                    try:
                        if 'Last' not in row.index or 'Strike' not in row.index:
                            call_elasticities.append(np.nan)
                            call_estimated_deltas.append(np.nan)
                            continue

                        call_price = row['Last']
                        strike = row['Strike']

                        if pd.isna(call_price) or call_price <= 0:
                            call_elasticities.append(np.nan)
                            call_estimated_deltas.append(np.nan)
                            continue

                        # Try to get delta from DataFrame first
                        call_delta = None
                        estimated_delta = np.nan

                        if 'Delta' in row.index and pd.notna(row['Delta']):
                            call_delta = row['Delta']
                            # Still calculate estimated delta for comparison
                            if 'IV' in row.index and pd.notna(row['IV']) and row['IV'] > 0:
                                iv = row['IV']
                                estimated_delta = self._calculate_call_delta_bs(underlying_price, strike, iv, T)
                        else:
                            # Fallback: calculate delta using Black-Scholes
                            if 'IV' in row.index and pd.notna(row['IV']) and row['IV'] > 0:
                                iv = row['IV']
                                call_delta = self._calculate_call_delta_bs(underlying_price, strike, iv, T)
                                estimated_delta = call_delta  # This is the estimated value
                            else:
                                call_elasticities.append(np.nan)
                                call_estimated_deltas.append(np.nan)
                                continue

                        # Calculate elasticity: (underlying_price * delta) / option_price
                        elasticity = (underlying_price * call_delta) / call_price
                        call_elasticities.append(elasticity)
                        call_estimated_deltas.append(estimated_delta)

                    except Exception as e:
                        self.logger.warning(f"Error calculating call elasticity for strike {row.get('Strike', 'unknown')}: {e}")
                        call_elasticities.append(np.nan)
                        call_estimated_deltas.append(np.nan)

                call_elasticity_df = pd.DataFrame({
                    'call_elasticity': call_elasticities,
                    'call_estimated_delta': call_estimated_deltas
                }, index=matched_calls.index)

                # Calculate put elasticity
                put_elasticities = []
                put_estimated_deltas = []  # Track calculated deltas
                for _, row in matched_puts.iterrows():
                    try:
                        if 'Last' not in row.index or 'Strike' not in row.index:
                            put_elasticities.append(np.nan)
                            put_estimated_deltas.append(np.nan)
                            continue

                        put_price = row['Last']
                        strike = row['Strike']

                        if pd.isna(put_price) or put_price <= 0:
                            put_elasticities.append(np.nan)
                            put_estimated_deltas.append(np.nan)
                            continue

                        # Try to get delta from DataFrame first
                        put_delta = None
                        estimated_delta = np.nan

                        if 'Delta' in row.index and pd.notna(row['Delta']):
                            put_delta = row['Delta']
                            # Still calculate estimated delta for comparison
                            if 'IV' in row.index and pd.notna(row['IV']) and row['IV'] > 0:
                                iv = row['IV']
                                estimated_delta = self._calculate_put_delta_bs(underlying_price, strike, iv, T)
                        else:
                            # Fallback: calculate delta using Black-Scholes
                            if 'IV' in row.index and pd.notna(row['IV']) and row['IV'] > 0:
                                iv = row['IV']
                                put_delta = self._calculate_put_delta_bs(underlying_price, strike, iv, T)
                                estimated_delta = put_delta  # This is the estimated value
                            else:
                                put_elasticities.append(np.nan)
                                put_estimated_deltas.append(np.nan)
                                continue

                        # Calculate elasticity: (underlying_price * |delta|) / option_price
                        elasticity = (underlying_price * abs(put_delta)) / put_price
                        put_elasticities.append(elasticity)
                        put_estimated_deltas.append(estimated_delta)

                    except Exception as e:
                        self.logger.warning(f"Error calculating put elasticity for strike {row.get('Strike', 'unknown')}: {e}")
                        put_elasticities.append(np.nan)
                        put_estimated_deltas.append(np.nan)

                put_elasticity_df = pd.DataFrame({
                    'put_elasticity': put_elasticities,
                    'put_estimated_delta': put_estimated_deltas
                }, index=matched_puts.index)

            except Exception as e:
                self.logger.error(f"Error in elasticity calculation: {e}")
                call_elasticity_df = pd.DataFrame({
                    'call_elasticity': np.nan,
                    'call_estimated_delta': np.nan
                }, index=matched_calls.index)
                put_elasticity_df = pd.DataFrame({
                    'put_elasticity': np.nan,
                    'put_estimated_delta': np.nan
                }, index=matched_puts.index)

        return matched_calls, matched_puts, pcOI, pcVol, pcIV, call_elasticity_df, put_elasticity_df
    
    def calculate_option_greeks(self, option_data: pd.DataFrame, 
                               underlying_price: float, risk_free_rate: float = 0.02
                               ) -> pd.DataFrame:
        """
        Calculate option Greeks (Delta, Gamma, Theta, Vega).
        
        Args:
            option_data: DataFrame with option data
            underlying_price: Current underlying asset price
            risk_free_rate: Risk-free interest rate
            
        Returns:
            DataFrame with Greeks added
        """
        # This is a placeholder for Greeks calculation
        # Implementation would require additional mathematical libraries
        
        if option_data.empty:
            return option_data
        
        # Add placeholder columns for Greeks
        option_data_with_greeks = option_data.copy()
        option_data_with_greeks['Delta'] = np.nan
        option_data_with_greeks['Gamma'] = np.nan
        option_data_with_greeks['Theta'] = np.nan
        option_data_with_greeks['Vega'] = np.nan
        
        return option_data_with_greeks
    
    def filter_by_criteria(self, option_data: pd.DataFrame, 
                          min_volume: int = 0, min_open_interest: int = 0,
                          max_bid_ask_spread: float = float('inf')) -> pd.DataFrame:
        """
        Filter options by various criteria.
        
        Args:
            option_data: DataFrame with option data
            min_volume: Minimum volume threshold
            min_open_interest: Minimum open interest threshold
            max_bid_ask_spread: Maximum bid-ask spread threshold
            
        Returns:
            Filtered DataFrame
        """
        if option_data.empty:
            return option_data
        
        filtered_data = option_data.copy()
        
        # Filter by volume
        if 'Vol' in filtered_data.columns and min_volume > 0:
            filtered_data = filtered_data[
                (filtered_data['Vol'].fillna(0) >= min_volume)
            ]
        
        # Filter by open interest
        if 'OI' in filtered_data.columns and min_open_interest > 0:
            filtered_data = filtered_data[
                (filtered_data['OI'].fillna(0) >= min_open_interest)
            ]
        
        # Filter by bid-ask spread
        if ('Bid' in filtered_data.columns and 'Ask' in filtered_data.columns and 
            max_bid_ask_spread < float('inf')):
            spread = filtered_data['Ask'] - filtered_data['Bid']
            filtered_data = filtered_data[spread <= max_bid_ask_spread]
        
        self.logger.info(f"Filtered from {len(option_data)} to {len(filtered_data)} options")
        
        return filtered_data
    
    def get_atm_options(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame, 
                       underlying_price: float, tolerance: float = 0.05) -> Dict[str, pd.DataFrame]:
        """
        Get at-the-money (ATM) options within tolerance.
        
        Args:
            calls_df: DataFrame of call options
            puts_df: DataFrame of put options
            underlying_price: Current underlying price
            tolerance: Tolerance for ATM selection (as fraction of price)
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames of ATM options
        """
        tolerance_amount = underlying_price * tolerance
        lower_bound = underlying_price - tolerance_amount
        upper_bound = underlying_price + tolerance_amount
        
        atm_calls = calls_df[
            (calls_df['Strike'] >= lower_bound) & 
            (calls_df['Strike'] <= upper_bound)
        ] if not calls_df.empty and 'Strike' in calls_df.columns else pd.DataFrame()
        
        atm_puts = puts_df[
            (puts_df['Strike'] >= lower_bound) & 
            (puts_df['Strike'] <= upper_bound)
        ] if not puts_df.empty and 'Strike' in puts_df.columns else pd.DataFrame()
        
        return {'calls': atm_calls, 'puts': atm_puts}
    
    def get_option_summary(self, option_chains: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate summary statistics for option chains.
        
        Args:
            option_chains: Dictionary of option chain data
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for ticker, ticker_data in option_chains.items():
            for expiry_date, expiry_data in ticker_data.items():
                calls_df = expiry_data.get('calls', pd.DataFrame())
                puts_df = expiry_data.get('puts', pd.DataFrame())
                
                summary_row = {
                    'ticker': ticker,
                    'expiry_date': expiry_date,
                    'num_calls': len(calls_df),
                    'num_puts': len(puts_df),
                    'call_volume': calls_df['Vol'].sum() if not calls_df.empty and 'Vol' in calls_df.columns else 0,
                    'put_volume': puts_df['Vol'].sum() if not puts_df.empty and 'Vol' in puts_df.columns else 0,
                    'call_oi': calls_df['OI'].sum() if not calls_df.empty and 'OI' in calls_df.columns else 0,
                    'put_oi': puts_df['OI'].sum() if not puts_df.empty and 'OI' in puts_df.columns else 0,
                    'min_strike': min(calls_df['Strike'].min() if not calls_df.empty and 'Strike' in calls_df.columns else float('inf'),
                                     puts_df['Strike'].min() if not puts_df.empty and 'Strike' in puts_df.columns else float('inf')),
                    'max_strike': max(calls_df['Strike'].max() if not calls_df.empty and 'Strike' in calls_df.columns else 0,
                                     puts_df['Strike'].max() if not puts_df.empty and 'Strike' in puts_df.columns else 0)
                }
                
                # Handle inf values
                if summary_row['min_strike'] == float('inf'):
                    summary_row['min_strike'] = np.nan
                
                summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)


    
    # ============================================================================
    # ADVANCED ANALYSIS METHODS (from op01.r and op05.r)
    # ============================================================================
    
    def plot_option_analysis(self, option_chains: Dict[str, Any], prices: Dict[str, float],
                           plot_folder: str = './plots5', specific_ticker: Optional[str] = None) -> List[str]:
        """
        Create comprehensive option plots (replicates plotOptions function from op01.r).
        
        Args:
            option_chains: Dictionary of option chain data
            prices: Dictionary of current prices
            plot_folder: Folder for saving plots
            specific_ticker: Optional specific ticker to plot
            
        Returns:
            List of created plot file paths
        """
        nv, nh = 2, 3  # Grid layout
        npp = nv * nh  # Plots per page
        
        date_str = datetime.now().strftime("%y%m%d")
        plot_dir = Path(plot_folder) / date_str
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        tickers = [specific_ticker] if specific_ticker else list(option_chains.keys())

        # ipdb.set_trace() 
        for ticker in tickers:
            if ticker not in option_chains or ticker not in prices:
                continue
                
            ticker_data = option_chains[ticker]
            price = prices[ticker]
            
            for expiry_name, expiry_data in ticker_data.items():
                try:
                    plot_filename = f"{ticker}_{expiry_name.replace('.', '_')}.png"
                    plot_path = plot_dir / plot_filename
                    
                    fig, axes = plt.subplots(nv, nh, figsize=(18, 12))
                    fig.suptitle(f"{ticker} - ${price:.2f}", fontsize=16)
                    
                    # Try original_calls/original_puts first, fallback to calls/puts
                    calls_df = expiry_data.get('original_calls', expiry_data.get('calls', pd.DataFrame()))
                    puts_df = expiry_data.get('original_puts', expiry_data.get('puts', pd.DataFrame()))
                    
                    if calls_df.empty or puts_df.empty:
                        continue
                    
                    # Parse expiry date for subtitle
                    try:
                        exp_date = datetime.strptime(expiry_name, '%b.%d.%Y').strftime('%Y-%m-%d')
                    except:
                        exp_date = expiry_name
                    
                    # Plot 1: Call Price (full range)
                    if 'Strike' in calls_df.columns and 'Last' in calls_df.columns:
                        axes[0, 0].plot(calls_df['Strike'].to_numpy(), calls_df['Last'].to_numpy(), 'b-', linewidth=2)
                        axes[0, 0].axvline(x=price, color='red', linestyle='--', alpha=0.7)
                        axes[0, 0].set_title('CALL Price')
                        axes[0, 0].set_xlabel('Strike')
                        axes[0, 0].set_ylabel('Price')
                        axes[0, 0].grid(True, alpha=0.3)
                    
                    # Plot 2: Call Price (limited range 0-10)
                    axes[0, 1].plot(calls_df['Strike'].to_numpy(), calls_df['Last'].to_numpy(), 'b-', linewidth=2)
                    axes[0, 1].axvline(x=price, color='red', linestyle='--', alpha=0.7)
                    axes[0, 1].set_title('CALL Price (0-10)')
                    axes[0, 1].set_xlabel('Strike')
                    axes[0, 1].set_ylabel('Price')
                    axes[0, 1].set_ylim(0, 10)
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Plot 3: Call IV
                    if 'IV' in calls_df.columns:
                        axes[0, 2].plot(calls_df['Strike'].to_numpy(), calls_df['IV'].to_numpy(), 'g-', linewidth=2)
                        axes[0, 2].axvline(x=price, color='red', linestyle='--', alpha=0.7)
                        axes[0, 2].set_title('CALL IV')
                        axes[0, 2].set_xlabel('Strike')
                        axes[0, 2].set_ylabel('Implied Volatility')
                        axes[0, 2].grid(True, alpha=0.3)
                    
                    # Plot 4: Put Price (full range)
                    if 'Strike' in puts_df.columns and 'Last' in puts_df.columns:
                        axes[1, 0].plot(puts_df['Strike'].to_numpy(), puts_df['Last'].to_numpy(), 'r-', linewidth=2)
                        axes[1, 0].axvline(x=price, color='red', linestyle='--', alpha=0.7)
                        axes[1, 0].set_title('PUT Price')
                        axes[1, 0].set_xlabel('Strike')
                        axes[1, 0].set_ylabel('Price')
                        axes[1, 0].grid(True, alpha=0.3)
                    
                    # Plot 5: Put Price (limited range 0-10)
                    axes[1, 1].plot(puts_df['Strike'].to_numpy(), puts_df['Last'].to_numpy(), 'r-', linewidth=2)
                    axes[1, 1].axvline(x=price, color='red', linestyle='--', alpha=0.7)
                    axes[1, 1].set_title('PUT Price (0-10)')
                    axes[1, 1].set_xlabel('Strike')
                    axes[1, 1].set_ylabel('Price')
                    axes[1, 1].set_ylim(0, 10)
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Plot 6: Put IV
                    if 'IV' in puts_df.columns:
                        axes[1, 2].plot(puts_df['Strike'].to_numpy(), puts_df['IV'].to_numpy(), 'orange', linewidth=2)
                        axes[1, 2].axvline(x=price, color='red', linestyle='--', alpha=0.7)
                        axes[1, 2].set_title('PUT IV')
                        axes[1, 2].set_xlabel('Strike')
                        axes[1, 2].set_ylabel('Implied Volatility')
                        axes[1, 2].grid(True, alpha=0.3)
                    
                    # Add expiry date as subtitle
                    fig.text(0.5, 0.02, f"Expiry: {exp_date}", ha='center', fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    created_files.append(str(plot_path))
                    
                except Exception as e:
                    self.logger.error(f"Error creating plot for {ticker} {expiry_name}: {e}")
                    continue
        
        return created_files
    
    def create_cross_expiry_analysis(self, option_chains: Dict[str, Any], prices: Dict[str, float],
                                   ticker: str, plot_folder: str = './plots2') -> Dict[str, str]:
        """
        Create cross-expiry analysis plots (replicates ggplot analysis from op01.r).
        
        Args:
            option_chains: Dictionary of option chain data
            prices: Dictionary of current prices
            ticker: Specific ticker to analyze
            plot_folder: Folder for saving plots
            
        Returns:
            Dictionary with plot file paths
        """
        if ticker not in option_chains or ticker not in prices:
            self.logger.warning(f"No data for ticker {ticker}")
            return {}
        
        date_str = datetime.now().strftime("%y%m%d")
        plot_dir = Path(plot_folder) / date_str
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        ticker_data = option_chains[ticker]
        current_price = prices[ticker]
        
        created_plots = {}
        
        try:
            # Prepare call price data across expiries
            call_price_data = []
            call_iv_data = []
            put_price_data = []
            put_iv_data = []
            
            for expiry_name, expiry_data in ticker_data.items():
                calls_df = expiry_data.get('original_calls', expiry_data.get('calls', pd.DataFrame()))
                puts_df = expiry_data.get('original_puts', expiry_data.get('puts', pd.DataFrame()))
                
                if not calls_df.empty and 'Strike' in calls_df.columns and 'Last' in calls_df.columns:
                    for _, row in calls_df.iterrows():
                        call_price_data.append({
                            'Strike': row['Strike'],
                            'Price': row['Last'],
                            'Expiry': expiry_name,
                            'Type': 'Call'
                        })
                        
                        if 'IV' in calls_df.columns and pd.notna(row['IV']):
                            call_iv_data.append({
                                'Strike': row['Strike'],
                                'IV': row['IV'],
                                'Expiry': expiry_name,
                                'Type': 'Call'
                            })
                
                if not puts_df.empty and 'Strike' in puts_df.columns and 'Last' in puts_df.columns:
                    for _, row in puts_df.iterrows():
                        put_price_data.append({
                            'Strike': row['Strike'],
                            'Price': row['Last'],
                            'Expiry': expiry_name,
                            'Type': 'Put'
                        })
                        
                        if 'IV' in puts_df.columns and pd.notna(row['IV']):
                            put_iv_data.append({
                                'Strike': row['Strike'],
                                'IV': row['IV'],
                                'Expiry': expiry_name,
                                'Type': 'Put'
                            })
            
            # Create Call Price Plot
            if call_price_data:
                df_calls = pd.DataFrame(call_price_data)
                
                plt.figure(figsize=(12, 8))
                for expiry in df_calls['Expiry'].unique():
                    expiry_data = df_calls[df_calls['Expiry'] == expiry]
                    plt.plot(expiry_data['Strike'].to_numpy(), expiry_data['Price'].to_numpy(), 'o-', 
                            label=expiry, alpha=0.7, markersize=4)
                
                plt.axvline(x=current_price, color='red', linestyle='--', 
                           label=f'Current: ${current_price:.2f}')
                plt.xlabel('Strike Price')
                plt.ylabel('Call Price')
                plt.title(f'{ticker} CALLS - Price Across Expiries')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                
                call_price_file = plot_dir / f"{ticker}_prices_call2.png"
                plt.savefig(call_price_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_plots['call_prices'] = str(call_price_file)
            
            # Create Call IV Plot
            if call_iv_data:
                df_call_iv = pd.DataFrame(call_iv_data)
                
                plt.figure(figsize=(12, 8))
                for expiry in df_call_iv['Expiry'].unique():
                    expiry_data = df_call_iv[df_call_iv['Expiry'] == expiry]
                    plt.plot(expiry_data['Strike'].to_numpy(), expiry_data['IV'].to_numpy(), 'o-', 
                            label=expiry, alpha=0.7, markersize=4)
                
                plt.axvline(x=current_price, color='red', linestyle='--', 
                           label=f'Current: ${current_price:.2f}')
                plt.xlabel('Strike Price')
                plt.ylabel('Implied Volatility')
                plt.title(f'{ticker} CALLS - IV Across Expiries')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                
                call_iv_file = plot_dir / f"{ticker}IV_call2.png"
                plt.savefig(call_iv_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_plots['call_iv'] = str(call_iv_file)
            
            # Create Put Price Plot
            if put_price_data:
                df_puts = pd.DataFrame(put_price_data)
                
                plt.figure(figsize=(12, 8))
                for expiry in df_puts['Expiry'].unique():
                    expiry_data = df_puts[df_puts['Expiry'] == expiry]
                    plt.plot(expiry_data['Strike'].to_numpy(), expiry_data['Price'].to_numpy(), 'o-', 
                            label=expiry, alpha=0.7, markersize=4)
                
                plt.axvline(x=current_price, color='red', linestyle='--', 
                           label=f'Current: ${current_price:.2f}')
                plt.xlabel('Strike Price')
                plt.ylabel('Put Price')
                plt.title(f'{ticker} PUTS - Price Across Expiries')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                
                put_price_file = plot_dir / f"{ticker}_prices_put2.png"
                plt.savefig(put_price_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_plots['put_prices'] = str(put_price_file)
            
            # Create Put IV Plot
            if put_iv_data:
                df_put_iv = pd.DataFrame(put_iv_data)
                
                plt.figure(figsize=(12, 8))
                for expiry in df_put_iv['Expiry'].unique():
                    expiry_data = df_put_iv[df_put_iv['Expiry'] == expiry]
                    plt.plot(expiry_data['Strike'].to_numpy(), expiry_data['IV'].to_numpy(), 'o-', 
                            label=expiry, alpha=0.7, markersize=4)
                
                plt.axvline(x=current_price, color='red', linestyle='--', 
                           label=f'Current: ${current_price:.2f}')
                plt.xlabel('Strike Price')
                plt.ylabel('Implied Volatility')
                plt.title(f'{ticker} PUTS - IV Across Expiries')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                
                put_iv_file = plot_dir / f"{ticker}_IV_put2.png"
                plt.savefig(put_iv_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_plots['put_iv'] = str(put_iv_file)
            
            
        except Exception as e:
            self.logger.error(f"Error creating cross-expiry analysis for {ticker}: {e}")
        
        return created_plots
    
    def analyze_put_strategies(self, option_chains: Dict[str, Any], prices: Dict[str, float],
                             analysis_date: str, plot_folder: str = './plots2', 
                             min_tickers_for_plot: int = 3) -> pd.DataFrame:
        """
        Analyze put strategies across tickers (replicates put analysis from op01.r).
        
        Args:
            option_chains: Dictionary of option chain data
            prices: Dictionary of current prices
            analysis_date: Date string for analysis
            plot_folder: Folder for saving plots
            min_tickers_for_plot: Minimum number of tickers required to generate plots (default: 3)
            
        Returns:
            DataFrame with put strategy analysis
        """
        plot_dir = Path(plot_folder) / analysis_date
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        tickers = list(option_chains.keys())
        
        # Get all available expiry dates
        expiry_dates = set()
        for ticker_data in option_chains.values():
            expiry_dates.update(ticker_data.keys())
        
        all_results = []
        
        for expiry_date in expiry_dates:
            strategy_data = []
            
            for ticker in tickers:
                if ticker not in option_chains or ticker not in prices:
                    continue
                
                ticker_data = option_chains[ticker]
                if expiry_date not in ticker_data:
                    continue
                
                puts_df = ticker_data[expiry_date].get('original_puts', ticker_data[expiry_date].get('puts', pd.DataFrame()))
                
                if puts_df.empty or 'Strike' not in puts_df.columns:
                    continue
                
                try:
                    current_price = prices[ticker]
                    cost = current_price * 100  # Cost basis (100 shares)
                    
                    # Find ATM put
                    strikes = puts_df['Strike'].values
                    idx_atm = np.argmin(np.abs(strikes - current_price))
                    
                    atm_put = puts_df.iloc[idx_atm]
                    
                    if 'Last' in puts_df.columns and pd.notna(atm_put['Last']):
                        gain_max = (atm_put['Last'] / current_price) * 100  # Max gain percentage
                        iv = atm_put.get('IV', 0) if pd.notna(atm_put.get('IV', 0)) else 0
                        
                        strategy_data.append({
                            'ticker': ticker,
                            'cost': cost,
                            'gain_max': gain_max,
                            'IV': iv,
                            'strike': atm_put['Strike'],
                            'put_price': atm_put['Last'],
                            'expiry': expiry_date
                        })
                
                except Exception as e:
                    self.logger.warning(f"Error analyzing {ticker} for {expiry_date}: {e}")
                    continue
            
            if not strategy_data:
                continue
            
            df = pd.DataFrame(strategy_data)
            
            # Check if we have enough tickers for plotting
            if len(df) < min_tickers_for_plot:
                self.logger.info(f"Skipping plot for {expiry_date}: only {len(df)} tickers (minimum {min_tickers_for_plot} required)")
                # Still add results to analysis data even if we skip plotting
                df_sorted = df.sort_values('gain_max', ascending=False)
                df_sorted['row_num'] = range(1, len(df_sorted) + 1)
                df_sorted['analysis_date'] = analysis_date
                all_results.append(df_sorted)
                continue
            
            plt.figure(figsize=(18, 7))
            try:
                # Create scatter plot (cost vs gain_max colored by IV)
                scatter = plt.scatter(df['cost'], df['gain_max'], c=df['IV']*4, 
                                    alpha=0.7, s=60, cmap='coolwarm')
                
                # Add ticker labels
                for _, row in df.iterrows():
                    plt.annotate(row['ticker'], 
                               (row['cost'], row['gain_max']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
                
                plt.xlim(0, 70000)
                plt.xlabel('Cost (Stock Price * 100)')
                plt.ylabel('Max Gain (%)')
                plt.title(f'PUT Strategy Analysis: {expiry_date} on {analysis_date}')
                plt.colorbar(scatter, label='IV')
                plt.grid(True, alpha=0.3)
                
                ratio_file = plot_dir / f"puts_ratio{expiry_date}.png"
                plt.savefig(ratio_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Create ranked plot
                df_sorted = df.sort_values('gain_max', ascending=False)
                
                plt.figure(figsize=(12, 7))
                plt.plot(range(len(df_sorted)), df_sorted['gain_max'].to_numpy(), 'bo', markersize=6)
                plt.xlabel('Ticker Rank')
                plt.ylabel('Max Gain (%)')
                plt.title(f'PUT Strategy Ranking: {expiry_date} on {analysis_date}')
                plt.xticks(range(len(df_sorted)), df_sorted['ticker'].to_numpy(), rotation=90)
                plt.grid(True, alpha=0.3)
                
                rank_file = plot_dir / f"puts_{expiry_date}.png"
                
                # Add row numbers and print statistics
                df_sorted['row_num'] = range(1, len(df_sorted) + 1)
                
                # Calculate IV quantiles
                iv_quantiles = df['IV'].quantile([0.1*i for i in range(11)])
                
                self.logger.info(f"Put analysis for {expiry_date}: {len(df)} tickers")
                self.logger.info(f"IV quantiles: {iv_quantiles.to_dict()}")
                
                # Store results
                df_sorted['analysis_date'] = analysis_date
                all_results.append(df_sorted)
                
            except Exception as e:
                self.logger.error(f"Error creating plots for {expiry_date}: {e}")
                continue
        
            plt.savefig(rank_file, dpi=150, bbox_inches='tight')
            plt.close()

        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Create multi-index with (expiry, ticker)
            if not combined_df.empty and 'expiry' in combined_df.columns and 'ticker' in combined_df.columns:
                # Set multi-index with expiry first, then ticker
                combined_df = combined_df.set_index(['expiry', 'ticker'])
                combined_df = combined_df.sort_index()  # Sort by expiry, then ticker
            
            return combined_df
        else:
            return pd.DataFrame()
    
    def calculate_option_elasticity(self, puts_df: pd.DataFrame, current_price: float,
                                  expiry_date: str, current_time: datetime) -> pd.DataFrame:
        """
        Calculate option elasticity (replicates elasticity analysis from op05.r).
        
        Args:
            puts_df: DataFrame of put options
            current_price: Current stock price
            expiry_date: Expiry date string
            current_time: Current datetime
            
        Returns:
            DataFrame with elasticity calculations
        """
        if puts_df.empty:
            return pd.DataFrame()
        
        try:
            # Calculate days to expiry (business days)
            expiry_dt = datetime.strptime(expiry_date, '%b.%d.%Y')
            days_to_go = max((expiry_dt - current_time).days, 1)
            
            # Convert to years (using 252 business days)
            years_to_go = days_to_go / 252.0
            
            results = []
            
            for _, put_row in puts_df.iterrows():
                try:
                    if not all(col in put_row.index for col in ['Strike', 'Last', 'IV']):
                        continue
                    
                    if pd.isna(put_row['Last']) or pd.isna(put_row['IV']) or put_row['Last'] <= 0:
                        continue
                    
                    strike = put_row['Strike']
                    put_price = put_row['Last']
                    iv = put_row['IV']
                    
                    # Calculate put delta using Black-Scholes
                    delta_put = self._calculate_put_delta(current_price, strike, iv, years_to_go)
                    
                    # Calculate elasticity
                    elasticity = (current_price * abs(delta_put)) / put_price
                    
                    results.append({
                        'strike': strike,
                        'put_price': put_price,
                        'IV': iv,
                        'delta': delta_put,
                        'elasticity': elasticity,
                        'days_to_expiry': days_to_go,
                        'years_to_expiry': years_to_go
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating elasticity for strike {put_row.get('Strike', 'unknown')}: {e}")
                    continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Error in elasticity calculation: {e}")
            return pd.DataFrame()
    
    def _calculate_put_delta(self, S: float, K: float, sigma: float, T: float, r: float = 0.0) -> float:
        """
        Calculate put delta using Black-Scholes formula.
        
        Args:
            S: Current stock price
            K: Strike price
            sigma: Implied volatility
            T: Time to expiry in years
            r: Risk-free rate
            
        Returns:
            Put delta
        """
        try:
            if T <= 0 or sigma <= 0:
                return 0.0
            
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            delta_put = norm.cdf(d1) - 1
            
            return delta_put
            
        except Exception:
            return 0.0
    
    def analyze_elasticity_strategies(self, option_chains: Dict[str, Any], prices: Dict[str, float],
                                    analysis_date: str, current_time: datetime,
                                    plot_folder: str = './plots2') -> pd.DataFrame:
        """
        Comprehensive elasticity analysis across all tickers and expiries (from op05.r).
        
        Args:
            option_chains: Dictionary of option chain data
            prices: Dictionary of current prices
            analysis_date: Date string for analysis
            current_time: Current datetime
            plot_folder: Folder for saving plots
            
        Returns:
            DataFrame with elasticity analysis
        """
        plot_dir = Path(plot_folder) / analysis_date
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        tickers = list(option_chains.keys())
        
        # Get all available expiry dates
        expiry_dates = set()
        for ticker_data in option_chains.values():
            expiry_dates.update(ticker_data.keys())
        
        all_results = []
        
        for expiry_date in expiry_dates:
            elasticity_data = []
            
            for ticker in tickers:
                if ticker not in option_chains or ticker not in prices:
                    continue
                
                ticker_data = option_chains[ticker]
                if expiry_date not in ticker_data:
                    continue
                
                puts_df = ticker_data[expiry_date].get('original_puts', ticker_data[expiry_date].get('puts', pd.DataFrame()))
                
                if puts_df.empty:
                    continue
                
                try:
                    current_price = prices[ticker]
                    cost = current_price * 100  # Cost basis
                    
                    # Calculate elasticity for all puts
                    elasticity_df = self.calculate_option_elasticity(
                        puts_df, current_price, expiry_date, current_time
                    )
                    
                    if elasticity_df.empty:
                        continue
                    
                    # Find ATM put for analysis
                    strikes = elasticity_df['strike'].values
                    idx_atm = np.argmin(np.abs(strikes - current_price))
                    atm_put = elasticity_df.iloc[idx_atm]
                    
                    elasticity_data.append({
                        'ticker': ticker,
                        'cost': cost,
                        'strike': atm_put['strike'],
                        'elasticity': atm_put['elasticity'],
                        'IV': atm_put['IV'],
                        'delta': atm_put['delta'],
                        'expiry': expiry_date
                    })
                
                except Exception as e:
                    self.logger.warning(f"Error in elasticity analysis for {ticker} {expiry_date}: {e}")
                    continue
            
            if not elasticity_data:
                continue
            
            df = pd.DataFrame(elasticity_data)
            
            try:
                # Create elasticity scatter plot
                plt.figure(figsize=(18, 7))
                scatter = plt.scatter(df['cost'], df['elasticity'], c=df['IV']*4, 
                                    alpha=0.7, s=60, cmap='coolwarm')
                
                # Add ticker labels
                for _, row in df.iterrows():
                    plt.annotate(row['ticker'], 
                               (row['cost'], row['elasticity']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
                
                plt.xlabel('Cost (Stock Price * 100)')
                plt.ylabel('Elasticity')
                plt.title(f'PUT Elasticity Analysis: {expiry_date} on {analysis_date}')
                plt.colorbar(scatter, label='IV')
                plt.grid(True, alpha=0.3)
                
                elasticity_ratio_file = plot_dir / f"puts_ratio{expiry_date}.png"
                plt.savefig(elasticity_ratio_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Create ranked elasticity plot
                df_sorted = df.sort_values('elasticity', ascending=False)
                df_sorted['ticker'] = pd.Categorical(df_sorted['ticker'], 
                                                   categories=df_sorted['ticker'], 
                                                   ordered=True)
                
                plt.figure(figsize=(12, 7))
                plt.plot(range(len(df_sorted)), df_sorted['elasticity'].to_numpy(), 'bo-', markersize=4)
                plt.xlabel('Ticker Rank')
                plt.ylabel('Elasticity')
                plt.title(f'PUT Elasticity Ranking: {expiry_date} on {analysis_date}')
                plt.xticks(range(len(df_sorted)), df_sorted['ticker'].to_numpy(), rotation=90)
                plt.grid(True, alpha=0.3)
                
                elasticity_rank_file = plot_dir / f"puts_{expiry_date}.png"
                plt.savefig(elasticity_rank_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add analysis metadata
                df_sorted['row_num'] = range(1, len(df_sorted) + 1)
                df_sorted['analysis_date'] = analysis_date
                
                # Print IV statistics
                iv_quantiles = df['IV'].quantile([0.1*i for i in range(11)])
                self.logger.info(f"Elasticity analysis for {expiry_date}: {len(df)} tickers")
                self.logger.info(f"IV quantiles: {iv_quantiles.to_dict()}")
                
                all_results.append(df_sorted)
                
            except Exception as e:
                self.logger.error(f"Error creating elasticity plots for {expiry_date}: {e}")
                continue
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Create multi-index with (expiry, ticker)
            if not combined_df.empty and 'expiry' in combined_df.columns and 'ticker' in combined_df.columns:
                # Set multi-index with expiry first, then ticker
                combined_df = combined_df.set_index(['expiry', 'ticker'])
                combined_df = combined_df.sort_index()  # Sort by expiry, then ticker
            
            return combined_df
        else:
            return pd.DataFrame()
    
    def load_and_analyze_historical_data(self, data_folder: str = None) -> Dict[str, Any]:
        """
        Load and analyze historical option data from multiple files (from op01.r and op05.r).
        
        Args:
            data_folder: Folder containing historical data files (uses self.data_folder if None)
            
        Returns:
            Dictionary with historical analysis results
        """
        if data_folder is None:
            data_folder = str(self.data_folder)
        
        data_path = Path(data_folder)
        if not data_path.exists():
            self.logger.warning(f"Data folder {data_folder} does not exist")
            return {}
        
        # Find all RData files (for now, we'll work with pickle files)
        data_files = list(data_path.glob('*.pkl'))
        
        if not data_files:
            self.logger.warning(f"No data files found in {data_folder}")
            return {}
        
        historical_data = {}
        
        for file_path in data_files:
            try:
                # Extract date from filename
                filename = file_path.stem
                date_match = re.search(r'oc_(\d{6})', filename)
                if date_match:
                    date_str = date_match.group(1)
                else:
                    date_str = filename
                
                # Load data (assuming pickle format for now)
                # In a real implementation, you'd load the actual data
                self.logger.info(f"Would load historical data from {file_path}")
                historical_data[date_str] = {
                    'filename': str(file_path),
                    'date': date_str
                }
                
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue
        
        self.logger.info(f"Loaded metadata for {len(historical_data)} historical datasets")
        return historical_data
    
    def analyze_put_elasticity_op05(self, option_chains: Dict[str, Any], prices: Dict[str, float],
                                   curr_time: datetime, config: 'Config' = None) -> Dict[str, pd.DataFrame]:
        """
        Replicate the exact functionality from op05.r for loop (lines 65-147).
        
        This method processes option chains data to calculate put elasticity analysis
        exactly as implemented in the R script.
        
        Args:
            option_chains: Dictionary of option chain data by ticker and expiry
            prices: Dictionary of current prices by ticker
            curr_time: Current datetime for calculating time to expiry
            config: Configuration object with path settings (uses default if None)
            
        Returns:
            Dictionary of DataFrames with elasticity analysis results by expiry date
        """
        # Import Config here to avoid circular imports
        from .config import Config
        
        # Use provided config or create default
        if config is None:
            config = Config()
        
        # Use config's elasticity plots path
        plot_dir = config.get_elasticity_plots_path()
        
        # Get tickers and expiry dates
        tickers = list(option_chains.keys())
        
        # Get all expiry dates from first ticker (assumes all tickers have same expiries)
        if not tickers:
            self.logger.warning("No tickers found in option_chains")
            return {}
        
        first_ticker = list(option_chains.keys())[0]
        dates_exp = list(option_chains[first_ticker].keys())
        
        results_by_expiry = {}
        
        # Main loop through expiry dates (equivalent to R's for (date_exp in dates_exp))
        for date_exp in dates_exp:
            self.logger.info(f"Processing expiry date: {date_exp}")
            
            # Initialize dataframe (equivalent to R's df initialization)
            df_data = []
            
            # Loop through tickers (equivalent to R's for(ticker in tickers))
            for ticker in tickers:
                if ticker not in option_chains or ticker not in prices:
                    continue
                
                # Get puts data
                ticker_data = option_chains[ticker]
                if date_exp not in ticker_data:
                    continue
                
                puts = ticker_data[date_exp].get('puts', pd.DataFrame())
                
                if puts is None or puts.empty:
                    continue
                
                try:
                    # Calculate days to go (equivalent to R lines 72-79)
                    expiry_dt = datetime.strptime(date_exp, "%b.%d.%Y")
                    days_to_go = self._calculate_business_days(curr_time, expiry_dt)
                    
                    if days_to_go <= 0:
                        days_to_go = 1
                    
                    year_to_go = days_to_go / 252.0  # Business days to years
                    
                    # Get current price and cost
                    price = prices[ticker]
                    cost = price * 100
                    
                    # Find ITM put closest to current price (equivalent to R line 86)
                    if 'Strike' not in puts.columns:
                        continue
                    
                    strikes = puts['Strike'].values
                    idx_itm = np.argmin(np.abs(strikes - price))
                    atm_put = puts.iloc[idx_itm]
                    
                    # Check required columns exist
                    if 'Last' not in puts.columns or 'IV' not in puts.columns:
                        continue
                    
                    if pd.isna(atm_put['Last']) or pd.isna(atm_put['IV']) or atm_put['Last'] <= 0:
                        continue
                    
                    # Extract values for Black-Scholes calculation
                    k = atm_put['Strike']  # Strike
                    v = atm_put['IV']      # IV (volatility)
                    s = price              # Current price
                    r = 0                  # Risk-free rate
                    tt = year_to_go        # Time to expiry
                    d = 0                  # Dividend yield
                    
                    # Calculate put delta using Black-Scholes (equivalent to R lines 103-104)
                    delta_put = self._calculate_put_delta_bs(s, k, v, tt, r, d)
                    
                    # Calculate elasticity (equivalent to R lines 107-108)
                    elasticity = (price * abs(delta_put)) / atm_put['Last']
                    
                    # Append to dataframe (equivalent to R line 110)
                    df_data.append({
                        'ticker': ticker,
                        'strike': k,
                        'cost': cost,
                        'elas': elasticity,
                        'IV': v,
                        'delta': delta_put,
                        'prob': 0.5,  # Placeholder as in R
                        'EV': 0       # Placeholder as in R
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing {ticker} for {date_exp}: {e}")
                    continue
            
            # Convert to DataFrame
            if not df_data:
                self.logger.warning(f"No valid data for expiry {date_exp}")
                continue
            
            df = pd.DataFrame(df_data)
            df.set_index('ticker', inplace=True)
            
            # Create plots (equivalent to R lines 120-143)
            try:
                # Plot 1: Scatter plot (cost vs elasticity colored by IV)
                plt.figure(figsize=(18, 7))
                scatter = plt.scatter(df['cost'], df['elas'], c=df['IV']*4, 
                                    alpha=0.7, s=60, cmap='coolwarm')
                
                # Add ticker labels
                for ticker in df.index:
                    row = df.loc[ticker]
                    plt.annotate(ticker, 
                               (row['cost'], row['elas']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
                
                plt.xlabel('Cost')
                plt.ylabel('Elasticity')
                plt.title(f'PUTS: {date_exp} On {config.date_string}')
                plt.colorbar(scatter, label='IV')
                plt.grid(True, alpha=0.3)
                
                # Save scatter plot using config path method
                scatter_file = config.get_elasticity_scatter_plot_path(date_exp)
                plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Plot 2: Ranked elasticity plot
                df_sorted = df.sort_values('elas', ascending=False)
                df_sorted['ticker'] = df_sorted.index
                
                plt.figure(figsize=(12, 7))
                plt.plot(range(len(df_sorted)), df_sorted['elas'].to_numpy(), 'bo-', markersize=4)
                plt.xlabel('Ticker')
                plt.ylabel('Elasticity')
                plt.title(f'PUTS: {date_exp} On {config.date_string}')
                plt.xticks(range(len(df_sorted)), df_sorted['ticker'].to_numpy(), rotation=90)
                plt.grid(True, alpha=0.3)
                
                # Save ranked plot using config path method
                rank_file = config.get_elasticity_ranking_plot_path(date_exp)
                plt.savefig(rank_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Add row numbers and print statistics (equivalent to R lines 139-143)
                df_sorted['row_num'] = range(1, len(df_sorted) + 1)
                
                # Print quantiles
                iv_quantiles = df['IV'].quantile([0.1*i for i in range(11)])
                self.logger.info(f"IV quantiles for {date_exp}: {iv_quantiles.to_dict()}")
                
                # Store results with the same variable pattern as R
                results_by_expiry[date_exp] = df_sorted
                
                self.logger.info(f"Completed analysis for {date_exp}: {len(df)} tickers")
                
            except Exception as e:
                self.logger.error(f"Error creating plots for {date_exp}: {e}")
                results_by_expiry[date_exp] = df
        
        return results_by_expiry
    
    def _calculate_business_days(self, start_date: datetime, end_date: datetime) -> int:
        """
        Calculate business days between two dates (replicates R's businessDaysBetween).
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of business days
        """
        # Simple implementation using pandas business day functionality
        return len(pd.bdate_range(start_date.date(), end_date.date())) - 1
    
    def _calculate_put_delta_bs(self, S: float, K: float, sigma: float, T: float,
                               r: float = 0.0, d: float = 0.0) -> float:
        """
        Calculate put delta using Black-Scholes formula (replicates R's putdelta function).

        Args:
            S: Current stock price
            K: Strike price
            sigma: Volatility (IV)
            T: Time to expiry in years
            r: Risk-free rate
            d: Dividend yield

        Returns:
            Put delta
        """
        try:
            if T <= 0 or sigma <= 0:
                return 0.0

            # Black-Scholes put delta calculation
            d1 = (math.log(S / K) + (r - d + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            delta_put = norm.cdf(d1) - 1

            return delta_put

        except Exception as e:
            self.logger.warning(f"Error calculating put delta: {e}")
            return 0.0

    def _calculate_call_delta_bs(self, S: float, K: float, sigma: float, T: float,
                                r: float = 0.0, d: float = 0.0) -> float:
        """
        Calculate call delta using Black-Scholes formula.

        Args:
            S: Current stock price
            K: Strike price
            sigma: Volatility (IV)
            T: Time to expiry in years
            r: Risk-free rate
            d: Dividend yield

        Returns:
            Call delta
        """
        try:
            if T <= 0 or sigma <= 0:
                return 0.0

            # Black-Scholes call delta calculation
            d1 = (math.log(S / K) + (r - d + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            delta_call = norm.cdf(d1)

            return delta_call

        except Exception as e:
            self.logger.warning(f"Error calculating call delta: {e}")
            return 0.0        