"""Put-call parity analysis module."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class PutCallParityAnalyzer:
    """Analyzes put-call parity relationships in option chains."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize PutCallParityAnalyzer.
        
        Args:
            risk_free_rate: Risk-free interest rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_put_call_parity(self, option_chains: Dict[str, Any], 
                                 price_vector: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate put-call parity for option chains.
        
        This replicates the R putCallParity function logic.
        
        Args:
            option_chains: Dictionary of processed option chain data
            price_vector: Dictionary of current stock prices by ticker
            
        Returns:
            Dictionary with put-call parity analysis results
        """
        pcp_results = {}
        
        tickers = list(option_chains.keys())
        
        for i, ticker in enumerate(tickers):
            if ticker not in price_vector:
                self.logger.warning(f"No price data for {ticker}, skipping")
                continue
                
            current_price = price_vector[ticker]
            ticker_options = option_chains[ticker]
            
            pcp_results[ticker] = {}
            
            for expiry_name, expiry_data in ticker_options.items():
                try:
                    pcp_data = self._calculate_parity_for_expiry(
                        expiry_data, current_price, ticker, expiry_name
                    )
                    
                    if pcp_data is not None and not pcp_data.empty:
                        pcp_results[ticker][expiry_name] = pcp_data
                    
                except Exception as e:
                    self.logger.error(f"Error calculating parity for {ticker} {expiry_name}: {e}")
                    continue
        
        return pcp_results
    
    def _calculate_parity_for_expiry(self, expiry_data: Dict[str, Any], 
                                   current_price: float, ticker: str, 
                                   expiry_name: str) -> Optional[pd.DataFrame]:
        """
        Calculate put-call parity for a specific expiry.
        
        Args:
            expiry_data: Option data for specific expiry
            current_price: Current stock price
            ticker: Stock ticker
            expiry_name: Expiry date name
            
        Returns:
            DataFrame with parity calculations or None if insufficient data
        """
        calls_df = expiry_data.get('calls', pd.DataFrame())
        puts_df = expiry_data.get('puts', pd.DataFrame())
        
        if calls_df.empty or puts_df.empty:
            self.logger.debug(f"No call/put data for {ticker} {expiry_name}")
            return None
        
        # Ensure we have the required columns
        required_cols = ['Strike', 'Bid', 'Ask']
        if not all(col in calls_df.columns for col in required_cols):
            self.logger.warning(f"Missing required columns in calls for {ticker} {expiry_name}")
            return None
        if not all(col in puts_df.columns for col in required_cols):
            self.logger.warning(f"Missing required columns in puts for {ticker} {expiry_name}")
            return None
        
        if len(calls_df) == 0 or len(puts_df) == 0:
            return None
        
        # Calculate parity for each matched call/put pair
        parity_rows = []
        
        # Assuming calls and puts are already matched by the processor
        num_pairs = min(len(calls_df), len(puts_df))
        
        for k in range(num_pairs):
            try:
                call_row = calls_df.iloc[k]
                put_row = puts_df.iloc[k]
                
                # Extract values with error checking
                strike = call_row['Strike']
                call_bid = call_row['Bid'] if pd.notna(call_row['Bid']) else 0
                call_ask = call_row['Ask'] if pd.notna(call_row['Ask']) else 0
                put_bid = put_row['Bid'] if pd.notna(put_row['Bid']) else 0
                put_ask = put_row['Ask'] if pd.notna(put_row['Ask']) else 0
                
                # Calculate put-call parity components
                # P + S = C + K (put + stock = call + strike)
                # Rearranged: P + S vs C + K
                
                put_plus_stock = put_bid + current_price  # pps in R
                call_plus_strike = call_ask + strike      # cpk in R
                
                parity_row = {
                    'price': current_price,
                    'strike': strike,
                    'pps': put_plus_stock,     # Put + Stock
                    'cpk': call_plus_strike,   # Call + Strike  
                    'call': call_ask,
                    'put': put_bid,
                    'parity_diff': put_plus_stock - call_plus_strike,
                    'call_bid': call_bid,
                    'call_ask': call_ask,
                    'put_bid': put_bid,
                    'put_ask': put_ask
                }
                
                parity_rows.append(parity_row)
                
            except Exception as e:
                self.logger.warning(f"Error processing pair {k} for {ticker} {expiry_name}: {e}")
                continue
        
        if not parity_rows:
            return None
        
        # Create DataFrame with parity calculations
        parity_df = pd.DataFrame(parity_rows)
        
        # Set column names to match R output
        parity_df.columns = ['price', 'strike', 'pps', 'cpk', 'call', 'put', 
                           'parity_diff', 'call_bid', 'call_ask', 'put_bid', 'put_ask']
        
        
        return parity_df
    
    def analyze_arbitrage_opportunities(self, pcp_results: Dict[str, Any], 
                                      threshold: float = 0.05) -> pd.DataFrame:
        """
        Identify potential arbitrage opportunities based on put-call parity violations.
        
        Args:
            pcp_results: Put-call parity results
            threshold: Minimum parity difference threshold for arbitrage
            
        Returns:
            DataFrame with arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        for ticker, ticker_data in pcp_results.items():
            for expiry, parity_df in ticker_data.items():
                if parity_df is None or parity_df.empty:
                    continue
                
                # Find significant parity violations
                violations = parity_df[abs(parity_df['parity_diff']) > threshold].copy()
                
                for _, row in violations.iterrows():
                    opportunity = {
                        'ticker': ticker,
                        'expiry': expiry,
                        'strike': row['strike'],
                        'parity_diff': row['parity_diff'],
                        'current_price': row['price'],
                        'strategy': 'Buy Put + Stock, Sell Call' if row['parity_diff'] > 0 
                                   else 'Buy Call, Sell Put + Stock',
                        'expected_profit': abs(row['parity_diff']),
                        'call_price': row['call'],
                        'put_price': row['put']
                    }
                    arbitrage_opportunities.append(opportunity)
        
        arbitrage_df = pd.DataFrame(arbitrage_opportunities)
        
        if not arbitrage_df.empty:
            arbitrage_df = arbitrage_df.sort_values('expected_profit', ascending=False)
            self.logger.info(f"Found {len(arbitrage_df)} potential arbitrage opportunities")
        
        return arbitrage_df
    
    def calculate_parity_statistics(self, pcp_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate summary statistics for put-call parity analysis.
        
        Args:
            pcp_results: Put-call parity results
            
        Returns:
            DataFrame with summary statistics
        """
        stats_data = []
        
        for ticker, ticker_data in pcp_results.items():
            for expiry, parity_df in ticker_data.items():
                if parity_df is None or parity_df.empty:
                    continue
                
                stats = {
                    'ticker': ticker,
                    'expiry': expiry,
                    'num_pairs': len(parity_df),
                    'mean_parity_diff': parity_df['parity_diff'].mean(),
                    'std_parity_diff': parity_df['parity_diff'].std(),
                    'max_parity_diff': parity_df['parity_diff'].max(),
                    'min_parity_diff': parity_df['parity_diff'].min(),
                    'violations_5pct': len(parity_df[abs(parity_df['parity_diff']) > 0.05]),
                    'violations_1pct': len(parity_df[abs(parity_df['parity_diff']) > 0.01]),
                    'mean_strike': parity_df['strike'].mean(),
                    'strike_range': parity_df['strike'].max() - parity_df['strike'].min()
                }
                stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        return stats_df
    
    def get_parity_summary(self, pcp_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get overall summary of put-call parity analysis.
        
        Args:
            pcp_results: Put-call parity results
            
        Returns:
            Dictionary with summary information
        """
        total_pairs = 0
        total_violations = 0
        all_diffs = []
        
        for ticker, ticker_data in pcp_results.items():
            for expiry, parity_df in ticker_data.items():
                if parity_df is not None and not parity_df.empty:
                    total_pairs += len(parity_df)
                    total_violations += len(parity_df[abs(parity_df['parity_diff']) > 0.01])
                    all_diffs.extend(parity_df['parity_diff'].tolist())
        
        summary = {
            'total_tickers': len(pcp_results),
            'total_pairs_analyzed': total_pairs,
            'total_violations_1pct': total_violations,
            'violation_rate': total_violations / total_pairs if total_pairs > 0 else 0,
            'mean_parity_diff': np.mean(all_diffs) if all_diffs else 0,
            'std_parity_diff': np.std(all_diffs) if all_diffs else 0,
            'max_violation': max(abs(d) for d in all_diffs) if all_diffs else 0
        }
        
        return summary