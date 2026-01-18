"""High-level convenience API for Options Analysis Toolkit.

This module provides a simplified interface for common options analysis tasks,
wrapping the orchestrator and individual components for ease of use.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime

from .orchestrator import OptionsAnalysisOrchestrator
from .config import Config


class AnalysisToolkit:
    """High-level convenience API for options analysis.

    This class provides simplified methods for common analysis tasks,
    making it easier to use the toolkit programmatically.

    Examples:
        >>> toolkit = AnalysisToolkit()
        >>> elasticity = toolkit.get_elasticity('AAPL')
        >>> opportunities = toolkit.find_arbitrage(min_profit=0.05)
        >>> toolkit.export_to_excel('AAPL', 'output.xlsx')
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize the toolkit.

        Args:
            config: Optional configuration object
        """
        self.orchestrator = OptionsAnalysisOrchestrator(config)
        self.logger = logging.getLogger(__name__)
        self._last_market_data = None
        self._last_processed = None

    def _fetch_market_data(self, tickers: List[str]):
        """Fetch market data for given tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            MarketData object
        """
        fetcher = self.orchestrator.data_fetcher
        option_chains = fetcher.get_options_batch(tickers)
        prices, _ = fetcher.get_quotes_batch(tickers)

        from .orchestrator import MarketData
        return MarketData(
            option_chains=option_chains,
            prices=prices,
            tickers=tickers,
            timestamp=datetime.now()
        )

    def analyze(self, ticker: str, generate_plots: bool = True) -> Dict[str, Any]:
        """Analyze a single ticker with full analysis.

        Args:
            ticker: Stock ticker symbol
            generate_plots: Whether to generate visualization plots

        Returns:
            Dictionary containing analysis results including:
            - quote: Current price and market data
            - option_chain: Processed options chain
            - tables: Pivot tables for strikes and expiries
            - elasticity: Option elasticity calculations

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> results = toolkit.analyze('AAPL')
            >>> print(f"Current price: ${results['quote']['price']:.2f}")
        """
        try:
            results = self.orchestrator.run_ticker_analysis(ticker)
            return results
        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {e}")
            return {}

    def get_elasticity(
        self,
        ticker: str,
        expiry: Optional[str] = None,
        option_type: str = 'call',
        top_n: int = 10
    ) -> pd.DataFrame:
        """Get option elasticity for a ticker.

        Elasticity measures leverage: how much the option price changes
        for a 1% change in the underlying stock price.

        Args:
            ticker: Stock ticker symbol
            expiry: Specific expiry date (e.g., "Jan.16.2026"). If None, uses nearest expiry
            option_type: 'call' or 'put'
            top_n: Number of top options to return (sorted by elasticity)

        Returns:
            DataFrame with strikes, prices, IV, and elasticity sorted by elasticity

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> # Get top 10 call elasticities for nearest expiry
            >>> calls = toolkit.get_elasticity('AAPL')
            >>> # Get top 5 put elasticities for specific expiry
            >>> puts = toolkit.get_elasticity('NVDA', expiry='Mar.20.2026',
            ...                               option_type='put', top_n=5)
        """
        try:
            # Fetch market data
            market_data = self._fetch_market_data([ticker])
            if not market_data.is_valid():
                self.logger.error(f"Failed to fetch data for {ticker}")
                return pd.DataFrame()

            # Process with elasticity
            processed = self.orchestrator.processor.extract_contract_identifiers(
                market_data.option_chains,
                prices=market_data.prices,
                current_time=market_data.timestamp
            )

            if ticker not in processed:
                self.logger.error(f"No options data for {ticker}")
                return pd.DataFrame()

            ticker_data = processed[ticker]

            # Select expiry
            if expiry and expiry in ticker_data:
                selected_expiry = expiry
            elif expiry:
                available = ', '.join(list(ticker_data.keys())[:5])
                self.logger.error(f"Expiry '{expiry}' not found. Available: {available}")
                return pd.DataFrame()
            else:
                # Use first (nearest) expiry
                selected_expiry = list(ticker_data.keys())[0]

            exp_data = ticker_data[selected_expiry]

            # Get options and elasticity
            if option_type.lower() == 'call':
                options_df = exp_data.get('calls')
                elasticity_df = exp_data.get('call_elasticity')
                elasticity_col = 'call_elasticity'
            else:  # put
                options_df = exp_data.get('puts')
                elasticity_df = exp_data.get('put_elasticity')
                elasticity_col = 'put_elasticity'

            if elasticity_df is None or elasticity_df.empty:
                self.logger.warning(f"No elasticity data for {ticker} {option_type}s")
                return pd.DataFrame()

            # Merge and sort
            merged = options_df.join(elasticity_df)
            merged = merged[merged[elasticity_col].notna()]
            merged = merged.nlargest(top_n, elasticity_col)

            # Select relevant columns
            columns = ['Strike', 'Last', 'Bid', 'Ask', 'IV', elasticity_col]
            if option_type.lower() == 'call':
                columns.append('call_estimated_delta')
            else:
                columns.append('put_estimated_delta')

            result_cols = [col for col in columns if col in merged.columns]

            return merged[result_cols]

        except Exception as e:
            self.logger.error(f"Error getting elasticity for {ticker}: {e}")
            return pd.DataFrame()

    def find_arbitrage(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        ticker_file: Optional[str] = None,
        min_profit: float = 0.01,
        max_results: int = 20
    ) -> pd.DataFrame:
        """Find put-call parity arbitrage opportunities.

        Args:
            tickers: Single ticker or list of tickers to analyze. If None, uses default file
            ticker_file: Path to file containing tickers (one per line)
            min_profit: Minimum profit threshold (e.g., 0.01 = 1%)
            max_results: Maximum number of results to return

        Returns:
            DataFrame with arbitrage opportunities sorted by expected profit

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> # Find arbitrage in specific tickers
            >>> arb = toolkit.find_arbitrage(['AAPL', 'MSFT', 'NVDA'], min_profit=0.05)
            >>> # Find arbitrage using ticker file
            >>> arb = toolkit.find_arbitrage(ticker_file='tickers.txt')
            >>> # Use default ticker file
            >>> arb = toolkit.find_arbitrage()
        """
        try:
            # Run full analysis
            if ticker_file:
                results = self.orchestrator.run_full_analysis(ticker_file)
            elif tickers:
                if isinstance(tickers, str):
                    tickers = [tickers]
                # Create temporary file with tickers
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                    f.write('\n'.join(tickers))
                    temp_file = f.name
                try:
                    results = self.orchestrator.run_full_analysis(temp_file)
                finally:
                    Path(temp_file).unlink()
            else:
                results = self.orchestrator.run_full_analysis()

            arbitrage_df = results.get('arbitrage')

            if arbitrage_df is None or arbitrage_df.empty:
                self.logger.info("No arbitrage opportunities found")
                return pd.DataFrame()

            # Filter by min_profit
            filtered = arbitrage_df[arbitrage_df['expected_profit'] >= min_profit * 100]
            filtered = filtered.head(max_results)

            return filtered

        except Exception as e:
            self.logger.error(f"Error finding arbitrage: {e}")
            return pd.DataFrame()

    def find_best_puts(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        budget: float = 1000,
        min_elasticity: float = 2.0,
        max_results: int = 10
    ) -> pd.DataFrame:
        """Find best put options based on budget and elasticity criteria.

        Args:
            tickers: Single ticker or list of tickers. If None, uses default
            budget: Maximum budget per contract (in dollars)
            min_elasticity: Minimum elasticity threshold
            max_results: Maximum number of results to return

        Returns:
            DataFrame with best put options sorted by elasticity

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> # Find cheap, high-leverage puts
            >>> puts = toolkit.find_best_puts(budget=500, min_elasticity=3.0)
            >>> # Find puts for specific stocks
            >>> puts = toolkit.find_best_puts(['AAPL', 'NVDA'], budget=1000)
        """
        try:
            if tickers is None:
                # Use default ticker file
                tickers_to_analyze = self.orchestrator.config.get_tickers()
            elif isinstance(tickers, str):
                tickers_to_analyze = [tickers]
            else:
                tickers_to_analyze = tickers

            # Fetch and process data
            market_data = self._fetch_market_data(tickers_to_analyze)
            if not market_data.is_valid():
                self.logger.error("Failed to fetch market data")
                return pd.DataFrame()

            processed = self.orchestrator.processor.extract_contract_identifiers(
                market_data.option_chains,
                prices=market_data.prices,
                current_time=market_data.timestamp
            )

            # Collect all puts with elasticity
            all_puts = []

            for ticker in processed:
                ticker_data = processed[ticker]
                current_price = market_data.prices[ticker]

                for expiry, exp_data in ticker_data.items():
                    puts = exp_data.get('puts')
                    put_elasticity = exp_data.get('put_elasticity')

                    if puts is None or put_elasticity is None:
                        continue

                    # Merge and filter
                    merged = puts.join(put_elasticity)
                    merged = merged[merged['put_elasticity'].notna()]
                    merged = merged[merged['put_elasticity'] >= min_elasticity]
                    merged = merged[merged['Last'] * 100 <= budget]  # Convert to per-contract price

                    # Add metadata
                    merged['ticker'] = ticker
                    merged['expiry'] = expiry
                    merged['current_price'] = current_price
                    merged['contract_cost'] = merged['Last'] * 100

                    all_puts.append(merged)

            if not all_puts:
                self.logger.info("No puts found matching criteria")
                return pd.DataFrame()

            # Combine and sort
            result = pd.concat(all_puts, ignore_index=False)
            result = result.nlargest(max_results, 'put_elasticity')

            # Select relevant columns
            columns = ['ticker', 'Strike', 'Last', 'contract_cost', 'IV',
                      'put_elasticity', 'expiry', 'current_price']
            result_cols = [col for col in columns if col in result.columns]

            return result[result_cols]

        except Exception as e:
            self.logger.error(f"Error finding best puts: {e}")
            return pd.DataFrame()

    def find_best_calls(
        self,
        tickers: Optional[Union[str, List[str]]] = None,
        budget: float = 1000,
        min_elasticity: float = 2.0,
        max_results: int = 10
    ) -> pd.DataFrame:
        """Find best call options based on budget and elasticity criteria.

        Args:
            tickers: Single ticker or list of tickers. If None, uses default
            budget: Maximum budget per contract (in dollars)
            min_elasticity: Minimum elasticity threshold
            max_results: Maximum number of results to return

        Returns:
            DataFrame with best call options sorted by elasticity

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> # Find cheap, high-leverage calls
            >>> calls = toolkit.find_best_calls(budget=500, min_elasticity=3.0)
            >>> # Find calls for specific stocks
            >>> calls = toolkit.find_best_calls(['AAPL', 'NVDA'], budget=1000)
        """
        try:
            if tickers is None:
                # Use default ticker file
                tickers_to_analyze = self.orchestrator.config.get_tickers()
            elif isinstance(tickers, str):
                tickers_to_analyze = [tickers]
            else:
                tickers_to_analyze = tickers

            # Fetch and process data
            market_data = self._fetch_market_data(tickers_to_analyze)
            if not market_data.is_valid():
                self.logger.error("Failed to fetch market data")
                return pd.DataFrame()

            processed = self.orchestrator.processor.extract_contract_identifiers(
                market_data.option_chains,
                prices=market_data.prices,
                current_time=market_data.timestamp
            )

            # Collect all calls with elasticity
            all_calls = []

            for ticker in processed:
                ticker_data = processed[ticker]
                current_price = market_data.prices[ticker]

                for expiry, exp_data in ticker_data.items():
                    calls = exp_data.get('calls')
                    call_elasticity = exp_data.get('call_elasticity')

                    if calls is None or call_elasticity is None:
                        continue

                    # Merge and filter
                    merged = calls.join(call_elasticity)
                    merged = merged[merged['call_elasticity'].notna()]
                    merged = merged[merged['call_elasticity'] >= min_elasticity]
                    merged = merged[merged['Last'] * 100 <= budget]  # Convert to per-contract price

                    # Add metadata
                    merged['ticker'] = ticker
                    merged['expiry'] = expiry
                    merged['current_price'] = current_price
                    merged['contract_cost'] = merged['Last'] * 100

                    all_calls.append(merged)

            if not all_calls:
                self.logger.info("No calls found matching criteria")
                return pd.DataFrame()

            # Combine and sort
            result = pd.concat(all_calls, ignore_index=False)
            result = result.nlargest(max_results, 'call_elasticity')

            # Select relevant columns
            columns = ['ticker', 'Strike', 'Last', 'contract_cost', 'IV',
                      'call_elasticity', 'expiry', 'current_price']
            result_cols = [col for col in columns if col in result.columns]

            return result[result_cols]

        except Exception as e:
            self.logger.error(f"Error finding best calls: {e}")
            return pd.DataFrame()

    def get_option_tables(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get pivot tables for a ticker (strikes × expiries).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary of pivot tables including:
            - call_price_table, put_price_table
            - call_iv_table, put_iv_table
            - call_elasticity_table, put_elasticity_table
            - And more...

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> tables = toolkit.get_option_tables('AAPL')
            >>> print(tables['call_elasticity_table'])
        """
        try:
            # Fetch and process data
            market_data = self._fetch_market_data([ticker])
            if not market_data.is_valid():
                self.logger.error(f"Failed to fetch data for {ticker}")
                return {}

            processed = self.orchestrator.processor.extract_contract_identifiers(
                market_data.option_chains,
                prices=market_data.prices,
                current_time=market_data.timestamp
            )

            # Create tables
            tables = self.orchestrator.processor.create_option_tables(processed)

            return tables.get(ticker, {})

        except Exception as e:
            self.logger.error(f"Error getting tables for {ticker}: {e}")
            return {}

    def export_to_csv(self, ticker: str, output_dir: str = 'exports') -> List[str]:
        """Export options data to CSV files.

        Args:
            ticker: Stock ticker symbol
            output_dir: Directory to save CSV files

        Returns:
            List of file paths created

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> files = toolkit.export_to_csv('AAPL', 'my_exports')
            >>> print(f"Created {len(files)} files")
        """
        try:
            tables = self.get_option_tables(ticker)
            if not tables:
                self.logger.error(f"No data to export for {ticker}")
                return []

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            created_files = []
            for table_name, df in tables.items():
                csv_file = output_path / f"{ticker}_{table_name}.csv"
                df.to_csv(csv_file)
                created_files.append(str(csv_file))
                self.logger.info(f"Saved: {csv_file}")

            return created_files

        except Exception as e:
            self.logger.error(f"Error exporting to CSV for {ticker}: {e}")
            return []

    def export_to_excel(self, ticker: str, output_file: str) -> bool:
        """Export options data to Excel file with multiple sheets.

        Args:
            ticker: Stock ticker symbol
            output_file: Path to output Excel file

        Returns:
            True if successful, False otherwise

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> success = toolkit.export_to_excel('AAPL', 'aapl_analysis.xlsx')
        """
        try:
            tables = self.get_option_tables(ticker)
            if not tables:
                self.logger.error(f"No data to export for {ticker}")
                return False

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for table_name, df in tables.items():
                    # Truncate sheet name to 31 chars (Excel limit)
                    sheet_name = table_name[:31]
                    df.to_excel(writer, sheet_name=sheet_name)

            self.logger.info(f"Saved: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting to Excel for {ticker}: {e}")
            return False

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """Get current quote and market data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with price, volume, and other market data

        Examples:
            >>> toolkit = AnalysisToolkit()
            >>> quote = toolkit.get_quote('AAPL')
            >>> print(f"Price: ${quote['price']:.2f}")
        """
        try:
            market_data = self._fetch_market_data([ticker])
            if not market_data.is_valid():
                return {}

            return {
                'ticker': ticker,
                'price': market_data.prices[ticker],
                'timestamp': market_data.timestamp
            }

        except Exception as e:
            self.logger.error(f"Error getting quote for {ticker}: {e}")
            return {}
