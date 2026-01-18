"""Main orchestrator for the options analysis system."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

from .config import Config
from .data_fetcher import DataFetcher
from .option_processor import OptionChainProcessor
from .put_call_parity import PutCallParityAnalyzer
from .black_scholes import BlackScholesCalculator
from .visualizer import Visualizer
from .data_persistence import DataPersistence


@dataclass
class MarketData:
    """Container for market data fetched for analysis."""
    option_chains: Dict[str, Any]
    prices: Dict[str, float]
    tickers: List[str]
    timestamp: datetime
    
    def is_valid(self) -> bool:
        """Check if market data is complete and valid."""
        return (
            bool(self.option_chains) and 
            bool(self.prices) and 
            bool(self.tickers) and
            set(self.tickers) <= set(self.option_chains.keys()) and
            set(self.tickers) <= set(self.prices.keys())
        )


@dataclass
class IndividualResults:
    """Container for individual ticker analysis results."""
    ticker_results: Dict[str, Dict[str, Any]]
    processed_chains: Dict[str, Any]
    pcp_results: Dict[str, Any]
    plot_files: List[str]
    option_tables: Dict[str, Dict[str, pd.DataFrame]]
    timestamp: datetime

    def get_ticker_result(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific ticker."""
        return self.ticker_results.get(ticker)

    def get_all_tickers(self) -> List[str]:
        """Get list of all analyzed tickers."""
        return list(self.ticker_results.keys())

    def get_ticker_tables(self, ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Get option tables for a specific ticker."""
        return self.option_tables.get(ticker)


@dataclass
class ComprehensiveResults:
    """Container for comprehensive multi-ticker analysis results."""
    arbitrage_opportunities: Any  # DataFrame
    put_strategy_analysis: Dict[str, Any]
    cross_ticker_plots: List[str]
    summary_statistics: Dict[str, Any]
    timestamp: datetime
    
    def get_top_arbitrage(self, n: int = 5) -> Any:
        """Get top N arbitrage opportunities."""
        if self.arbitrage_opportunities is not None and not self.arbitrage_opportunities.empty:
            return self.arbitrage_opportunities.head(n)
        return None


class OptionsAnalysisOrchestrator:
    """Main orchestrator for the options analysis workflow."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the orchestrator with all components.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.processor = OptionChainProcessor(data_folder=self.config.data_folder)
        self.parity_analyzer = PutCallParityAnalyzer()
        self.bs_calculator = BlackScholesCalculator()
        self.visualizer = Visualizer()
        self.persistence = DataPersistence(self.config.data_folder)
        
        self.logger.info("Options Analysis Orchestrator initialized")
    
    def run_full_analysis(self, ticker_file: Optional[str] = None, 
                         use_cached: bool = True, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run complete options analysis workflow using 3-step approach.
        
        Args:
            ticker_file: Optional ticker file path
            use_cached: Whether to use cached data if available
            force_refresh: Force refresh of all data
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            self.logger.info("🚀 Starting full options analysis workflow")
            
            # Load tickers
            tickers = self._load_tickers(ticker_file)
            self.logger.info(f"📝 Loaded {len(tickers)} tickers for analysis")
            self.logger.info(f"🎨 {self.config.get_plot_summary()}")
            
            # Step 1: Fetch market data (with caching logic)
            self.logger.info("🔄 Step 1/3: Fetching market data")
            market_data = self._fetch_market_data(tickers, use_cached, force_refresh)
            
            # Step 2: Run individual ticker analysis
            self.logger.info("🔄 Step 2/3: Running individual ticker analysis")
            individual_results = self._run_individual_analysis(market_data)
            
            # Step 3: Run comprehensive multi-ticker analysis
            self.logger.info("🔄 Step 3/3: Running comparative analysis")
            comprehensive_results = self._run_comparative_analysis(market_data, individual_results)
            
            # Merge all results for backward compatibility
            final_results = self._merge_all_results(
                market_data, individual_results, comprehensive_results
            )
            
            # Note: Individual step results are already saved in step-specific folders
            # No need to save consolidated results anymore (following Option A)
            self.logger.info("✅ Full analysis completed successfully!")
            
            # Log final summary
            summary = comprehensive_results.summary_statistics
            self.logger.info(f"📊 Final Summary: {summary.get('total_tickers', 0)} tickers, "
                           f"{len(individual_results.plot_files)} plots generated")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in full analysis: {e}")
            raise
    
    def _load_tickers(self, ticker_file: Optional[str] = None) -> list:
        """Load ticker symbols."""
        try:
            # Try to find ticker file in parent directory if not absolute path
            if ticker_file is None:
                ticker_file = self.config.ticker_file
            
            if not os.path.isabs(ticker_file):
                # Look in parent directory (where R files are)
                parent_dir = Path(__file__).parent.parent.parent.parent
                ticker_path = parent_dir / ticker_file
                if ticker_path.exists():
                    ticker_file = str(ticker_path)
            
            tickers = self.config.load_tickers(ticker_file)
            self.logger.info(f"Loaded {len(tickers)} tickers from {ticker_file}")
            return tickers
            
        except Exception as e:
            self.logger.error(f"Error loading tickers: {e}")
            raise
    
    def _fetch_current_prices(self, tickers: List[str]) -> tuple:
        """Fetch current stock prices."""
        try:
            self.logger.info("Fetching current stock prices")
            prices, price_df = self.data_fetcher.get_quotes_batch(tickers)
            self.logger.info(f"Retrieved prices for {len(prices)} tickers")
            return prices, price_df
        except Exception as e:
            self.logger.error(f"Error fetching prices: {e}")
            raise
    
    def _fetch_option_chains(self, tickers: List[str]) -> Dict[str, Any]:
        """Fetch option chains for all tickers."""
        try:
            self.logger.info("Fetching option chains")
            option_chains = {}
            
            for ticker in tickers:
                chain_data = self.data_fetcher.get_option_chain(ticker)
                if chain_data:
                    option_chains[ticker] = chain_data
            
            self.logger.info(f"Retrieved option chains for {len(option_chains)} tickers")
            return option_chains
        except Exception as e:
            self.logger.error(f"Error fetching option chains: {e}")
            raise
    
    def _is_cache_valid(self, cached_data: Dict[str, Any], tickers: List[str]) -> bool:
        """Check if cached data is still valid."""
        try:
            # Check if same tickers
            cached_tickers = cached_data.get('tickers', [])
            if set(cached_tickers) != set(tickers):
                return False
            
            # Check if data is from today (simple check)
            timestamp = cached_data.get('timestamp')
            if timestamp and timestamp.date() == datetime.now().date():
                return True
                
            return False
        except Exception as e:
            self.logger.warning(f"Error validating cache: {e}")
            return False
    
    def _is_cache_valid_distributed(self, cached_data: Dict[str, Any], tickers: List[str]) -> bool:
        """Check if distributed cached data is still valid."""
        try:
            # Check if same tickers
            cached_tickers = cached_data.get('tickers', [])
            if set(cached_tickers) != set(tickers):
                return False
            
            # Check if data is from today (simple check)
            timestamp = cached_data.get('timestamp')
            if timestamp and timestamp.date() == datetime.now().date():
                return True
                
            return False
        except Exception as e:
            self.logger.warning(f"Error validating distributed cache: {e}")
            return False
    
    def _run_analysis_with_cached_data(self, cached_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using cached data with 3-step workflow."""
        try:
            self.logger.info("🔄 Running analysis with cached data using 3-step workflow")
            
            # Create MarketData from cached data
            market_data = MarketData(
                option_chains=cached_data.get('option_chains', {}),
                prices=cached_data.get('prices', {}),
                tickers=cached_data.get('tickers', []),
                timestamp=cached_data.get('timestamp', datetime.now())
            )
            
            # Run 3-step analysis
            individual_results = self._run_individual_analysis(market_data)
            comprehensive_results = self._run_comparative_analysis(market_data, individual_results)
            
            # Merge results
            return self._merge_all_results(market_data, individual_results, comprehensive_results)
            
        except Exception as e:
            self.logger.error(f"Error running cached analysis: {e}")
            raise
    
    def _fetch_market_data(self, tickers: List[str], use_cached: bool = True, 
                          force_refresh: bool = False) -> MarketData:
        """
        Step 1: Fetch market data with caching logic.
        
        Args:
            tickers: List of ticker symbols to fetch data for
            use_cached: Whether to use cached data if available
            force_refresh: Force refresh of all data
            
        Returns:
            MarketData object containing option chains, prices, and tickers
        """
        try:
            self.logger.info(f"📡 Fetching market data for {len(tickers)} tickers")
            
            # Check for cached data in distributed files
            if use_cached and not force_refresh:
                cached_data = self.persistence.load_distributed_analysis_data()
                
                if cached_data and self._is_cache_valid_distributed(cached_data, tickers):
                    self.logger.info("Using cached market data from distributed files")
                    return MarketData(
                        option_chains=cached_data.get('option_chains', {}),
                        prices=cached_data.get('prices', {}),
                        tickers=cached_data.get('tickers', []),
                        timestamp=cached_data.get('timestamp', datetime.now())
                    )
            
            # Fetch fresh data
            self.logger.info("Fetching fresh market data")
            prices, price_df = self._fetch_current_prices(tickers)
            option_chains = self._fetch_option_chains(tickers)
            
            # Create MarketData object
            market_data = MarketData(
                option_chains=option_chains,
                prices=prices,
                tickers=tickers,
                timestamp=datetime.now()
            )
            
            # Save raw market data to step-specific folder
            market_data_dict = {
                'option_chains': option_chains,
                'prices': prices,
                'price_df': price_df,
                'tickers': tickers,
                'timestamp': market_data.timestamp
            }
            self.persistence.save_market_data(market_data_dict)
            self.logger.info("Saved market data to raw folder")

            # Log fetched data details for debugging
            self.logger.info(f"Fetched {len(option_chains)} option chains")
            self.logger.info(f"Fetched {len(prices)} prices")
            self.logger.info(f"Requested {len(tickers)} tickers")

            # Filter to only include tickers with complete data
            valid_tickers = [t for t in tickers if t in option_chains and t in prices]
            missing_tickers = set(tickers) - set(valid_tickers)

            if missing_tickers:
                self.logger.warning(f"Skipping {len(missing_tickers)} ticker(s) with incomplete data: {missing_tickers}")

            # Check if we have ANY valid tickers
            if not valid_tickers:
                raise ValueError("No tickers have complete market data available")

            # Update market_data to only include valid tickers
            market_data = MarketData(
                option_chains={k: v for k, v in option_chains.items() if k in valid_tickers},
                prices={k: v for k, v in prices.items() if k in valid_tickers},
                tickers=valid_tickers,
                timestamp=datetime.now()
            )

            self.logger.info(f"Successfully fetched market data for {len(valid_tickers)} tickers")
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            raise
    
    def _run_individual_analysis(self, market_data: MarketData) -> IndividualResults:
        """
        Step 2: Run individual ticker analysis.
        
        Args:
            market_data: MarketData object containing fetched data
            
        Returns:
            IndividualResults object containing individual ticker analysis
        """
        try:
            self.logger.info(f"Running individual analysis for {len(market_data.tickers)} tickers")
            
            # Process option chains (shared step) - now with elasticity calculation
            self.logger.info("Processing option chains with elasticity calculation")
            processed_chains = self.processor.extract_contract_identifiers(
                market_data.option_chains,
                prices=market_data.prices,
                current_time=market_data.timestamp
            )

            # Create option tables (pivot tables by strike and expiry)
            self.logger.info("Creating option pivot tables")
            option_tables = self.processor.create_option_tables(processed_chains)

            # Calculate put-call parity (shared step)
            self.logger.info("Calculating put-call parity")
            pcp_results = self.parity_analyzer.calculate_put_call_parity(
                processed_chains, market_data.prices
            )
            
            # Individual ticker analysis
            ticker_results = {}
            all_plot_files = []
            successful_tickers = 0
            total_tickers = len(market_data.tickers)
            
            for i, ticker in enumerate(market_data.tickers, 1):
                try:
                    self.logger.info(f"Analyzing ticker {i}/{total_tickers}: {ticker}")
                    ticker_plot_files = []

                    if ticker in pcp_results and ticker in market_data.prices:
                        # Only generate plots if enabled for this ticker
                        if self.config.should_plot(ticker):
                            self.logger.info(f"Generating plots for {ticker}")

                            # Put-call parity plots for this ticker
                            parity_plots = self.visualizer.create_put_call_parity_plots(
                                ticker, pcp_results[ticker], market_data.prices[ticker]
                            )
                            ticker_plot_files.extend(parity_plots)

                            # Put-call ratio plots for this ticker
                            ratio_plots = self.visualizer.create_putcall_ratios_plots(
                                processed_chains, ticker
                            )
                            ticker_plot_files.extend(ratio_plots)

                            # Put-call IV plots for this ticker
                            iv_plots = self.visualizer.create_putcall_iv_plots(
                                processed_chains, ticker
                            )
                            ticker_plot_files.extend(iv_plots)

                            # Cross-expiry analysis plots for this ticker (Step 2: Individual analysis)
                            if ticker in market_data.option_chains:
                                cross_plots = self.processor.create_cross_expiry_analysis(
                                    option_chains={ticker: market_data.option_chains[ticker]},
                                    prices={ticker: market_data.prices[ticker]},
                                    ticker=ticker,
                                    plot_folder='plots2'
                                )
                                ticker_plot_files.extend(cross_plots.values())

                                # 6-panel option analysis plots for this ticker (Step 2: Individual analysis)
                                option_plots = self.processor.plot_option_analysis(
                                    option_chains={ticker: market_data.option_chains[ticker]},
                                    prices={ticker: market_data.prices[ticker]},
                                    plot_folder='plots2',
                                    specific_ticker=ticker
                                )
                                ticker_plot_files.extend(option_plots)
                        else:
                            self.logger.debug(f"Skipping plots for {ticker} (plot flag disabled)")
                    
                    # Store individual ticker results
                    ticker_results[ticker] = {
                        'pcp_results': pcp_results.get(ticker, {}),
                        'processed_chain': processed_chains.get(ticker, {}),
                        'plot_files': ticker_plot_files,
                        'price': market_data.prices.get(ticker)
                    }
                    
                    all_plot_files.extend(ticker_plot_files)
                    successful_tickers += 1
                    self.logger.info(f"✅ Completed analysis for {ticker} ({i}/{total_tickers})")
                    
                except Exception as e:
                    self.logger.warning(f"❌ Failed to analyze {ticker}: {e}")
                    # Store empty result for failed ticker to maintain structure
                    ticker_results[ticker] = {
                        'pcp_results': {},
                        'processed_chain': {},
                        'plot_files': [],
                        'price': market_data.prices.get(ticker),
                        'error': str(e)
                    }
                    continue
            
            # Progress summary
            self.logger.info(f"Individual analysis completed: {successful_tickers}/{total_tickers} tickers successful")
            if successful_tickers == 0:
                raise ValueError("No tickers were successfully analyzed")
            
            individual_results = IndividualResults(
                ticker_results=ticker_results,
                processed_chains=processed_chains,
                pcp_results=pcp_results,
                plot_files=all_plot_files,
                option_tables=option_tables,
                timestamp=datetime.now()
            )
            
            # Save individual analysis results to step-specific folder
            individual_data_dict = {
                'ticker_results': ticker_results,
                'processed_chains': processed_chains,
                'pcp_results': pcp_results,
                'plot_files': all_plot_files,
                'option_tables': option_tables,
                'timestamp': individual_results.timestamp
            }
            self.persistence.save_individual_results(individual_data_dict)
            self.logger.info("Saved individual results to individual folder")
            
            self.logger.info(f"Completed individual analysis for {len(market_data.tickers)} tickers")
            return individual_results
            
        except Exception as e:
            self.logger.error(f"Error in individual analysis: {e}")
            raise
    
    def _run_comparative_analysis(self, market_data: MarketData, 
                                 individual_results: IndividualResults) -> ComprehensiveResults:
        """
        Step 3: Run multi-ticker comparative analysis.
        
        Args:
            market_data: MarketData object containing fetched data
            individual_results: IndividualResults from step 2
            
        Returns:
            ComprehensiveResults object containing comparative analysis
        """
        try:
            self.logger.info(f"🔄 Running comparative analysis for {len(market_data.tickers)} tickers")
            
            # Analyze arbitrage opportunities across all tickers
            self.logger.info("📊 Analyzing arbitrage opportunities across all tickers")
            arbitrage = self.parity_analyzer.analyze_arbitrage_opportunities(
                individual_results.pcp_results
            )
            
            # Generate put strategy analysis plots for ALL tickers (Step 3: Comparative analysis)
            self.logger.info("🎯 Running put strategy analysis across all tickers")
            import datetime as dt
            analysis_date = dt.datetime.now().strftime('%y%m%d')
            put_strategy_analysis = self.processor.analyze_put_strategies(
                option_chains=market_data.option_chains,
                prices=market_data.prices,
                analysis_date=analysis_date,
                plot_folder='plots3'
            )
            
            # Generate cross-ticker summary statistics
            self.logger.info("📈 Generating summary statistics")
            summary_stats = self._generate_comparative_summary(
                market_data, individual_results, arbitrage
            )
            
            comprehensive_results = ComprehensiveResults(
                arbitrage_opportunities=arbitrage,
                put_strategy_analysis=put_strategy_analysis,
                cross_ticker_plots=[],  # Put strategy plots are handled in put_strategy_analysis
                summary_statistics=summary_stats,
                timestamp=datetime.now()
            )
            
            # Save comprehensive analysis results to step-specific folder
            comprehensive_data_dict = {
                'arbitrage_opportunities': arbitrage,
                'put_strategy_analysis': put_strategy_analysis,
                'cross_ticker_plots': [],
                'summary_statistics': summary_stats,
                'timestamp': comprehensive_results.timestamp
            }
            self.persistence.save_comprehensive_results(comprehensive_data_dict)
            self.logger.info("Saved comprehensive results to comprehensive folder")
            
            self.logger.info(f"Completed comparative analysis for {len(market_data.tickers)} tickers")
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Error in comparative analysis: {e}")
            raise
    
    def _generate_comparative_summary(self, market_data: MarketData, 
                                     individual_results: IndividualResults,
                                     arbitrage: Any) -> Dict[str, Any]:
        """Generate summary statistics for comparative analysis."""
        try:
            total_pairs = sum(
                len(ticker_data.get('pcp_results', {})) 
                for ticker_data in individual_results.ticker_results.values()
            )
            
            violations_1pct = 0
            if arbitrage is not None and not arbitrage.empty:
                violations_1pct = len(arbitrage[arbitrage['expected_profit'].abs() > 1.0])
            
            violation_rate = violations_1pct / total_pairs if total_pairs > 0 else 0
            
            return {
                'total_tickers': len(market_data.tickers),
                'total_pairs_analyzed': total_pairs,
                'total_violations_1pct': violations_1pct,
                'violation_rate': violation_rate,
                'analysis_timestamp': market_data.timestamp,
                'individual_plot_count': len(individual_results.plot_files)
            }
            
        except Exception as e:
            self.logger.warning(f"Error generating comparative summary: {e}")
            return {
                'total_tickers': len(market_data.tickers),
                'total_pairs_analyzed': 0,
                'total_violations_1pct': 0,
                'violation_rate': 0,
                'analysis_timestamp': market_data.timestamp,
                'individual_plot_count': len(individual_results.plot_files)
            }
    
    def _merge_all_results(self, market_data: MarketData, 
                          individual_results: IndividualResults,
                          comprehensive_results: ComprehensiveResults) -> Dict[str, Any]:
        """
        Merge results from all 3 steps for backward compatibility.
        
        Args:
            market_data: Results from step 1
            individual_results: Results from step 2  
            comprehensive_results: Results from step 3
            
        Returns:
            Dictionary with combined results matching original format
        """
        try:
            # Combine all plot files
            all_plot_files = individual_results.plot_files.copy()
            all_plot_files.extend(comprehensive_results.cross_ticker_plots)
            
            # Create backward-compatible results structure
            merged_results = {
                'option_chains': market_data.option_chains,
                'processed_chains': individual_results.processed_chains,
                'prices': market_data.prices,
                'tickers': market_data.tickers,
                'pcp_results': individual_results.pcp_results,
                'arbitrage': comprehensive_results.arbitrage_opportunities,
                'plot_files': all_plot_files,
                'summary': comprehensive_results.summary_statistics,
                'timestamp': comprehensive_results.timestamp,
                
                # Additional structured data for future use
                'market_data': market_data,
                'individual_results': individual_results,
                'comprehensive_results': comprehensive_results
            }
            
            return merged_results
            
        except Exception as e:
            self.logger.error(f"Error merging results: {e}")
            # Return basic structure on error
            return {
                'option_chains': market_data.option_chains,
                'prices': market_data.prices,
                'tickers': market_data.tickers,
                'plot_files': individual_results.plot_files,
                'timestamp': datetime.now(),
                'summary': {}
            }
    
    def run_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
        """
        Run analysis for a single ticker using 3-step workflow.
        
        Args:
            ticker: Single ticker symbol to analyze
            
        Returns:
            Dictionary with single ticker analysis results
        """
        try:
            self.logger.info(f"Running analysis for {ticker}")
            
            # Step 1: Fetch market data
            market_data = self._fetch_market_data([ticker], use_cached=True, force_refresh=False)
            
            # Step 2: Run individual analysis
            individual_results = self._run_individual_analysis(market_data)
            
            # Format results for single ticker return (maintaining compatibility)
            ticker_result = individual_results.get_ticker_result(ticker)
            if not ticker_result:
                raise ValueError(f"No analysis results generated for {ticker}")
            
            results = {
                'ticker': ticker,
                'quote': {'price': market_data.prices.get(ticker)},
                'option_chain': market_data.option_chains.get(ticker, {}),
                'processed_chain': ticker_result.get('processed_chain', {}),
                'pcp_results': ticker_result.get('pcp_results', {}),
                'plot_files': ticker_result.get('plot_files', []),
                'timestamp': individual_results.timestamp
            }
            
            self.logger.info(f"Completed analysis for {ticker}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {e}")
            raise
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        try:
            tickers = results.get('tickers', [])
            pcp_results = results.get('pcp_results', {})
            arbitrage = results.get('arbitrage')
            plot_files = results.get('plot_files', [])
            
            # Count total pairs analyzed
            total_pairs = 0
            for ticker_data in pcp_results.values():
                for expiry_data in ticker_data.values():
                    if expiry_data is not None:
                        total_pairs += len(expiry_data)
            
            # Count violations
            violations_1pct = 0
            if arbitrage is not None and not arbitrage.empty:
                violations_1pct = len(arbitrage[arbitrage['expected_profit'] > 0.01])
            
            violation_rate = violations_1pct / total_pairs if total_pairs > 0 else 0
            
            summary = {
                'total_tickers': len(tickers),
                'total_pairs_analyzed': total_pairs,
                'total_violations_1pct': violations_1pct,
                'violation_rate': violation_rate,
                'plot_files_created': len(plot_files),
                'analysis_timestamp': datetime.now()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting analysis summary: {e}")
            return {'error': str(e)}