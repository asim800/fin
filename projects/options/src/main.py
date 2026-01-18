#!/usr/bin/env python3
"""
Main entry point for Options Analysis System.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis.orchestrator import OptionsAnalysisOrchestrator


def main():
    """Main entry point for the application."""
    # Ensure log directory exists
    log_dir = Path('options_analysis')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to show important messages on stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Output to stdout
            logging.FileHandler(log_dir / 'options_analysis.log', 'a')  # Append to log file
        ]
    )
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Options Analysis System')
    parser.add_argument('--file', '--tickers', dest='ticker_file', 
                       help='Ticker file to use (default: tickersv1.txt)')
    parser.add_argument('--ticker', dest='single_ticker',
                       help='Analyze a single ticker instead of a file')
    
    args = parser.parse_args()
    
    try:
        # Initialize orchestrator
        orchestrator = OptionsAnalysisOrchestrator()

        # Determine what to analyze
        if args.single_ticker:
            # Single ticker analysis
            print(f"🎯 Analyzing single ticker: {args.single_ticker}")
            results = orchestrator.run_ticker_analysis(args.single_ticker)
        else:
            # Full analysis with ticker file (default: tickersv1.txt)
            ticker_file = args.ticker_file  # None means use default
            if ticker_file:
                print(f"📊 Using ticker file: {ticker_file}")
            else:
                print(f"📊 Using default ticker file: tickersv1.txt")
            results = orchestrator.run_full_analysis(ticker_file)
        
        # Print summary
        summary = results.get('summary', {})
        print("\n" + "="*50)
        print("OPTIONS ANALYSIS COMPLETE")
        print("="*50)
        print(f"Total tickers analyzed: {summary.get('total_tickers', 0)}")
        print(f"Total option pairs: {summary.get('total_pairs_analyzed', 0)}")
        print(f"Parity violations (1%): {summary.get('total_violations_1pct', 0)}")
        print(f"Violation rate: {summary.get('violation_rate', 0):.2%}")
        print(f"Plot files created: {len(results.get('plot_files', []))}")
        
        # Show top arbitrage opportunities
        arbitrage = results.get('arbitrage')
        if arbitrage is not None and not arbitrage.empty:
            print(f"\nTop arbitrage opportunities:")
            top_arb = arbitrage.head(5)
            for _, row in top_arb.iterrows():
                print(f"  {row['ticker']} ${row['strike']}: ${row['expected_profit']:.2f}")
        
        print("\nAnalysis saved to data folder")
        print("Plots saved to plots folder")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.exception("Full error details:")
        sys.exit(1)

    return results, orchestrator


if __name__ == "__main__":
    [opt, orch] = main()



'''
run ./src/main.py
run ./src/main.py --file tickers.txt

opt['option_chains']['SPY']['Nov.10.2025']['calls'].keys()

'''