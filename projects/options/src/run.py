#!/usr/bin/env python3
"""
Simple runner for options analysis - No installation needed!

Usage:
    python run.py AAPL                    # Quick elasticity check
    python run.py AAPL --puts             # Check put elasticity
    python run.py AAPL --export           # Export to Excel
    python run.py --find-cheap            # Find cheap options
"""

import sys
from pathlib import Path

# Add current directory to path (we're already in src/)
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis import AnalysisToolkit


def quick_check(ticker, option_type='call'):
    """Quick elasticity check for a ticker."""
    print(f"\n{'='*60}")
    print(f"{ticker} - Top 10 {option_type.upper()} Options by Elasticity")
    print(f"{'='*60}\n")

    toolkit = AnalysisToolkit()

    # Get quote
    quote = toolkit.get_quote(ticker)
    if quote:
        print(f"Current Price: ${quote['price']:.2f}\n")

    # Get elasticity
    options = toolkit.get_elasticity(ticker, option_type=option_type, top_n=10)

    if not options.empty:
        print(options.to_string())
    else:
        print(f"No {option_type} options found")

    print(f"\n{'='*60}\n")


def export_data(ticker):
    """Export options data to Excel."""
    print(f"\nExporting {ticker} data to Excel...")

    toolkit = AnalysisToolkit()
    filename = f"{ticker}_options.xlsx"

    success = toolkit.export_to_excel(ticker, filename)

    if success:
        print(f"✓ Exported to {filename}")
    else:
        print(f"✗ Export failed")


def find_cheap_options():
    """Find cheap high-leverage options."""
    tickers = ['AAPL', 'MSFT', 'NVDA']
    budget = 500
    min_leverage = 3.0

    print(f"\n{'='*60}")
    print(f"Finding options < ${budget} with leverage > {min_leverage}x")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"{'='*60}\n")

    toolkit = AnalysisToolkit()

    # Find calls
    print("🔍 Searching for calls...")
    calls = toolkit.find_best_calls(
        tickers=tickers,
        budget=budget,
        min_elasticity=min_leverage,
        max_results=5
    )

    if not calls.empty:
        print(f"\n✓ Found {len(calls)} calls:\n")
        print(calls[['ticker', 'Strike', 'contract_cost', 'call_elasticity']].to_string())
    else:
        print("✗ No calls found")

    # Find puts
    print("\n🔍 Searching for puts...")
    puts = toolkit.find_best_puts(
        tickers=tickers,
        budget=budget,
        min_elasticity=min_leverage,
        max_results=5
    )

    if not puts.empty:
        print(f"\n✓ Found {len(puts)} puts:\n")
        print(puts[['ticker', 'Strike', 'contract_cost', 'put_elasticity']].to_string())
    else:
        print("✗ No puts found")

    print(f"\n{'='*60}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    # Check for flags
    if arg == '--find-cheap':
        find_cheap_options()
    elif arg.startswith('--'):
        print(f"Unknown option: {arg}")
        print(__doc__)
        sys.exit(1)
    else:
        # Ticker provided
        ticker = arg.upper()

        if '--puts' in sys.argv:
            quick_check(ticker, option_type='put')
        elif '--export' in sys.argv:
            export_data(ticker)
        else:
            quick_check(ticker, option_type='call')


if __name__ == '__main__':
    main()
