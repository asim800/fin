"""
Quick utility to check option elasticity for a single ticker.

Usage:
    python quick_elasticity_check.py AAPL
    python quick_elasticity_check.py NVDA --expiry "Jan.16.2026"
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from options_analysis.orchestrator import OptionsAnalysisOrchestrator
from options_analysis.config import Config
import pandas as pd


def format_elasticity_report(ticker, expiry, calls, puts, call_elast, put_elast, current_price):
    """Format a nice elasticity report."""

    print("\n" + "=" * 80)
    print(f"ELASTICITY REPORT: {ticker} (Current Price: ${current_price:.2f})")
    print(f"Expiry: {expiry}")
    print("=" * 80)

    # Merge calls with elasticity
    call_data = calls.copy()
    if not call_elast.empty:
        call_data = call_data.join(call_elast)
        call_data = call_data[call_data['call_elasticity'].notna()]

        if not call_data.empty:
            print("\n📞 CALL OPTIONS - Top 10 by Elasticity")
            print("-" * 80)
            print(f"{'Strike':<10} {'Price':<10} {'IV':<8} {'Volume':<10} {'OI':<10} {'Elasticity':<12}")
            print("-" * 80)

            # Sort by elasticity and show top 10
            top_calls = call_data.nlargest(10, 'call_elasticity')

            for _, row in top_calls.iterrows():
                strike = row.get('Strike', 0)
                price = row.get('Last', 0)
                iv = row.get('IV', 0)
                vol = row.get('Vol', 0)
                oi = row.get('OI', 0)
                elast = row.get('call_elasticity', 0)

                # Highlight ATM strikes
                marker = " ← ATM" if abs(strike - current_price) / current_price < 0.05 else ""

                print(f"{strike:<10.2f} ${price:<9.2f} {iv:<8.2%} {vol:<10.0f} {oi:<10.0f} "
                      f"{elast:<12.2f}{marker}")

            # Summary statistics
            print(f"\nSummary:")
            print(f"  Max Elasticity: {call_data['call_elasticity'].max():.2f}")
            print(f"  Min Elasticity: {call_data['call_elasticity'].min():.2f}")
            print(f"  Mean Elasticity: {call_data['call_elasticity'].mean():.2f}")

    # Merge puts with elasticity
    put_data = puts.copy()
    if not put_elast.empty:
        put_data = put_data.join(put_elast)
        put_data = put_data[put_data['put_elasticity'].notna()]

        if not put_data.empty:
            print("\n📉 PUT OPTIONS - Top 10 by Elasticity")
            print("-" * 80)
            print(f"{'Strike':<10} {'Price':<10} {'IV':<8} {'Volume':<10} {'OI':<10} {'Elasticity':<12}")
            print("-" * 80)

            # Sort by elasticity and show top 10
            top_puts = put_data.nlargest(10, 'put_elasticity')

            for _, row in top_puts.iterrows():
                strike = row.get('Strike', 0)
                price = row.get('Last', 0)
                iv = row.get('IV', 0)
                vol = row.get('Vol', 0)
                oi = row.get('OI', 0)
                elast = row.get('put_elasticity', 0)

                # Highlight ATM strikes
                marker = " ← ATM" if abs(strike - current_price) / current_price < 0.05 else ""

                print(f"{strike:<10.2f} ${price:<9.2f} {iv:<8.2%} {vol:<10.0f} {oi:<10.0f} "
                      f"{elast:<12.2f}{marker}")

            # Summary statistics
            print(f"\nSummary:")
            print(f"  Max Elasticity: {put_data['put_elasticity'].max():.2f}")
            print(f"  Min Elasticity: {put_data['put_elasticity'].min():.2f}")
            print(f"  Mean Elasticity: {put_data['put_elasticity'].mean():.2f}")

    print("\n" + "=" * 80)


def main():
    """Main function."""

    parser = argparse.ArgumentParser(description='Check option elasticity for a ticker')
    parser.add_argument('ticker', help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--expiry', help='Specific expiry date (e.g., "Jan.16.2026")', default=None)
    parser.add_argument('--all-expiries', action='store_true', help='Show all expiries')

    args = parser.parse_args()

    ticker = args.ticker.upper()

    print(f"Fetching data for {ticker}...")

    # Initialize orchestrator
    config = Config()
    orchestrator = OptionsAnalysisOrchestrator(config)

    # Fetch market data
    try:
        market_data = orchestrator.fetch_market_data(ticker_list=[ticker])

        if not market_data.is_valid():
            print(f"❌ Failed to fetch valid data for {ticker}")
            return

        current_price = market_data.prices.get(ticker)
        print(f"✓ Current price: ${current_price:.2f}")

        # Process with elasticity
        print("Calculating elasticity...")
        processed_chains = orchestrator.processor.extract_contract_identifiers(
            market_data.option_chains,
            prices=market_data.prices,
            current_time=market_data.timestamp
        )

        if ticker not in processed_chains:
            print(f"❌ No options data for {ticker}")
            return

        ticker_data = processed_chains[ticker]
        expiries = list(ticker_data.keys())

        print(f"✓ Found {len(expiries)} expiry dates")

        # If specific expiry requested
        if args.expiry:
            if args.expiry in ticker_data:
                expiry_data = ticker_data[args.expiry]
                format_elasticity_report(
                    ticker,
                    args.expiry,
                    expiry_data.get('calls', pd.DataFrame()),
                    expiry_data.get('puts', pd.DataFrame()),
                    expiry_data.get('call_elasticity', pd.DataFrame()),
                    expiry_data.get('put_elasticity', pd.DataFrame()),
                    current_price
                )
            else:
                print(f"❌ Expiry '{args.expiry}' not found")
                print(f"Available expiries: {', '.join(expiries)}")
        else:
            # Show first expiry or all if requested
            expiries_to_show = expiries if args.all_expiries else expiries[:1]

            for expiry in expiries_to_show:
                expiry_data = ticker_data[expiry]
                format_elasticity_report(
                    ticker,
                    expiry,
                    expiry_data.get('calls', pd.DataFrame()),
                    expiry_data.get('puts', pd.DataFrame()),
                    expiry_data.get('call_elasticity', pd.DataFrame()),
                    expiry_data.get('put_elasticity', pd.DataFrame()),
                    current_price
                )

            if not args.all_expiries and len(expiries) > 1:
                print(f"\n💡 Tip: Use --all-expiries to see all {len(expiries)} expiry dates")
                print(f"💡 Tip: Use --expiry '{expiries[1]}' to see a specific date")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
