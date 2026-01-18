#!/usr/bin/env python3
"""Export monthly options price tables to CSV files.

This script exports call and put price tables for monthly expiries only.
Monthly options typically expire on the third Friday of each month.
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis.toolkit import AnalysisToolkit
from plot_monthly_options import identify_monthly_expiries


def export_monthly_tables(ticker, output_dir='exports', n_around=10):
    """Export monthly options price tables to CSV.

    Args:
        ticker: Stock ticker symbol
        output_dir: Directory to save CSV files
        n_around: Number of strikes around ATM to include

    Returns:
        List of created file paths
    """
    print("=" * 80)
    print(f"Exporting Monthly Options Tables for {ticker}")
    print("=" * 80)
    print()

    # Initialize toolkit
    toolkit = AnalysisToolkit()

    # Get quote
    print(f"🔍 Fetching quote for {ticker}...")
    quote = toolkit.get_quote(ticker)
    if not quote:
        print(f"❌ Failed to get quote for {ticker}")
        return []

    current_price = quote['price']
    print(f"✓ Current Price: ${current_price:.2f}")
    print()

    # Get option tables
    print("📊 Fetching option tables...")
    tables = toolkit.get_option_tables(ticker)
    if not tables:
        print(f"❌ Failed to get option tables for {ticker}")
        return []

    # Get put and call price tables
    put_price_table = tables.get('put_price_table')
    call_price_table = tables.get('call_price_table')

    if put_price_table is None or call_price_table is None:
        print("❌ Price tables not found")
        return []

    print(f"✓ Retrieved tables")
    print(f"  Put prices: {len(put_price_table)} expiries × {len(put_price_table.columns)} strikes")
    print(f"  Call prices: {len(call_price_table)} expiries × {len(call_price_table.columns)} strikes")
    print()

    # Identify monthly expiries
    all_expiries = put_price_table.index.tolist()
    monthly_expiries = identify_monthly_expiries(all_expiries)

    # Sort monthly expiries by date
    from plot_monthly_options import sort_expiries_by_date
    monthly_expiries_sorted = sort_expiries_by_date(monthly_expiries)

    print(f"📅 Identified {len(monthly_expiries_sorted)} monthly expiries (3rd Friday):")
    for exp in monthly_expiries_sorted[:10]:
        print(f"   • {exp}")
    if len(monthly_expiries_sorted) > 10:
        print(f"   ... and {len(monthly_expiries_sorted) - 10} more")
    print()

    # Filter to monthly expiries (with sorted order)
    monthly_puts = put_price_table.loc[monthly_expiries_sorted]
    monthly_calls = call_price_table.loc[monthly_expiries_sorted]

    # Filter strikes around ATM
    import numpy as np
    all_strikes = monthly_puts.columns.tolist()

    # Find ATM strike
    strikes_arr = np.array(all_strikes)
    atm_idx = np.argmin(np.abs(strikes_arr - current_price))

    # Get strikes around ATM
    start_idx = max(0, atm_idx - n_around)
    end_idx = min(len(all_strikes), atm_idx + n_around + 1)
    selected_strikes = all_strikes[start_idx:end_idx]

    # Filter tables
    monthly_puts = monthly_puts[selected_strikes]
    monthly_calls = monthly_calls[selected_strikes]

    print(f"📊 Filtered to {len(selected_strikes)} strikes around ATM:")
    print(f"   Strike range: ${selected_strikes[0]} to ${selected_strikes[-1]}")
    print(f"   ATM strike: ${all_strikes[atm_idx]} (current: ${current_price:.2f})")
    print()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export to CSV
    created_files = []

    # Export put prices
    put_file = output_path / f"{ticker}_monthly_put_prices.csv"
    monthly_puts.to_csv(put_file)
    created_files.append(str(put_file))
    print(f"✓ Saved: {put_file}")
    print(f"  {len(monthly_puts)} expiries × {len(monthly_puts.columns)} strikes")

    # Export call prices
    call_file = output_path / f"{ticker}_monthly_call_prices.csv"
    monthly_calls.to_csv(call_file)
    created_files.append(str(call_file))
    print(f"✓ Saved: {call_file}")
    print(f"  {len(monthly_calls)} expiries × {len(monthly_calls.columns)} strikes")

    print()
    print("=" * 80)
    print(f"✓ Exported {len(created_files)} files to {output_dir}/")
    print("=" * 80)

    return created_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export monthly options price tables to CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export AAPL monthly prices with ±10 strikes around ATM
  python src/export_monthly_tables.py AAPL

  # Export SPY with wider strike range
  python src/export_monthly_tables.py SPY --around 15

  # Export to custom directory
  python src/export_monthly_tables.py NVDA --output my_data

  # Export multiple tickers
  python src/export_monthly_tables.py AAPL SPY NVDA QQQ
        """
    )

    parser.add_argument(
        'tickers',
        nargs='+',
        help='Stock ticker symbol(s) (e.g., AAPL, SPY, NVDA)'
    )

    parser.add_argument(
        '--output',
        '-o',
        default='exports',
        help='Output directory for CSV files (default: exports/)'
    )

    parser.add_argument(
        '--around',
        '-n',
        type=int,
        default=10,
        help='Number of strikes around ATM to include (default: 10)'
    )

    args = parser.parse_args()

    # Process each ticker
    all_files = []
    for ticker in args.tickers:
        try:
            files = export_monthly_tables(
                ticker.upper(),
                output_dir=args.output,
                n_around=args.around
            )
            all_files.extend(files)
            print()
        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            print()
            continue

    # Summary
    if all_files:
        print()
        print("=" * 80)
        print(f"✓ Successfully exported {len(all_files)} files")
        print("=" * 80)
        print()
        print("Files created:")
        for f in all_files:
            print(f"  • {f}")


if __name__ == '__main__':
    main()
