#!/usr/bin/env python3
"""
Show option pivot tables with control over display.

Usage:
    python src/show_tables.py AAPL
    python src/show_tables.py AAPL --expiries 3 --strikes 10
    python src/show_tables.py AAPL --expiry "Dec.05.2025"
    python src/show_tables.py AAPL --export
"""

import sys
from pathlib import Path
import pandas as pd

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis import AnalysisToolkit


def clean_table(table, max_expiries=None, max_strikes=None):
    """Remove all-NaN rows and columns, return cleaned table."""
    # First, remove rows (expiries) that are all NaN
    table = table.dropna(how='all', axis=0)

    # Remove columns (strikes) that are all NaN
    table = table.dropna(how='all', axis=1)

    # Now limit to requested size
    if max_expiries and len(table) > max_expiries:
        table = table.iloc[:max_expiries]

    if max_strikes and len(table.columns) > max_strikes:
        table = table.iloc[:, :max_strikes]

    return table


def show_tables(ticker, max_expiries=5, max_strikes=10, specific_expiry=None, export=False):
    """Get and display option tables with filtering."""

    print(f"\n{'='*80}")
    print(f"Option Tables for {ticker}")
    print(f"{'='*80}\n")

    toolkit = AnalysisToolkit()

    # Get all tables
    print(f"🔍 Fetching option data for {ticker}...")
    tables = toolkit.get_option_tables(ticker)

    if not tables:
        print(f"✗ No tables found for {ticker}")
        return

    print(f"✓ Retrieved {len(tables)} tables\n")

    # Get quote for reference
    quote = toolkit.get_quote(ticker)
    if quote:
        current_price = quote['price']
        print(f"Current Price: ${current_price:.2f}\n")

    # Show samples
    if specific_expiry:
        # Show specific expiry only
        show_specific_expiry(tables, specific_expiry, current_price)
    else:
        # Show multiple expiries
        show_multiple_expiries(tables, max_expiries, max_strikes, current_price)

    # Export if requested
    if export:
        print(f"\n{'='*80}")
        print("EXPORTING TO EXCEL")
        print(f"{'='*80}")

        filename = f"{ticker}_option_tables.xlsx"
        success = toolkit.export_to_excel(ticker, filename)

        if success:
            print(f"✓ Exported all tables to {filename}")
            print(f"  All 16 tables included (no filtering)")
        else:
            print(f"✗ Export failed")

    print(f"\n{'='*80}\n")


def show_specific_expiry(tables, expiry, current_price):
    """Show all strikes for a specific expiry."""

    print(f"{'='*80}")
    print(f"Showing data for expiry: {expiry}")
    print(f"{'='*80}\n")

    # Get data for this expiry
    for table_name in ['call_price_table', 'put_price_table',
                       'call_elasticity_table', 'put_elasticity_table']:
        if table_name not in tables:
            continue

        table = tables[table_name]

        if expiry not in table.index:
            print(f"✗ Expiry '{expiry}' not found in {table_name}")
            print(f"  Available expiries: {', '.join(table.index.tolist()[:5])}...")
            continue

        # Get row for this expiry
        row = table.loc[expiry]

        # Remove NaN values
        row = row.dropna()

        if len(row) == 0:
            print(f"No data for {expiry} in {table_name}")
            continue

        print(f"\n{table_name.replace('_', ' ').upper()}")
        print(f"{'-'*80}")

        # Format as DataFrame for nice display
        df = pd.DataFrame({
            'Strike': row.index,
            'Value': row.values
        })

        # Add indicator for ATM
        if current_price:
            df['ATM'] = df['Strike'].apply(
                lambda x: '←' if abs(x - current_price) < 10 else ''
            )

        print(df.to_string(index=False))
        print(f"\nTotal strikes: {len(row)}")


def show_multiple_expiries(tables, max_expiries, max_strikes, current_price):
    """Show multiple expiries with limited strikes."""

    # Show call prices
    print(f"{'='*80}")
    print(f"CALL PRICES (showing {max_expiries} expiries × {max_strikes} strikes)")
    print(f"{'='*80}")

    if 'call_price_table' in tables:
        call_prices = clean_table(tables['call_price_table'], max_expiries, max_strikes)

        if not call_prices.empty:
            print(call_prices.to_string())
            print(f"\n(Filtered: {call_prices.shape[0]} expiries × {call_prices.shape[1]} strikes)")
            print(f"Strike range: ${call_prices.columns[0]:.0f} to ${call_prices.columns[-1]:.0f}")
            if current_price:
                print(f"Current price: ${current_price:.2f}")
        else:
            print("No data after filtering")

    # Show put prices
    print(f"\n{'='*80}")
    print(f"PUT PRICES (showing {max_expiries} expiries × {max_strikes} strikes)")
    print(f"{'='*80}")

    if 'put_price_table' in tables:
        put_prices = clean_table(tables['put_price_table'], max_expiries, max_strikes)

        if not put_prices.empty:
            print(put_prices.to_string())
            print(f"\n(Filtered: {put_prices.shape[0]} expiries × {put_prices.shape[1]} strikes)")
            print(f"Strike range: ${put_prices.columns[0]:.0f} to ${put_prices.columns[-1]:.0f}")

    # Show call elasticity
    print(f"\n{'='*80}")
    print(f"CALL ELASTICITY (showing {max_expiries} expiries × {max_strikes} strikes)")
    print(f"{'='*80}")

    if 'call_elasticity_table' in tables:
        call_elast = clean_table(tables['call_elasticity_table'], max_expiries, max_strikes)

        if not call_elast.empty:
            print(call_elast.to_string())
            print(f"\n(Filtered: {call_elast.shape[0]} expiries × {call_elast.shape[1]} strikes)")

    # Show put elasticity
    print(f"\n{'='*80}")
    print(f"PUT ELASTICITY (showing {max_expiries} expiries × {max_strikes} strikes)")
    print(f"{'='*80}")

    if 'put_elasticity_table' in tables:
        put_elast = clean_table(tables['put_elasticity_table'], max_expiries, max_strikes)

        if not put_elast.empty:
            print(put_elast.to_string())
            print(f"\n(Filtered: {put_elast.shape[0]} expiries × {put_elast.shape[1]} strikes)")

    # Show available expiries
    if 'call_price_table' in tables:
        all_expiries = tables['call_price_table'].index.tolist()
        print(f"\n{'='*80}")
        print(f"ALL AVAILABLE EXPIRIES ({len(all_expiries)} total)")
        print(f"{'='*80}")
        for i, exp in enumerate(all_expiries, 1):
            print(f"{i:2d}. {exp}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Show option tables with filtering')
    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--expiries', type=int, default=5, help='Number of expiries to show (default: 5)')
    parser.add_argument('--strikes', type=int, default=10, help='Number of strikes to show (default: 10)')
    parser.add_argument('--expiry', type=str, help='Show specific expiry only (e.g., "Dec.05.2025")')
    parser.add_argument('--export', action='store_true', help='Export to Excel')

    args = parser.parse_args()

    show_tables(
        args.ticker.upper(),
        max_expiries=args.expiries,
        max_strikes=args.strikes,
        specific_expiry=args.expiry,
        export=args.export
    )


if __name__ == '__main__':
    main()
