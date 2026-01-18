#!/usr/bin/env python3
"""
View option tables centered around current stock price (ATM).

Usage:
    python src/view_tables.py AAPL                    # Default: ±5 strikes around ATM
    python src/view_tables.py AAPL --around 10        # Show 10 strikes above and below ATM
    python src/view_tables.py AAPL --around 3         # Show 3 strikes above and below ATM
    python src/view_tables.py AAPL --expiry "Dec.05.2025" --around 8
    python src/view_tables.py AAPL --export
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis import AnalysisToolkit


def find_atm_strike(strikes, current_price):
    """Find strike price closest to current stock price."""
    strikes = np.array(strikes)
    idx = np.argmin(np.abs(strikes - current_price))
    return strikes[idx]


def get_strikes_around_atm(table, current_price, n_around=5):
    """Get N strikes above and below ATM strike.

    Args:
        table: DataFrame with strikes as columns
        current_price: Current stock price
        n_around: Number of strikes to show above and below ATM

    Returns:
        List of selected strike prices
    """
    # Get all available strikes
    all_strikes = table.columns.tolist()

    if len(all_strikes) == 0:
        return []

    # Find ATM (strike closest to current price)
    atm_strike = find_atm_strike(all_strikes, current_price)
    atm_idx = all_strikes.index(atm_strike)

    # Get strikes around ATM
    start_idx = max(0, atm_idx - n_around)
    end_idx = min(len(all_strikes), atm_idx + n_around + 1)

    selected_strikes = all_strikes[start_idx:end_idx]

    return selected_strikes


def show_table_around_atm(table, table_name, current_price, n_around, max_expiries=None):
    """Show table with strikes centered around ATM."""

    # Get strikes around ATM
    selected_strikes = get_strikes_around_atm(table, current_price, n_around)

    if len(selected_strikes) == 0:
        print(f"No strikes available for {table_name}")
        return

    # Get ATM strike for marking
    atm_strike = find_atm_strike(selected_strikes, current_price)

    # Filter table to selected strikes
    filtered_table = table[selected_strikes]

    # Remove rows that are all NaN
    filtered_table = filtered_table.dropna(how='all', axis=0)

    # Limit expiries if requested
    if max_expiries and len(filtered_table) > max_expiries:
        filtered_table = filtered_table.iloc[:max_expiries]

    if filtered_table.empty:
        print(f"No data available for {table_name}")
        return

    print(f"\n{'='*90}")
    print(f"{table_name.replace('_', ' ').upper()}")
    print(f"Current Price: ${current_price:.2f} | ATM Strike: ${atm_strike:.2f}")
    print(f"Showing {n_around} strikes below and {n_around} above ATM ({len(selected_strikes)} total)")
    print(f"{'='*90}")

    # Create column headers with ATM marker
    col_headers = []
    for strike in selected_strikes:
        if strike == atm_strike:
            col_headers.append(f"{strike:.1f}*")  # Mark ATM with *
        else:
            col_headers.append(f"{strike:.1f}")

    # Create a copy with renamed columns for display
    display_table = filtered_table.copy()
    display_table.columns = col_headers

    print(display_table.to_string())
    print(f"\n(* = ATM strike closest to current price)")
    print(f"Showing {len(filtered_table)} expiries × {len(selected_strikes)} strikes")
    print(f"Strike range: ${selected_strikes[0]:.0f} to ${selected_strikes[-1]:.0f}")


def show_specific_expiry_around_atm(tables, expiry, current_price, n_around):
    """Show specific expiry with strikes around ATM."""

    print(f"\n{'='*90}")
    print(f"EXPIRY: {expiry}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Showing {n_around} strikes below and {n_around} above ATM")
    print(f"{'='*90}")

    for table_name in ['call_price_table', 'call_elasticity_table',
                       'put_price_table', 'put_elasticity_table',
                       'call_iv_table', 'put_iv_table']:

        if table_name not in tables:
            continue

        table = tables[table_name]

        if expiry not in table.index:
            print(f"\n✗ Expiry '{expiry}' not found in {table_name}")
            continue

        # Get row for this expiry
        row = table.loc[expiry]

        # Get all strikes (columns)
        all_strikes = row.index.tolist()

        # Find strikes around ATM
        atm_strike = find_atm_strike(all_strikes, current_price)
        atm_idx = all_strikes.index(atm_strike)

        start_idx = max(0, atm_idx - n_around)
        end_idx = min(len(all_strikes), atm_idx + n_around + 1)

        selected_strikes = all_strikes[start_idx:end_idx]
        selected_data = row[selected_strikes]

        # Remove NaN
        selected_data = selected_data.dropna()

        if len(selected_data) == 0:
            continue

        print(f"\n{table_name.replace('_', ' ').upper()}")
        print(f"{'-'*90}")

        # Format as table
        df = pd.DataFrame({
            'Strike': selected_data.index,
            'Value': selected_data.values
        })

        # Mark ATM
        df[''] = df['Strike'].apply(lambda x: '← ATM' if x == atm_strike else '')

        print(df.to_string(index=False))
        print(f"({len(selected_data)} strikes)")


def view_tables(ticker, n_around=5, max_expiries=5, specific_expiry=None, export=False):
    """View tables centered around ATM strike."""

    print(f"\n{'='*90}")
    print(f"Option Tables for {ticker} - Centered Around ATM")
    print(f"{'='*90}\n")

    toolkit = AnalysisToolkit()

    # Get current price
    print(f"🔍 Fetching quote for {ticker}...")
    quote = toolkit.get_quote(ticker)

    if not quote:
        print(f"✗ Failed to get quote for {ticker}")
        return

    current_price = quote['price']
    print(f"✓ Current Price: ${current_price:.2f}\n")

    # Get tables
    print(f"📊 Fetching option tables...")
    tables = toolkit.get_option_tables(ticker)

    if not tables:
        print(f"✗ No tables found for {ticker}")
        return

    print(f"✓ Retrieved {len(tables)} tables\n")

    if specific_expiry:
        # Show specific expiry
        show_specific_expiry_around_atm(tables, specific_expiry, current_price, n_around)
    else:
        # Show multiple expiries
        # Call Prices
        if 'call_price_table' in tables:
            show_table_around_atm(
                tables['call_price_table'],
                'call_price_table',
                current_price,
                n_around,
                max_expiries
            )

        # Put Prices
        if 'put_price_table' in tables:
            show_table_around_atm(
                tables['put_price_table'],
                'put_price_table',
                current_price,
                n_around,
                max_expiries
            )

        # Call Elasticity
        if 'call_elasticity_table' in tables:
            show_table_around_atm(
                tables['call_elasticity_table'],
                'call_elasticity_table',
                current_price,
                n_around,
                max_expiries
            )

        # Put Elasticity
        if 'put_elasticity_table' in tables:
            show_table_around_atm(
                tables['put_elasticity_table'],
                'put_elasticity_table',
                current_price,
                n_around,
                max_expiries
            )

        # Call IV
        if 'call_iv_table' in tables:
            show_table_around_atm(
                tables['call_iv_table'],
                'call_iv_table',
                current_price,
                n_around,
                max_expiries
            )

        # Put IV
        if 'put_iv_table' in tables:
            show_table_around_atm(
                tables['put_iv_table'],
                'put_iv_table',
                current_price,
                n_around,
                max_expiries
            )

        # Show available expiries
        if 'call_price_table' in tables:
            all_expiries = tables['call_price_table'].index.tolist()
            print(f"\n{'='*90}")
            print(f"ALL AVAILABLE EXPIRIES ({len(all_expiries)} total)")
            print(f"{'='*90}")
            for i, exp in enumerate(all_expiries, 1):
                print(f"{i:2d}. {exp}")

    # Export if requested
    if export:
        print(f"\n{'='*90}")
        print("EXPORTING TO EXCEL")
        print(f"{'='*90}")

        filename = f"{ticker}_option_tables.xlsx"
        success = toolkit.export_to_excel(ticker, filename)

        if success:
            print(f"✓ Exported all tables to {filename}")
            print(f"  All 16 tables with full data (no ATM filtering)")
        else:
            print(f"✗ Export failed")

    print(f"\n{'='*90}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='View option tables centered around ATM (current stock price)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/view_tables.py AAPL                    # Show ±5 strikes around ATM
  python src/view_tables.py AAPL --around 10        # Show ±10 strikes around ATM
  python src/view_tables.py AAPL --around 3 --expiries 3
  python src/view_tables.py AAPL --expiry "Dec.05.2025" --around 8
  python src/view_tables.py NVDA --around 7 --export
        """
    )

    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--around', type=int, default=5,
                       help='Number of strikes to show above and below ATM (default: 5)')
    parser.add_argument('--expiries', type=int, default=5,
                       help='Number of expiries to show (default: 5)')
    parser.add_argument('--expiry', type=str,
                       help='Show specific expiry only (e.g., "Dec.05.2025")')
    parser.add_argument('--export', action='store_true',
                       help='Export full tables to Excel')

    args = parser.parse_args()

    view_tables(
        args.ticker.upper(),
        n_around=args.around,
        max_expiries=args.expiries,
        specific_expiry=args.expiry,
        export=args.export
    )


if __name__ == '__main__':
    main()
