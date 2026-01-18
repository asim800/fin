#!/usr/bin/env python3
"""
Get option pivot tables (expiries × strikes) for a ticker.

Usage:
    python src/get_tables.py AAPL
    python src/get_tables.py NVDA --export
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis import AnalysisToolkit


def show_tables(ticker, export=False):
    """Get and display option tables."""

    print(f"\n{'='*70}")
    print(f"Option Tables for {ticker}")
    print(f"{'='*70}\n")

    toolkit = AnalysisToolkit()

    # Get all tables
    print(f"🔍 Fetching option data for {ticker}...")
    tables = toolkit.get_option_tables(ticker)

    if not tables:
        print(f"✗ No tables found for {ticker}")
        return

    print(f"✓ Retrieved {len(tables)} tables\n")

    # Show available tables
    print("Available tables:")
    for i, table_name in enumerate(tables.keys(), 1):
        rows, cols = tables[table_name].shape
        print(f"  {i:2d}. {table_name:30s} ({rows} expiries × {cols} strikes)")

    # Helper function to show table with actual data
    def show_table_sample(table, title):
        """Show table sample with strikes that have data."""
        print(f"\n{'='*70}")
        print(title)
        print(f"{'='*70}")

        # Get first 5 rows
        sample_rows = table.iloc[:5]

        # Find columns that have data in these specific rows
        cols_with_data_in_sample = sample_rows.columns[sample_rows.notna().any()]

        if len(cols_with_data_in_sample) > 0:
            # Show first 8 strikes that actually have data in these rows
            sample = sample_rows.loc[:, cols_with_data_in_sample[:8]]
            print(sample.to_string())

            # Overall stats
            total_strikes = table.notna().sum(axis=0)
            total_with_data = (total_strikes > 0).sum()
            print(f"\n(Showing {len(sample.columns)} of {len(cols_with_data_in_sample)} strikes with data in these expiries)")
            print(f"Total strikes with data across all expiries: {total_with_data}")
            if len(cols_with_data_in_sample) > 0:
                print(f"Strike range shown: {cols_with_data_in_sample[0]:.0f} to {cols_with_data_in_sample[min(7, len(cols_with_data_in_sample)-1)]:.0f}")
        else:
            print("No data in first 5 expiries")

    # Show samples of main tables
    if 'call_price_table' in tables:
        show_table_sample(tables['call_price_table'],
                         "CALL PRICE TABLE (first 5 expiries × 8 strikes)")

    if 'put_price_table' in tables:
        show_table_sample(tables['put_price_table'],
                         "PUT PRICE TABLE (first 5 expiries × 8 strikes)")

    if 'call_elasticity_table' in tables:
        show_table_sample(tables['call_elasticity_table'],
                         "CALL ELASTICITY TABLE (first 5 expiries × 8 strikes)")

    if 'put_elasticity_table' in tables:
        show_table_sample(tables['put_elasticity_table'],
                         "PUT ELASTICITY TABLE (first 5 expiries × 8 strikes)")

    # Export if requested
    if export:
        print(f"\n{'='*70}")
        print("EXPORTING TO EXCEL")
        print(f"{'='*70}")

        filename = f"{ticker}_option_tables.xlsx"
        success = toolkit.export_to_excel(ticker, filename)

        if success:
            print(f"✓ Exported all tables to {filename}")
            print(f"  {len(tables)} sheets created")
        else:
            print(f"✗ Export failed")

    print(f"\n{'='*70}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python src/get_tables.py AAPL")
        print("  python src/get_tables.py NVDA --export")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    export = '--export' in sys.argv

    show_tables(ticker, export)


if __name__ == '__main__':
    main()
    
    
'''
# Get tables for AAPL
python3 src/get_tables.py AAPL

# Get tables and export to Excel
python3 src/get_tables.py AAPL --export

'''
