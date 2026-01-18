#!/usr/bin/env python3
"""
Plot monthly options as line charts (each expiry = one line).

This script creates line plots instead of heatmaps, making it easier to
compare pricing across strikes for different expiration dates.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis import AnalysisToolkit
from plot_monthly_options import (
    identify_monthly_expiries,
    sort_expiries_by_date,
    filter_strikes_around_atm
)


def plot_option_lines(ticker, table_name='put_price_table', n_around=10,
                      asymmetric=False, save=True, show=True, print_data=True):
    """
    Plot monthly options table as line chart.

    Args:
        ticker: Stock ticker symbol
        table_name: Table to plot
        n_around: Number of strikes around ATM
        asymmetric: Use asymmetric strike filtering
        save: Whether to save the plot
        show: Whether to display the plot
        print_data: Whether to print the data to stdout

    Each expiry date is plotted as a separate line, making it easy to
    compare option prices across strikes.
    """

    print(f"\n{'='*80}")
    print(f"Line Plot: {table_name} for {ticker} - Monthly Expiries")
    print(f"{'='*80}\n")

    # Initialize toolkit
    toolkit = AnalysisToolkit()

    # Get quote
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

    if not tables or table_name not in tables:
        print(f"✗ Table '{table_name}' not found")
        return

    table = tables[table_name]
    print(f"✓ Retrieved table: {table.shape[0]} expiries × {table.shape[1]} strikes\n")

    # Identify and sort monthly expiries
    all_expiries = table.index.tolist()
    monthly_expiries = identify_monthly_expiries(all_expiries)

    if len(monthly_expiries) == 0:
        print("✗ No monthly expiries found")
        return

    monthly_expiries_sorted = sort_expiries_by_date(monthly_expiries)

    print(f"📅 Identified {len(monthly_expiries_sorted)} monthly expiries (3rd Friday):")
    for exp in monthly_expiries_sorted[:10]:
        print(f"   • {exp}")
    if len(monthly_expiries_sorted) > 10:
        print(f"   ... and {len(monthly_expiries_sorted) - 10} more")
    print()

    # Filter table to monthly expiries
    monthly_table = table.loc[monthly_expiries_sorted]

    # Determine option type
    option_type = 'put' if 'put' in table_name.lower() else 'call'

    # Filter strikes
    all_strikes = monthly_table.columns.tolist()
    selected_strikes = filter_strikes_around_atm(
        all_strikes,
        current_price,
        n_around,
        asymmetric=asymmetric,
        option_type=option_type
    )

    # Print filtering info
    if asymmetric:
        direction = "at/below" if option_type == 'put' else "at/above"
        reference = "+1 above" if option_type == 'put' else "+1 below"
        print(f"📍 Using asymmetric filtering for {option_type}s ({direction} current, {reference} for reference)")
    else:
        print(f"📍 Using symmetric filtering (±{n_around} around ATM)")

    # Filter sparse columns before subselecting
    threshold = 0.8
    valid_strikes = []
    for strike in monthly_table.columns:
        nan_ratio = monthly_table[strike].isna().sum() / len(monthly_table)
        if nan_ratio <= threshold:
            valid_strikes.append(strike)

    selected_strikes_with_data = [s for s in selected_strikes if s in valid_strikes]

    if not selected_strikes_with_data:
        print("✗ No strikes with sufficient data after filtering")
        return

    # Get filtered data
    plot_data = monthly_table[selected_strikes_with_data]

    # Remove rows that are all NaN
    plot_data = plot_data.dropna(how='all', axis=0)

    # Final cleanup: remove any remaining all-NaN columns
    plot_data = plot_data.dropna(how='all', axis=1)

    if plot_data.empty:
        print("✗ No data to plot after filtering")
        return

    strikes_removed = len(selected_strikes) - len(plot_data.columns)
    print(f"📊 Plotting {plot_data.shape[0]} expiries × {plot_data.shape[1]} strikes")
    print(f"   Strike range: ${plot_data.columns[0]:.0f} to ${plot_data.columns[-1]:.0f}")
    if strikes_removed > 0:
        print(f"   (Removed {strikes_removed} strikes with >80% missing data)")
    print()

    # Print data to stdout if requested
    if print_data:
        print("=" * 80)
        print("DATA TABLE")
        print("=" * 80)
        print()
        print(plot_data.to_string())
        print()
        print("=" * 80)
        print()

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each expiry as a line
    for expiry in plot_data.index:
        # Get data for this expiry
        expiry_data = plot_data.loc[expiry]

        # Remove NaN values for cleaner lines
        valid_data = expiry_data.dropna()

        if len(valid_data) > 0:
            # Convert to numpy arrays for matplotlib compatibility
            strikes = np.array(valid_data.index.tolist())
            values = np.array(valid_data.values)
            ax.plot(strikes, values,
                   marker='o', label=expiry, linewidth=2, markersize=6)

    # Mark current price with vertical line
    ax.axvline(x=current_price, color='red', linestyle='--',
              linewidth=2, label=f'Current Price (${current_price:.2f})')

    # Formatting
    option_type_label = 'Put' if 'put' in table_name.lower() else 'Call'

    if 'price' in table_name.lower():
        ylabel = 'Option Price ($)'
        title_suffix = 'Prices'
    elif 'elasticity' in table_name.lower():
        ylabel = 'Elasticity'
        title_suffix = 'Elasticity'
    elif 'iv' in table_name.lower():
        ylabel = 'Implied Volatility'
        title_suffix = 'Implied Volatility'
    elif 'volume' in table_name.lower():
        ylabel = 'Volume'
        title_suffix = 'Volume'
    elif 'oi' in table_name.lower():
        ylabel = 'Open Interest'
        title_suffix = 'Open Interest'
    else:
        ylabel = table_name.replace('_', ' ').title()
        title_suffix = ''

    ax.set_title(
        f'{ticker} Monthly {option_type_label} Options - {title_suffix}\n'
        f'Current Price: ${current_price:.2f} | Each Line = One Expiry',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Strike Price ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # Add grid for easier reading
    ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
             fontsize=9, framealpha=0.9)

    # Add current date
    today = datetime.now().strftime('%Y-%m-%d')
    fig.text(0.99, 0.01, f'Generated: {today}', ha='right',
            fontsize=8, style='italic')

    plt.tight_layout()

    # Save if requested
    if save:
        filename = f"{ticker}_monthly_{table_name}_lines.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {filename}\n")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot monthly options as line charts (each expiry = one line)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic line plot for put prices
  python src/plot_monthly_lines.py AAPL

  # Call prices with asymmetric filtering
  python src/plot_monthly_lines.py AAPL --table call_price_table --asymmetric

  # Put elasticity with more strikes
  python src/plot_monthly_lines.py AAPL --table put_elasticity_table --around 15

  # Don't print data table
  python src/plot_monthly_lines.py AAPL --no-print
        """
    )

    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--table', type=str, default='put_price_table',
                       help='Table to plot (default: put_price_table)')
    parser.add_argument('--around', type=int, default=10,
                       help='Number of strikes around ATM (default: 10)')
    parser.add_argument('--asymmetric', action='store_true',
                       help='Use asymmetric strike filtering')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plot to file')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')
    parser.add_argument('--no-print', action='store_true',
                       help='Do not print data table to stdout')

    args = parser.parse_args()

    plot_option_lines(
        args.ticker.upper(),
        table_name=args.table,
        n_around=args.around,
        asymmetric=args.asymmetric,
        save=not args.no_save,
        show=not args.no_show,
        print_data=not args.no_print
    )


if __name__ == '__main__':
    main()
