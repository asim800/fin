#!/usr/bin/env python3
"""
Plot monthly options tables (third Friday expiries).

Monthly options typically expire on the third Friday of each month.

Usage:
    python src/plot_monthly_options.py AAPL
    python src/plot_monthly_options.py AAPL --table call_price
    python src/plot_monthly_options.py AAPL --table put_elasticity
    python src/plot_monthly_options.py AAPL --around 10
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from options_analysis import AnalysisToolkit


def identify_monthly_expiries(expiries):
    """
    Identify monthly expiries (third Friday of the month).

    Monthly options typically expire on the third Friday.
    We'll identify them by looking for expiries that are likely monthly.

    Args:
        expiries: List of expiry date strings (e.g., "Jan.16.2026")

    Returns:
        List of monthly expiry strings
    """
    monthly = []

    for expiry in expiries:
        try:
            # Parse expiry date (format: "Dec.05.2025")
            parts = expiry.split('.')
            if len(parts) != 3:
                continue

            month_str, day_str, year_str = parts
            day = int(day_str)

            # Third Friday is typically between 15th-21st
            # We'll accept 15-21 as monthly expiries
            if 15 <= day <= 21:
                monthly.append(expiry)
        except:
            continue

    return monthly


def sort_expiries_by_date(expiries):
    """
    Sort expiry dates chronologically.

    Args:
        expiries: List of expiry date strings (e.g., "Jan.16.2026")

    Returns:
        List of expiries sorted by date (earliest first)
    """
    def parse_expiry(expiry_str):
        """Parse expiry string to datetime."""
        try:
            # Format: "Dec.05.2025"
            return datetime.strptime(expiry_str, "%b.%d.%Y")
        except:
            # Return far future date if parsing fails
            return datetime(2099, 12, 31)

    # Sort by parsed date
    sorted_expiries = sorted(expiries, key=parse_expiry)
    return sorted_expiries


def filter_strikes_around_atm(strikes, current_price, n_around=10, asymmetric=False,
                               option_type='put'):
    """
    Filter strikes around ATM.

    Args:
        strikes: List of strike prices
        current_price: Current stock price
        n_around: Number of strikes around ATM (symmetric) or directional (asymmetric)
        asymmetric: If True, filter based on option type
        option_type: 'put' or 'call' - only used if asymmetric=True

    Returns:
        List of filtered strikes

    When asymmetric=True:
        - For puts: Returns n_around strikes at or below current price, plus 1 above for reference
        - For calls: Returns n_around strikes at or above current price, plus 1 below for reference
    When asymmetric=False:
        - Returns n_around strikes on each side of ATM (symmetric)
    """
    strikes_array = np.array(strikes)

    # Find ATM strike
    atm_idx = np.argmin(np.abs(strikes_array - current_price))
    atm_strike = strikes_array[atm_idx]

    if asymmetric:
        # Asymmetric filtering based on option type
        if option_type.lower() == 'put':
            # For puts: get strikes at or below current price, plus at least 1 above
            below_strikes = strikes_array[strikes_array <= atm_strike]
            above_strikes = strikes_array[strikes_array > atm_strike]

            # Get n_around strikes below (including ATM)
            selected_below = below_strikes[-n_around:] if len(below_strikes) > n_around else below_strikes

            # Add at least 1 strike above current price for reference
            if len(above_strikes) > 0:
                selected_above = above_strikes[:1]  # Just the first one above
                selected = np.concatenate([selected_below, selected_above])
            else:
                selected = selected_below
        else:  # call
            # For calls: get strikes at or above current price, plus at least 1 below
            below_strikes = strikes_array[strikes_array < atm_strike]
            above_strikes = strikes_array[strikes_array >= atm_strike]

            # Get n_around strikes above (including ATM)
            selected_above = above_strikes[:n_around] if len(above_strikes) > n_around else above_strikes

            # Add at least 1 strike below current price for reference
            if len(below_strikes) > 0:
                selected_below = below_strikes[-1:]  # Just the last one below
                selected = np.concatenate([selected_below, selected_above])
            else:
                selected = selected_above

        return selected.tolist()
    else:
        # Symmetric filtering (original behavior)
        start_idx = max(0, atm_idx - n_around)
        end_idx = min(len(strikes), atm_idx + n_around + 1)
        return strikes[start_idx:end_idx]


def plot_option_table(ticker, table_name='put_price_table', n_around=10,
                     save=True, show=True, asymmetric=False):
    """
    Plot monthly options table as heatmap.

    Args:
        ticker: Stock ticker symbol
        table_name: Table to plot (put_price_table, call_price_table,
                    put_elasticity_table, call_elasticity_table, etc.)
        n_around: Number of strikes to show around ATM (or directionally if asymmetric)
        save: Whether to save the plot
        show: Whether to display the plot
        asymmetric: If True, filter strikes based on option type:
                   - Puts: show strikes at/below current price
                   - Calls: show strikes at/above current price
    """

    print(f"\n{'='*80}")
    print(f"Plotting {table_name} for {ticker} - Monthly Expiries")
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

    # Identify monthly expiries
    all_expiries = table.index.tolist()
    monthly_expiries = identify_monthly_expiries(all_expiries)

    if len(monthly_expiries) == 0:
        print("✗ No monthly expiries found")
        return

    # Sort monthly expiries by date
    monthly_expiries_sorted = sort_expiries_by_date(monthly_expiries)

    print(f"📅 Identified {len(monthly_expiries_sorted)} monthly expiries (3rd Friday):")
    for exp in monthly_expiries_sorted[:10]:  # Show first 10
        print(f"   • {exp}")
    if len(monthly_expiries_sorted) > 10:
        print(f"   ... and {len(monthly_expiries_sorted) - 10} more")
    print()

    # Filter table to monthly expiries (with sorted order)
    monthly_table = table.loc[monthly_expiries_sorted]

    # Determine option type from table name
    option_type = 'put' if 'put' in table_name.lower() else 'call'

    # Filter strikes around ATM (asymmetric if requested)
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

    # First, identify which strikes actually have data across expiries
    # Remove columns (strikes) with too many NaN values BEFORE subselecting
    threshold = 0.8
    valid_strikes = []
    for strike in monthly_table.columns:
        nan_ratio = monthly_table[strike].isna().sum() / len(monthly_table)
        if nan_ratio <= threshold:  # Keep if ≤80% NaN
            valid_strikes.append(strike)

    # Filter selected_strikes to only include strikes with sufficient data
    selected_strikes_with_data = [s for s in selected_strikes if s in valid_strikes]

    if not selected_strikes_with_data:
        print("✗ No strikes with sufficient data after filtering")
        return

    # Now get filtered data with only valid strikes
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

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Determine color scheme based on table type
    if 'elasticity' in table_name.lower():
        cmap = 'RdYlGn'
        fmt = '.2f'
        title_suffix = 'Elasticity'
    elif 'iv' in table_name.lower():
        cmap = 'YlOrRd'
        fmt = '.3f'
        title_suffix = 'Implied Volatility'
    elif 'price' in table_name.lower():
        cmap = 'Blues'
        fmt = '.2f'
        title_suffix = 'Prices'
    elif 'volume' in table_name.lower():
        cmap = 'Purples'
        fmt = '.0f'
        title_suffix = 'Volume'
    else:
        cmap = 'viridis'
        fmt = '.2f'
        title_suffix = ''

    # Create heatmap
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar_kws={'label': table_name.replace('_', ' ').title()},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    # Find ATM strike index for marking
    strike_values = plot_data.columns.tolist()
    atm_strike = min(strike_values, key=lambda x: abs(x - current_price))
    atm_idx = strike_values.index(atm_strike)

    # Draw vertical line at ATM
    ax.axvline(x=atm_idx + 0.5, color='red', linestyle='--', linewidth=2,
               label=f'ATM (${current_price:.2f})')

    # Formatting
    option_type = 'Put' if 'put' in table_name.lower() else 'Call'

    ax.set_title(
        f'{ticker} Monthly {option_type} Options - {title_suffix}\n'
        f'Current Price: ${current_price:.2f} | Monthly Expiries Only',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('Strike Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expiry Date', fontsize=12, fontweight='bold')

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Add legend
    ax.legend(loc='upper right')

    # Add current date
    today = datetime.now().strftime('%Y-%m-%d')
    fig.text(0.99, 0.01, f'Generated: {today}', ha='right', fontsize=8, style='italic')

    plt.tight_layout()

    # Save if requested
    if save:
        filename = f"{ticker}_monthly_{table_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {filename}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    # Now create line plot with the same data
    print(f"\n{'='*80}")
    print(f"Creating line plot with same data...")
    print(f"{'='*80}\n")

    fig_line, ax_line = plt.subplots(figsize=(12, 8))

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
            ax_line.plot(strikes, values,
                   marker='o', label=expiry, linewidth=2, markersize=6)

    # Mark current price with vertical line
    ax_line.axvline(x=current_price, color='red', linestyle='--',
              linewidth=2, label=f'Current Price (${current_price:.2f})')

    # Determine y-axis label
    if 'price' in table_name.lower():
        ylabel = 'Option Price ($)'
    elif 'elasticity' in table_name.lower():
        ylabel = 'Elasticity'
    elif 'iv' in table_name.lower():
        ylabel = 'Implied Volatility'
    elif 'volume' in table_name.lower():
        ylabel = 'Volume'
    elif 'oi' in table_name.lower():
        ylabel = 'Open Interest'
    else:
        ylabel = table_name.replace('_', ' ').title()

    ax_line.set_title(
        f'{ticker} Monthly {option_type} Options - {title_suffix}\n'
        f'Current Price: ${current_price:.2f} | Each Line = One Expiry',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    ax_line.set_xlabel('Strike Price ($)', fontsize=12, fontweight='bold')
    ax_line.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # Add grid for easier reading
    ax_line.grid(True, alpha=0.3, linestyle='--')

    # Legend
    ax_line.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
             fontsize=9, framealpha=0.9)

    # Add current date
    fig_line.text(0.99, 0.01, f'Generated: {today}', ha='right',
            fontsize=8, style='italic')

    plt.tight_layout()

    # Save line plot if requested
    if save:
        filename_line = f"{ticker}_monthly_{table_name}_lines.png"
        plt.savefig(filename_line, dpi=300, bbox_inches='tight')
        print(f"✓ Saved line plot to {filename_line}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

    print(f"\n{'='*80}\n")


def plot_multiple_tables(ticker, n_around=10, save=True, asymmetric=False):
    """Plot multiple tables for monthly options."""

    tables_to_plot = [
        'put_price_table',
        'call_price_table',
        'put_elasticity_table',
        'call_elasticity_table',
        'put_iv_table',
        'call_iv_table'
    ]

    print(f"\n{'='*80}")
    print(f"Plotting {len(tables_to_plot)} tables for {ticker}")
    print(f"{'='*80}\n")

    for table_name in tables_to_plot:
        try:
            plot_option_table(
                ticker,
                table_name=table_name,
                n_around=n_around,
                save=save,
                show=False,  # Don't show interactively when plotting multiple
                asymmetric=asymmetric
            )
        except Exception as e:
            print(f"✗ Error plotting {table_name}: {e}\n")

    print(f"{'='*80}")
    print(f"✓ Completed plotting {len(tables_to_plot)} tables")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot monthly options tables (third Friday expiries)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic symmetric plotting (default)
  python src/plot_monthly_options.py AAPL
  python src/plot_monthly_options.py AAPL --table put_price_table
  python src/plot_monthly_options.py AAPL --table call_elasticity_table
  python src/plot_monthly_options.py AAPL --around 15

  # Asymmetric plotting (puts: at/below, calls: at/above current price)
  python src/plot_monthly_options.py AAPL --asymmetric --around 15
  python src/plot_monthly_options.py AAPL --table call_price_table --asymmetric
  python src/plot_monthly_options.py AAPL --all --asymmetric
        """
    )

    parser.add_argument('ticker', help='Stock ticker symbol')
    parser.add_argument('--table', type=str, default='put_price_table',
                       help='Table to plot (default: put_price_table)')
    parser.add_argument('--around', type=int, default=10,
                       help='Number of strikes around ATM (default: 10)')
    parser.add_argument('--asymmetric', action='store_true',
                       help='Use asymmetric strike filtering: puts show strikes at/below '
                            'current price, calls show strikes at/above current price')
    parser.add_argument('--all', action='store_true',
                       help='Plot all 6 main tables')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plot to file')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot')

    args = parser.parse_args()

    if args.all:
        # Plot all main tables
        plot_multiple_tables(
            args.ticker.upper(),
            n_around=args.around,
            save=not args.no_save,
            asymmetric=args.asymmetric
        )
    else:
        # Plot single table
        plot_option_table(
            args.ticker.upper(),
            table_name=args.table,
            n_around=args.around,
            save=not args.no_save,
            show=not args.no_show,
            asymmetric=args.asymmetric
        )


if __name__ == '__main__':
    main()
