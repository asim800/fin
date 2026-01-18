#!/usr/bin/env python3
"""
Demonstration of the AnalysisToolkit high-level API.

This shows how to use the simplified toolkit interface for common tasks.
"""

from options_analysis import AnalysisToolkit


def main():
    """Demonstrate toolkit usage."""

    # Initialize toolkit
    print("🚀 Initializing Options Analysis Toolkit...\n")
    toolkit = AnalysisToolkit()

    # Example 1: Get current quote
    print("=" * 60)
    print("Example 1: Get Current Quote")
    print("=" * 60)
    quote = toolkit.get_quote('AAPL')
    if quote:
        print(f"Ticker: {quote['ticker']}")
        print(f"Price: ${quote['price']:.2f}")
        print(f"Timestamp: {quote['timestamp']}")
    print()

    # Example 2: Get call elasticity
    print("=" * 60)
    print("Example 2: Get Call Elasticity (Top 5)")
    print("=" * 60)
    elasticity = toolkit.get_elasticity('AAPL', option_type='call', top_n=5)
    if not elasticity.empty:
        print(elasticity.to_string())
    else:
        print("No elasticity data available")
    print()

    # Example 3: Find best calls within budget
    print("=" * 60)
    print("Example 3: Find Best Calls (Budget: $500, Min Elasticity: 2.5)")
    print("=" * 60)
    best_calls = toolkit.find_best_calls(
        tickers=['AAPL', 'MSFT'],
        budget=500,
        min_elasticity=2.5,
        max_results=5
    )
    if not best_calls.empty:
        print(best_calls.to_string())
    else:
        print("No calls found matching criteria")
    print()

    # Example 4: Find best puts
    print("=" * 60)
    print("Example 4: Find Best Puts (Budget: $300, Min Elasticity: 3.0)")
    print("=" * 60)
    best_puts = toolkit.find_best_puts(
        tickers=['AAPL', 'MSFT'],
        budget=300,
        min_elasticity=3.0,
        max_results=5
    )
    if not best_puts.empty:
        print(best_puts.to_string())
    else:
        print("No puts found matching criteria")
    print()

    # Example 5: Find arbitrage opportunities
    print("=" * 60)
    print("Example 5: Find Arbitrage Opportunities (Min Profit: 5%)")
    print("=" * 60)
    arbitrage = toolkit.find_arbitrage(
        tickers=['AAPL', 'MSFT', 'NVDA'],
        min_profit=0.05,
        max_results=5
    )
    if not arbitrage.empty:
        print(arbitrage.to_string())
    else:
        print("No arbitrage opportunities found")
    print()

    # Example 6: Export to Excel
    print("=" * 60)
    print("Example 6: Export Data to Excel")
    print("=" * 60)
    success = toolkit.export_to_excel('AAPL', 'aapl_analysis.xlsx')
    if success:
        print("✓ Successfully exported AAPL data to aapl_analysis.xlsx")
    else:
        print("✗ Failed to export data")
    print()

    # Example 7: Export to CSV
    print("=" * 60)
    print("Example 7: Export Data to CSV")
    print("=" * 60)
    files = toolkit.export_to_csv('AAPL', 'aapl_exports')
    if files:
        print(f"✓ Created {len(files)} CSV files:")
        for f in files[:3]:  # Show first 3
            print(f"  - {f}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")
    else:
        print("✗ Failed to export data")
    print()

    # Example 8: Get option tables
    print("=" * 60)
    print("Example 8: Get Option Pivot Tables")
    print("=" * 60)
    tables = toolkit.get_option_tables('AAPL')
    if tables:
        print(f"✓ Retrieved {len(tables)} tables:")
        for table_name in list(tables.keys())[:5]:
            print(f"  - {table_name}")
        if len(tables) > 5:
            print(f"  ... and {len(tables) - 5} more")

        # Show sample of call elasticity table
        if 'call_elasticity_table' in tables:
            print("\nSample of call_elasticity_table (first 3 expiries × first 5 strikes):")
            sample = tables['call_elasticity_table'].iloc[:3, :5]
            print(sample.to_string())
    else:
        print("✗ Failed to retrieve tables")
    print()

    print("=" * 60)
    print("✓ Toolkit Demo Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
