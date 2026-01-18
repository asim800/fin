"""
Demo script showing how to use option elasticity and pivot tables features.

This script demonstrates:
1. How elasticity is automatically calculated for all options
2. How to access elasticity values from processed chains
3. How to create and use pivot tables organized by expiry and strike
4. How to analyze elasticity across different strikes and expiries
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from options_analysis.orchestrator import OptionsAnalysisOrchestrator
from options_analysis.config import Config


def main():
    """Demonstrate elasticity and pivot table functionality."""

    print("=" * 80)
    print("OPTION ELASTICITY AND PIVOT TABLES DEMONSTRATION")
    print("=" * 80)

    # Initialize orchestrator
    config = Config()
    orchestrator = OptionsAnalysisOrchestrator(config)

    # Fetch market data
    print("\n1. Fetching market data...")
    print("-" * 40)

    tickers = ['AAPL', 'MSFT', 'NVDA']  # Small subset for demo
    market_data = orchestrator.fetch_market_data(ticker_list=tickers)

    if not market_data.is_valid():
        print("❌ Failed to fetch valid market data")
        return

    print(f"✓ Fetched data for {len(market_data.tickers)} tickers")
    print(f"✓ Prices: {market_data.prices}")

    # Process chains with elasticity
    print("\n2. Processing option chains with elasticity calculation...")
    print("-" * 40)

    processed_chains = orchestrator.processor.extract_contract_identifiers(
        market_data.option_chains,
        prices=market_data.prices,
        current_time=market_data.timestamp
    )

    print(f"✓ Processed {len(processed_chains)} tickers with elasticity")

    # Display elasticity for first ticker
    if tickers and tickers[0] in processed_chains:
        ticker = tickers[0]
        print(f"\n3. Elasticity values for {ticker}:")
        print("-" * 40)

        ticker_data = processed_chains[ticker]
        expiry_count = 0

        for expiry, data in ticker_data.items():
            expiry_count += 1
            if expiry_count > 2:  # Show only first 2 expiries
                break

            print(f"\nExpiry: {expiry}")

            # Show call elasticity
            call_elasticity = data.get('call_elasticity')
            if call_elasticity is not None and not call_elasticity.empty:
                calls = data.get('calls')
                print("\nCall Options (showing first 5):")
                print(f"{'Strike':<10} {'Price':<10} {'Elasticity':<12}")
                print("-" * 35)

                count = 0
                for idx in call_elasticity.index[:5]:
                    if idx in calls.index:
                        strike = calls.loc[idx, 'Strike']
                        price = calls.loc[idx, 'Last']
                        elast = call_elasticity.loc[idx, 'call_elasticity']
                        print(f"{strike:<10.2f} ${price:<9.2f} {elast:<12.2f}")
                        count += 1
                        if count >= 5:
                            break

            # Show put elasticity
            put_elasticity = data.get('put_elasticity')
            if put_elasticity is not None and not put_elasticity.empty:
                puts = data.get('puts')
                print("\nPut Options (showing first 5):")
                print(f"{'Strike':<10} {'Price':<10} {'Elasticity':<12}")
                print("-" * 35)

                count = 0
                for idx in put_elasticity.index[:5]:
                    if idx in puts.index:
                        strike = puts.loc[idx, 'Strike']
                        price = puts.loc[idx, 'Last']
                        elast = put_elasticity.loc[idx, 'put_elasticity']
                        print(f"{strike:<10.2f} ${price:<9.2f} {elast:<12.2f}")
                        count += 1
                        if count >= 5:
                            break

    # Create pivot tables
    print("\n4. Creating pivot tables (expiry x strike)...")
    print("-" * 40)

    option_tables = orchestrator.processor.create_option_tables(processed_chains)

    print(f"✓ Created pivot tables for {len(option_tables)} tickers")

    # Display sample pivot table
    if tickers and tickers[0] in option_tables:
        ticker = tickers[0]
        tables = option_tables[ticker]

        print(f"\n5. Sample pivot tables for {ticker}:")
        print("-" * 40)

        print("\nAvailable tables:")
        for table_name in tables.keys():
            print(f"  - {table_name}")

        # Show call price table (truncated)
        if 'call_price_table' in tables:
            call_price_table = tables['call_price_table']
            print(f"\nCall Price Table (first 3 expiries, first 5 strikes):")
            print(call_price_table.iloc[:3, :5].to_string())

        # Show call elasticity table (truncated)
        if 'call_elasticity_table' in tables:
            call_elast_table = tables['call_elasticity_table']
            print(f"\nCall Elasticity Table (first 3 expiries, first 5 strikes):")
            print(call_elast_table.iloc[:3, :5].to_string())

        # Show put price table (truncated)
        if 'put_price_table' in tables:
            put_price_table = tables['put_price_table']
            print(f"\nPut Price Table (first 3 expiries, first 5 strikes):")
            print(put_price_table.iloc[:3, :5].to_string())

        # Show put elasticity table (truncated)
        if 'put_elasticity_table' in tables:
            put_elast_table = tables['put_elasticity_table']
            print(f"\nPut Elasticity Table (first 3 expiries, first 5 strikes):")
            print(put_elast_table.iloc[:3, :5].to_string())

    # Demonstrate using the full orchestrator
    print("\n6. Running full individual analysis (with elasticity & tables)...")
    print("-" * 40)

    try:
        individual_results = orchestrator.run_individual_analysis(market_data)

        print(f"✓ Analysis completed for {len(individual_results.get_all_tickers())} tickers")
        print(f"✓ Option tables available for {len(individual_results.option_tables)} tickers")

        # Access tables from results
        if tickers and tickers[0] in individual_results.option_tables:
            ticker = tickers[0]
            ticker_tables = individual_results.get_ticker_tables(ticker)

            if ticker_tables:
                print(f"\nAccessed {len(ticker_tables)} tables for {ticker} from results")

                # Example: Find strikes with highest elasticity
                if 'call_elasticity_table' in ticker_tables:
                    elast_table = ticker_tables['call_elasticity_table']

                    # Get mean elasticity across expiries for each strike
                    mean_elast_by_strike = elast_table.mean(axis=0).dropna()

                    if not mean_elast_by_strike.empty:
                        top_strikes = mean_elast_by_strike.nlargest(5)
                        print(f"\nTop 5 strikes by average call elasticity for {ticker}:")
                        for strike, elast in top_strikes.items():
                            print(f"  Strike ${strike:.2f}: Avg Elasticity = {elast:.2f}")

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\nKey Takeaways:")
        print("1. Elasticity = (underlying_price * option_delta) / option_price")
        print("2. Higher elasticity = more leverage per dollar invested")
        print("3. Pivot tables organize options by expiry (rows) and strike (columns)")
        print("4. Tables available for: price, bid, ask, IV, volume, OI, elasticity")
        print("5. Access via individual_results.option_tables[ticker][table_name]")

    except Exception as e:
        print(f"❌ Error in full analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
