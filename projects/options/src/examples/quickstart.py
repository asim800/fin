#!/usr/bin/env python3
"""
Quick Start Example for Options Analysis System

This is the simplest way to get started with the options analysis system.
Run this to see the basic functionality in action.
"""

import sys
from pathlib import Path

# Add src directory to Python path (go up from examples/ to src/)
sys.path.insert(0, str(Path(__file__).parent.parent))

def quick_demo():
    """Quick demonstration of the options analysis system."""
    
    print("🚀 QUICK START - Options Analysis System")
    print("="*50)
    
    try:
        # Import the main orchestrator
        from options_analysis.orchestrator import OptionsAnalysisOrchestrator
        
        print("✓ Successfully imported options analysis modules")
        
        # Initialize the system
        orchestrator = OptionsAnalysisOrchestrator()
        print("✓ Initialized options analysis orchestrator")
        
        # Test with a single ticker (faster for demo)
        print("\n📊 Running analysis for AAPL...")
        results = orchestrator.run_ticker_analysis('AAPL')
        
        if results:
            quote = results['quote']
            option_chain = results['option_chain']
            
            print(f"✅ Analysis complete for AAPL!")
            print(f"   Current Price: ${quote['price']:.2f}")
            print(f"   Volume: {quote['volume']:,}")
            print(f"   Option Expiries: {len(option_chain)}")
            
            # Show available expiry dates
            print(f"\n📅 Available Option Expiries:")
            for expiry_date in list(option_chain.keys())[:5]:  # Show first 5
                print(f"   • {expiry_date}")
                
            if len(option_chain) > 5:
                print(f"   ... and {len(option_chain) - 5} more")
            
            print(f"\n📁 Plots saved to: plots/ folder")
            
        else:
            print("❌ Could not retrieve data for AAPL")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be due to market hours or network connectivity")

def full_demo():
    """Full demonstration with multiple tickers."""
    
    print("\n" + "🚀 FULL DEMO - Complete Analysis Pipeline")
    print("="*50)
    
    try:
        from options_analysis.orchestrator import OptionsAnalysisOrchestrator
        
        # Initialize
        orchestrator = OptionsAnalysisOrchestrator()
        
        print("Running full analysis (this may take a few minutes)...")
        print("Analyzing multiple tickers and generating comprehensive reports...")
        
        # Run full analysis with default tickers
        results = orchestrator.run_full_analysis()
        
        if results:
            summary = results.get('summary', {})
            
            print(f"\n✅ FULL ANALYSIS COMPLETE!")
            print(f"   📊 Tickers analyzed: {summary.get('total_tickers', 0)}")
            print(f"   🔗 Option pairs: {summary.get('total_pairs_analyzed', 0):,}")
            print(f"   ⚠️  Parity violations: {summary.get('total_violations_1pct', 0)}")
            print(f"   📈 Violation rate: {summary.get('violation_rate', 0):.2%}")
            
            # Show arbitrage opportunities
            arbitrage = results.get('arbitrage')
            if arbitrage is not None and not arbitrage.empty:
                print(f"\n💰 Top Arbitrage Opportunities:")
                top_arb = arbitrage.head(3)
                for i, (_, row) in enumerate(top_arb.iterrows(), 1):
                    print(f"   {i}. {row['ticker']} ${row['strike']}: ${row['expected_profit']:.2f} profit")
            
            print(f"\n📁 All results saved to:")
            print(f"   • data/ - Analysis data files")
            print(f"   • plots/ - Visualization files")
            
        else:
            print("❌ Full analysis failed")
            
    except Exception as e:
        print(f"❌ Full demo error: {e}")

def advanced_features_demo():
    """Demonstrate advanced features from op01.r and op05.r."""
    
    print("\n" + "🚀 ADVANCED FEATURES DEMO")
    print("="*50)
    
    try:
        from options_analysis.option_processor import OptionChainProcessor
        from options_analysis.data_fetcher import DataFetcher
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data for demonstration
        print("Creating sample data for advanced features demo...")
        
        # Initialize components
        processor = OptionChainProcessor()  # Uses config's data folder
        data_fetcher = DataFetcher()
        
        # Try to get real data for one ticker
        print("Fetching real data for SPY...")
        spy_quote = data_fetcher.get_quote('SPY')
        spy_options = data_fetcher.get_option_chain('SPY')
        
        if spy_quote and spy_options:
            prices = {'SPY': spy_quote['price']}
            option_chains = {'SPY': spy_options}
            
            print(f"✓ Retrieved SPY data - Price: ${spy_quote['price']:.2f}")
            
            # Demonstrate cross-expiry analysis
            print("\n📈 Creating cross-expiry analysis...")
            cross_plots = processor.create_cross_expiry_analysis(
                option_chains=option_chains,
                prices=prices,
                ticker='SPY',
                plot_folder='./demo_plots'
            )
            
            print(f"✓ Created {len(cross_plots)} cross-expiry plots")
            for plot_type, file_path in cross_plots.items():
                print(f"   📊 {plot_type}: {file_path}")
            
            # Demonstrate elasticity analysis
            print("\n⚡ Analyzing option elasticity...")
            current_time = datetime.now()
            analysis_date = current_time.strftime('%y%m%d')
            
            elasticity_results = processor.analyze_elasticity_strategies(
                option_chains=option_chains,
                prices=prices,
                analysis_date=analysis_date,
                current_time=current_time,
                plot_folder='./demo_plots'
            )
            
            if not elasticity_results.empty:
                print(f"✓ Elasticity analysis complete")
                print(f"   Top elasticity: {elasticity_results['elasticity'].max():.2f}")
                print(f"   Results saved to: ./demo_plots/{analysis_date}/")
            
            print(f"\n🎯 Advanced features demonstrated successfully!")
            
        else:
            print("⚠️ Could not fetch real data, using simulated data...")
            
            # Create minimal simulated data
            ticker = 'DEMO'
            current_price = 150.0
            
            # Create basic option data
            strikes = [140, 145, 150, 155, 160]
            puts_data = []
            
            for strike in strikes:
                puts_data.append({
                    'Strike': strike,
                    'Last': max(strike - current_price + 2, 0.5),
                    'IV': 0.25 + 0.05 * abs(strike - current_price) / current_price,
                    'Bid': 1.0,
                    'Ask': 1.2
                })
            
            puts_df = pd.DataFrame(puts_data)
            puts_df.index = [f"DEMO250321P{int(s*1000):08d}" for s in strikes]
            
            expiry_name = (datetime.now() + timedelta(days=30)).strftime('%b.%d.%Y')
            
            # Demonstrate elasticity calculation with simulated data
            elasticity_df = processor.calculate_option_elasticity(
                puts_df=puts_df,
                current_price=current_price,
                expiry_date=expiry_name,
                current_time=datetime.now()
            )
            
            if not elasticity_df.empty:
                print(f"✓ Simulated elasticity calculation:")
                print(elasticity_df[['strike', 'elasticity', 'delta']].round(3))
        
    except Exception as e:
        print(f"❌ Advanced features demo error: {e}")

def main():
    """Main entry point - show menu and run selected demo."""
    
    print("🎯 OPTIONS ANALYSIS SYSTEM - QUICK START")
    print("Choose a demonstration:")
    print("1. Quick Demo (single ticker - fastest)")
    print("2. Full Demo (multiple tickers - comprehensive)")  
    print("3. Advanced Features Demo (op01.r & op05.r features)")
    print("4. Run all demos")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        quick_demo()
    elif choice == "2":
        full_demo()
    elif choice == "3":
        advanced_features_demo()
    elif choice == "4":
        quick_demo()
        full_demo()
        advanced_features_demo()
    else:
        print("Invalid choice, running quick demo...")
        quick_demo()
    
    print("\n" + "="*50)
    print("🎉 DEMO COMPLETE!")
    print("="*50)
    print("Next steps:")
    print("📖 Check example_advanced_analysis.py for comprehensive examples")
    print("🔧 Check example_individual_features.py for feature-specific examples")
    print("📚 See README.md for full documentation")
    print("🚀 Modify main.py to customize for your needs")

if __name__ == "__main__":
    main()