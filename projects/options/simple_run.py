#!/usr/bin/env python3
"""
Simple script to run options analysis - NO INSTALLATION NEEDED!
Just run: python simple_run.py
"""

import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent / "src"))

from options_analysis import AnalysisToolkit

def main():
    """Run simple analysis."""

    # Change this ticker to whatever you want
    ticker = 'AAPL'

    print(f"\n{'='*60}")
    print(f"Options Analysis for {ticker}")
    print(f"{'='*60}\n")

    # Initialize toolkit
    toolkit = AnalysisToolkit()

    # Get current price
    print("📊 Getting quote...")
    quote = toolkit.get_quote(ticker)
    if quote:
        print(f"✓ Current Price: ${quote['price']:.2f}\n")

    # Get top 10 call options by elasticity
    print("📈 Getting top 10 calls by elasticity...")
    calls = toolkit.get_elasticity(ticker, option_type='call', top_n=10)
    if not calls.empty:
        print("\nTop Call Options:")
        print(calls.to_string())

    # Get top 10 put options by elasticity
    print("\n📉 Getting top 10 puts by elasticity...")
    puts = toolkit.get_elasticity(ticker, option_type='put', top_n=10)
    if not puts.empty:
        print("\nTop Put Options:")
        print(puts.to_string())

    print(f"\n{'='*60}")
    print("✓ Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
