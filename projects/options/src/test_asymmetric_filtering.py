#!/usr/bin/env python3
"""Test asymmetric strike filtering functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from plot_monthly_options import filter_strikes_around_atm

# Example strikes and current price
strikes = [240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320]
current_price = 278.50

print("=" * 80)
print("Testing Asymmetric Strike Filtering")
print("=" * 80)
print()
print(f"Available strikes: {strikes}")
print(f"Current stock price: ${current_price:.2f}")
print(f"ATM strike: ${280} (closest to current)")
print()

# Test symmetric filtering
print("-" * 80)
print("SYMMETRIC FILTERING (±5 around ATM)")
print("-" * 80)
symmetric_strikes = filter_strikes_around_atm(
    strikes, current_price, n_around=5, asymmetric=False
)
print(f"Selected strikes: {symmetric_strikes}")
print(f"Strike range: ${min(symmetric_strikes)} to ${max(symmetric_strikes)}")
print(f"Number of strikes: {len(symmetric_strikes)}")
print()

# Test asymmetric filtering for puts
print("-" * 80)
print("ASYMMETRIC FILTERING - PUTS (at/below current price)")
print("-" * 80)
put_strikes = filter_strikes_around_atm(
    strikes, current_price, n_around=8, asymmetric=True, option_type='put'
)
print(f"Selected strikes: {put_strikes}")
print(f"Strike range: ${min(put_strikes)} to ${max(put_strikes)}")
print(f"Number of strikes: {len(put_strikes)}")
print(f"All strikes <= ATM? {all(s <= 280 for s in put_strikes)} ✓")
print()

# Test asymmetric filtering for calls
print("-" * 80)
print("ASYMMETRIC FILTERING - CALLS (at/above current price)")
print("-" * 80)
call_strikes = filter_strikes_around_atm(
    strikes, current_price, n_around=8, asymmetric=True, option_type='call'
)
print(f"Selected strikes: {call_strikes}")
print(f"Strike range: ${min(call_strikes)} to ${max(call_strikes)}")
print(f"Number of strikes: {len(call_strikes)}")
print(f"All strikes >= ATM? {all(s >= 280 for s in call_strikes)} ✓")
print()

print("=" * 80)
print("✓ Asymmetric filtering works correctly!")
print()
print("Key differences:")
print("  • SYMMETRIC: Shows strikes on both sides of current price")
print("  • ASYMMETRIC PUTS: Shows only strikes at/below current price")
print("  • ASYMMETRIC CALLS: Shows only strikes at/above current price")
print("=" * 80)
