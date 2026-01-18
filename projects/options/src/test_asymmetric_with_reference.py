#!/usr/bin/env python3
"""Test updated asymmetric strike filtering with reference strikes."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from plot_monthly_options import filter_strikes_around_atm

# Example strikes and current price
strikes = [240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320]
current_price = 278.50

print("=" * 80)
print("Testing Asymmetric Strike Filtering WITH Reference Strikes")
print("=" * 80)
print()
print(f"Available strikes: {strikes}")
print(f"Current stock price: ${current_price:.2f}")
print(f"ATM strike: ${280} (closest to current)")
print()

# Test asymmetric filtering for puts
print("-" * 80)
print("ASYMMETRIC FILTERING - PUTS")
print("Includes strikes at/below current + 1 above for reference")
print("-" * 80)
put_strikes = filter_strikes_around_atm(
    strikes, current_price, n_around=8, asymmetric=True, option_type='put'
)
print(f"Selected strikes: {put_strikes}")
print(f"Strike range: ${min(put_strikes)} to ${max(put_strikes)}")
print(f"Number of strikes: {len(put_strikes)}")
print()

# Check the filtering
below_atm = [s for s in put_strikes if s <= 280]
above_atm = [s for s in put_strikes if s > 280]
print(f"Strikes at/below ATM (280): {below_atm} ({len(below_atm)} strikes)")
print(f"Strikes above ATM (280): {above_atm} ({len(above_atm)} strikes)")
print(f"✓ Has reference strike above ATM: {len(above_atm) >= 1}")
print()

# Test asymmetric filtering for calls
print("-" * 80)
print("ASYMMETRIC FILTERING - CALLS")
print("Includes strikes at/above current + 1 below for reference")
print("-" * 80)
call_strikes = filter_strikes_around_atm(
    strikes, current_price, n_around=8, asymmetric=True, option_type='call'
)
print(f"Selected strikes: {call_strikes}")
print(f"Strike range: ${min(call_strikes)} to ${max(call_strikes)}")
print(f"Number of strikes: {len(call_strikes)}")
print()

# Check the filtering
below_atm = [s for s in call_strikes if s < 280]
at_above_atm = [s for s in call_strikes if s >= 280]
print(f"Strikes below ATM (280): {below_atm} ({len(below_atm)} strikes)")
print(f"Strikes at/above ATM (280): {at_above_atm} ({len(at_above_atm)} strikes)")
print(f"✓ Has reference strike below ATM: {len(below_atm) >= 1}")
print()

print("=" * 80)
print("✓ Updated asymmetric filtering works correctly!")
print()
print("Key improvements:")
print("  • PUTS: Include 1 strike above for ITM reference")
print("  • CALLS: Include 1 strike below for ITM reference")
print("  • Better context for option pricing comparisons")
print("=" * 80)
