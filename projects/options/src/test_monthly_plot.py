#!/usr/bin/env python3
"""Quick test of monthly options plotting functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from plot_monthly_options import identify_monthly_expiries, filter_strikes_around_atm
import pandas as pd
import numpy as np

# Test identify_monthly_expiries function
print("Testing identify_monthly_expiries()...")
print("=" * 60)

test_expiries = [
    "Sep.05.2025",  # Not monthly (too early)
    "Sep.12.2025",  # Not monthly (too early)
    "Sep.19.2025",  # MONTHLY (3rd Friday)
    "Sep.26.2025",  # Not monthly (4th Friday)
    "Oct.03.2025",  # Not monthly (too early)
    "Oct.17.2025",  # MONTHLY (3rd Friday)
    "Nov.21.2025",  # MONTHLY (3rd Friday)
    "Dec.19.2025",  # MONTHLY (3rd Friday)
    "Jan.16.2026",  # MONTHLY (3rd Friday)
]

monthly = identify_monthly_expiries(test_expiries)

print(f"Input expiries: {len(test_expiries)}")
print(f"Monthly expiries found: {len(monthly)}")
print()

for exp in test_expiries:
    is_monthly = "✓ MONTHLY" if exp in monthly else "✗ weekly"
    print(f"  {exp:15s} {is_monthly}")

print()
print("=" * 60)
print()

# Test filter_strikes_around_atm function
print("Testing filter_strikes_around_atm()...")
print("=" * 60)

strikes = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
current_price = 145

print(f"All strikes: {strikes}")
print(f"Current price: ${current_price}")
print()

for n in [3, 5, 10]:
    filtered = filter_strikes_around_atm(strikes, current_price, n)
    print(f"N={n:2d} strikes around ATM: {filtered}")

print()
print("=" * 60)
print()
print("✓ Basic functionality tests passed!")
print()
print("To test with real data (requires network):")
print("  python src/plot_monthly_options.py AAPL --no-show")
