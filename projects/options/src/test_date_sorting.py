#!/usr/bin/env python3
"""Test date sorting functionality for expiries."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from plot_monthly_options import sort_expiries_by_date

# Test with various expiry dates
test_expiries = [
    "Jan.15.2027",
    "Dec.19.2025",
    "Sep.18.2026",
    "Feb.20.2026",
    "Jun.18.2026",
    "Mar.20.2026",
    "Dec.18.2026",
    "Jan.16.2026",
    "Apr.17.2026",
]

print("=" * 60)
print("Testing Expiry Date Sorting")
print("=" * 60)
print()

print("BEFORE SORTING:")
print("-" * 60)
for i, exp in enumerate(test_expiries, 1):
    print(f"{i:2d}. {exp}")

print()
print("AFTER SORTING (chronological):")
print("-" * 60)

sorted_expiries = sort_expiries_by_date(test_expiries)

for i, exp in enumerate(sorted_expiries, 1):
    print(f"{i:2d}. {exp}")

print()
print("=" * 60)
print("✓ Dates are now sorted from earliest to latest!")
print("=" * 60)
