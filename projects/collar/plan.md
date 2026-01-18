

I'll implement the Option Collar Framework Phase 0 according to the specification. Let me start by setting up the project structure and working through the implementation order.
Update Todos
￼
Create project structure and pyproject.toml
￼
Implement src/collar/pricing/base.py - OptionGreeks and PricingEngine
￼
Implement src/collar/pricing/bsm.py - BSMPricer class
￼
Implement src/collar/core/structures.py - CollarLeg, CollarPosition, CollarPricing
￼
Implement src/collar/core/builder.py - CollarBuilder class
￼
Implement src/collar/core/scenario.py - Scenario, ScenarioResult, ScenarioAnalyzer
￼
Implement src/collar/core/greeks.py - GreeksAnalyzer class
￼
Implement src/collar/analysis/comparator.py - CollarComparator class
￼
Implement tests/test_pricing.py - BSM validation tests
￼
Implement tests/test_collar.py - Collar construction tests
￼
Implement tests/test_scenario.py - Scenario analysis tests
￼
Implement demo.py - CLI demonstration script
￼
Run all tests and validate against success criteria

# Option Collar Framework - Phase 0.1 Enhancements


## Overview
Three enhancements to add realistic volatility skew, opportunity cost analysis, and interactive visualizations.

---

## Feature 1: SABR Volatility Skew Model

### Problem
Current BSMPricer uses flat volatility across all strikes. Real SPY options have negative skew (puts trade at higher IV than calls).

### Solution
Implement SABR stochastic volatility model with pluggable interface.

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `lib/volatility.py` | **Create** | VolatilityModel protocol, FlatVolatility, SABRVolatility, SABRParameters |
| `lib/pricing.py` | Modify | Add optional `volatility_model` parameter to BSMPricer |
| `lib/__init__.py` | Modify | Export new volatility classes |
| `src/builder.py` | Modify | Pass volatility model through to pricing engine |
| `tests/test_volatility.py` | **Create** | SABR accuracy, calibration, edge case tests |

### Key Classes

```python
# lib/volatility.py
class VolatilityModel(Protocol):
    def implied_vol(self, strike: float, forward: float, expiry_years: float) -> float: ...

class FlatVolatility:
    volatility: float  # Backward compatible

class SABRParameters:
    alpha: float  # ATM vol level
    beta: float = 0.5  # CEV exponent
    rho: float = -0.35  # Correlation (negative = skew)
    nu: float = 0.4  # Vol-of-vol

class SABRVolatility:
    def implied_vol(...) -> float  # Hagan approximation
    @classmethod
    def from_atm_vol(cls, atm_vol, forward, expiry_years, ...) -> SABRVolatility
```

### Integration Point
```python
# In BSMPricer.__init__:
if volatility_model is not None:
    self.volatility = volatility_model.implied_vol(strike, forward, expiry_years)
```

---

## Feature 2: Capped Gains (Opportunity Cost) Analysis

### Problem
No quantification of opportunity cost when underlying exceeds call strike.

### Solution
Analyze discrete upside scenarios with probability-weighted opportunity costs.

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/capped_gains.py` | **Create** | CappedGainsAnalyzer, UpsideScenario, CappedGainsResult |
| `src/__init__.py` | Modify | Export new classes |
| `tests/test_capped_gains.py` | **Create** | Probability and cost calculation tests |

### Output Table Format
```
Upside Level | Target Price | Probability | Uncapped Gain | Capped Gain | Opp Cost | Expected Cost
-------------|--------------|-------------|---------------|-------------|----------|---------------
5%           | $525.00      | 42.3%       | $25.00        | $25.00      | $0.00    | $0.00
10%          | $550.00      | 28.1%       | $50.00        | $25.00      | $25.00   | $7.03
15%          | $575.00      | 17.2%       | $75.00        | $25.00      | $50.00   | $8.60
20%          | $600.00      | 9.8%        | $100.00       | $25.00      | $75.00   | $7.35
25%          | $625.00      | 5.2%        | $125.00       | $25.00      | $100.00  | $5.20
```

### Probability Models
1. **Lognormal** (default): Use BSM d2 formula for P(S_T > K)
2. **Empirical SPY** (future): Hardcoded historical return distribution

### Key Methods
```python
class CappedGainsAnalyzer:
    def analyze(collar, pricing, expiry_years, upside_levels=[0.05, 0.10, 0.15, 0.20, 0.25]) -> CappedGainsResult
    def to_dataframe(result) -> pd.DataFrame
    def calculate_breakeven_upside(collar, pricing) -> float
```

---

## Feature 3: Plotly Visualizations

### Problem
No visual representation of collar analysis results.

### Solution
Create CollarVisualizer with Plotly charts (Streamlit-compatible).

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/visualizer.py` | **Create** | CollarVisualizer with 6 chart methods |
| `src/__init__.py` | Modify | Export CollarVisualizer |
| `pyproject.toml` | Modify | Add `plotly>=5.18` dependency |
| `tests/test_visualizer.py` | **Create** | Figure generation tests |

### Visualization Methods

| Method | Description |
|--------|-------------|
| `payoff_diagram()` | Collar payoff at expiry with stock/put/call/net lines |
| `scenario_comparison()` | Bar chart comparing collar vs unhedged P&L |
| `greeks_profile()` | Delta/gamma across price range vs stock-only |
| `greeks_comparison()` | Bar chart: collar Greeks vs stock (delta=1, others=0) |
| `opportunity_cost_analysis()` | Capped vs uncapped gains with expected cost |
| `volatility_skew()` | IV curve from SABR model |

### Color Scheme
```python
DEFAULT_COLORS = {
    "stock": "#2E86AB",      # Blue
    "put": "#A23B72",        # Magenta
    "call": "#F18F01",       # Orange
    "collar_net": "#C73E1D", # Red
}
```

---

## Implementation Order

### Step 1: SABR Volatility
1. Create `lib/volatility.py` with VolatilityModel protocol and SABR implementation
2. Modify `lib/pricing.py` to accept volatility_model parameter
3. Update `lib/__init__.py` exports
4. Create `tests/test_volatility.py`
5. Modify `src/builder.py` to pass volatility model

### Step 2: Capped Gains Analysis
1. Create `src/capped_gains.py` with analyzer and dataclasses
2. Update `src/__init__.py` exports
3. Create `tests/test_capped_gains.py`

### Step 3: Visualizations
1. Add plotly to `pyproject.toml` dependencies
2. Create `src/visualizer.py` with all chart methods
3. Update `src/__init__.py` exports
4. Create `tests/test_visualizer.py`

### Step 4: Integration
1. Update `demo.py` to showcase all features
2. Run full test suite

---

## Critical Files Summary

| File | Purpose |
|------|---------|
| `lib/volatility.py` | New - SABR model |
| `lib/pricing.py` | Modify - Accept vol model |
| `src/capped_gains.py` | New - Opportunity cost analysis |
| `src/visualizer.py` | New - Plotly charts |
| `src/builder.py` | Modify - Vol model integration |
| `demo.py` | Modify - Showcase all features |

---

## Dependencies to Add

```toml
# pyproject.toml
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "plotly>=5.18",  # NEW
]
```

---

## Success Criteria

- [ ] SABR skew produces higher IV for OTM puts vs OTM calls
- [ ] Put-call parity still holds with skewed volatilities
- [ ] Capped gains table shows correct probability calculations
- [ ] Total expected opportunity cost sums correctly
- [ ] All 6 visualizations render without error
- [ ] Existing 45 tests still pass
- [ ] New tests pass for volatility, capped gains, visualizer
- [ ] Demo runs end-to-end with all new features
