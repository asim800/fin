# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace Overview

This is a financial analysis workspace containing shared libraries and multiple application projects. All projects share a single virtual environment managed by uv.

## Directory Structure

```
fin/
├── lib/                    # Shared Python packages (3 packages)
│   ├── findata/            # Data fetching, caching, persistence
│   ├── fincalc/            # Calculations: pricing, metrics, risk, covariance
│   └── finviz/             # Visualization utilities
│
├── projects/               # Application projects
│   ├── portfolio/          # Monte Carlo retirement simulation
│   ├── cover/              # Covered call strategies
│   ├── collar/             # Option collar strategies
│   └── options/            # Options pricing analysis
│
├── configs/                # Centralized configuration
│   ├── portfolio/          # Portfolio project configs
│   └── shared/             # Shared parameters (mean returns, covariance)
│
├── data/                   # All data files
│   ├── cache/              # Price data caches (.pkl)
│   ├── raw/                # Raw downloaded data
│   ├── processed/          # Processed datasets
│   └── tickers/            # Ticker list files
│
├── output/                 # Generated outputs
│   ├── plots/              # Visualization outputs
│   ├── results/            # Analysis results
│   ├── mc/                 # Monte Carlo results
│   └── logs/               # Log files
│
├── notebooks/              # Jupyter notebooks
├── experiments/            # Experimental scripts
└── docs/                   # Cross-project documentation
```

## Quick Start

```bash
# Install all dependencies (including dev tools like pytest)
uv sync --extra dev

# Run portfolio Monte Carlo simulation
uv run python -m projects.portfolio.src.run_mc --config configs/portfolio/test_simple_buyhold.json

# Run tests
uv run python -m pytest projects/portfolio/tests/
uv run python -m pytest projects/collar/tests/
```

## Shared Libraries

### findata - Data Management
```python
from findata import FinDataFetcher, CacheManager
from findata import save_pickle, load_pickle, save_csv, load_csv
from findata import calculate_returns, resample_returns, compound_returns

# Fetch market data
fetcher = FinDataFetcher(start_date='2020-01-01', end_date='2024-01-01')
prices = fetcher.fetch_prices(['SPY', 'AGG', 'NVDA'])

# Calculate returns
returns = calculate_returns(prices)
```

### fincalc - Financial Calculations
```python
from fincalc import BlackScholesCalculator, SABRVolatility, SABRParameters
from fincalc import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
from fincalc import calculate_var, calculate_cvar
from fincalc import CovarianceEstimator

# Option pricing
bs = BlackScholesCalculator()
price = bs.price(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')
greeks = bs.greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.2)

# Performance metrics
sharpe = sharpe_ratio(portfolio_values, rf=0.02)

# Covariance estimation
cov_est = CovarianceEstimator()
cov_matrix = cov_est.estimate(returns, method='shrunk')
```

### finviz - Visualization
```python
from finviz import create_figure, save_figure, format_currency, format_percentage
from finviz import set_style, DEFAULT_COLORS

set_style()
fig, ax = create_figure(nrows=1, ncols=2)
# ... plot ...
save_figure(fig, 'output/plots/my_chart.png')
```

## Projects

### portfolio/ - Retirement Monte Carlo Simulation
Main entry point: `projects/portfolio/src/run_mc.py`
- Monte Carlo lifecycle simulation
- Parametric and bootstrap sampling
- Accumulation and decumulation phases
- See `projects/portfolio/CLAUDE.md` for detailed docs

### collar/ - Option Collar Framework
- SABR volatility model
- Option pricing with Greeks
- Collar strategy analysis

### cover/ - Covered Call Strategies
- Covered call analysis
- ML-based models

### options/ - Options Analysis
- Options pricing and Greeks
- Put-call parity analysis

## Common Commands

```bash
# Install dependencies
uv sync --extra dev

# Run portfolio MC simulation
uv run python -m projects.portfolio.src.run_mc --config configs/portfolio/test_simple_buyhold.json

# Run with parameter sweep
uv run python -m projects.portfolio.src.run_mc --config configs/portfolio/test_simple_buyhold.json --sweep

# Start FastAPI backend (portfolio)
uv run uvicorn projects.portfolio.src.api.main:app --host 0.0.0.0 --port 8001

# Run tests
uv run python -m pytest projects/portfolio/tests/
uv run python -m pytest projects/collar/tests/

# Test lib package imports
uv run python -c "from findata import FinDataFetcher; print('OK')"
uv run python -c "from fincalc import BlackScholesCalculator; print('OK')"
uv run python -c "from finviz import create_figure; print('OK')"
```

## Key Concepts

### Covariance Scales Linearly (NOT sqrt)
```python
# CORRECT
period_cov = annual_cov / periods_per_year

# WRONG
period_cov = annual_cov / np.sqrt(periods_per_year)
```

### Compound Returns (NOT sum)
```python
# CORRECT
annual_return = np.prod(1 + weekly_returns) - 1

# WRONG
annual_return = np.sum(weekly_returns)
```

## Development Guidelines

1. Use shared lib packages (`findata`, `fincalc`, `finviz`) for common functionality
2. Keep project-specific code in `projects/<project>/`
3. Store configs in `configs/`, data in `data/`, outputs in `output/`
4. Don't remove `import ipdb` - used for debugging
5. Use relative imports within project packages (e.g., `from .config import SystemConfig`, `from ..data import FinData`)
6. Use WORKSPACE_ROOT for paths to configs/, data/, output/ directories
