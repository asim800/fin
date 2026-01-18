# Options Analysis Toolkit

Comprehensive Python toolkit for options trading analysis with elasticity calculations, put-call parity analysis, and arbitrage detection.

## 🚀 Quick Start

```bash
# Quick elasticity check
python src/run.py AAPL

# Check puts instead of calls
python src/run.py AAPL --puts

# Export to Excel
python src/run.py AAPL --export

# Find cheap options
python src/run.py --find-cheap
```

## 📂 Project Structure

```
options/python/
├── src/                    # All source code
│   ├── run.py             # Simple runner script
│   ├── main.py            # Main analysis script
│   ├── options_analysis/  # Core package
│   └── examples/          # Example scripts
├── docs/                   # All documentation
│   ├── HOW_TO_RUN.md      # Complete usage guide
│   ├── QUICKSTART.md      # Quick start guide
│   └── ...
├── data/                   # Data storage
└── pyproject.toml         # Package configuration
```

## 📖 Documentation

**Start Here**: [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) - Complete usage guide

**Other Docs**:
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Installation and setup
- [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md) - Integration guide
- [docs/TOOLKIT_IMPLEMENTATION.md](docs/TOOLKIT_IMPLEMENTATION.md) - API reference
- [docs/ELASTICITY_TABLES_FEATURES.md](docs/ELASTICITY_TABLES_FEATURES.md) - Feature docs

## 🎯 Common Tasks

### Run Examples
```bash
# Toolkit demo (best for learning)
python src/examples/toolkit_demo.py

# Elasticity demo
python src/examples/elasticity_tables_demo.py

# Quick check
python src/examples/quick_elasticity_check.py AAPL
```

### Interactive Python
```bash
python
```
```python
import sys
sys.path.insert(0, 'src')

from options_analysis import AnalysisToolkit

toolkit = AnalysisToolkit()
quote = toolkit.get_quote('AAPL')
elasticity = toolkit.get_elasticity('AAPL')
```

### Analyze Multiple Tickers
```bash
python src/main.py --ticker NVDA
python src/main.py --file tickers.txt
```

## ✨ Features

- **Option Elasticity** - Calculate leverage ratios
- **Put-Call Parity** - Detect arbitrage opportunities
- **Smart Filtering** - Find best options by budget
- **Pivot Tables** - Organize by strikes and expiries
- **Data Export** - CSV, Excel, JSON formats
- **Visualizations** - Comprehensive plots

## 🔧 No Installation Needed

Just run Python scripts directly. All imports are handled automatically.

## 📚 Learn More

See [docs/HOW_TO_RUN.md](docs/HOW_TO_RUN.md) for complete documentation.
