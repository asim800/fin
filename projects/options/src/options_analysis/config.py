"""Configuration module for options analysis system."""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for options analysis system."""
    
    # Base directory paths (using os.path for better compatibility)
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    src_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_folder: str = "data"
    plot_folder: str = "plots"
    ticker_file: str = "tickersv1.csv"
    
    # Subfolder organization
    processed_data_subfolder: str = "processed"
    raw_data_subfolder: str = "raw"
    greeks_subfolder: str = "greeks"
    prices_subfolder: str = "prices"
    
    # Plot subfolder organization  
    elasticity_plots_subfolder: str = "elasticity"
    parity_plots_subfolder: str = "parity"
    strategy_plots_subfolder: str = "strategy"
    iv_plots_subfolder: str = "iv_analysis"
    
    # Data settings
    tickers: List[str] = field(default_factory=list)
    prices: Dict[str, float] = field(default_factory=dict)
    plot_enabled: Dict[str, bool] = field(default_factory=dict)
    
    # Plot settings
    plot_width: int = 160
    plots_per_page: int = 6  # 2x3 grid
    plot_rows: int = 2
    plot_cols: int = 3
    
    # Special limits for specific tickers
    xlimits: Dict[str, float] = field(default_factory=lambda: {'NVDA': 220})
    
    # Yahoo Finance settings
    request_delay: float = 1.0  # seconds between requests
    timeout: int = 30
    
    # File naming
    save_file_prefix: str = "oc_"
    date_format: str = "%y%m%d"
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        self.current_time = datetime.now()
        self.date_string = self.current_time.strftime(self.date_format)
        self.save_filename = f"{self.save_file_prefix}{self.date_string}.pkl"
        
        # Convert relative paths to absolute paths using os.path
        # Use project_dir to put folders at the same level as src
        if not os.path.isabs(self.data_folder):
            self.data_folder = os.path.join(self.project_dir, self.data_folder)
        if not os.path.isabs(self.plot_folder):
            self.plot_folder = os.path.join(self.project_dir, self.plot_folder)
        
        # Create all necessary directories
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create complete directory structure using os.makedirs."""
        # Main directories
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.plot_folder, exist_ok=True)
        
        # Data subdirectories (3-step architecture folders)
        os.makedirs(self.get_processed_data_path(), exist_ok=True)
        os.makedirs(self.get_raw_data_path(), exist_ok=True)
        os.makedirs(self.get_individual_data_path(), exist_ok=True)
        os.makedirs(self.get_comprehensive_data_path(), exist_ok=True)
        # Legacy folders (keep for transition)
        os.makedirs(self.get_greeks_data_path(), exist_ok=True)
        os.makedirs(self.get_prices_data_path(), exist_ok=True)
        
        # Plot subdirectories (will be created with date when needed)
        date_plot_folder = os.path.join(self.plot_folder, self.date_string)
        os.makedirs(date_plot_folder, exist_ok=True)
        
        # Plot type subdirectories
        os.makedirs(self.get_elasticity_plots_path(), exist_ok=True)
        os.makedirs(self.get_parity_plots_path(), exist_ok=True)
        os.makedirs(self.get_strategy_plots_path(), exist_ok=True)
        os.makedirs(self.get_iv_plots_path(), exist_ok=True)
        
    def load_tickers(self, ticker_file_path: Optional[str] = None) -> List[str]:
        """Load ticker symbols and plot flags from CSV file."""
        import csv

        file_path = ticker_file_path or self.ticker_file

        # Convert to absolute path if relative
        if not os.path.isabs(file_path):
            # Try current directory first
            if os.path.exists(file_path):
                file_path = os.path.abspath(file_path)
            else:
                # Try parent directories
                parent_path = os.path.join(os.path.dirname(self.base_dir), file_path)
                if os.path.exists(parent_path):
                    file_path = parent_path
                else:
                    # Try grandparent directory
                    grandparent_path = os.path.join(os.path.dirname(os.path.dirname(self.base_dir)), file_path)
                    if os.path.exists(grandparent_path):
                        file_path = grandparent_path
                    else:
                        raise FileNotFoundError(f"Ticker file not found: {file_path}")

        tickers = []
        self.plot_enabled = {}

        with open(file_path, 'r') as f:
            # Check if CSV format by looking at first line
            first_line = f.readline().strip()
            f.seek(0)

            if ',' in first_line:
                # CSV format with header
                reader = csv.DictReader(f)
                for row in reader:
                    ticker = row['ticker'].strip()
                    if ticker:
                        tickers.append(ticker)
                        # Parse plot flag (default to 0)
                        plot_flag = row.get('plot', '0').strip()
                        self.plot_enabled[ticker] = (plot_flag == '1')
            else:
                # Legacy format (plain text, one ticker per line)
                f.seek(0)
                for line in f:
                    ticker = line.strip()
                    if ticker:
                        tickers.append(ticker)
                        self.plot_enabled[ticker] = False  # Default to no plots

        self.tickers = tickers
        return tickers

    def should_plot(self, ticker: str) -> bool:
        """Check if plots should be generated for ticker."""
        return self.plot_enabled.get(ticker, False)

    def get_plot_enabled_tickers(self) -> List[str]:
        """Get list of tickers with plotting enabled."""
        return [t for t in self.tickers if self.should_plot(t)]

    def get_plot_summary(self) -> str:
        """Get summary of plot settings."""
        enabled = sum(1 for t in self.tickers if self.should_plot(t))
        return f"{enabled}/{len(self.tickers)} tickers have plotting enabled"

    # =============================================================================
    # DATA PATH METHODS (using os.path)
    # =============================================================================
    
    def get_save_file_path(self) -> str:
        """Get full path for main save file."""
        return os.path.join(self.data_folder, self.save_filename)
    
    def get_processed_data_path(self) -> str:
        """Get path for processed data."""
        return os.path.join(self.data_folder, self.processed_data_subfolder)
        
    def get_raw_data_path(self) -> str:
        """Get path for raw data."""
        return os.path.join(self.data_folder, self.raw_data_subfolder)
        
    def get_greeks_data_path(self) -> str:
        """Get path for Greeks calculations."""
        return os.path.join(self.data_folder, self.greeks_subfolder)
        
    def get_prices_data_path(self) -> str:
        """Get path for price data."""
        return os.path.join(self.data_folder, self.prices_subfolder)
    
    def get_individual_data_path(self) -> str:
        """Get path for individual analysis results (Step 2)."""
        return os.path.join(self.data_folder, "individual")
    
    def get_comprehensive_data_path(self) -> str:
        """Get path for comprehensive analysis results (Step 3)."""
        return os.path.join(self.data_folder, "comprehensive")
    
    def get_processed_chains_file_path(self, suffix: str = "") -> str:
        """Get path for processed option chains file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_chains_{timestamp}{suffix}.pkl"
        return os.path.join(self.get_processed_data_path(), filename)
    
    def get_greeks_file_path(self, ticker: str, date_str: str = None) -> str:
        """Get path for Greeks data file."""
        date_str = date_str or self.date_string
        filename = f"greeks_{ticker}_{date_str}.pkl"
        return os.path.join(self.get_greeks_data_path(), filename)
    
    def get_prices_file_path(self, date_str: str = None) -> str:
        """Get path for prices data file."""
        date_str = date_str or self.date_string
        filename = f"prices_{date_str}.pkl"
        return os.path.join(self.get_prices_data_path(), filename)
    
    # =============================================================================
    # PLOT PATH METHODS (using os.path)
    # =============================================================================
    
    def get_plot_folder_path(self) -> str:
        """Get plot folder path with date subdirectory."""
        date_folder = os.path.join(self.plot_folder, self.date_string)
        os.makedirs(date_folder, exist_ok=True)
        return date_folder
    
    def get_elasticity_plots_path(self) -> str:
        """Get path for elasticity plots."""
        path = os.path.join(self.get_plot_folder_path(), self.elasticity_plots_subfolder)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_parity_plots_path(self) -> str:
        """Get path for put-call parity plots."""
        path = os.path.join(self.get_plot_folder_path(), self.parity_plots_subfolder)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_strategy_plots_path(self) -> str:
        """Get path for strategy plots."""
        path = os.path.join(self.get_plot_folder_path(), self.strategy_plots_subfolder)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_iv_plots_path(self) -> str:
        """Get path for IV analysis plots."""
        path = os.path.join(self.get_plot_folder_path(), self.iv_plots_subfolder)
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_elasticity_scatter_plot_path(self, expiry_date: str) -> str:
        """Get path for elasticity scatter plot."""
        clean_expiry = expiry_date.replace('.', '')
        filename = f"puts_ratio_{clean_expiry}.png"
        return os.path.join(self.get_elasticity_plots_path(), filename)
    
    def get_elasticity_ranking_plot_path(self, expiry_date: str) -> str:
        """Get path for elasticity ranking plot."""
        clean_expiry = expiry_date.replace('.', '')
        filename = f"puts_{clean_expiry}.png"
        return os.path.join(self.get_elasticity_plots_path(), filename)
    
    def update_prices(self, ticker: str, price: float) -> None:
        """Update price for a ticker."""
        self.prices[ticker] = price
    
    def get_xlimit(self, ticker: str) -> Optional[float]:
        """Get x-axis limit for specific ticker plots."""
        return self.xlimits.get(ticker)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config instance from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to dictionary."""
        return {
            'data_folder': self.data_folder,
            'plot_folder': self.plot_folder,
            'ticker_file': self.ticker_file,
            'tickers': self.tickers,
            'prices': self.prices,
            'plot_width': self.plot_width,
            'plots_per_page': self.plots_per_page,
            'plot_rows': self.plot_rows,
            'plot_cols': self.plot_cols,
            'xlimits': self.xlimits,
            'request_delay': self.request_delay,
            'timeout': self.timeout,
            'save_file_prefix': self.save_file_prefix,
            'date_format': self.date_format
        }