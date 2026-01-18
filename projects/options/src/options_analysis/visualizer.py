"""Visualization module for options analysis plots."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime


class Visualizer:
    """Creates visualizations for options analysis."""
    
    def __init__(self, plot_folder: str = "plots", plot_width: int = 160):
        """
        Initialize Visualizer.
        
        Args:
            plot_folder: Base folder for saving plots
            plot_width: Width setting for plots
        """
        self.plot_folder = plot_folder
        self.plot_width = plot_width
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
    def create_put_call_parity_plots(self, ticker: str, pcp_data: Dict[str, Any], 
                                   current_price: float, xlimit: Optional[float] = None,
                                   plots_per_page: int = 6, plot_rows: int = 2, 
                                   plot_cols: int = 3) -> List[str]:
        """
        Create put-call parity plots for a ticker.
        
        This replicates the R pcpplot function.
        
        Args:
            ticker: Stock ticker symbol
            pcp_data: Put-call parity data for ticker
            current_price: Current stock price
            xlimit: X-axis limit for plots
            plots_per_page: Number of plots per page
            plot_rows: Number of rows in subplot grid
            plot_cols: Number of columns in subplot grid
            
        Returns:
            List of created plot file paths
        """
        if not pcp_data:
            self.logger.warning(f"No PCP data for {ticker}")
            return []
        
        # Create date-specific folder
        date_str = datetime.now().strftime("%y%m%d")
        plot_dir = Path(self.plot_folder) / date_str
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        expiry_names = list(pcp_data.keys())
        num_expiries = len(expiry_names)
        
        if num_expiries == 0:
            return []
        
        # Calculate number of pages needed
        num_pages = int(np.ceil(num_expiries / plots_per_page))
        created_files = []
        
        
        for page in range(num_pages):
            try:
                plot_filename = f"{ticker}_{page+1:02d}.png"
                plot_path = plot_dir / plot_filename
                
                # Create figure with subplots
                fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(15, 10))
                fig.suptitle(f"{ticker} Put-Call Parity - ${current_price:.2f}", fontsize=16)
                
                # Flatten axes array for easier indexing
                if plot_rows * plot_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                # Determine which expiries to plot on this page
                start_idx = page * plots_per_page
                end_idx = min(start_idx + plots_per_page, num_expiries)
                expiries_this_page = expiry_names[start_idx:end_idx]
                
                for i, expiry_name in enumerate(expiries_this_page):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    parity_df = pcp_data[expiry_name]
                    
                    if parity_df is None or parity_df.empty:
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                               transform=ax.transAxes)
                        ax.set_title(expiry_name, fontsize=10)
                        continue
                    
                    try:
                        self._plot_single_parity(ax, parity_df, expiry_name, 
                                               current_price, xlimit)
                    except Exception as e:
                        self.logger.error(f"Error plotting {expiry_name}: {e}")
                        ax.text(0.5, 0.5, f'Error: {str(e)[:20]}...', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(expiry_name, fontsize=10)
                
                # Hide unused subplots
                for i in range(len(expiries_this_page), len(axes)):
                    axes[i].set_visible(False)
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                created_files.append(str(plot_path))
                
            except Exception as e:
                self.logger.error(f"Error creating plot page {page+1} for {ticker}: {e}")
                continue
        
        return created_files
    
    def _plot_single_parity(self, ax, parity_df: pd.DataFrame, expiry_name: str,
                           current_price: float, xlimit: Optional[float] = None):
        """
        Plot a single put-call parity chart.
        
        Args:
            ax: Matplotlib axis object
            parity_df: Put-call parity DataFrame
            expiry_name: Expiry date name
            current_price: Current stock price
            xlimit: X-axis limit
        """
        if 'strike' not in parity_df.columns or 'pps' not in parity_df.columns or 'cpk' not in parity_df.columns:
            ax.text(0.5, 0.5, 'Missing columns', ha='center', va='center',
                   transform=ax.transAxes)
            return
        
        strikes = parity_df['strike']
        pps_values = parity_df['pps']  # Put + Stock
        cpk_values = parity_df['cpk']  # Call + Strike
        
        # Determine x-axis limit
        if xlimit is None:
            strikes_array = strikes.to_numpy()
            strikes_finite = strikes_array[np.isfinite(strikes_array)]
            if len(strikes_finite) > 0:
                xlimit = strikes_finite.max()
            else:
                xlimit = current_price * 1.5
        
        # Plot lines
        ax.plot(strikes.to_numpy(), pps_values.to_numpy(), 'b-', label='P+S', linewidth=2)
        ax.plot(strikes.to_numpy(), cpk_values.to_numpy(), 'r-', label='C+K', linewidth=2)
        
        # Add vertical line at current price
        ax.axvline(x=current_price, color='blue', linestyle='--', alpha=0.7, 
                  label=f'Current: ${current_price:.2f}')
        
        # Set limits and labels  
        # Ensure xlimit is finite
        if not np.isfinite(xlimit):
            xlimit = current_price * 1.5
        ax.set_xlim(0, xlimit)
        
        # Handle NaN and Inf values in y-axis limits
        pps_array = pps_values.to_numpy()
        cpk_array = cpk_values.to_numpy()
        
        # Filter out NaN and Inf values
        pps_finite = pps_array[np.isfinite(pps_array)]
        cpk_finite = cpk_array[np.isfinite(cpk_array)]
        
        if len(pps_finite) > 0 and len(cpk_finite) > 0:
            y_min = min(pps_finite.min(), cpk_finite.min()) * 0.95
            y_max = max(pps_finite.max(), cpk_finite.max()) * 1.05
            ax.set_ylim(y_min, y_max)
        else:
            # Default limits if no finite values
            ax.set_ylim(-1, 1)
        
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Put-Call Parity Value')
        ax.set_title(expiry_name, fontsize=10, color='blue')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def create_implied_volatility_surface(self, option_chains: Dict[str, Any], 
                                        ticker: str) -> Optional[str]:
        """
        Create implied volatility surface plot.
        
        Args:
            option_chains: Option chain data
            ticker: Stock ticker
            
        Returns:
            Path to created plot file or None
        """
        if ticker not in option_chains:
            return None
        
        ticker_data = option_chains[ticker]
        
        # Collect IV data
        iv_data = []
        
        for expiry_name, expiry_data in ticker_data.items():
            calls_df = expiry_data.get('calls', pd.DataFrame())
            puts_df = expiry_data.get('puts', pd.DataFrame())
            
            # Process calls
            if not calls_df.empty and 'IV' in calls_df.columns and 'Strike' in calls_df.columns:
                for _, row in calls_df.iterrows():
                    if pd.notna(row['IV']) and row['IV'] > 0:
                        iv_data.append({
                            'expiry': expiry_name,
                            'strike': row['Strike'],
                            'iv': row['IV'],
                            'type': 'call'
                        })
            
            # Process puts  
            if not puts_df.empty and 'IV' in puts_df.columns and 'Strike' in puts_df.columns:
                for _, row in puts_df.iterrows():
                    if pd.notna(row['IV']) and row['IV'] > 0:
                        iv_data.append({
                            'expiry': expiry_name,
                            'strike': row['Strike'],
                            'iv': row['IV'],
                            'type': 'put'
                        })
        
        if not iv_data:
            self.logger.warning(f"No IV data for {ticker}")
            return None
        
        # Create DataFrame
        iv_df = pd.DataFrame(iv_data)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot calls IV
        call_data = iv_df[iv_df['type'] == 'call']
        if not call_data.empty:
            for expiry in call_data['expiry'].unique():
                expiry_data = call_data[call_data['expiry'] == expiry]
                ax1.plot(expiry_data['strike'].to_numpy(), expiry_data['iv'].to_numpy(), 
                        'o-', label=expiry, alpha=0.7)
        
        ax1.set_title(f'{ticker} Call Implied Volatility')
        ax1.set_xlabel('Strike Price')
        ax1.set_ylabel('Implied Volatility')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot puts IV
        put_data = iv_df[iv_df['type'] == 'put']
        if not put_data.empty:
            for expiry in put_data['expiry'].unique():
                expiry_data = put_data[put_data['expiry'] == expiry]
                ax2.plot(expiry_data['strike'].to_numpy(), expiry_data['iv'].to_numpy(), 
                        'o-', label=expiry, alpha=0.7)
        
        ax2.set_title(f'{ticker} Put Implied Volatility')
        ax2.set_xlabel('Strike Price')
        ax2.set_ylabel('Implied Volatility')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Save plot
        date_str = datetime.now().strftime("%y%m%d")
        plot_dir = Path(self.plot_folder) / date_str
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = plot_dir / f"{ticker}_IV_surface.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(plot_path)
    
    def create_volume_analysis_plot(self, option_chains: Dict[str, Any], 
                                  ticker: str) -> Optional[str]:
        """
        Create option volume analysis plot.
        
        Args:
            option_chains: Option chain data
            ticker: Stock ticker
            
        Returns:
            Path to created plot file or None
        """
        if ticker not in option_chains:
            return None
        
        ticker_data = option_chains[ticker]
        
        # Collect volume data
        volume_data = []
        
        for expiry_name, expiry_data in ticker_data.items():
            calls_df = expiry_data.get('calls', pd.DataFrame())
            puts_df = expiry_data.get('puts', pd.DataFrame())
            
            # Process calls
            if not calls_df.empty and 'Vol' in calls_df.columns and 'Strike' in calls_df.columns:
                for _, row in calls_df.iterrows():
                    volume = row['Vol'] if pd.notna(row['Vol']) else 0
                    volume_data.append({
                        'expiry': expiry_name,
                        'strike': row['Strike'],
                        'volume': volume,
                        'type': 'call'
                    })
            
            # Process puts
            if not puts_df.empty and 'Vol' in puts_df.columns and 'Strike' in puts_df.columns:
                for _, row in puts_df.iterrows():
                    volume = row['Vol'] if pd.notna(row['Vol']) else 0
                    volume_data.append({
                        'expiry': expiry_name,
                        'strike': row['Strike'],
                        'volume': volume,
                        'type': 'put'
                    })
        
        if not volume_data:
            return None
        
        # Create DataFrame
        vol_df = pd.DataFrame(volume_data)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Total volume by expiry
        expiry_volume = vol_df.groupby(['expiry', 'type'])['volume'].sum().unstack(fill_value=0)
        if not expiry_volume.empty:
            expiry_volume.plot(kind='bar', ax=ax1, alpha=0.7)
            ax1.set_title(f'{ticker} Option Volume by Expiry')
            ax1.set_ylabel('Total Volume')
            ax1.legend(title='Option Type')
            ax1.tick_params(axis='x', rotation=45)
        
        # Volume by strike (for nearest expiry)
        if not vol_df.empty:
            nearest_expiry = vol_df['expiry'].iloc[0]  # Assuming first is nearest
            nearest_data = vol_df[vol_df['expiry'] == nearest_expiry]
            
            call_vol = nearest_data[nearest_data['type'] == 'call']
            put_vol = nearest_data[nearest_data['type'] == 'put']
            
            if not call_vol.empty:
                ax2.bar(call_vol['strike'].to_numpy(), call_vol['volume'].to_numpy(), 
                       alpha=0.7, label='Calls', color='green')
            if not put_vol.empty:
                ax2.bar(put_vol['strike'].to_numpy(), -put_vol['volume'].to_numpy(), 
                       alpha=0.7, label='Puts', color='red')
            
            ax2.set_title(f'{ticker} Volume by Strike ({nearest_expiry})')
            ax2.set_xlabel('Strike Price')
            ax2.set_ylabel('Volume (Calls +, Puts -)')
            ax2.legend()
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Save plot
        date_str = datetime.now().strftime("%y%m%d")
        plot_dir = Path(self.plot_folder) / date_str
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        plot_path = plot_dir / f"{ticker}_volume_analysis.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return str(plot_path)
    
    def create_summary_dashboard(self, analysis_results: Dict[str, Any]) -> Optional[str]:
        """
        Create a summary dashboard with key metrics.
        
        Args:
            analysis_results: Dictionary with all analysis results
            
        Returns:
            Path to created dashboard file or None
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Options Analysis Dashboard', fontsize=16)
            
            # Plot 1: Top arbitrage opportunities
            if 'arbitrage' in analysis_results and not analysis_results['arbitrage'].empty:
                arb_df = analysis_results['arbitrage'].head(10)
                axes[0, 0].barh(range(len(arb_df)), arb_df['expected_profit'])
                axes[0, 0].set_yticks(range(len(arb_df)))
                axes[0, 0].set_yticklabels([f"{row['ticker']} {row['strike']}" 
                                           for _, row in arb_df.iterrows()])
                axes[0, 0].set_title('Top Arbitrage Opportunities')
                axes[0, 0].set_xlabel('Expected Profit')
            else:
                axes[0, 0].text(0.5, 0.5, 'No arbitrage data', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Top Arbitrage Opportunities')
            
            # Plot 2: Parity violations by ticker
            if 'pcp_stats' in analysis_results and not analysis_results['pcp_stats'].empty:
                pcp_stats = analysis_results['pcp_stats']
                ticker_violations = pcp_stats.groupby('ticker')['violations_1pct'].sum()
                ticker_violations.plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Parity Violations by Ticker')
                axes[0, 1].set_ylabel('Number of Violations')
                axes[0, 1].tick_params(axis='x', rotation=45)
            else:
                axes[0, 1].text(0.5, 0.5, 'No parity stats', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Parity Violations by Ticker')
            
            # Plot 3: Summary statistics
            if 'summary' in analysis_results:
                summary = analysis_results['summary']
                metrics = ['total_pairs_analyzed', 'total_violations_1pct', 'violation_rate']
                values = [summary.get(m, 0) for m in metrics]
                labels = ['Total Pairs', 'Violations', 'Violation Rate']
                
                axes[1, 0].bar(labels, values)
                axes[1, 0].set_title('Overall Statistics')
                axes[1, 0].tick_params(axis='x', rotation=45)
            else:
                axes[1, 0].text(0.5, 0.5, 'No summary data', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Overall Statistics')
            
            # Plot 4: Price distribution
            if 'prices' in analysis_results:
                prices = list(analysis_results['prices'].values())
                axes[1, 1].hist(prices, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 1].set_title('Stock Price Distribution')
                axes[1, 1].set_xlabel('Stock Price')
                axes[1, 1].set_ylabel('Frequency')
            else:
                axes[1, 1].text(0.5, 0.5, 'No price data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Stock Price Distribution')
            
            # Save dashboard
            date_str = datetime.now().strftime("%y%m%d")
            plot_dir = Path(self.plot_folder) / date_str
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            plot_path = plot_dir / "options_analysis_dashboard.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error creating dashboard: {e}")
            return None
    
    def create_putcall_ratios_plots(self, processed_chains: Dict[str, Any], 
                                   ticker: str, plots_per_page: int = 6, 
                                   plot_rows: int = 2, plot_cols: int = 3) -> List[str]:
        """
        Create put-call ratio plots for OI and Volume (with markers).
        
        Args:
            processed_chains: Processed option chain data with ratios
            ticker: Stock ticker symbol
            plots_per_page: Number of plots per page
            plot_rows: Number of rows in subplot grid
            plot_cols: Number of columns in subplot grid
            
        Returns:
            List of created plot file paths
        """
        if ticker not in processed_chains:
            self.logger.warning(f"No processed data for {ticker}")
            return []
        
        ticker_data = processed_chains[ticker]
        expiry_names = list(ticker_data.keys())
        
        if not expiry_names:
            return []
        
        # Create date-specific folder
        date_str = datetime.now().strftime("%y%m%d")
        plot_dir = Path(self.plot_folder) / date_str
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of pages needed
        num_pages = int(np.ceil(len(expiry_names) / plots_per_page))
        created_files = []
        
        for page in range(num_pages):
            try:
                plot_filename = f"{ticker}_putcall_ratios_{page+1:02d}.png"
                plot_path = plot_dir / plot_filename
                
                # Create figure with subplots
                fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(18, 12))
                fig.suptitle(f"{ticker} Put-Call Ratios (OI & Volume)", fontsize=16)
                
                # Flatten axes array for easier indexing
                if plot_rows * plot_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                # Determine which expiries to plot on this page
                start_idx = page * plots_per_page
                end_idx = min(start_idx + plots_per_page, len(expiry_names))
                expiries_this_page = expiry_names[start_idx:end_idx]
                
                for i, expiry_name in enumerate(expiries_this_page):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    expiry_data = ticker_data[expiry_name]
                    
                    self._plot_single_putcall_oi_vol(ax, expiry_data, expiry_name)
                
                # Hide unused subplots
                for i in range(len(expiries_this_page), len(axes)):
                    axes[i].set_visible(False)
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                created_files.append(str(plot_path))
                
            except Exception as e:
                self.logger.error(f"Error creating ratios plot page {page+1} for {ticker}: {e}")
                continue
        
        return created_files
    
    def _plot_single_putcall_oi_vol(self, ax, expiry_data: Dict[str, Any], expiry_name: str):
        """
        Plot put-call OI and Volume ratios for a single expiry (with markers).
        
        Args:
            ax: Matplotlib axis object
            expiry_data: Expiry data containing ratio information
            expiry_name: Expiry date name
        """
        try:
            # Get the ratio data (only OI and Volume)
            pc_oi_df = expiry_data.get('putcallOI', pd.DataFrame())
            pc_vol_df = expiry_data.get('putcallVol', pd.DataFrame())
            pc_oi = pc_oi_df['putcallOI'] if not pc_oi_df.empty else pd.Series(dtype=float)
            pc_vol = pc_vol_df['putcallVol'] if not pc_vol_df.empty else pd.Series(dtype=float) 
            
            # Get strike prices from calls or puts
            calls_df = expiry_data.get('calls', pd.DataFrame())
            if not calls_df.empty and 'Strike' in calls_df.columns:
                strikes = calls_df['Strike']
            else:
                puts_df = expiry_data.get('puts', pd.DataFrame())
                if not puts_df.empty and 'Strike' in puts_df.columns:
                    strikes = puts_df['Strike']
                else:
                    ax.text(0.5, 0.5, 'No strike data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(expiry_name, fontsize=10)
                    return
            
            # Ensure all series have the same index
            if not pc_oi.empty:
                pc_oi_values = pc_oi.reindex(strikes.index, fill_value=np.nan)
            else:
                pc_oi_values = pd.Series([np.nan] * len(strikes), index=strikes.index)
                
            if not pc_vol.empty:
                pc_vol_values = pc_vol.reindex(strikes.index, fill_value=np.nan)
            else:
                pc_vol_values = pd.Series([np.nan] * len(strikes), index=strikes.index)
            
            # Filter out NaN and infinite values for plotting
            valid_mask = (np.isfinite(strikes) & 
                         (np.isfinite(pc_oi_values) | 
                          np.isfinite(pc_vol_values)))
            
            if not valid_mask.any():
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(expiry_name, fontsize=10)
                return
            
            strikes_plot = strikes.to_numpy()[valid_mask]
            
            # Plot the ratios with markers only (no lines)
            if np.isfinite(pc_oi_values.to_numpy()[valid_mask]).any():
                ax.plot(strikes_plot, pc_oi_values.to_numpy()[valid_mask], 'bo', 
                       label='P/C OI', markersize=8, alpha=0.7)
            
            if np.isfinite(pc_vol_values.to_numpy()[valid_mask]).any():
                ax.plot(strikes_plot, pc_vol_values.to_numpy()[valid_mask], 'rs', 
                       label='P/C Vol', markersize=8, alpha=0.7)
            
            # Add horizontal line at 1.0 (parity)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
                      label='Parity (1.0)')
            
            # Set labels and formatting
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('Put/Call Ratio')
            ax.set_title(expiry_name, fontsize=10, color='blue')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set reasonable y-axis limits
            all_ratios = np.concatenate([
                pd.Series(pc_oi_values.to_numpy()[valid_mask]).dropna(),
                pd.Series(pc_vol_values.to_numpy()[valid_mask]).dropna()
            ])
            
            if len(all_ratios) > 0:
                finite_ratios = all_ratios[np.isfinite(all_ratios)]
                if len(finite_ratios) > 0:
                    y_min = max(0, np.percentile(finite_ratios, 5) * 0.9)
                    y_max = np.percentile(finite_ratios, 95) * 1.1
                    ax.set_ylim(y_min, y_max)
            
        except Exception as e:
            self.logger.error(f"Error plotting OI/Vol ratios for {expiry_name}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(expiry_name, fontsize=10)
    
    def create_putcall_iv_plots(self, processed_chains: Dict[str, Any], 
                               ticker: str, plots_per_page: int = 6, 
                               plot_rows: int = 2, plot_cols: int = 3) -> List[str]:
        """
        Create put-call IV ratio plots separately.
        
        Args:
            processed_chains: Processed option chain data with ratios
            ticker: Stock ticker symbol
            plots_per_page: Number of plots per page
            plot_rows: Number of rows in subplot grid
            plot_cols: Number of columns in subplot grid
            
        Returns:
            List of created plot file paths
        """
        if ticker not in processed_chains:
            self.logger.warning(f"No processed data for {ticker}")
            return []
        
        ticker_data = processed_chains[ticker]
        expiry_names = list(ticker_data.keys())
        
        if not expiry_names:
            return []
        
        # Create date-specific folder
        date_str = datetime.now().strftime("%y%m%d")
        plot_dir = Path(self.plot_folder) / date_str
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of pages needed
        num_pages = int(np.ceil(len(expiry_names) / plots_per_page))
        created_files = []
        
        for page in range(num_pages):
            try:
                plot_filename = f"{ticker}_putcall_iv_{page+1:02d}.png"
                plot_path = plot_dir / plot_filename
                
                # Create figure with subplots
                fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(18, 12))
                fig.suptitle(f"{ticker} Put-Call IV Ratios", fontsize=16)
                
                # Flatten axes array for easier indexing
                if plot_rows * plot_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                # Determine which expiries to plot on this page
                start_idx = page * plots_per_page
                end_idx = min(start_idx + plots_per_page, len(expiry_names))
                expiries_this_page = expiry_names[start_idx:end_idx]
                
                for i, expiry_name in enumerate(expiries_this_page):
                    if i >= len(axes):
                        break
                        
                    ax = axes[i]
                    expiry_data = ticker_data[expiry_name]
                    
                    self._plot_single_putcall_iv(ax, expiry_data, expiry_name)
                
                # Hide unused subplots
                for i in range(len(expiries_this_page), len(axes)):
                    axes[i].set_visible(False)
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                created_files.append(str(plot_path))
                
            except Exception as e:
                self.logger.error(f"Error creating IV plot page {page+1} for {ticker}: {e}")
                continue
        
        return created_files
    
    def _plot_single_putcall_iv(self, ax, expiry_data: Dict[str, Any], expiry_name: str):
        """
        Plot put-call IV ratio for a single expiry.
        
        Args:
            ax: Matplotlib axis object
            expiry_data: Expiry data containing ratio information
            expiry_name: Expiry date name
        """
        try:
            # Get the IV ratio data
            pc_iv_df = expiry_data.get('putcallIV', pd.DataFrame())
            pc_iv = pc_iv_df['putcallIV'] if not pc_iv_df.empty else pd.Series(dtype=float)
            
            # Get strike prices from calls or puts
            calls_df = expiry_data.get('calls', pd.DataFrame())
            if not calls_df.empty and 'Strike' in calls_df.columns:
                strikes = calls_df['Strike']
            else:
                puts_df = expiry_data.get('puts', pd.DataFrame())
                if not puts_df.empty and 'Strike' in puts_df.columns:
                    strikes = puts_df['Strike']
                else:
                    ax.text(0.5, 0.5, 'No strike data', ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(expiry_name, fontsize=10)
                    return
            
            # Ensure series has the same index
            if not pc_iv.empty:
                pc_iv_values = pc_iv.reindex(strikes.index, fill_value=np.nan)
            else:
                pc_iv_values = pd.Series([np.nan] * len(strikes), index=strikes.index)
            
            # Filter out NaN and infinite values for plotting
            valid_mask = (np.isfinite(strikes) & np.isfinite(pc_iv_values))
            
            if not valid_mask.any():
                ax.text(0.5, 0.5, 'No valid IV data', ha='center', va='center',
                       transform=ax.transAxes)
                ax.set_title(expiry_name, fontsize=10)
                return
            
            strikes_plot = strikes.to_numpy()[valid_mask]
            iv_values_plot = pc_iv_values.to_numpy()[valid_mask]
            
            # Plot the IV ratio with line
            ax.plot(strikes_plot, iv_values_plot, 'g-', 
                   label='P/C IV', linewidth=2, alpha=0.8)
            
            # Add horizontal line at 1.0 (parity)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, 
                      label='Parity (1.0)')
            
            # Set labels and formatting
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('Put/Call IV Ratio')
            ax.set_title(expiry_name, fontsize=10, color='green')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits to 0-10 for IV ratios
            ax.set_ylim(0, 10)
            
        except Exception as e:
            self.logger.error(f"Error plotting IV ratio for {expiry_name}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(expiry_name, fontsize=10)
    
    def create_put_elasticity_plots(self, elasticity_data: Dict[str, pd.DataFrame],
                                  config: 'Config' = None) -> List[str]:
        """
        Create put elasticity plots (replicates op05.r visualization functionality).
        
        This method creates scatter plots and ranking plots for put elasticity analysis,
        exactly as implemented in the R script lines 120-143.
        
        Args:
            elasticity_data: Dictionary of elasticity DataFrames by expiry date
            config: Configuration object with path settings (uses default if None)
            
        Returns:
            List of created plot file paths
        """
        # Import Config here to avoid circular imports
        from config import Config
        
        # Use provided config or create default
        if config is None:
            config = Config()
        
        # Use config's elasticity plots path - no need to create plot_dir separately
        # as config methods handle directory creation
        
        created_files = []
        
        for expiry_date, df in elasticity_data.items():
            if df is None or df.empty:
                continue
            
            try:
                # Create scatter plot (cost vs elasticity colored by IV)
                # Equivalent to R: ggplot(df, aes(x=cost, y=elas, color=IV*4))+geom_point()
                plt.figure(figsize=(18, 7))
                scatter = plt.scatter(df['cost'], df['elas'], c=df['IV']*4, 
                                    alpha=0.7, s=60, cmap='coolwarm')
                
                # Add ticker labels (equivalent to R: geom_text(aes(label=ticker)))
                for ticker in df.index:
                    if ticker in df.index:
                        row = df.loc[ticker]
                        plt.annotate(ticker, 
                                   (row['cost'], row['elas']),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
                
                plt.xlabel('Cost')
                plt.ylabel('Elasticity')
                plt.title(f'PUTS: {expiry_date} On {config.date_string}')
                plt.colorbar(scatter, label='IV')
                plt.grid(True, alpha=0.3)
                
                # Save scatter plot using config path method
                scatter_file = config.get_elasticity_scatter_plot_path(expiry_date)
                plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_files.append(scatter_file)
                
                # Create ranked elasticity plot
                # Equivalent to R: df = df[order(-df$elas),] and subsequent ggplot
                df_sorted = df.sort_values('elas', ascending=False)
                df_sorted['ticker'] = df_sorted.index
                
                plt.figure(figsize=(12, 7))
                plt.plot(range(len(df_sorted)), df_sorted['elas'].to_numpy(), 'bo-', markersize=4)
                plt.xlabel('Ticker')
                plt.ylabel('Elasticity')
                plt.title(f'PUTS: {expiry_date} On {config.date_string}')
                plt.xticks(range(len(df_sorted)), df_sorted['ticker'].to_numpy(), rotation=90)
                plt.grid(True, alpha=0.3)
                
                # Save ranked plot using config path method
                rank_file = config.get_elasticity_ranking_plot_path(expiry_date)
                plt.savefig(rank_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_files.append(rank_file)
                
                
            except Exception as e:
                self.logger.error(f"Error creating elasticity plots for {expiry_date}: {e}")
                continue
        
        return created_files
    
    def create_put_strategy_analysis_plots(self, strategy_data: Dict[str, pd.DataFrame],
                                         analysis_date: str, plot_folder: str = './plots3') -> List[str]:
        """
        Create put strategy analysis plots (replicates op01.r put analysis visualization).
        
        Args:
            strategy_data: Dictionary of strategy DataFrames by expiry date
            analysis_date: Date string for analysis
            plot_folder: Folder for saving plots
            
        Returns:
            List of created plot file paths
        """
        plot_dir = Path(plot_folder) / analysis_date
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        for expiry_date, df in strategy_data.items():
            if df is None or df.empty:
                continue
            
            try:
                # Create scatter plot (cost vs gain_max colored by IV)
                plt.figure(figsize=(18, 7))
                scatter = plt.scatter(df['cost'], df['gain_max'], c=df['IV']*4, 
                                    alpha=0.7, s=60, cmap='coolwarm')
                
                # Add ticker labels
                for _, row in df.iterrows():
                    plt.annotate(row['ticker'], 
                               (row['cost'], row['gain_max']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
                
                plt.xlim(0, 70000)
                plt.xlabel('Cost (Stock Price * 100)')
                plt.ylabel('Max Gain (%)')
                plt.title(f'PUT Strategy Analysis: {expiry_date} on {analysis_date}')
                plt.colorbar(scatter, label='IV')
                plt.grid(True, alpha=0.3)
                
                ratio_file = plot_dir / f"puts_ratio{expiry_date.replace('.', '')}.png"
                plt.savefig(ratio_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_files.append(str(ratio_file))
                
                # Create ranked plot
                df_sorted = df.sort_values('gain_max', ascending=False)
                
                plt.figure(figsize=(12, 7))
                plt.plot(range(len(df_sorted)), df_sorted['gain_max'].to_numpy(), 'bo-', markersize=4)
                plt.xlabel('Ticker Rank')
                plt.ylabel('Max Gain (%)')
                plt.title(f'PUT Strategy Ranking: {expiry_date} on {analysis_date}')
                plt.xticks(range(len(df_sorted)), df_sorted['ticker'].to_numpy(), rotation=90)
                plt.grid(True, alpha=0.3)
                
                rank_file = plot_dir / f"puts_{expiry_date.replace('.', '')}.png"
                plt.savefig(rank_file, dpi=150, bbox_inches='tight')
                plt.close()
                created_files.append(str(rank_file))
                
                
            except Exception as e:
                self.logger.error(f"Error creating strategy plots for {expiry_date}: {e}")
                continue
        
        return created_files
    
    def create_iv_quantile_plot(self, iv_data: pd.Series, title: str,
                               plot_folder: str = './plots3', analysis_date: str = None) -> Optional[str]:
        """
        Create IV quantile plot (replicates R's quantile and plot functionality).
        
        Args:
            iv_data: Series of IV values
            title: Plot title
            plot_folder: Folder for saving plots
            analysis_date: Date string for analysis
            
        Returns:
            Path to created plot file or None
        """
        if iv_data.empty:
            return None
        
        try:
            if analysis_date is None:
                analysis_date = datetime.now().strftime("%y%m%d")
            
            plot_dir = Path(plot_folder) / analysis_date
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate quantiles (equivalent to R: quantile(df$IV, probs=seq(0,1,0.1)))
            quantiles = iv_data.quantile([0.1*i for i in range(11)])
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Plot line chart of IV values
            plt.subplot(2, 1, 1)
            plt.plot(range(len(iv_data)), iv_data.values, 'b-', linewidth=1)
            plt.title(f'{title} - IV Values')
            plt.xlabel('Index')
            plt.ylabel('Implied Volatility')
            plt.grid(True, alpha=0.3)
            
            # Plot quantiles as bar chart
            plt.subplot(2, 1, 2)
            quantile_labels = [f'{int(p*100)}%' for p in quantiles.index]
            plt.bar(quantile_labels, quantiles.values, alpha=0.7)
            plt.title(f'{title} - IV Quantiles')
            plt.xlabel('Quantile')
            plt.ylabel('Implied Volatility')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"iv_analysis_{title.replace(' ', '_').lower()}.png"
            plot_path = plot_dir / plot_filename
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.error(f"Error creating IV quantile plot: {e}")
            return None