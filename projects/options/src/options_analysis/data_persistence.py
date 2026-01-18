"""Data persistence module for saving and loading analysis results."""

import logging
import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd


class DataPersistence:
    """Handles saving and loading of analysis data and results."""
    
    def __init__(self, data_folder: str = None):
        """
        Initialize DataPersistence.
        
        Args:
            data_folder: Folder for saving data files (uses config default if None)
        """
        # Use config's data folder if none provided
        if data_folder is None:
            from .config import Config
            config = Config()
            data_folder = config.data_folder
            
        self.data_folder = Path(data_folder)
        self.data_folder.mkdir(exist_ok=True)
        
        # Create step-specific folders
        self.raw_folder = self.data_folder / "raw"
        self.individual_folder = self.data_folder / "individual"
        self.comprehensive_folder = self.data_folder / "comprehensive"
        
        self.raw_folder.mkdir(exist_ok=True)
        self.individual_folder.mkdir(exist_ok=True)
        self.comprehensive_folder.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def save_analysis_data(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save complete analysis data to pickle file.
        
        This replicates the R save() functionality.
        
        Args:
            data: Dictionary containing all analysis data
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            date_str = datetime.now().strftime("%y%m%d")
            filename = f"oc_{date_str}.pkl"
        
        file_path = self.data_folder / filename
        
        try:
            # Prepare data for saving
            save_data = {
                'timestamp': datetime.now(),
                'version': '1.0',
                **data
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Saved analysis data to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {e}")
            raise
    
    def load_analysis_data(self, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load analysis data from pickle file.
        
        This replicates the R load() functionality.
        
        Args:
            filename: Optional filename to load, defaults to latest
            
        Returns:
            Dictionary with loaded data or None if file not found
        """
        if filename is None:
            # Find the most recent file
            date_str = datetime.now().strftime("%y%m%d")
            filename = f"oc_{date_str}.pkl"
        
        file_path = self.data_folder / filename
        
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.info(f"Loaded analysis data from {file_path}")
            return data
            
        except (ImportError, ModuleNotFoundError) as e:
            if "numpy._core" in str(e) or "numpy.core" in str(e):
                self.logger.warning(f"Numpy version compatibility issue with {file_path}. "
                                  f"Removing incompatible cached file: {e}")
                try:
                    file_path.unlink()
                    self.logger.info(f"Removed incompatible cached file: {file_path}")
                except Exception as rm_e:
                    self.logger.error(f"Failed to remove incompatible file: {rm_e}")
                return None
            else:
                self.logger.error(f"Import error loading data from {file_path}: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            return None
    
    def save_option_chains(self, option_chains: Dict[str, Any], 
                          filename: Optional[str] = None) -> str:
        """
        Save option chains data specifically.
        
        Args:
            option_chains: Option chains dictionary
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            date_str = datetime.now().strftime("%y%m%d")
            filename = f"option_chains_{date_str}.pkl"
        
        return self.save_analysis_data({'option_chains': option_chains}, filename)
    
    def save_prices(self, prices: Dict[str, float], price_df: pd.DataFrame,
                   filename: Optional[str] = None) -> str:
        """
        Save price data.
        
        Args:
            prices: Price dictionary
            price_df: Price DataFrame
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            date_str = datetime.now().strftime("%y%m%d")
            filename = f"prices_{date_str}.pkl"
        
        price_data = {
            'prices': prices,
            'price_df': price_df,
            'timestamp': datetime.now()
        }
        
        return self.save_analysis_data(price_data, filename)
    
    def save_pcp_results(self, pcp_results: Dict[str, Any], 
                        filename: Optional[str] = None) -> str:
        """
        Save put-call parity results.
        
        Args:
            pcp_results: Put-call parity analysis results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            date_str = datetime.now().strftime("%y%m%d")
            filename = f"pcp_results_{date_str}.pkl"
        
        return self.save_analysis_data({'pcp_results': pcp_results}, filename)
    
    def save_market_data(self, market_data: Dict[str, Any], date_str: Optional[str] = None) -> str:
        """
        Save Step 1 market data to raw folder.
        
        Args:
            market_data: Dictionary containing option chains, prices, etc.
            date_str: Optional date string, defaults to today
            
        Returns:
            Path to saved file
        """
        if date_str is None:
            date_str = datetime.now().strftime("%y%m%d")
            
        filename = f"market_data_{date_str}.pkl"
        file_path = self.raw_folder / filename
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    **market_data
                }, f)
            
            self.logger.info(f"Saved market data to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving market data: {e}")
            raise
    
    def save_individual_results(self, individual_results: Dict[str, Any], date_str: Optional[str] = None) -> str:
        """
        Save Step 2 individual analysis results to individual folder.
        
        Args:
            individual_results: Dictionary containing individual analysis results
            date_str: Optional date string, defaults to today
            
        Returns:
            Path to saved file
        """
        if date_str is None:
            date_str = datetime.now().strftime("%y%m%d")
            
        filename = f"individual_results_{date_str}.pkl"
        file_path = self.individual_folder / filename
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    **individual_results
                }, f)
            
            self.logger.info(f"Saved individual results to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving individual results: {e}")
            raise
    
    def save_comprehensive_results(self, comprehensive_results: Dict[str, Any], date_str: Optional[str] = None) -> str:
        """
        Save Step 3 comprehensive analysis results to comprehensive folder.
        
        Args:
            comprehensive_results: Dictionary containing comprehensive analysis results
            date_str: Optional date string, defaults to today
            
        Returns:
            Path to saved file
        """
        if date_str is None:
            date_str = datetime.now().strftime("%y%m%d")
            
        filename = f"comprehensive_results_{date_str}.pkl"
        file_path = self.comprehensive_folder / filename
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    **comprehensive_results
                }, f)
            
            self.logger.info(f"Saved comprehensive results to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving comprehensive results: {e}")
            raise
    
    def load_distributed_analysis_data(self, date_str: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load analysis data from distributed step-specific files and merge.
        
        Args:
            date_str: Optional date string, defaults to today
            
        Returns:
            Dictionary with merged data from all steps or None if not found
        """
        if date_str is None:
            date_str = datetime.now().strftime("%y%m%d")
        
        # Define file paths
        market_file = self.raw_folder / f"market_data_{date_str}.pkl"
        individual_file = self.individual_folder / f"individual_results_{date_str}.pkl"
        comprehensive_file = self.comprehensive_folder / f"comprehensive_results_{date_str}.pkl"
        
        # Check if all files exist
        if not all([market_file.exists(), individual_file.exists(), comprehensive_file.exists()]):
            missing_files = []
            if not market_file.exists(): missing_files.append(str(market_file))
            if not individual_file.exists(): missing_files.append(str(individual_file))
            if not comprehensive_file.exists(): missing_files.append(str(comprehensive_file))
            
            self.logger.warning(f"Missing step-specific files: {missing_files}")
            return None
        
        try:
            # Load all step files
            with open(market_file, 'rb') as f:
                market_data = pickle.load(f)
            
            with open(individual_file, 'rb') as f:
                individual_data = pickle.load(f)
                
            with open(comprehensive_file, 'rb') as f:
                comprehensive_data = pickle.load(f)
            
            # Merge data (similar to _merge_all_results format)
            merged_data = {
                'option_chains': market_data.get('option_chains', {}),
                'prices': market_data.get('prices', {}),
                'tickers': market_data.get('tickers', []),
                'processed_chains': individual_data.get('processed_chains', {}),
                'pcp_results': individual_data.get('pcp_results', {}),
                'plot_files': individual_data.get('plot_files', []),
                'arbitrage': comprehensive_data.get('arbitrage_opportunities'),
                'summary': comprehensive_data.get('summary_statistics', {}),
                'timestamp': comprehensive_data.get('timestamp', datetime.now()),
                
                # Include original structured data
                'market_data': market_data,
                'individual_results': individual_data,
                'comprehensive_results': comprehensive_data
            }
            
            self.logger.info(f"Loaded distributed analysis data for {date_str}")
            return merged_data
            
        except Exception as e:
            self.logger.error(f"Error loading distributed analysis data: {e}")
            return None
    
    def export_to_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Export data to JSON format for interoperability.
        
        Args:
            data: Data to export
            filename: JSON filename
            
        Returns:
            Path to exported file
        """
        file_path = self.data_folder / filename
        
        try:
            # Convert non-JSON serializable objects
            json_data = self._prepare_for_json(data)
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported data to JSON: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON {file_path}: {e}")
            raise
    
    def export_to_csv(self, dataframes: Dict[str, pd.DataFrame], 
                     base_filename: str) -> List[str]:
        """
        Export DataFrames to CSV files.
        
        Args:
            dataframes: Dictionary of DataFrames to export
            base_filename: Base filename (will append keys)
            
        Returns:
            List of exported file paths
        """
        exported_files = []
        
        for key, df in dataframes.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            
            filename = f"{base_filename}_{key}.csv"
            file_path = self.data_folder / filename
            
            try:
                df.to_csv(file_path, index=True)
                exported_files.append(str(file_path))
                self.logger.info(f"Exported {key} to CSV: {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error exporting {key} to CSV: {e}")
                continue
        
        return exported_files
    
    def _prepare_for_json(self, data: Any) -> Any:
        """
        Recursively prepare data for JSON serialization.
        
        Args:
            data: Data to prepare
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            return {
                'type': 'DataFrame',
                'data': data.to_dict('records'),
                'index': data.index.tolist(),
                'columns': data.columns.tolist()
            }
        elif isinstance(data, pd.Series):
            return {
                'type': 'Series',
                'data': data.to_dict(),
                'index': data.index.tolist()
            }
        elif isinstance(data, (datetime, pd.Timestamp)):
            return data.isoformat()
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        else:
            # Convert other types to string
            return str(data)
    
    def get_available_files(self, pattern: str = "*.pkl") -> List[str]:
        """
        Get list of available data files.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of available file paths
        """
        try:
            files = list(self.data_folder.glob(pattern))
            return [str(f) for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)]
        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            return []
    
    def cleanup_old_files(self, days_old: int = 30, pattern: str = "*.pkl") -> int:
        """
        Remove data files older than specified days.
        
        Args:
            days_old: Number of days old for cleanup threshold
            pattern: File pattern to match
            
        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
        deleted_count = 0
        
        try:
            for file_path in self.data_folder.glob(pattern):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted old file: {file_path}")
            
            self.logger.info(f"Cleaned up {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return 0
    
    def backup_data(self, backup_folder: str) -> str:
        """
        Create backup of all data files.
        
        Args:
            backup_folder: Folder for backup
            
        Returns:
            Path to backup folder
        """
        import shutil
        
        backup_path = Path(backup_folder)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_backup = backup_path / f"backup_{timestamp}"
        
        try:
            timestamped_backup.mkdir(parents=True, exist_ok=True)
            
            # Copy all files from data folder
            for file_path in self.data_folder.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, timestamped_backup)
            
            self.logger.info(f"Backup created: {timestamped_backup}")
            return str(timestamped_backup)
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise
    
    def get_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a data file.
        
        Args:
            filename: Name of file to inspect
            
        Returns:
            Dictionary with file information or None
        """
        file_path = self.data_folder / filename
        
        if not file_path.exists():
            return None
        
        try:
            stat = file_path.stat()
            info = {
                'filename': filename,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'is_pickle': filename.endswith('.pkl'),
                'is_json': filename.endswith('.json'),
                'is_csv': filename.endswith('.csv')
            }
            
            # Try to get data info for pickle files
            if info['is_pickle']:
                try:
                    data = self.load_analysis_data(filename)
                    if data:
                        info['data_keys'] = list(data.keys())
                        info['data_timestamp'] = data.get('timestamp')
                except:
                    pass
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {filename}: {e}")
            return None