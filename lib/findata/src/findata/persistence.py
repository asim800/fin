"""
Data persistence utilities for pickle, CSV, and JSON formats.
"""

import os
import pickle
import json
import logging
from typing import Any, Optional
import pandas as pd


def save_pickle(data: Any, path: str) -> bool:
    """
    Save data to pickle file.

    Parameters:
    -----------
    data : Any
        Data to save
    path : str
        Output file path

    Returns:
    --------
    bool: True if successful
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved pickle: {path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save pickle {path}: {e}")
        return False


def load_pickle(path: str) -> Optional[Any]:
    """
    Load data from pickle file.

    Parameters:
    -----------
    path : str
        Input file path

    Returns:
    --------
    Loaded data or None if failed
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Loaded pickle: {path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load pickle {path}: {e}")
        return None


def save_csv(data: pd.DataFrame, path: str, **kwargs) -> bool:
    """
    Save DataFrame to CSV file.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame to save
    path : str
        Output file path
    **kwargs : dict
        Additional arguments to to_csv()

    Returns:
    --------
    bool: True if successful
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data.to_csv(path, **kwargs)
        logging.info(f"Saved CSV: {path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save CSV {path}: {e}")
        return False


def load_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from CSV file.

    Parameters:
    -----------
    path : str
        Input file path
    **kwargs : dict
        Additional arguments to read_csv()

    Returns:
    --------
    pd.DataFrame or None if failed
    """
    try:
        data = pd.read_csv(path, **kwargs)
        logging.info(f"Loaded CSV: {path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load CSV {path}: {e}")
        return None


def save_json(data: dict, path: str, indent: int = 2) -> bool:
    """
    Save dictionary to JSON file.

    Parameters:
    -----------
    data : dict
        Dictionary to save
    path : str
        Output file path
    indent : int
        JSON indentation

    Returns:
    --------
    bool: True if successful
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logging.info(f"Saved JSON: {path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON {path}: {e}")
        return False


def load_json(path: str) -> Optional[dict]:
    """
    Load dictionary from JSON file.

    Parameters:
    -----------
    path : str
        Input file path

    Returns:
    --------
    dict or None if failed
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded JSON: {path}")
        return data
    except Exception as e:
        logging.error(f"Failed to load JSON {path}: {e}")
        return None
