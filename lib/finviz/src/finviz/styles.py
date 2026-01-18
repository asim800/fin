"""
Style management for matplotlib and plotly.
"""

import matplotlib.pyplot as plt
from typing import List

# Default matplotlib color palette
DEFAULT_COLORS: List[str] = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]

# Plotly-compatible color palette
PLOTLY_COLORS: List[str] = [
    'rgb(31, 119, 180)',   # Blue
    'rgb(255, 127, 14)',   # Orange
    'rgb(44, 160, 44)',    # Green
    'rgb(214, 39, 40)',    # Red
    'rgb(148, 103, 189)',  # Purple
    'rgb(140, 86, 75)',    # Brown
    'rgb(227, 119, 194)',  # Pink
    'rgb(127, 127, 127)',  # Gray
    'rgb(188, 189, 34)',   # Olive
    'rgb(23, 190, 207)',   # Cyan
]

# Success rate color thresholds
SUCCESS_COLORS = {
    'high': '#2ca02c',      # Green (>= 80%)
    'medium': '#ff7f0e',    # Orange (>= 60%)
    'low_medium': '#d62728', # Red-orange (>= 40%)
    'low': '#d62728',       # Red (< 40%)
}


def set_style(style: str = 'seaborn-v0_8-whitegrid') -> None:
    """
    Set matplotlib style.

    Parameters:
    -----------
    style : str
        Matplotlib style name
    """
    try:
        plt.style.use(style)
    except OSError:
        # Fallback for older matplotlib versions
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            # Final fallback to default
            plt.style.use('default')


def get_success_color(rate: float) -> str:
    """
    Get color based on success rate.

    Parameters:
    -----------
    rate : float
        Success rate (0 to 1)

    Returns:
    --------
    str: Color hex code
    """
    if rate >= 0.8:
        return SUCCESS_COLORS['high']
    elif rate >= 0.6:
        return SUCCESS_COLORS['medium']
    elif rate >= 0.4:
        return SUCCESS_COLORS['low_medium']
    else:
        return SUCCESS_COLORS['low']


def apply_financial_style() -> None:
    """Apply financial chart styling defaults."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
    })
