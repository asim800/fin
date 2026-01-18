"""
Chart creation and manipulation utilities.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional


def create_figure(nrows: int = 1, ncols: int = 1,
                  figsize: Optional[Tuple[float, float]] = None,
                  **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with subplots.

    Parameters:
    -----------
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    figsize : tuple, optional
        Figure size (width, height)
    **kwargs : dict
        Additional arguments to plt.subplots

    Returns:
    --------
    tuple: (fig, axes)
    """
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def save_figure(fig: plt.Figure, path: str, dpi: int = 300, **kwargs) -> None:
    """
    Save figure to file.

    Parameters:
    -----------
    fig : plt.Figure
        Figure to save
    path : str
        Output path
    dpi : int
        Resolution
    **kwargs : dict
        Additional arguments to savefig
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"Saved: {path}")


def add_grid(ax: plt.Axes, alpha: float = 0.3) -> None:
    """Add grid to axes."""
    ax.grid(True, alpha=alpha)


def add_legend(ax: plt.Axes, loc: str = 'best', **kwargs) -> None:
    """Add legend to axes."""
    ax.legend(loc=loc, **kwargs)


def get_color_palette(n_colors: int) -> List[str]:
    """
    Get a color palette with specified number of colors.

    Parameters:
    -----------
    n_colors : int
        Number of colors needed

    Returns:
    --------
    list: List of color hex codes
    """
    from finviz.styles import DEFAULT_COLORS

    if n_colors <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:n_colors]

    # Generate more colors using colormap
    cmap = plt.cm.get_cmap('tab20')
    return [cmap(i / n_colors) for i in range(n_colors)]


def plot_fan_chart(ax: plt.Axes, dates, values: np.ndarray,
                   percentiles: List[int] = [5, 25, 50, 75, 95],
                   colors: Optional[List[str]] = None,
                   alpha: float = 0.3) -> None:
    """
    Plot fan chart showing percentile bands.

    Parameters:
    -----------
    ax : plt.Axes
        Axes to plot on
    dates : array-like
        X-axis values (dates or periods)
    values : np.ndarray
        Values array (n_simulations, n_periods)
    percentiles : list
        Percentiles to plot
    colors : list, optional
        Colors for bands
    alpha : float
        Band transparency
    """
    if colors is None:
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    pct_values = np.percentile(values, percentiles, axis=0)

    # Plot bands (outer to inner)
    n_bands = len(percentiles) // 2
    for i in range(n_bands):
        lower_idx = i
        upper_idx = len(percentiles) - 1 - i
        color_idx = min(i, len(colors) - 1)
        ax.fill_between(dates, pct_values[lower_idx], pct_values[upper_idx],
                        alpha=alpha, color=colors[color_idx],
                        label=f'{percentiles[lower_idx]}-{percentiles[upper_idx]}%')

    # Plot median line
    if len(percentiles) % 2 == 1:
        median_idx = len(percentiles) // 2
        ax.plot(dates, pct_values[median_idx], 'k-', linewidth=2,
                label=f'{percentiles[median_idx]}% (Median)')


def plot_heatmap(ax: plt.Axes, data: np.ndarray, row_labels: List[str],
                 col_labels: List[str], cmap: str = 'coolwarm',
                 fmt: str = '.2f', annot: bool = True) -> None:
    """
    Plot a heatmap.

    Parameters:
    -----------
    ax : plt.Axes
        Axes to plot on
    data : np.ndarray
        2D array of values
    row_labels : list
        Labels for rows
    col_labels : list
        Labels for columns
    cmap : str
        Colormap name
    fmt : str
        Format string for annotations
    annot : bool
        Whether to annotate cells
    """
    im = ax.imshow(data, cmap=cmap, aspect='auto')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if annot:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, format(data[i, j], fmt),
                               ha="center", va="center", color="black")

    plt.colorbar(im, ax=ax)
