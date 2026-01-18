"""
finviz - Financial visualization utilities.

This package provides:
- Chart utilities (fan charts, heatmaps)
- Formatting utilities (currency, percentage)
- Style management (color palettes, themes)
"""

from finviz.charts import (
    create_figure,
    save_figure,
    add_grid,
    add_legend,
    get_color_palette,
)
from finviz.formatters import (
    format_currency,
    format_percentage,
    format_number,
)
from finviz.styles import (
    set_style,
    DEFAULT_COLORS,
    PLOTLY_COLORS,
)

__all__ = [
    # Charts
    'create_figure',
    'save_figure',
    'add_grid',
    'add_legend',
    'get_color_palette',
    # Formatters
    'format_currency',
    'format_percentage',
    'format_number',
    # Styles
    'set_style',
    'DEFAULT_COLORS',
    'PLOTLY_COLORS',
]
