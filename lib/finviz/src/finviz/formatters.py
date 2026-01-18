"""
Formatting utilities for currency, percentages, and numbers.
"""


def format_currency(value: float, prefix: str = '$', decimals: int = 0) -> str:
    """
    Format value as currency string.

    Parameters:
    -----------
    value : float
        Value to format
    prefix : str
        Currency prefix
    decimals : int
        Decimal places for small values

    Returns:
    --------
    str: Formatted currency string
    """
    if abs(value) >= 1e9:
        return f'{prefix}{value/1e9:.1f}B'
    elif abs(value) >= 1e6:
        return f'{prefix}{value/1e6:.1f}M'
    elif abs(value) >= 1e3:
        return f'{prefix}{value/1e3:.0f}K'
    else:
        if decimals > 0:
            return f'{prefix}{value:.{decimals}f}'
        return f'{prefix}{value:.0f}'


def format_percentage(value: float, decimals: int = 1, as_ratio: bool = True) -> str:
    """
    Format value as percentage string.

    Parameters:
    -----------
    value : float
        Value to format
    decimals : int
        Number of decimal places
    as_ratio : bool
        If True, value is a ratio (0.05 = 5%)
        If False, value is already a percentage (5.0 = 5%)

    Returns:
    --------
    str: Formatted percentage string
    """
    if as_ratio:
        return f'{value*100:.{decimals}f}%'
    return f'{value:.{decimals}f}%'


def format_number(value: float, decimals: int = 2, with_sign: bool = False) -> str:
    """
    Format number with specified decimals.

    Parameters:
    -----------
    value : float
        Value to format
    decimals : int
        Number of decimal places
    with_sign : bool
        Whether to include + for positive values

    Returns:
    --------
    str: Formatted number string
    """
    if with_sign and value > 0:
        return f'+{value:.{decimals}f}'
    return f'{value:.{decimals}f}'


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format a ratio (e.g., Sharpe ratio).

    Parameters:
    -----------
    value : float
        Ratio value
    decimals : int
        Number of decimal places

    Returns:
    --------
    str: Formatted ratio string
    """
    if abs(value) > 100:
        return f'{value:.0f}'
    return f'{value:.{decimals}f}'


def format_date(date, fmt: str = '%Y-%m-%d') -> str:
    """
    Format date object to string.

    Parameters:
    -----------
    date : datetime-like
        Date to format
    fmt : str
        Format string

    Returns:
    --------
    str: Formatted date string
    """
    try:
        return date.strftime(fmt)
    except AttributeError:
        return str(date)
