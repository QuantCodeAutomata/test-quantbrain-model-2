"""
Sharpe Ratio Calculator

This module provides a function to compute the Sharpe ratio given a list of daily returns
and a risk-free rate.
"""

import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate):
    """
    Calculate the Sharpe ratio given a list of daily returns and a risk-free rate.

    The Sharpe ratio is calculated as:
    Sharpe Ratio = (Mean of returns - Risk-free rate) / Standard deviation of returns

    Args:
        returns (list or array-like): A list of daily returns (as decimals, e.g., 0.01 for 1%).
        risk_free_rate (float): The risk-free rate (as a decimal, e.g., 0.02 for 2%).

    Returns:
        float: The Sharpe ratio.

    Raises:
        ValueError: If returns is empty or if standard deviation is zero.

    Example:
        >>> returns = [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01]
        >>> risk_free_rate = 0.02
        >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
        >>> print(f"Sharpe Ratio: {sharpe:.4f}")
        Sharpe Ratio: 1.2345
    """
    if not returns:
        raise ValueError("Returns list cannot be empty.")

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)

    if std_return == 0:
        raise ValueError("Standard deviation of returns is zero. Cannot calculate Sharpe ratio.")

    sharpe_ratio = (mean_return - risk_free_rate) / std_return
    return sharpe_ratio


# Usage example
if __name__ == "__main__":
    # Example daily returns (as decimals)
    daily_returns = [0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.01]
    risk_free_rate = 0.02  # 2% annual risk-free rate

    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    print(f"Sharpe Ratio: {sharpe:.4f}")
