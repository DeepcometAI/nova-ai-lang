"""
cosmos.stats — Statistical analysis library
Unit-aware correlation, regression, and statistical functions.

All return types are dimensionless (Float[1]) at the type level.
At runtime, this module returns raw Python floats.
The NOVA compiler guarantees unit correctness.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from typing import Tuple, List

__all__ = [
    'pearson', 'spearman', 'linear_fit', 'polyfit',
    'mean', 'median', 'std', 'variance', 'quantile',
    'min', 'max', 'sum', 'product'
]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation coefficient between x and y.
    
    Args:
        x: First array
        y: Second array (must be same length as x)
    
    Returns:
        Correlation coefficient in [-1, 1] (Float[1] in NOVA)
        
    Raises:
        ValueError: If arrays have different lengths
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) != len(y):
        raise ValueError(f"Array length mismatch: {len(x)} vs {len(y)}")
    
    r, _ = pearsonr(x, y)
    return float(r)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman rank correlation coefficient between x and y.
    
    Args:
        x: First array
        y: Second array (must be same length as x)
    
    Returns:
        Rank correlation coefficient in [-1, 1] (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) != len(y):
        raise ValueError(f"Array length mismatch: {len(x)} vs {len(y)}")
    
    rho, _ = spearmanr(x, y)
    return float(rho)


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Linear regression: fit y = slope * x + intercept
    
    Args:
        x: Independent variable
        y: Dependent variable
    
    Returns:
        (slope, intercept) as a tuple of Float[1] in NOVA
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) != len(y):
        raise ValueError(f"Array length mismatch: {len(x)} vs {len(y)}")
    
    result = linregress(x, y)
    return (float(result.slope), float(result.intercept))


def polyfit(x: np.ndarray, y: np.ndarray, degree: int) -> List[float]:
    """
    Polynomial fit: fit y = p[0]*x^n + p[1]*x^(n-1) + ... + p[n]
    
    Args:
        x: Independent variable
        y: Dependent variable
        degree: Polynomial degree
    
    Returns:
        Array of coefficients (Array[Float[1]] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    if len(x) != len(y):
        raise ValueError(f"Array length mismatch: {len(x)} vs {len(y)}")
    
    coeffs = np.polyfit(x, y, degree)
    return [float(c) for c in coeffs]


def mean(x: np.ndarray) -> float:
    """
    Arithmetic mean of x.
    
    Args:
        x: Array
    
    Returns:
        Mean value (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x))


def median(x: np.ndarray) -> float:
    """
    Median of x.
    
    Args:
        x: Array
    
    Returns:
        Median value (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(x))


def std(x: np.ndarray) -> float:
    """
    Standard deviation of x (population std, ddof=0).
    
    Args:
        x: Array
    
    Returns:
        Standard deviation (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x, ddof=0))


def variance(x: np.ndarray) -> float:
    """
    Variance of x (population variance, ddof=0).
    
    Args:
        x: Array
    
    Returns:
        Variance (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.var(x, ddof=0))


def quantile(x: np.ndarray, q: float) -> float:
    """
    Quantile of x at probability q.
    
    Args:
        x: Array
        q: Quantile in [0, 1] (Float[1] in NOVA)
    
    Returns:
        Quantile value (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    q = float(q)
    
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"Quantile must be in [0, 1], got {q}")
    
    return float(np.quantile(x, q))


def min(x: np.ndarray) -> float:
    """
    Minimum value in x.
    
    Args:
        x: Array
    
    Returns:
        Minimum (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.min(x))


def max(x: np.ndarray) -> float:
    """
    Maximum value in x.
    
    Args:
        x: Array
    
    Returns:
        Maximum (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(x))


def sum(x: np.ndarray) -> float:
    """
    Sum of all elements in x.
    
    Args:
        x: Array
    
    Returns:
        Sum (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x))


def product(x: np.ndarray) -> float:
    """
    Product of all elements in x.
    
    Args:
        x: Array
    
    Returns:
        Product (Float[1] in NOVA)
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.prod(x))
