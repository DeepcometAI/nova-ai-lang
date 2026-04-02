"""
cosmos.stats — Scientific statistics constellation
"""

from .stats import (
    pearson, spearman, linear_fit, polyfit,
    mean, median, std, variance, quantile,
    min, max, sum, product
)

__all__ = [
    'pearson', 'spearman', 'linear_fit', 'polyfit',
    'mean', 'median', 'std', 'variance', 'quantile',
    'min', 'max', 'sum', 'product'
]
