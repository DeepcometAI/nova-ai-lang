"""
cosmos.plot — Scientific visualization constellation
"""

from .plot import (
    scatter, histogram, plot, regression_line, hr_diagram,
    heatmap, bar_chart, legend,
    set_xlabel, set_ylabel, set_title, save, show,
    _ensure_fig
)

__all__ = [
    'scatter', 'histogram', 'plot', 'regression_line', 'hr_diagram',
    'heatmap', 'bar_chart', 'legend',
    'set_xlabel', 'set_ylabel', 'set_title', 'save', 'show',
    '_ensure_fig'
]
