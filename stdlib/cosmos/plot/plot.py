"""
cosmos.plot — Scientific visualization constellation
Matplotlib backend for plotting astronomical and scientific data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

__all__ = [
    'scatter', 'histogram', 'plot', 'regression_line', 'hr_diagram',
    'heatmap', 'bar_chart', 'legend',
    'set_xlabel', 'set_ylabel', 'set_title', 'save', 'show'
]

# Global figure and axes for stateful API
_current_fig = None
_current_ax = None


def _ensure_fig():
    """Ensure we have a current figure and axes."""
    global _current_fig, _current_ax
    if _current_fig is None:
        _current_fig, _current_ax = plt.subplots(figsize=(10, 6))
    return _current_fig, _current_ax


def scatter(x: np.ndarray, y: np.ndarray, 
           title: str = "", xlabel: str = "", ylabel: str = "",
           color: Optional[np.ndarray] = None, s: int = 50) -> None:
    """
    Scatter plot.
    
    Args:
        x: X coordinates (Array[Float[1]])
        y: Y coordinates (Array[Float[1]])
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Optional color array for each point
        s: Marker size
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    fig, ax = _ensure_fig()
    
    if color is not None:
        color = np.asarray(color, dtype=np.float64)
        ax.scatter(x, y, c=color, s=s, cmap='viridis', alpha=0.6)
    else:
        ax.scatter(x, y, s=s, alpha=0.6)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def histogram(data: np.ndarray, bins: int = 20,
             title: str = "", xlabel: str = "", ylabel: str = "Frequency") -> None:
    """
    Histogram.
    
    Args:
        data: Data values (Array[Float[1]])
        bins: Number of bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    data = np.asarray(data, dtype=np.float64)
    
    fig, ax = _ensure_fig()
    ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def plot(x: np.ndarray, y: np.ndarray,
        title: str = "", xlabel: str = "", ylabel: str = "",
        line_style: str = "-", color: str = "blue", label: Optional[str] = None) -> None:
    """
    Line plot.
    
    Args:
        x: X coordinates
        y: Y coordinates
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        line_style: '-', '--', '-.', ':'
        color: Line color name or hex
        label: Legend label
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    fig, ax = _ensure_fig()
    ax.plot(x, y, linestyle=line_style, color=color, label=label, linewidth=2)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def regression_line(slope: float, intercept: float,
                   x_range: Tuple[float, float] = (-1, 1),
                   color: str = "red", label: str = "Linear fit") -> None:
    """
    Overlay a regression line: y = slope * x + intercept
    
    Args:
        slope: Regression slope
        intercept: Y-intercept
        x_range: (x_min, x_max) for line endpoints
        color: Line color
        label: Legend label
    """
    slope = float(slope)
    intercept = float(intercept)
    x_min, x_max = float(x_range[0]), float(x_range[1])
    
    fig, ax = _ensure_fig()
    x_line = np.array([x_min, x_max])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=2, label=label)


def hr_diagram(magnitude: np.ndarray, color_index: np.ndarray,
              title: str = "Hertzsprung-Russell Diagram") -> None:
    """
    Plot a Hertzsprung-Russell (H-R) diagram.
    X-axis: Color index (e.g., B-V)
    Y-axis: Absolute magnitude (inverted, brighter = lower)
    
    Args:
        magnitude: Absolute magnitudes
        color_index: Color indices (B-V, etc.)
        title: Plot title
    """
    magnitude = np.asarray(magnitude, dtype=np.float64)
    color_index = np.asarray(color_index, dtype=np.float64)
    
    fig, ax = _ensure_fig()
    # Invert Y-axis (brightness convention: higher magnitude = fainter = higher on plot)
    ax.scatter(color_index, magnitude, alpha=0.6, s=50)
    ax.invert_yaxis()
    
    ax.set_xlabel("Color Index (B-V)")
    ax.set_ylabel("Absolute Magnitude")
    ax.set_title(title)


def heatmap(data: np.ndarray, title: str = "",
           cmap: str = "viridis", vmin: Optional[float] = None,
           vmax: Optional[float] = None) -> None:
    """
    Plot a 2D heatmap.
    
    Args:
        data: 2D array (Tensor[Float[1], _])
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
    """
    data = np.asarray(data, dtype=np.float64)
    
    fig, ax = _ensure_fig()
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    
    if title:
        ax.set_title(title)


def bar_chart(labels: List[str], values: np.ndarray,
             title: str = "", ylabel: str = "") -> None:
    """
    Vertical bar chart.
    
    Args:
        labels: Category labels
        values: Bar heights
        title: Plot title
        ylabel: Y-axis label
    """
    values = np.asarray(values, dtype=np.float64)
    
    fig, ax = _ensure_fig()
    ax.bar(labels, values, alpha=0.7, edgecolor='black')
    
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def legend(labels: Optional[List[str]] = None, loc: str = "best") -> None:
    """
    Display legend.
    
    Args:
        labels: Label list (if None, uses auto labels from plot commands)
        loc: Legend location
    """
    fig, ax = _ensure_fig()
    ax.legend(loc=loc)


def set_xlabel(label: str) -> None:
    """Set X-axis label."""
    fig, ax = _ensure_fig()
    ax.set_xlabel(label)


def set_ylabel(label: str) -> None:
    """Set Y-axis label."""
    fig, ax = _ensure_fig()
    ax.set_ylabel(label)


def set_title(title: str) -> None:
    """Set plot title."""
    fig, ax = _ensure_fig()
    ax.set_title(title)


def save(filename: str, dpi: int = 150, bbox_inches: str = "tight") -> None:
    """
    Save plot to file.
    
    Args:
        filename: Output path (.png, .pdf, .svg, etc.)
        dpi: Resolution
        bbox_inches: 'tight' removes whitespace
    """
    global _current_fig
    if _current_fig is not None:
        _current_fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)


def show() -> None:
    """Display plot in window."""
    global _current_fig
    if _current_fig is not None:
        plt.show()
