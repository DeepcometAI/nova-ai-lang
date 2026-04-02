"""
cosmos.data — Data loading and I/O constellation
Handles CSV, FITS, Parquet, Arrow formats with Wave (lazy) semantics.
"""

import pandas as pd
import numpy as np
from typing import Iterator, List, Callable, Any, Optional
import os

# Try to import polars for Arrow support (optional)
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

__all__ = [
    'read_csv', 'read_fits', 'read_parquet', 'read_arrow',
    'write_csv', 'write_parquet',
    'Wave'
]


class Wave:
    """
    Lazy sequence of records (pull-based iterator).
    Similar to Rust Iterator or Python generator.
    """
    def __init__(self, generator: Iterator[Any]):
        self.generator = generator
    
    def __iter__(self):
        return self.generator
    
    def filter(self, predicate: Callable[[Any], bool]) -> 'Wave':
        """Filter elements passing predicate."""
        return Wave(x for x in self.generator if predicate(x))
    
    def map(self, transform: Callable[[Any], Any]) -> 'Wave':
        """Apply transformation to each element."""
        return Wave(transform(x) for x in self.generator)
    
    def collect(self) -> List[Any]:
        """Collect all elements into a list (forces evaluation)."""
        return list(self.generator)
    
    def batch(self, size: int) -> 'Wave':
        """Batch elements into groups of `size`."""
        def batch_gen():
            batch_list = []
            for item in self.generator:
                batch_list.append(item)
                if len(batch_list) == size:
                    yield batch_list
                    batch_list = []
            if batch_list:
                yield batch_list
        return Wave(batch_gen())


def read_csv(path: str) -> Wave:
    """
    Read CSV file lazily.
    
    Args:
        path: Path to CSV file
    
    Returns:
        Wave of records (rows as dicts)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    def csv_generator():
        df = pd.read_csv(path)
        for record in df.to_dict(orient='records'):
            yield record
    
    return Wave(csv_generator())


def read_fits(path: str) -> Wave:
    """
    Read FITS file lazily.
    Requires astropy.io.fits (optional)
    
    Args:
        path: Path to FITS file (or CSV as fallback)
    
    Returns:
        Wave of records (table rows)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Try astropy first, fallback to CSV if FITS file not readable
    try:
        from astropy.io import fits
        
        def fits_generator():
            try:
                with fits.open(path) as hdul:
                    # Assume first HDU has table data
                    table = hdul[1].data
                    for row in table:
                        yield dict(zip(table.names, row))
            except (IndexError, TypeError):
                # If FITS table not readable, try as CSV
                df = pd.read_csv(path)
                for record in df.to_dict(orient='records'):
                    yield record
        
        return Wave(fits_generator())
    except ImportError:
        # Fallback: treat as CSV
        def csv_fallback_generator():
            if path.endswith('.fits') or path.endswith('.fit'):
                raise ImportError(
                    f"Cannot read FITS file {path}. "
                    "Install astropy: pip install astropy pyerfa --only-binary :all: "
                    "Or convert to CSV format instead."
                )
            df = pd.read_csv(path)
            for record in df.to_dict(orient='records'):
                yield record
        
        return Wave(csv_fallback_generator())


def read_parquet(path: str) -> Wave:
    """
    Read Parquet file lazily.
    
    Args:
        path: Path to Parquet file
    
    Returns:
        Wave of records
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    def parquet_generator():
        df = pd.read_parquet(path)
        for record in df.to_dict(orient='records'):
            yield record
    
    return Wave(parquet_generator())


def read_arrow(path: str) -> Wave:
    """
    Read Arrow file lazily (Feather or Parquet via Polars).
    
    Args:
        path: Path to Arrow/Feather file
    
    Returns:
        Wave of records
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    if not HAS_POLARS:
        raise ImportError(
            f"Cannot read Arrow file {path}. "
            "Install polars: pip install polars"
        )
    
    def arrow_generator():
        # Polars can read both Feather (.arrow) and Parquet formats
        if path.endswith('.feather') or path.endswith('.arrow'):
            df = pl.read_ipc(path)
        else:
            # Fallback to parquet
            df = pl.read_parquet(path)
        # Convert to pandas for record iteration
        pdf = df.to_pandas()
        for record in pdf.to_dict(orient='records'):
            yield record
    
    return Wave(arrow_generator())


def write_csv(data: List[dict], path: str) -> None:
    """
    Write records to CSV file.
    
    Args:
        data: List of records (dicts)
        path: Output path
    """
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def write_parquet(data: List[dict], path: str) -> None:
    """
    Write records to Parquet file.
    
    Args:
        data: List of records (dicts)
        path: Output path
    """
    df = pd.DataFrame(data)
    df.to_parquet(path, index=False)


def filter_wave(predicate: Callable[[Any], bool], data: Wave) -> Wave:
    """Filter Wave by predicate (curried for pipeline use)."""
    return data.filter(predicate)


def map_wave(transform: Callable[[Any], Any], data: Wave) -> Wave:
    """Map transformation over Wave (curried for pipeline use)."""
    return data.map(transform)


def collect_wave(data: Wave) -> List[Any]:
    """Collect Wave into list."""
    return data.collect()


def batch_wave(size: int, data: Wave) -> Wave:
    """Batch Wave into groups (curried for pipeline use)."""
    return data.batch(size)
