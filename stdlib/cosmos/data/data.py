"""
cosmos.data — Data loading and I/O constellation
Handles CSV, FITS, Parquet, Arrow formats with Wave (lazy) semantics.
"""

import pandas as pd
import numpy as np
from typing import Iterator, List, Callable, Any, Optional, Iterable
import os

# Try to import polars for Arrow support (optional)
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

__all__ = [
    'read_csv', 'read_fits', 'read_parquet', 'read_arrow',
    'read_hdf5', 'read_netcdf',
    'write_csv', 'write_parquet',
    'Wave',
    'pipeline', 'filter', 'map', 'collect', 'batch', 'drop_outliers', 'sort_by'
]


class Wave:
    """
    Lazy sequence of records (pull-based iterator).
    Similar to Rust Iterator or Python generator.
    """
    def __init__(self, generator: Iterator[Any] | Iterable[Any]):
        # The integration tests expect a Wave to be reusable (you can call
        # filter/map/collect multiple times). A bare Python generator is
        # single-use, so we cache materialized values on first use.
        self._source = generator
        self._cache: Optional[List[Any]] = None
    
    def __iter__(self):
        return iter(self._materialize())

    def _materialize(self) -> List[Any]:
        if self._cache is None:
            self._cache = list(self._source)
        return self._cache
    
    def filter(self, predicate: Callable[[Any], bool]) -> 'Wave':
        """Filter elements passing predicate."""
        return Wave(x for x in self._materialize() if predicate(x))
    
    def map(self, transform: Callable[[Any], Any]) -> 'Wave':
        """Apply transformation to each element."""
        return Wave(transform(x) for x in self._materialize())
    
    def collect(self) -> List[Any]:
        """Collect all elements into a list (forces evaluation)."""
        return list(self._materialize())
    
    def batch(self, size: int) -> 'Wave':
        """Batch elements into groups of `size`."""
        def batch_gen():
            batch_list = []
            for item in self._materialize():
                batch_list.append(item)
                if len(batch_list) == size:
                    yield batch_list
                    batch_list = []
            if batch_list:
                yield batch_list
        return Wave(batch_gen())

    def sort_by(self, key: Callable[[Any], Any], reverse: bool = False) -> "Wave":
        return Wave(sorted(self._materialize(), key=key, reverse=reverse))

    def drop_outliers(self, field: str, sigma: float = 3.0) -> "Wave":
        data = self._materialize()
        vals = np.array([float(x[field]) for x in data], dtype=np.float64)
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=0))
        if sd == 0.0:
            return Wave(list(data))
        lo = mu - float(sigma) * sd
        hi = mu + float(sigma) * sd
        return Wave([x for x, v in zip(data, vals) if lo <= v <= hi])


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


def read_hdf5(path: str, dataset: str) -> Wave:
    """
    Read an HDF5 dataset into a Wave of records.

    Requires `h5py`. For table-like datasets, each row is returned as a dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required for HDF5 support: pip install h5py") from e

    def gen():
        with h5py.File(path, "r") as f:
            ds = f[dataset]
            arr = ds[()]
            # structured array -> dict rows
            if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
                names = list(arr.dtype.names)
                for row in arr:
                    yield {n: row[n].item() if hasattr(row[n], "item") else row[n] for n in names}
            else:
                for x in np.asarray(arr):
                    yield x

    return Wave(gen())


def read_netcdf(path: str, variable: str) -> Wave:
    """
    Read a NetCDF variable into a Wave of values.

    Requires `netCDF4`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        import netCDF4  # type: ignore
    except ImportError as e:
        raise ImportError("netCDF4 required for NetCDF support: pip install netCDF4") from e

    def gen():
        with netCDF4.Dataset(path, "r") as ds:  # type: ignore[attr-defined]
            v = ds.variables[variable][:]
            for x in np.asarray(v):
                yield x

    return Wave(gen())


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


def pipeline(data: Wave, steps: List[Callable[[Wave], Wave]]) -> Wave:
    out = data
    for step in steps:
        out = step(out)
    return out


def filter(predicate: Callable[[Any], bool]) -> Callable[[Wave], Wave]:
    return lambda w: w.filter(predicate)


def map(transform: Callable[[Any], Any]) -> Callable[[Wave], Wave]:
    return lambda w: w.map(transform)


def collect() -> Callable[[Wave], List[Any]]:
    return lambda w: w.collect()


def batch(size: int) -> Callable[[Wave], Wave]:
    return lambda w: w.batch(size)


def sort_by(field: str, reverse: bool = False) -> Callable[[Wave], Wave]:
    return lambda w: w.sort_by(lambda r: r[field], reverse=reverse)


def drop_outliers(field: str, sigma: float = 3.0) -> Callable[[Wave], Wave]:
    return lambda w: w.drop_outliers(field=field, sigma=sigma)


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
