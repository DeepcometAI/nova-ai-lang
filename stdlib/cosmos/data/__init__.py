"""
cosmos.data — Data loading and I/O constellation
"""

from .data import (
    read_csv, read_fits, read_parquet, read_arrow,
    write_csv, write_parquet,
    Wave
)

__all__ = [
    'read_csv', 'read_fits', 'read_parquet', 'read_arrow',
    'write_csv', 'write_parquet',
    'Wave'
]
