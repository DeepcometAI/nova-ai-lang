"""
cosmos.astro — Astronomy helpers (prototype)
"""

from __future__ import annotations

import math


def parallax_distance_pc(parallax_arcsec: float) -> float:
    """Distance in parsec from parallax in arcseconds: d = 1/p."""
    if parallax_arcsec <= 0:
        raise ValueError("parallax_arcsec must be > 0")
    return float(1.0 / parallax_arcsec)


def magnitude_distance_modulus(apparent_m: float, absolute_m: float) -> float:
    """Distance (parsec) from distance modulus: m - M = 5 log10(d) - 5."""
    mu = apparent_m - absolute_m
    return float(10 ** ((mu + 5.0) / 5.0))


def read_fits(path: str):
    """
    Read a FITS table as a lazy Wave of records.

    This delegates to `cosmos.data.read_fits` so users can `absorb cosmos.astro.{ read_fits }`
    as shown in the docs.
    """
    from cosmos.data import read_fits as _read_fits

    return _read_fits(path)


