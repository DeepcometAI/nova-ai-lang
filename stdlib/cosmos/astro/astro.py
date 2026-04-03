"""
cosmos.astro — Astronomy constellation (Python prototype)

Coordinate transforms, distance calculations, catalogue utilities,
spectral classification, and FITS file I/O.

NOVA type annotations are in comments.  At runtime all values are
raw Python floats / numpy arrays.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "parallax_distance_pc",
    "distance_modulus_to_pc",
    "magnitude_from_flux",
    "absolute_magnitude",
    "luminosity_from_magnitude",
    "spectral_class",
    "bv_to_temperature_k",
    "ra_dec_to_cartesian",
    "cartesian_to_ra_dec",
    "angular_separation_deg",
    "proper_motion_velocity",
    "read_fits",
    "wien_displacement",
    "stefan_boltzmann_luminosity",
    "hubble_distance_mpc",
]

# Physical constants
C_M_S        = 299_792_458.0       # speed of light, m/s
PC_TO_M      = 3.085677581e16      # parsec → metres
AU_TO_M      = 1.495978707e11      # AU → metres
L_SUN_W      = 3.828e26            # solar luminosity, W
M_SUN_KG     = 1.989e30            # solar mass, kg
SIGMA_SB     = 5.670374419e-8      # Stefan-Boltzmann, W m⁻² K⁻⁴
WIEN_B       = 2.897771955e-3      # Wien displacement constant, m·K


def parallax_distance_pc(parallax_arcsec: float) -> float:
    """
    Distance in parsec from parallax angle in arcseconds.
    d [pc] = 1 / p [arcsec]
    """
    if parallax_arcsec <= 0.0:
        raise ValueError(f"parallax must be > 0, got {parallax_arcsec}")
    return float(1.0 / parallax_arcsec)


def distance_modulus_to_pc(apparent_m: float, absolute_m: float) -> float:
    """
    Distance in parsec from the distance modulus μ = m − M.
    d = 10^((μ + 5) / 5)
    """
    mu = float(apparent_m) - float(absolute_m)
    return float(10.0 ** ((mu + 5.0) / 5.0))


def magnitude_from_flux(flux: float, flux_zero: float = 3631.0) -> float:
    """
    AB magnitude from flux in Janskys.
    m = -2.5 * log10(flux / flux_zero)
    """
    if flux <= 0.0:
        raise ValueError("flux must be positive")
    return float(-2.5 * math.log10(float(flux) / float(flux_zero)))


def absolute_magnitude(apparent_m: float, distance_pc: float) -> float:
    """
    Absolute magnitude from apparent magnitude and distance in parsec.
    M = m - 5 * log10(d/10)
    """
    return float(float(apparent_m) - 5.0 * math.log10(float(distance_pc) / 10.0))


def luminosity_from_magnitude(absolute_m: float, m_sun: float = 4.74) -> float:
    """
    Luminosity in solar luminosities from absolute magnitude.
    L / L_sun = 10^((M_sun - M) / 2.5)
    """
    return float(10.0 ** ((float(m_sun) - float(absolute_m)) / 2.5))


def spectral_class(bv_color: float) -> str:
    """
    OBAFGKM spectral class from B-V colour index (approximate).
    """
    bv = float(bv_color)
    if bv < -0.30: return "O"
    if bv < -0.02: return "B"
    if bv <  0.30: return "A"
    if bv <  0.58: return "F"
    if bv <  0.81: return "G"
    if bv <  1.40: return "K"
    return "M"


def bv_to_temperature_k(bv_color: float) -> float:
    """
    Effective temperature in Kelvin from B-V colour (Ballesteros 2012 formula).
    T = 4600 * (1/(0.92*BV + 1.7) + 1/(0.92*BV + 0.62))
    """
    bv = float(bv_color)
    return float(4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62)))


def ra_dec_to_cartesian(
    ra_deg: float, dec_deg: float, distance: float = 1.0
) -> Tuple[float, float, float]:
    """
    Equatorial (RA, Dec) → Cartesian (x, y, z).
    RA and Dec in degrees; distance in any consistent unit.
    """
    ra  = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    d   = float(distance)
    x = d * math.cos(dec) * math.cos(ra)
    y = d * math.cos(dec) * math.sin(ra)
    z = d * math.sin(dec)
    return (x, y, z)


def cartesian_to_ra_dec(
    x: float, y: float, z: float
) -> Tuple[float, float]:
    """
    Cartesian (x, y, z) → (RA_deg, Dec_deg).
    """
    d   = math.sqrt(x*x + y*y + z*z)
    if d == 0.0:
        return (0.0, 0.0)
    dec = math.degrees(math.asin(z / d))
    ra  = math.degrees(math.atan2(y, x)) % 360.0
    return (ra, dec)


def angular_separation_deg(
    ra1: float, dec1: float, ra2: float, dec2: float
) -> float:
    """
    Angular separation in degrees between two points on the sky.
    Uses the Haversine formula for numerical stability.
    """
    r1 = math.radians(float(ra1));  r2 = math.radians(float(ra2))
    d1 = math.radians(float(dec1)); d2 = math.radians(float(dec2))
    dra  = r2 - r1
    ddec = d2 - d1
    a = (math.sin(ddec/2)**2
         + math.cos(d1) * math.cos(d2) * math.sin(dra/2)**2)
    return math.degrees(2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))


def proper_motion_velocity(
    proper_motion_mas_yr: float, distance_pc: float
) -> float:
    """
    Transverse velocity in km/s from proper motion (mas/yr) and distance (pc).
    v_T = 4.74047 * mu_mas_yr * d_pc  [km/s]
    """
    return float(4.74047 * float(proper_motion_mas_yr) * float(distance_pc))


def wien_displacement(temp_k: float) -> float:
    """
    Wien's displacement law: peak wavelength in metres.
    λ_max = b / T,   b = 2.898e-3 m·K
    """
    if float(temp_k) <= 0.0:
        raise ValueError("temperature must be positive")
    return float(WIEN_B / float(temp_k))


def stefan_boltzmann_luminosity(radius_m: float, temp_k: float) -> float:
    """
    Stellar luminosity in Watts from radius and effective temperature.
    L = 4π R² σ T⁴
    """
    return float(4.0 * math.pi * float(radius_m)**2 * SIGMA_SB * float(temp_k)**4)


def hubble_distance_mpc(
    recession_velocity_km_s: float, h0_km_s_mpc: float = 70.0
) -> float:
    """
    Hubble law distance in Mpc.
    d = v / H0
    """
    return float(float(recession_velocity_km_s) / float(h0_km_s_mpc))


def read_fits(path: str):
    """
    Read a FITS catalogue as a lazy Wave of row dicts.
    Delegates to cosmos.data.read_fits so users can:
        absorb cosmos.astro.{ read_fits }
    """
    from cosmos.data import read_fits as _read_fits
    return _read_fits(path)
