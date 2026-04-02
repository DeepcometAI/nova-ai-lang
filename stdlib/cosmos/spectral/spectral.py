"""
cosmos.spectral — Spectral helpers (prototype)
"""

from __future__ import annotations

import math


def doppler_shift(wavelength_m: float, radial_velocity_m_s: float, c_m_s: float = 299_792_458.0) -> float:
    """Non-relativistic Doppler shift: λ' = λ * (1 + v/c)."""
    return float(wavelength_m * (1.0 + radial_velocity_m_s / c_m_s))


def blackbody_peak_wavelength(temp_k: float) -> float:
    """Wien's displacement law peak wavelength in meters: λ_max = b/T."""
    b = 2.897771955e-3  # m·K
    return float(b / temp_k)


def redshift_from_velocity(radial_velocity_m_s: float, c_m_s: float = 299_792_458.0) -> float:
    """Non-relativistic: z ≈ v/c."""
    return float(radial_velocity_m_s / c_m_s)


def velocity_from_redshift(z: float, c_m_s: float = 299_792_458.0) -> float:
    """Non-relativistic: v ≈ z c."""
    return float(float(z) * c_m_s)

