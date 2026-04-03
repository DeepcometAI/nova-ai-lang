"""
cosmos.spectral — Spectral physics constellation (Python prototype)

Blackbody radiation, Doppler shifts, emission lines, redshift, and
spectral classification utilities.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

__all__ = [
    "blackbody_peak_wavelength",
    "blackbody_spectral_radiance",
    "blackbody_luminosity",
    "doppler_shift_wavelength",
    "doppler_shift_frequency",
    "redshift_from_velocity",
    "velocity_from_redshift",
    "relativistic_redshift",
    "line_equivalent_width",
    "hydrogen_balmer_wavelength",
    "emission_line_wavelengths",
    "luminosity_distance_mpc",
    "comoving_distance_mpc",
]

# Physical constants
C_M_S    = 299_792_458.0    # m/s
H_J_S    = 6.62607015e-34   # J·s
K_B      = 1.380649e-23     # J/K
SIGMA_SB = 5.670374419e-8   # W m⁻² K⁻⁴
WIEN_B   = 2.897771955e-3   # m·K

# Hydrogen Balmer series wavelengths (vacuum, nm)
_BALMER_NM: Dict[int, float] = {
    3: 656.279,  # Hα
    4: 486.133,  # Hβ
    5: 434.047,  # Hγ
    6: 410.174,  # Hδ
}

# Common emission lines (name → vacuum wavelength in nm)
EMISSION_LINES: Dict[str, float] = {
    "Lyman-alpha": 121.567,
    "H-alpha":     656.279,
    "H-beta":      486.133,
    "H-gamma":     434.047,
    "Ca-K":        393.366,
    "Ca-H":        396.847,
    "Na-D1":       589.592,
    "Na-D2":       588.995,
    "Mg-b1":       518.362,
    "O-III-4959":  495.891,
    "O-III-5007":  500.684,
    "N-II-6583":   658.345,
    "S-II-6716":   671.647,
}


def blackbody_peak_wavelength(temp_k: float) -> float:
    """Wien's displacement law peak wavelength in metres."""
    if float(temp_k) <= 0.0:
        raise ValueError("temperature must be > 0")
    return float(WIEN_B / float(temp_k))


def blackbody_spectral_radiance(
    wavelength_m: float, temp_k: float
) -> float:
    """
    Planck spectral radiance B_λ in W sr⁻¹ m⁻³.
    B_λ = (2hc²/λ⁵) / (exp(hc/λkT) − 1)
    """
    lam = float(wavelength_m)
    T   = float(temp_k)
    exponent = H_J_S * C_M_S / (lam * K_B * T)
    if exponent > 700:  # prevent overflow
        return 0.0
    return float(
        (2.0 * H_J_S * C_M_S**2 / lam**5)
        / (math.exp(exponent) - 1.0)
    )


def blackbody_luminosity(radius_m: float, temp_k: float) -> float:
    """Total luminosity of a blackbody sphere: L = 4π R² σ T⁴."""
    return float(4.0 * math.pi * float(radius_m)**2 * SIGMA_SB * float(temp_k)**4)


def doppler_shift_wavelength(
    rest_wavelength_m: float,
    radial_velocity_m_s: float,
    c_m_s: float = C_M_S,
) -> float:
    """
    Non-relativistic Doppler: λ_obs = λ_rest (1 + v/c).
    Positive v = recession (redshift).
    """
    return float(float(rest_wavelength_m) * (1.0 + float(radial_velocity_m_s) / float(c_m_s)))


def doppler_shift_frequency(
    rest_freq_hz: float,
    radial_velocity_m_s: float,
    c_m_s: float = C_M_S,
) -> float:
    """Non-relativistic Doppler: f_obs = f_rest (1 − v/c)."""
    return float(float(rest_freq_hz) * (1.0 - float(radial_velocity_m_s) / float(c_m_s)))


def redshift_from_velocity(
    radial_velocity_m_s: float, c_m_s: float = C_M_S
) -> float:
    """Non-relativistic redshift: z ≈ v/c."""
    return float(float(radial_velocity_m_s) / float(c_m_s))


def velocity_from_redshift(z: float, c_m_s: float = C_M_S) -> float:
    """Non-relativistic recession velocity from redshift: v = zc."""
    return float(float(z) * float(c_m_s))


def relativistic_redshift(z: float) -> float:
    """
    Recession velocity from redshift using relativistic formula.
    v = c ((1+z)² − 1) / ((1+z)² + 1)
    """
    zp1 = 1.0 + float(z)
    return float(C_M_S * (zp1**2 - 1.0) / (zp1**2 + 1.0))


def hydrogen_balmer_wavelength(n_upper: int) -> float:
    """Hydrogen Balmer series wavelength in metres for transition n_upper → 2."""
    if n_upper not in _BALMER_NM:
        # Rydberg formula
        r_h = 1.097e7  # m⁻¹
        inv_lam = r_h * (1.0/4.0 - 1.0/float(n_upper)**2)
        return float(1.0 / inv_lam)
    return float(_BALMER_NM[n_upper] * 1e-9)


def emission_line_wavelengths() -> Dict[str, float]:
    """Return dict of common emission lines (name → wavelength in metres)."""
    return {k: v * 1e-9 for k, v in EMISSION_LINES.items()}


def line_equivalent_width(
    continuum: np.ndarray,
    line_profile: np.ndarray,
    wavelengths_m: np.ndarray,
) -> float:
    """
    Equivalent width of a spectral line in metres.
    EW = ∫ (1 − F_line/F_continuum) dλ
    """
    c = np.asarray(continuum, dtype=np.float64)
    l = np.asarray(line_profile, dtype=np.float64)
    w = np.asarray(wavelengths_m, dtype=np.float64)
    integrand = 1.0 - l / np.where(c != 0, c, 1e-30)
    return float(np.trapz(integrand, w))


def luminosity_distance_mpc(
    z: float, h0_km_s_mpc: float = 70.0, omega_m: float = 0.3
) -> float:
    """
    Luminosity distance in Mpc for flat ΛCDM cosmology (Ω_Λ = 1 − Ω_m).
    Numerical integration of the comoving distance.
    """
    d_c = comoving_distance_mpc(z, h0_km_s_mpc, omega_m)
    return float(d_c * (1.0 + float(z)))


def comoving_distance_mpc(
    z: float,
    h0_km_s_mpc: float = 70.0,
    omega_m: float = 0.3,
    n_steps: int = 1000,
) -> float:
    """
    Comoving distance in Mpc for flat ΛCDM via trapezoidal integration.
    """
    c_km_s = C_M_S / 1000.0
    z_val  = float(z)
    omega_l = 1.0 - float(omega_m)
    dz = z_val / int(n_steps)
    total = 0.0
    for i in range(int(n_steps)):
        zi = (i + 0.5) * dz
        ez = math.sqrt(float(omega_m) * (1.0 + zi)**3 + omega_l)
        total += dz / ez
    return float((c_km_s / float(h0_km_s_mpc)) * total)
