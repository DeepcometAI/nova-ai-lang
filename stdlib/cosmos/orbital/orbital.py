"""
cosmos.orbital — Orbital mechanics constellation (prototype)

Unit safety is a compile-time feature in NOVA. This Python prototype assumes
inputs are already in compatible units and returns raw floats.
"""

from __future__ import annotations

import math


G = 6.674e-11  # m^3 / (kg s^2)


def delta_v(isp_s: float, m_wet_kg: float, m_dry_kg: float, g0: float = 9.80665) -> float:
    """Tsiolkovsky rocket equation: Δv = Isp * g0 * ln(m_wet / m_dry)."""
    return float(isp_s * g0 * math.log(m_wet_kg / m_dry_kg))


def kepler_period(a_m: float, m_central_kg: float) -> float:
    """Kepler's third law period for semi-major axis a and central mass."""
    return float(2.0 * math.pi * math.sqrt(a_m**3 / (G * m_central_kg)))


def gravitational_parameter(m_central_kg: float) -> float:
    """Standard gravitational parameter μ = G M."""
    return float(G * m_central_kg)


def hohmann_delta_v(r1_m: float, r2_m: float, mu_m3_s2: float) -> tuple[float, float, float]:
    """
    Hohmann transfer Δv from circular orbit r1 to r2 around a central body.

    Returns (dv1, dv2, dv_total) in m/s.
    """
    r1 = float(r1_m)
    r2 = float(r2_m)
    mu = float(mu_m3_s2)
    a = 0.5 * (r1 + r2)
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)
    v_peri = math.sqrt(mu * (2 / r1 - 1 / a))
    v_apo = math.sqrt(mu * (2 / r2 - 1 / a))
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    return (float(dv1), float(dv2), float(dv1 + dv2))

