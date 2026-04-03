"""
cosmos.orbital — Orbital mechanics constellation (Python prototype)

Tsiolkovsky rocket equation, Kepler's laws, Hohmann transfers,
vis-viva equation, orbital elements, and trajectory integration stubs.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "delta_v",
    "kepler_period",
    "kepler_period_yr",
    "gravitational_parameter",
    "hohmann_delta_v",
    "vis_viva_velocity",
    "escape_velocity",
    "circular_velocity",
    "semi_major_axis_from_period",
    "orbital_energy",
    "sphere_of_influence_m",
    "synodic_period",
    "launch_window_angle_deg",
]

G       = 6.67430e-11     # m³ kg⁻¹ s⁻²
G_EARTH = 9.80665         # m/s²  (standard gravity)
M_SUN   = 1.989e30        # kg
M_EARTH = 5.972e24        # kg
R_EARTH = 6.371e6         # m
AU_M    = 1.496e11        # m
YR_S    = 3.15576e7       # s


def delta_v(
    isp_s: float,
    m_wet_kg: float,
    m_dry_kg: float,
    g0: float = G_EARTH,
) -> float:
    """
    Tsiolkovsky rocket equation: Δv = Isp · g₀ · ln(m_wet / m_dry).
    Returns Δv in m/s.
    """
    if m_dry_kg <= 0:
        raise ValueError("m_dry_kg must be > 0")
    if m_wet_kg <= m_dry_kg:
        raise ValueError("m_wet_kg must be > m_dry_kg")
    return float(float(isp_s) * float(g0) * math.log(float(m_wet_kg) / float(m_dry_kg)))


def gravitational_parameter(m_central_kg: float) -> float:
    """Standard gravitational parameter μ = G M in m³/s²."""
    return float(G * float(m_central_kg))


def kepler_period(a_m: float, m_central_kg: float) -> float:
    """
    Orbital period in seconds for semi-major axis a (metres) around mass M.
    T = 2π √(a³ / μ)
    """
    mu = gravitational_parameter(m_central_kg)
    return float(2.0 * math.pi * math.sqrt(float(a_m)**3 / mu))


def kepler_period_yr(a_au: float) -> float:
    """
    Simplified Kepler period in years for orbit around the Sun.
    T_yr = a_AU^(3/2)   (exact for circular orbits around the Sun)
    """
    return float(float(a_au)**1.5)


def hohmann_delta_v(
    r1_m: float, r2_m: float, mu_m3_s2: float
) -> Tuple[float, float, float]:
    """
    Hohmann transfer Δv values from circular orbit r1 to r2.
    Returns (dv1, dv2, dv_total) in m/s.
    """
    r1, r2, mu = float(r1_m), float(r2_m), float(mu_m3_s2)
    a_transfer = 0.5 * (r1 + r2)
    v1     = math.sqrt(mu / r1)
    v2     = math.sqrt(mu / r2)
    v_peri = math.sqrt(mu * (2.0/r1 - 1.0/a_transfer))
    v_apo  = math.sqrt(mu * (2.0/r2 - 1.0/a_transfer))
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    return (float(dv1), float(dv2), float(dv1 + dv2))


def vis_viva_velocity(r_m: float, a_m: float, mu_m3_s2: float) -> float:
    """
    Vis-viva velocity at radius r on an orbit with semi-major axis a.
    v = √(μ (2/r − 1/a))
    """
    return float(math.sqrt(float(mu_m3_s2) * (2.0/float(r_m) - 1.0/float(a_m))))


def escape_velocity(r_m: float, m_central_kg: float) -> float:
    """
    Escape velocity at distance r from central mass M.
    v_esc = √(2GM/r)
    """
    mu = gravitational_parameter(m_central_kg)
    return float(math.sqrt(2.0 * mu / float(r_m)))


def circular_velocity(r_m: float, m_central_kg: float) -> float:
    """
    Circular orbital velocity at radius r around mass M.
    v_c = √(GM/r)
    """
    mu = gravitational_parameter(m_central_kg)
    return float(math.sqrt(mu / float(r_m)))


def semi_major_axis_from_period(period_s: float, m_central_kg: float) -> float:
    """
    Semi-major axis in metres from orbital period.
    a = (μ T² / 4π²)^(1/3)
    """
    mu = gravitational_parameter(m_central_kg)
    return float((mu * float(period_s)**2 / (4.0 * math.pi**2))**(1.0/3.0))


def orbital_energy(
    r_m: float, v_m_s: float, m_central_kg: float
) -> float:
    """
    Specific orbital energy (J/kg):  ε = v²/2 − μ/r
    Negative → bound orbit; zero → parabolic; positive → hyperbolic.
    """
    mu = gravitational_parameter(m_central_kg)
    return float(0.5 * float(v_m_s)**2 - mu / float(r_m))


def sphere_of_influence_m(
    a_m: float, m_body_kg: float, m_central_kg: float
) -> float:
    """
    Sphere of influence radius in metres (Laplace approximation).
    r_SOI = a (m/M)^(2/5)
    """
    return float(float(a_m) * (float(m_body_kg) / float(m_central_kg))**(2.0/5.0))


def synodic_period(t1_s: float, t2_s: float) -> float:
    """
    Synodic period in seconds for two orbiting bodies.
    1/T_syn = |1/T1 - 1/T2|
    """
    return float(1.0 / abs(1.0/float(t1_s) - 1.0/float(t2_s)))


def launch_window_angle_deg(
    t1_s: float, t2_s: float
) -> float:
    """
    Phase angle between orbits at next Hohmann-transfer launch window.
    φ = π (1 − (1/2 (1 + r1/r2))^(3/2))  in degrees
    (r1/r2 approximated as (T1/T2)^(2/3) for circular orbits)
    """
    ratio = (float(t1_s) / float(t2_s))**(2.0/3.0)
    phi = math.pi * (1.0 - (0.5 * (1.0 + ratio))**1.5)
    return float(math.degrees(phi))
