"""
cosmos.thermo — Thermodynamics constellation (Python prototype)

Ideal gas laws, heat transfer, thermodynamic cycles, entropy, and
thermochemistry helpers.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = [
    "ideal_gas_pressure",
    "ideal_gas_volume",
    "ideal_gas_temperature",
    "ideal_gas_n_moles",
    "heat_capacity_ideal",
    "adiabatic_temperature",
    "entropy_change_isothermal",
    "carnot_efficiency",
    "stefan_boltzmann_flux",
    "conduction_heat_flux",
    "convection_heat_transfer",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "fahrenheit_to_kelvin",
    "kelvin_to_fahrenheit",
    "speed_of_sound_ideal",
    "mean_free_path",
    "rms_speed",
]

R_J_MOL_K   = 8.314462618     # J mol⁻¹ K⁻¹
K_B         = 1.380649e-23    # J K⁻¹
SIGMA_SB    = 5.670374419e-8  # W m⁻² K⁻⁴
N_A         = 6.02214076e23   # mol⁻¹


def ideal_gas_pressure(n_mol: float, temp_k: float, volume_m3: float) -> float:
    """Ideal gas law: P = nRT/V  [Pa]."""
    return float(float(n_mol) * R_J_MOL_K * float(temp_k) / float(volume_m3))


def ideal_gas_volume(n_mol: float, temp_k: float, pressure_pa: float) -> float:
    """Ideal gas law: V = nRT/P  [m³]."""
    return float(float(n_mol) * R_J_MOL_K * float(temp_k) / float(pressure_pa))


def ideal_gas_temperature(pressure_pa: float, volume_m3: float, n_mol: float) -> float:
    """Ideal gas law: T = PV/(nR)  [K]."""
    return float(float(pressure_pa) * float(volume_m3) / (float(n_mol) * R_J_MOL_K))


def ideal_gas_n_moles(pressure_pa: float, volume_m3: float, temp_k: float) -> float:
    """Ideal gas law: n = PV/(RT)  [mol]."""
    return float(float(pressure_pa) * float(volume_m3) / (R_J_MOL_K * float(temp_k)))


def heat_capacity_ideal(
    n_mol: float, degrees_of_freedom: int = 5
) -> float:
    """
    Molar heat capacity at constant volume for an ideal gas.
    Cv = (f/2) R  where f = degrees of freedom (3 monatomic, 5 diatomic, 6 polyatomic)
    """
    return float(float(n_mol) * (int(degrees_of_freedom) / 2.0) * R_J_MOL_K)


def adiabatic_temperature(
    t_initial_k: float,
    p_initial_pa: float,
    p_final_pa: float,
    gamma: float = 1.4,
) -> float:
    """
    Final temperature after reversible adiabatic process.
    T₂ = T₁ (P₂/P₁)^((γ−1)/γ)
    """
    g = float(gamma)
    return float(float(t_initial_k)
                 * (float(p_final_pa) / float(p_initial_pa))**((g - 1.0) / g))


def entropy_change_isothermal(n_mol: float, v1_m3: float, v2_m3: float) -> float:
    """
    Entropy change for isothermal expansion/compression.
    ΔS = nR ln(V₂/V₁)  [J/K]
    """
    return float(float(n_mol) * R_J_MOL_K * math.log(float(v2_m3) / float(v1_m3)))


def carnot_efficiency(t_hot_k: float, t_cold_k: float) -> float:
    """
    Carnot (maximum) heat engine efficiency:  η = 1 − T_cold / T_hot.
    Returns value in [0, 1].
    """
    return float(1.0 - float(t_cold_k) / float(t_hot_k))


def stefan_boltzmann_flux(temp_k: float) -> float:
    """
    Blackbody radiation flux: j = σ T⁴  [W m⁻²].
    """
    return float(SIGMA_SB * float(temp_k)**4)


def conduction_heat_flux(
    k_w_m_k: float, delta_t_k: float, thickness_m: float
) -> float:
    """
    Fourier heat conduction:  q = k ΔT / d  [W m⁻²].
    k_w_m_k: thermal conductivity in W m⁻¹ K⁻¹
    """
    return float(float(k_w_m_k) * float(delta_t_k) / float(thickness_m))


def convection_heat_transfer(
    h_w_m2_k: float, area_m2: float, delta_t_k: float
) -> float:
    """
    Newton's law of cooling:  Q = h A ΔT  [W].
    h_w_m2_k: convection coefficient in W m⁻² K⁻¹
    """
    return float(float(h_w_m2_k) * float(area_m2) * float(delta_t_k))


def celsius_to_kelvin(c: float) -> float:
    """°C → K."""
    return float(c) + 273.15


def kelvin_to_celsius(k: float) -> float:
    """K → °C."""
    return float(k) - 273.15


def fahrenheit_to_kelvin(f: float) -> float:
    """°F → K."""
    return (float(f) - 32.0) * 5.0 / 9.0 + 273.15


def kelvin_to_fahrenheit(k: float) -> float:
    """K → °F."""
    return (float(k) - 273.15) * 9.0 / 5.0 + 32.0


def speed_of_sound_ideal(
    gamma: float, r_specific_j_kg_k: float, temp_k: float
) -> float:
    """
    Speed of sound in an ideal gas: c = √(γ R_specific T)  [m/s].
    r_specific_j_kg_k: specific gas constant = R/M_molar
    """
    return float(math.sqrt(float(gamma) * float(r_specific_j_kg_k) * float(temp_k)))


def mean_free_path(
    pressure_pa: float, temp_k: float, diameter_m: float
) -> float:
    """
    Mean free path for hard-sphere molecules: λ = kT / (√2 π d² P)  [m].
    """
    d = float(diameter_m)
    return float(K_B * float(temp_k) / (math.sqrt(2.0) * math.pi * d**2 * float(pressure_pa)))


def rms_speed(molar_mass_kg_mol: float, temp_k: float) -> float:
    """
    RMS molecular speed: v_rms = √(3RT/M)  [m/s].
    molar_mass_kg_mol: molar mass in kg/mol (e.g. N₂ = 0.028 kg/mol)
    """
    return float(math.sqrt(3.0 * R_J_MOL_K * float(temp_k) / float(molar_mass_kg_mol)))
