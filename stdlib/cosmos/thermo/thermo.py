"""
cosmos.thermo — Thermodynamics helpers (prototype)
"""

from __future__ import annotations


R = 8.314462618  # J/(mol·K)


def ideal_gas_pressure(n_mol: float, temp_k: float, volume_m3: float) -> float:
    """Ideal gas law: P = nRT / V."""
    return float((n_mol * R * temp_k) / volume_m3)


def ideal_gas_temperature(pressure_pa: float, volume_m3: float, n_mol: float) -> float:
    """Ideal gas law: T = PV / (nR)."""
    return float((pressure_pa * volume_m3) / (n_mol * R))

