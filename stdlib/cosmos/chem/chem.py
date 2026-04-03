"""
cosmos.chem — Chemistry constellation (Python prototype)

Periodic table lookup, stoichiometry, reaction energy, and
thermochemistry helpers.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

__all__ = [
    "element_symbol",
    "element_name",
    "atomic_mass_u",
    "atomic_number",
    "molar_mass",
    "avogadro",
    "moles_from_grams",
    "grams_from_moles",
    "ideal_gas_volume",
    "activation_energy",
    "arrhenius_rate",
    "balance_check",
    "energy_to_wavelength_m",
    "wavelength_to_energy_j",
    "ionization_shells",
]

# Physical constants
N_A      = 6.02214076e23      # Avogadro, mol⁻¹
R_GAS    = 8.314462618        # ideal gas constant, J mol⁻¹ K⁻¹
K_B      = 1.380649e-23       # Boltzmann, J K⁻¹
H_PLANCK = 6.62607015e-34     # Planck, J s
C_LIGHT  = 299_792_458.0      # speed of light, m s⁻¹

# Abbreviated periodic table (atomic number → properties)
_PT: Dict[int, Dict] = {
    1:  {"symbol":"H",  "name":"Hydrogen",   "mass_u":1.00794,  "group":1,  "period":1},
    2:  {"symbol":"He", "name":"Helium",     "mass_u":4.002602, "group":18, "period":1},
    3:  {"symbol":"Li", "name":"Lithium",    "mass_u":6.941,    "group":1,  "period":2},
    4:  {"symbol":"Be", "name":"Beryllium",  "mass_u":9.01218,  "group":2,  "period":2},
    5:  {"symbol":"B",  "name":"Boron",      "mass_u":10.81,    "group":13, "period":2},
    6:  {"symbol":"C",  "name":"Carbon",     "mass_u":12.0107,  "group":14, "period":2},
    7:  {"symbol":"N",  "name":"Nitrogen",   "mass_u":14.0067,  "group":15, "period":2},
    8:  {"symbol":"O",  "name":"Oxygen",     "mass_u":15.9994,  "group":16, "period":2},
    9:  {"symbol":"F",  "name":"Fluorine",   "mass_u":18.9984,  "group":17, "period":2},
    10: {"symbol":"Ne", "name":"Neon",       "mass_u":20.1797,  "group":18, "period":2},
    11: {"symbol":"Na", "name":"Sodium",     "mass_u":22.9898,  "group":1,  "period":3},
    12: {"symbol":"Mg", "name":"Magnesium",  "mass_u":24.305,   "group":2,  "period":3},
    13: {"symbol":"Al", "name":"Aluminium",  "mass_u":26.9815,  "group":13, "period":3},
    14: {"symbol":"Si", "name":"Silicon",    "mass_u":28.0855,  "group":14, "period":3},
    15: {"symbol":"P",  "name":"Phosphorus", "mass_u":30.9738,  "group":15, "period":3},
    16: {"symbol":"S",  "name":"Sulfur",     "mass_u":32.065,   "group":16, "period":3},
    17: {"symbol":"Cl", "name":"Chlorine",   "mass_u":35.453,   "group":17, "period":3},
    18: {"symbol":"Ar", "name":"Argon",      "mass_u":39.948,   "group":18, "period":3},
    19: {"symbol":"K",  "name":"Potassium",  "mass_u":39.0983,  "group":1,  "period":4},
    20: {"symbol":"Ca", "name":"Calcium",    "mass_u":40.078,   "group":2,  "period":4},
    26: {"symbol":"Fe", "name":"Iron",       "mass_u":55.845,   "group":8,  "period":4},
    29: {"symbol":"Cu", "name":"Copper",     "mass_u":63.546,   "group":11, "period":4},
    30: {"symbol":"Zn", "name":"Zinc",       "mass_u":65.38,    "group":12, "period":4},
    47: {"symbol":"Ag", "name":"Silver",     "mass_u":107.868,  "group":11, "period":5},
    79: {"symbol":"Au", "name":"Gold",       "mass_u":196.967,  "group":11, "period":6},
    82: {"symbol":"Pb", "name":"Lead",       "mass_u":207.2,    "group":14, "period":6},
    92: {"symbol":"U",  "name":"Uranium",    "mass_u":238.029,  "group":3,  "period":7},
}
# Reverse lookup: symbol → atomic number
_SYM2Z = {v["symbol"]: k for k, v in _PT.items()}


def _get(z: int) -> Dict:
    if z not in _PT:
        raise KeyError(f"element with atomic number {z} not in table")
    return _PT[z]


def element_symbol(z: int) -> str:
    """Return element symbol for atomic number z."""
    return str(_get(z)["symbol"])


def element_name(z: int) -> str:
    """Return element name for atomic number z."""
    return str(_get(z)["name"])


def atomic_mass_u(z: int) -> float:
    """Return standard atomic weight in unified atomic mass units (u)."""
    return float(_get(z)["mass_u"])


def atomic_number(symbol: str) -> int:
    """Return atomic number for element symbol (case-sensitive, e.g. 'Fe')."""
    if symbol not in _SYM2Z:
        raise KeyError(f"unknown element symbol: {symbol!r}")
    return int(_SYM2Z[symbol])


def avogadro() -> float:
    """Avogadro's number N_A in mol⁻¹."""
    return float(N_A)


def molar_mass(z: int) -> float:
    """Molar mass in g/mol (numerically equal to atomic mass in u)."""
    return float(_get(z)["mass_u"])


def moles_from_grams(mass_g: float, z: int) -> float:
    """Convert grams of element z to moles."""
    return float(mass_g) / molar_mass(z)


def grams_from_moles(moles: float, z: int) -> float:
    """Convert moles of element z to grams."""
    return float(moles) * molar_mass(z)


def ideal_gas_volume(n_mol: float, temp_k: float, pressure_pa: float) -> float:
    """Ideal gas volume in m³:  V = nRT/P."""
    return float(float(n_mol) * R_GAS * float(temp_k) / float(pressure_pa))


def activation_energy(
    k1: float, k2: float, t1_k: float, t2_k: float
) -> float:
    """
    Activation energy in J/mol from two rate constants k1, k2 at T1, T2.
    Arrhenius: Ea = R * ln(k2/k1) / (1/T1 - 1/T2)
    """
    return float(R_GAS * math.log(float(k2) / float(k1))
                 / (1.0 / float(t1_k) - 1.0 / float(t2_k)))


def arrhenius_rate(
    a_pre: float, ea_j_mol: float, temp_k: float
) -> float:
    """
    Arrhenius rate constant:  k = A * exp(-Ea / RT)
    """
    return float(float(a_pre) * math.exp(-float(ea_j_mol) / (R_GAS * float(temp_k))))


def balance_check(
    reactant_counts: Dict[str, int], product_counts: Dict[str, int]
) -> bool:
    """
    Check atom balance between reactants and products.
    Inputs are dicts mapping element symbol → total atom count.
    Returns True if balanced.
    """
    return reactant_counts == product_counts


def energy_to_wavelength_m(energy_j: float) -> float:
    """Photon wavelength in metres from energy in Joules. λ = hc/E."""
    return float(H_PLANCK * C_LIGHT / float(energy_j))


def wavelength_to_energy_j(wavelength_m: float) -> float:
    """Photon energy in Joules from wavelength in metres. E = hc/λ."""
    return float(H_PLANCK * C_LIGHT / float(wavelength_m))


def ionization_shells(z: int) -> List[int]:
    """
    Return electron shell configuration as a list [K, L, M, ...].
    Simplified Aufbau filling (does not handle transition metal exceptions).
    """
    electrons = z
    shells = []
    capacities = [2, 8, 18, 32, 32, 18, 8]
    for cap in capacities:
        if electrons <= 0:
            break
        filled = min(electrons, cap)
        shells.append(filled)
        electrons -= filled
    return shells
