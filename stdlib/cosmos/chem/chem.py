"""
cosmos.chem — Chemistry helpers (prototype)
"""

from __future__ import annotations

from typing import Dict


_ELEMENTS: Dict[int, Dict[str, float | str]] = {
    1: {"symbol": "H", "mass_u": 1.00784},
    2: {"symbol": "He", "mass_u": 4.002602},
    6: {"symbol": "C", "mass_u": 12.0107},
    7: {"symbol": "N", "mass_u": 14.0067},
    8: {"symbol": "O", "mass_u": 15.999},
    26: {"symbol": "Fe", "mass_u": 55.845},
}


def element_symbol(z: int) -> str:
    if z not in _ELEMENTS:
        raise KeyError(f"unknown atomic number: {z}")
    return str(_ELEMENTS[z]["symbol"])


def atomic_mass_u(z: int) -> float:
    if z not in _ELEMENTS:
        raise KeyError(f"unknown atomic number: {z}")
    return float(_ELEMENTS[z]["mass_u"])

