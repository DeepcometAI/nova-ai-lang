"""
cosmos.quantum — Quantum mechanics constellation (Python prototype)

Quantum gates, state vectors, expectation values, and basic QM helpers.
All state vectors are 1-D complex numpy arrays, normalised to |ψ|² = 1.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

__all__ = [
    "ket_zero",
    "ket_one",
    "ket_plus",
    "ket_minus",
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "hadamard",
    "phase_gate",
    "cnot",
    "apply_gate",
    "tensor_product",
    "inner_product",
    "expectation_value",
    "measure",
    "normalise",
    "bloch_angles",
    "hydrogen_energy_ev",
    "de_broglie_wavelength",
    "uncertainty_product",
]

# Reduced Planck constant
HBAR_J_S = 1.054571817e-34  # J·s
M_E_KG   = 9.1093837015e-31  # electron mass, kg
E_EV     = 1.602176634e-19   # eV in Joules
ALPHA    = 13.605693122      # Hydrogen ground state energy in eV


# ── Standard basis states ─────────────────────────────────────────────────────

def ket_zero() -> np.ndarray:
    """Computational basis state |0⟩."""
    return np.array([1.0, 0.0], dtype=complex)


def ket_one() -> np.ndarray:
    """Computational basis state |1⟩."""
    return np.array([0.0, 1.0], dtype=complex)


def ket_plus() -> np.ndarray:
    """|+⟩ = (|0⟩ + |1⟩) / √2 — Hadamard basis."""
    return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)


def ket_minus() -> np.ndarray:
    """|−⟩ = (|0⟩ − |1⟩) / √2 — Hadamard basis."""
    return np.array([1.0, -1.0], dtype=complex) / math.sqrt(2.0)


# ── Standard gates ────────────────────────────────────────────────────────────

def pauli_x() -> np.ndarray:
    """Pauli-X (NOT) gate."""
    return np.array([[0, 1], [1, 0]], dtype=complex)


def pauli_y() -> np.ndarray:
    """Pauli-Y gate."""
    return np.array([[0, -1j], [1j, 0]], dtype=complex)


def pauli_z() -> np.ndarray:
    """Pauli-Z gate."""
    return np.array([[1, 0], [0, -1]], dtype=complex)


def hadamard() -> np.ndarray:
    """Hadamard gate H = (X + Z) / √2."""
    return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)


def phase_gate(theta: float) -> np.ndarray:
    """Phase (P) gate: P(θ) = [[1,0],[0,exp(iθ)]]."""
    return np.array([[1, 0], [0, np.exp(1j * float(theta))]], dtype=complex)


def cnot() -> np.ndarray:
    """CNOT (CX) gate — 4×4 matrix on |control, target⟩."""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)


# ── Operations ────────────────────────────────────────────────────────────────

def apply_gate(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply a unitary gate to a state vector: |ψ'⟩ = U|ψ⟩."""
    g = np.asarray(gate, dtype=complex)
    s = np.asarray(state, dtype=complex)
    return g @ s


def tensor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Tensor (Kronecker) product of two state vectors or matrices.
    |a⟩ ⊗ |b⟩ or A ⊗ B.
    """
    return np.kron(
        np.asarray(a, dtype=complex),
        np.asarray(b, dtype=complex),
    )


def inner_product(bra: np.ndarray, ket: np.ndarray) -> complex:
    """⟨bra|ket⟩ = conjugate(bra) · ket."""
    return complex(np.dot(np.conj(np.asarray(bra, dtype=complex)),
                          np.asarray(ket, dtype=complex)))


def expectation_value(operator: np.ndarray, state: np.ndarray) -> float:
    """
    Real expectation value ⟨ψ|O|ψ⟩ for Hermitian operator O.
    """
    s = np.asarray(state, dtype=complex)
    o = np.asarray(operator, dtype=complex)
    return float(np.real(np.dot(np.conj(s), o @ s)))


def measure(state: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Simulate measurement in computational basis.
    Returns (outcome_index, collapsed_state).
    Probabilities = |amplitude|².
    """
    s   = np.asarray(state, dtype=complex)
    probs = np.abs(s)**2
    probs = probs / probs.sum()  # normalise for floating point safety
    idx = int(np.random.choice(len(probs), p=probs))
    collapsed = np.zeros_like(s)
    collapsed[idx] = 1.0 + 0j
    return (idx, collapsed)


def normalise(state: np.ndarray) -> np.ndarray:
    """Return normalised state vector |ψ⟩ / ‖|ψ⟩‖."""
    s = np.asarray(state, dtype=complex)
    norm = float(np.sqrt(np.real(np.dot(np.conj(s), s))))
    if norm < 1e-300:
        raise ValueError("cannot normalise the zero state")
    return s / norm


def bloch_angles(state: np.ndarray) -> Tuple[float, float]:
    """
    Bloch sphere angles (θ, φ) for a single-qubit state.
    |ψ⟩ = cos(θ/2)|0⟩ + exp(iφ)sin(θ/2)|1⟩
    Returns (theta_rad, phi_rad).
    """
    s = normalise(np.asarray(state, dtype=complex))
    alpha, beta = s[0], s[1]
    theta = float(2.0 * np.arccos(min(1.0, abs(alpha))))
    phi   = float(np.angle(beta) - np.angle(alpha))
    return (theta, phi)


# ── Atomic physics ────────────────────────────────────────────────────────────

def hydrogen_energy_ev(n: int) -> float:
    """
    Hydrogen atom energy for principal quantum number n (in eV).
    E_n = −13.606 / n²  eV
    """
    return float(-ALPHA / float(n)**2)


def de_broglie_wavelength(mass_kg: float, velocity_m_s: float) -> float:
    """
    de Broglie wavelength: λ = h/(mv)  [m].
    """
    h = 2.0 * math.pi * HBAR_J_S
    return float(h / (float(mass_kg) * float(velocity_m_s)))


def uncertainty_product(
    delta_x_m: float, delta_p_kg_m_s: float
) -> float:
    """
    Heisenberg uncertainty product Δx·Δp.
    Returns ℏ/2 ratio: actual / (ℏ/2).
    Values ≥ 1.0 satisfy the uncertainty principle.
    """
    return float(float(delta_x_m) * float(delta_p_kg_m_s) / (HBAR_J_S / 2.0))
