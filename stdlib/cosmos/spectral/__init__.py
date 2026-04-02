"""
cosmos.spectral — Spectral/EM helpers constellation (prototype)
"""

from .spectral import (
    doppler_shift,
    blackbody_peak_wavelength,
    redshift_from_velocity,
    velocity_from_redshift,
)

__all__ = ["doppler_shift", "blackbody_peak_wavelength", "redshift_from_velocity", "velocity_from_redshift"]

