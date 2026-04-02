"""
cosmos.signal — Signal processing constellation (prototype)
"""

from __future__ import annotations

import numpy as np


def fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.fft.fft(x)


def ifft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.fft.ifft(x)


def convolve(x: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    x = np.asarray(x)
    kernel = np.asarray(kernel)
    return np.convolve(x, kernel, mode=mode)

