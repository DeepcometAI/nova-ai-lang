"""
cosmos.signal — Signal processing constellation (Python prototype)

FFT, IFFT, convolution, windowing, filtering, and spectral analysis.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

__all__ = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fft_frequencies",
    "power_spectrum",
    "convolve",
    "correlate",
    "window_hann",
    "window_hamming",
    "window_blackman",
    "lowpass_filter",
    "highpass_filter",
    "bandpass_filter",
    "rms",
    "snr_db",
]


def fft(x: np.ndarray) -> np.ndarray:
    """1-D discrete Fourier transform (complex output)."""
    return np.fft.fft(np.asarray(x, dtype=np.float64))


def ifft(x: np.ndarray) -> np.ndarray:
    """Inverse 1-D DFT."""
    return np.fft.ifft(np.asarray(x))


def rfft(x: np.ndarray) -> np.ndarray:
    """Real-input FFT (returns N//2+1 complex coefficients)."""
    return np.fft.rfft(np.asarray(x, dtype=np.float64))


def irfft(x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """Inverse real-input FFT."""
    return np.fft.irfft(np.asarray(x), n=n)


def fft_frequencies(n: int, sample_rate_hz: float = 1.0) -> np.ndarray:
    """
    Frequency bins for an N-point FFT sampled at sample_rate_hz.
    Returns array of length N with frequencies in Hz.
    """
    return np.fft.fftfreq(int(n), d=1.0 / float(sample_rate_hz))


def power_spectrum(x: np.ndarray, sample_rate_hz: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-sided power spectral density.
    Returns (frequencies_hz, power) where len = N//2+1.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
    power = (np.abs(spec)**2) / n
    # Double one-sided except DC and Nyquist
    power[1:-1] *= 2.0
    return freqs, power


def convolve(x: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    1-D convolution.
    mode: 'full' | 'same' | 'valid'
    """
    return np.convolve(
        np.asarray(x, dtype=np.float64),
        np.asarray(kernel, dtype=np.float64),
        mode=mode,
    )


def correlate(x: np.ndarray, y: np.ndarray, mode: str = "full") -> np.ndarray:
    """1-D cross-correlation of x and y."""
    return np.correlate(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        mode=mode,
    )


def window_hann(n: int) -> np.ndarray:
    """Hann (raised-cosine) window of length n."""
    return np.hanning(int(n))


def window_hamming(n: int) -> np.ndarray:
    """Hamming window of length n."""
    return np.hamming(int(n))


def window_blackman(n: int) -> np.ndarray:
    """Blackman window of length n."""
    return np.blackman(int(n))


def lowpass_filter(
    x: np.ndarray, cutoff_hz: float, sample_rate_hz: float, order: int = 4
) -> np.ndarray:
    """
    Butterworth low-pass filter (requires scipy).
    Falls back to FFT-based ideal filter if scipy unavailable.
    """
    x = np.asarray(x, dtype=np.float64)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * float(sample_rate_hz)
        norm = float(cutoff_hz) / nyq
        b, a = butter(int(order), norm, btype="low")
        return filtfilt(b, a, x)
    except ImportError:
        # Ideal brick-wall filter in frequency domain
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate_hz))
        spec[np.abs(freqs) > float(cutoff_hz)] = 0.0
        return np.fft.irfft(spec, n=len(x))


def highpass_filter(
    x: np.ndarray, cutoff_hz: float, sample_rate_hz: float, order: int = 4
) -> np.ndarray:
    """Butterworth high-pass filter (requires scipy; falls back to FFT ideal)."""
    x = np.asarray(x, dtype=np.float64)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * float(sample_rate_hz)
        norm = float(cutoff_hz) / nyq
        b, a = butter(int(order), norm, btype="high")
        return filtfilt(b, a, x)
    except ImportError:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate_hz))
        spec[np.abs(freqs) < float(cutoff_hz)] = 0.0
        return np.fft.irfft(spec, n=len(x))


def bandpass_filter(
    x: np.ndarray,
    low_hz: float,
    high_hz: float,
    sample_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Band-pass filter between low_hz and high_hz."""
    x = np.asarray(x, dtype=np.float64)
    try:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * float(sample_rate_hz)
        low  = float(low_hz) / nyq
        high = float(high_hz) / nyq
        b, a = butter(int(order), [low, high], btype="band")
        return filtfilt(b, a, x)
    except ImportError:
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate_hz))
        mask = (np.abs(freqs) < float(low_hz)) | (np.abs(freqs) > float(high_hz))
        spec[mask] = 0.0
        return np.fft.irfft(spec, n=len(x))


def rms(x: np.ndarray) -> float:
    """Root mean square of signal."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x**2)))


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Signal-to-noise ratio in dB.
    SNR = 10 log10(RMS(signal)² / RMS(noise)²)
    """
    s_rms = rms(signal)
    n_rms = rms(noise)
    if n_rms == 0.0:
        return float("inf")
    return float(10.0 * np.log10((s_rms / n_rms)**2))
