"""
Wavelet Filters and Basic Operations
=====================================
Filter coefficient retrieval and basic convolution operations.
"""

import numpy as np
from typing import Tuple

# Try importing PyWavelets for filter coefficients (optional)
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


def get_wavelet_filters(wavelet_name: str = 'db4', use_pywt: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get QMF filter coefficients for specified wavelet.
    
    Uses PyWavelets Wavelet object as the primary (safer) method, with hardcoded
    coefficients as fallback only if PyWavelets is unavailable.
    
    Args:
        wavelet_name: Name of wavelet (any PyWavelets-supported wavelet, e.g., 
                     'haar', 'db2', 'db4', 'db6', 'db8', 'sym2', 'coif2', etc.)
        use_pywt: If True and PyWavelets is available, use its Wavelet object.
                  If False or PyWavelets unavailable, use hardcoded fallback.
    
    Returns:
        dec_lo: Decomposition lowpass filter
        dec_hi: Decomposition highpass filter
        rec_lo: Reconstruction lowpass filter
        rec_hi: Reconstruction highpass filter
    
    Raises:
        ValueError: If wavelet is not found in PyWavelets and not in hardcoded list
        ImportError: If use_pywt=True but PyWavelets is not available
    """
    wavelet = wavelet_name.lower()
    
    # Primary method: Use PyWavelets Wavelet object (safer and supports more wavelets)
    if use_pywt:
        if not PYWT_AVAILABLE:
            raise ImportError(
                f"PyWavelets is not available. Install with: pip install PyWavelets\n"
                f"Alternatively, set use_pywt=False to use hardcoded coefficients "
                f"(limited to: haar, db1, db2, db4, db6, db8)"
            )
        
        try:
            w = pywt.Wavelet(wavelet)
            return (
                np.array(w.dec_lo, dtype=np.float64),
                np.array(w.dec_hi, dtype=np.float64),
                np.array(w.rec_lo, dtype=np.float64),
                np.array(w.rec_hi, dtype=np.float64)
            )
        except ValueError as e:
            # Wavelet not found in PyWavelets, try fallback
            if "Unknown wavelet" in str(e) or "not found" in str(e).lower():
                pass  # Fall through to hardcoded fallback
            else:
                raise
        except Exception as e:
            raise RuntimeError(f"Error accessing PyWavelets Wavelet object: {e}")
    
    # Fallback: Hardcoded coefficients (only if PyWavelets unavailable or use_pywt=False)
    
    if wavelet in ['haar', 'db1']:
        # Haar wavelet
        dec_lo = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    elif wavelet == 'db2':
        # Daubechies-2
        dec_lo = np.array([
            (1 + np.sqrt(3)) / (4 * np.sqrt(2)),
            (3 + np.sqrt(3)) / (4 * np.sqrt(2)),
            (3 - np.sqrt(3)) / (4 * np.sqrt(2)),
            (1 - np.sqrt(3)) / (4 * np.sqrt(2))
        ])
    elif wavelet == 'db4':
        # Daubechies-4
        dec_lo = np.array([
            0.23037781330885523,
            0.7148465705525415,
            0.6308807679295904,
            -0.02798376941698385,
            -0.18703481171888114,
            0.03084138183556076,
            0.03288301166688519,
            -0.010597401784997278
        ])
    elif wavelet == 'db6':
        # Daubechies-6
        dec_lo = np.array([
            0.11154074335008017,
            0.4946238903984533,
            0.7511339080215775,
            0.3152503517092432,
            -0.22626469396516913,
            -0.12976686756709563,
            0.09750160558707936,
            0.02752286553001629,
            -0.031582039318031156,
            0.0005538422009938016,
            0.004777257511010651,
            -0.001077301085308479
        ])
    elif wavelet == 'db8':
        # Daubechies-8
        dec_lo = np.array([
            0.05441584224308161,
            0.3128715909144659,
            0.6756307362980128,
            0.5853546836548691,
            -0.015829105256023893,
            -0.2840155429624281,
            0.00047248457399797254,
            0.12854627665505343,
            -0.017440411845200255,
            -0.04408825393106472,
            0.013981027917015516,
            0.008746094047015655,
            -0.004870352993451574,
            -0.000391740372995977,
            0.0006754494059985568,
            -0.00011747678400228192
        ])
    else:
        # Wavelet not in hardcoded list
        if PYWT_AVAILABLE and use_pywt:
            # This shouldn't happen if PyWavelets worked, but just in case
            raise ValueError(
                f"Unknown wavelet: {wavelet_name}. "
                f"PyWavelets is available but wavelet not found. "
                f"Try: pywt.wavelist() to see available wavelets."
            )
        else:
            raise ValueError(
                f"Unknown wavelet: {wavelet_name}. "
                f"Hardcoded fallback only supports: haar, db1, db2, db4, db6, db8. "
                f"Install PyWavelets (pip install PyWavelets) to use more wavelets."
            )
    
    # Create QMF highpass filter from lowpass
    # Standard QMF relation: g[n] = (-1)^n * h[N-1-n]
    # But PyWavelets uses: g[n] = (-1)^(n+1) * h[N-1-n]
    N = len(dec_lo)
    dec_hi = np.array([(-1)**(n+1) * dec_lo[N-1-n] for n in range(N)])
    
    # Reconstruction filters (time-reversed)
    rec_lo = dec_lo[::-1]
    rec_hi = dec_hi[::-1]
    
    return dec_lo, dec_hi, rec_lo, rec_hi


def periodic_convolve(signal: np.ndarray, filter_coeffs: np.ndarray) -> np.ndarray:
    """
    Perform periodic (circular) convolution.
    
    Args:
        signal: Input signal
        filter_coeffs: Filter coefficients
        
    Returns:
        Convolution result with same length as signal
    """
    N = len(signal)
    M = len(filter_coeffs)
    result = np.zeros(N)
    
    for n in range(N):
        for k in range(M):
            idx = (n - k) % N
            result[n] += filter_coeffs[k] * signal[idx]
    
    return result


def morlet_wavelet(t: np.ndarray, omega0: float = 5.0) -> np.ndarray:
    """
    Generate Morlet mother wavelet.
    
    The Morlet wavelet is a complex sinusoid modulated by a Gaussian:
    ψ(t) = π^(-1/4) * exp(iω₀t) * exp(-t²/2)
    
    Args:
        t: Time array
        omega0: Central frequency (default 5.0 for good time-frequency localization)
    
    Returns:
        Complex Morlet wavelet values
    """
    normalization = np.pi ** (-0.25)
    oscillation = np.exp(1j * omega0 * t)
    envelope = np.exp(-t**2 / 2)
    return normalization * oscillation * envelope

