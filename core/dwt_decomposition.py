"""
DWT Decomposition - Mallat's Algorithm
========================================
Fast Discrete Wavelet Transform decomposition using filter banks.
"""

import numpy as np
from typing import List, Tuple
from .filters import get_wavelet_filters, periodic_convolve


def dwt_single_level(signal: np.ndarray, dec_lo: np.ndarray, 
                     dec_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single level DWT decomposition using Mallat's algorithm.
    
    Implements the filter bank:
    - Convolve with lowpass filter → downsample → approximation coefficients
    - Convolve with highpass filter → downsample → detail coefficients
    
    Args:
        signal: Input signal
        dec_lo: Decomposition lowpass filter
        dec_hi: Decomposition highpass filter
    
    Returns:
        approx: Approximation coefficients (low frequency)
        detail: Detail coefficients (high frequency)
    """
    # Pad odd-length signals to even
    if len(signal) % 2 == 1:
        signal = np.append(signal, 0)
    
    # Periodic convolution
    conv_lo = periodic_convolve(signal, dec_lo)
    conv_hi = periodic_convolve(signal, dec_hi)
    
    # Shift for proper phase alignment
    shift = len(dec_lo) // 2
    conv_lo = np.roll(conv_lo, -shift)
    conv_hi = np.roll(conv_hi, -shift)
    
    # Downsample by 2 (take even indices after shift)
    approx = conv_lo[::2]
    detail = conv_hi[::2]
    
    return approx, detail


def dwt(signal: np.ndarray, wavelet: str = 'db4', 
        level: int = None, verbose: bool = False) -> List[np.ndarray]:
    """
    Multi-level Discrete Wavelet Transform using Mallat's Algorithm.
    
    Decomposes signal into approximation and detail coefficients at each level.
    
    Args:
        signal: Input signal
        wavelet: Wavelet name
        level: Number of decomposition levels (default: maximum possible)
        verbose: Print decomposition progress
    
    Returns:
        List of coefficients: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        where cA_n is the final approximation and cD_i are details
    """
    # Get filters
    dec_lo, dec_hi, _, _ = get_wavelet_filters(wavelet)
    
    # Store original length
    original_length = len(signal)
    
    # Determine maximum level
    max_level = int(np.floor(np.log2(len(signal) / (len(dec_lo) - 1))))
    max_level = max(1, max_level)
    
    if level is None:
        level = max_level
    else:
        level = min(level, max_level)
    
    if verbose:
        print(f"  DWT decomposition: {len(signal)} samples → {level} levels (wavelet: {wavelet})")
    
    # Decomposition
    details = []
    approx = signal.copy()
    
    for j in range(level):
        if verbose:
            print(f"    Level {j+1}: length {len(approx)} → {len(approx)//2}")
        approx, detail = dwt_single_level(approx, dec_lo, dec_hi)
        details.append(detail)
    
    # Build coefficients list: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    # where cD_n is the coarsest detail (last computed) and cD_1 is finest (first computed)
    coefficients = [approx] + details[::-1]
    
    if verbose:
        print(f"  Final approximation length: {len(approx)}")
    
    return coefficients

