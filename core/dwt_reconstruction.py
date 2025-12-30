"""
DWT Reconstruction - Inverse Mallat's Algorithm
================================================
Inverse Fast Discrete Wavelet Transform using synthesis filter banks.
"""

import numpy as np
from typing import List
from .filters import get_wavelet_filters, periodic_convolve


def idwt_single_level(approx: np.ndarray, detail: np.ndarray,
                      rec_lo: np.ndarray, rec_hi: np.ndarray,
                      output_length: int = None) -> np.ndarray:
    """
    Single level inverse DWT reconstruction.
    
    Implements the synthesis filter bank:
    - Upsample coefficients → convolve with reconstruction filters → sum
    
    Args:
        approx: Approximation coefficients
        detail: Detail coefficients
        rec_lo: Reconstruction lowpass filter
        rec_hi: Reconstruction highpass filter
        output_length: Desired output length
    
    Returns:
        Reconstructed signal
    """
    n_coeffs = len(approx)
    if output_length is None:
        output_length = n_coeffs * 2
    
    # Upsample (insert zeros between samples)
    up_approx = np.zeros(output_length)
    up_detail = np.zeros(output_length)
    
    # Place coefficients at even indices
    for i in range(n_coeffs):
        if 2*i < output_length:
            up_approx[2*i] = approx[i]
            up_detail[2*i] = detail[i]
    
    # Periodic convolution with reconstruction filters
    rec_approx = periodic_convolve(up_approx, rec_lo)
    rec_detail = periodic_convolve(up_detail, rec_hi)
    
    # Shift for phase alignment
    shift = len(rec_lo) // 2 - 1
    rec_approx = np.roll(rec_approx, -shift)
    rec_detail = np.roll(rec_detail, -shift)
    
    # Sum contributions
    reconstructed = rec_approx + rec_detail
    
    return reconstructed


def idwt(coefficients: List[np.ndarray], wavelet: str = 'db4',
         original_length: int = None, verbose: bool = False) -> np.ndarray:
    """
    Multi-level Inverse Discrete Wavelet Transform.
    
    Reconstructs signal from wavelet coefficients.
    
    Args:
        coefficients: List [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        wavelet: Wavelet name
        original_length: Original signal length (for trimming)
        verbose: Print reconstruction progress
    
    Returns:
        Reconstructed signal
    """
    # Get filters
    _, _, rec_lo, rec_hi = get_wavelet_filters(wavelet)
    
    if verbose:
        print(f"  IDWT reconstruction: {len(coefficients)-1} levels (wavelet: {wavelet})")
    
    # Start with coarsest approximation
    approx = coefficients[0].copy()
    
    # Reconstruct from coarsest to finest
    for i in range(1, len(coefficients)):
        detail = coefficients[i]
        output_length = len(detail) * 2
        if verbose:
            print(f"    Level {i}: length {len(approx)} → {output_length}")
        approx = idwt_single_level(approx, detail, rec_lo, rec_hi, output_length)
    
    # Trim to original length if specified
    if original_length is not None and len(approx) > original_length:
        approx = approx[:original_length]
        if verbose:
            print(f"  Trimmed to original length: {original_length}")
    
    return approx

