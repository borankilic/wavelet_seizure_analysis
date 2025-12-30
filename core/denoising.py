"""
Wavelet Denoising
==================
Signal denoising using DWT soft/hard thresholding.
"""

import numpy as np
from .dwt_decomposition import dwt
from .dwt_reconstruction import idwt


def soft_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply soft thresholding to wavelet coefficients.
    
    Soft thresholding: sign(x) * max(|x| - threshold, 0)
    
    Args:
        coefficients: Wavelet coefficients
        threshold: Threshold value
    
    Returns:
        Thresholded coefficients
    """
    return np.sign(coefficients) * np.maximum(np.abs(coefficients) - threshold, 0)


def hard_threshold(coefficients: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply hard thresholding to wavelet coefficients.
    
    Hard thresholding: x if |x| > threshold, else 0
    
    Args:
        coefficients: Wavelet coefficients
        threshold: Threshold value
    
    Returns:
        Thresholded coefficients
    """
    result = coefficients.copy()
    result[np.abs(result) < threshold] = 0
    return result


def estimate_noise_std(detail_coeffs: np.ndarray) -> float:
    """
    Estimate noise standard deviation using MAD estimator.
    
    Uses the finest detail coefficients to estimate noise level.
    MAD = Median Absolute Deviation / 0.6745
    
    Args:
        detail_coeffs: Finest level detail coefficients
    
    Returns:
        Estimated noise standard deviation
    """
    return np.median(np.abs(detail_coeffs)) / 0.6745


def denoise_signal(signal: np.ndarray, wavelet: str = 'db4',
                   level: int = None, threshold_type: str = 'soft',
                   threshold_mode: str = 'universal', verbose: bool = False) -> np.ndarray:
    """
    Denoise signal using DWT soft/hard thresholding.
    
    Args:
        signal: Noisy input signal
        wavelet: Wavelet to use
        level: Decomposition level
        threshold_type: 'soft' or 'hard'
        threshold_mode: 'universal' (VisuShrink) or 'sure' (SureShrink)
        verbose: Print denoising information
    
    Returns:
        Denoised signal
    """
    original_length = len(signal)
    
    # Decompose
    coeffs = dwt(signal, wavelet, level, verbose=False)
    
    # Estimate noise from finest detail coefficients
    sigma = estimate_noise_std(coeffs[-1])
    
    # Calculate threshold
    n = len(signal)
    if threshold_mode == 'universal':
        # Universal threshold (VisuShrink)
        threshold = sigma * np.sqrt(2 * np.log(n))
    else:
        # Default to universal
        threshold = sigma * np.sqrt(2 * np.log(n))
    
    if verbose:
        print(f"  Noise std estimate: {sigma:.4f}")
        print(f"  Threshold: {threshold:.4f} ({threshold_type})")
    
    # Apply thresholding to detail coefficients (not approximation)
    thresholded_coeffs = [coeffs[0]]  # Keep approximation unchanged
    threshold_func = soft_threshold if threshold_type == 'soft' else hard_threshold
    
    for detail in coeffs[1:]:
        thresholded_coeffs.append(threshold_func(detail, threshold))
    
    # Reconstruct
    denoised = idwt(thresholded_coeffs, wavelet, original_length, verbose=False)
    
    return denoised

