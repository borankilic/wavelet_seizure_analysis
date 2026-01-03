"""
Continuous Wavelet Transform (CWT)
====================================
Manual CWT implementation using convolution with scaled wavelets.
"""

import numpy as np
from .filters import morlet_wavelet


def cwt(signal: np.ndarray, scales: np.ndarray, wavelet: str = 'morlet',
        sampling_rate: float = 1.0, omega0: float = 5.0) -> np.ndarray:
    """
    Continuous Wavelet Transform using convolution with scaled wavelets.
    
    Computes the scalogram by convolving the signal with dilated/compressed
    versions of the mother wavelet at each scale.
    
    Args:
        signal: Input signal (1D array)
        scales: Array of scales to compute (larger = lower frequency)
        wavelet: Wavelet type ('morlet')
        sampling_rate: Sampling rate of the signal
        omega0: Central frequency for Morlet wavelet
    
    Returns:
        CWT coefficients: 2D array of shape (n_scales, signal_length)
    """
    n_samples = len(signal)
    n_scales = len(scales)
    
    # Initialize output
    coefficients = np.zeros((n_scales, n_samples), dtype=complex)
    
    for i, scale in enumerate(scales):
        # Create time vector for wavelet at this scale
        # Wavelet support depends on scale
        wavelet_length = min(10 * int(scale) + 1, n_samples)
        wavelet_length = max(wavelet_length, 11)  # Minimum length
        
        # Make odd for symmetric centering
        if wavelet_length % 2 == 0:
            wavelet_length += 1
        
        half_len = wavelet_length // 2
        # IMPORTANT: Use a dimensionless time variable for the mother wavelet.
        # In the CWT, the scaled wavelet is ψ((n)/scale). The sampling_rate is
        # only needed when converting scales↔frequencies and for plotting axes.
        t = np.arange(-half_len, half_len + 1) / scale
        
        # Generate scaled wavelet
        if wavelet.lower() == 'morlet':
            psi = morlet_wavelet(t, omega0)
        else:
            raise ValueError(f"Unknown wavelet: {wavelet}")
        
        # Normalize by sqrt(scale) for energy preservation
        psi = psi / np.sqrt(scale)
        
        # Convolve signal with wavelet (use conjugate for proper CWT)
        conv_result = np.convolve(signal, np.conj(psi[::-1]), mode='same')
        # Ensure output length matches input
        coefficients[i, :] = conv_result[:n_samples]
    
    return coefficients


def scales_to_frequencies(scales: np.ndarray, wavelet: str = 'morlet',
                          sampling_rate: float = 1.0, omega0: float = 5.0) -> np.ndarray:
    """
    Convert CWT scales to corresponding frequencies.
    
    Args:
        scales: Array of scales
        wavelet: Wavelet type
        sampling_rate: Sampling rate
        omega0: Central frequency for Morlet
    
    Returns:
        Array of frequencies corresponding to each scale
    """
    if wavelet.lower() == 'morlet':
        # For Morlet: f = (omega0 / (2π)) * (sampling_rate / scale)
        # omega0 is the (dimensionless) central angular frequency of the Morlet
        # mother wavelet defined over the dimensionless time variable.
        frequencies = omega0 * sampling_rate / (2 * np.pi * scales)
    else:
        frequencies = sampling_rate / scales
    
    return frequencies

