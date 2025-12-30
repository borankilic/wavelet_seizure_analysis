"""
Core DSP Module
================
Custom wavelet transform implementations organized by functionality.
"""

# Filters and basic operations
from .filters import (
    get_wavelet_filters,
    periodic_convolve,
    morlet_wavelet
)

# Continuous Wavelet Transform
from .cwt import (
    cwt,
    scales_to_frequencies
)

# DWT Decomposition
from .dwt_decomposition import (
    dwt_single_level,
    dwt
)

# DWT Reconstruction
from .dwt_reconstruction import (
    idwt_single_level,
    idwt
)

# Denoising
from .denoising import (
    soft_threshold,
    hard_threshold,
    estimate_noise_std,
    denoise_signal
)

__all__ = [
    # Filters
    'get_wavelet_filters', 'periodic_convolve', 'morlet_wavelet',
    
    # CWT
    'cwt', 'scales_to_frequencies',
    
    # DWT Decomposition
    'dwt_single_level', 'dwt',
    
    # DWT Reconstruction
    'idwt_single_level', 'idwt',
    
    # Denoising
    'soft_threshold', 'hard_threshold', 'estimate_noise_std', 'denoise_signal'
]
