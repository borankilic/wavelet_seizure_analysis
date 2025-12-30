"""
Visualization Module
=====================
Signal and scalogram plotting functions organized by type.
"""

# Signal plots
from .signal_plots import (
    plot_signals,
    plot_signal_comparison
)

# Scalogram and wavelet coefficient plots
from .scalogram_plots import (
    plot_scalogram,
    plot_dwt_coefficients,
    plot_brain_wave_bands
)

# Feature and classification result plots
from .feature_plots import (
    plot_feature_projection,
    plot_confusion_matrix
)

__all__ = [
    # Signal plots
    'plot_signals', 'plot_signal_comparison',
    
    # Scalogram plots
    'plot_scalogram', 'plot_dwt_coefficients', 'plot_brain_wave_bands',
    
    # Feature plots
    'plot_feature_projection', 'plot_confusion_matrix'
]
