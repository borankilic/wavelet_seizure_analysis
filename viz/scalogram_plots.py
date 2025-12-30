"""
Scalogram and Wavelet Coefficient Plotting
===========================================
Visualization of CWT and DWT results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import List, Optional, Tuple


def plot_scalogram(coefficients: np.ndarray,
                   scales: np.ndarray = None,
                   frequencies: np.ndarray = None,
                   sampling_rate: float = 1.0,
                   title: str = "CWT Scalogram",
                   cmap: str = 'viridis',
                   log_scale: bool = False,
                   colorbar: bool = True,
                   figsize: Tuple[int, int] = (12, 6),
                   show_frequencies: bool = True) -> plt.Figure:
    """
    Plot CWT scalogram (time-frequency representation).
    
    Visualizes the magnitude of CWT coefficients to show time-frequency 
    "hotspots" where signal energy is concentrated.
    
    Args:
        coefficients: CWT coefficients (n_scales x n_samples)
        scales: Array of scales used
        frequencies: Corresponding frequencies (if None, uses scales)
        sampling_rate: Sampling rate
        title: Plot title
        cmap: Colormap
        log_scale: Use logarithmic color scale
        colorbar: Show colorbar
        figsize: Figure size
        show_frequencies: Show frequency axis instead of scales
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Take magnitude if complex
    if np.iscomplexobj(coefficients):
        power = np.abs(coefficients) ** 2
    else:
        power = coefficients ** 2
    
    # Time axis
    n_samples = coefficients.shape[1]
    t = np.arange(n_samples) / sampling_rate
    
    # Y-axis (frequencies or scales)
    if show_frequencies and frequencies is not None:
        y = frequencies
        ylabel = "Frequency (Hz)"
    elif scales is not None:
        y = scales
        ylabel = "Scale"
    else:
        y = np.arange(coefficients.shape[0])
        ylabel = "Scale Index"
    
    # Plot
    if log_scale:
        norm = LogNorm(vmin=power[power > 0].min(), vmax=power.max())
        im = ax.pcolormesh(t, y, power, cmap=cmap, norm=norm, shading='auto')
    else:
        im = ax.pcolormesh(t, y, power, cmap=cmap, shading='auto')
    
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, label='Power')
    
    plt.tight_layout()
    return fig


def plot_dwt_coefficients(coefficients: List[np.ndarray],
                          wavelet: str = 'db4',
                          sampling_rate: float = 1.0,
                          level_names: List[str] = None,
                          title: str = "DWT Decomposition",
                          figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot DWT decomposition coefficients at each level.
    
    Args:
        coefficients: List [cA_n, cD_n, ..., cD_1] from dwt()
        wavelet: Wavelet used
        sampling_rate: Sampling rate
        level_names: Names for each level (e.g., ['Approx', 'D5', 'D4', ...])
        title: Main title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    n_levels = len(coefficients)
    
    if level_names is None:
        level_names = ['Approx'] + [f'Detail {n_levels - 1 - i}' for i in range(1, n_levels)]
    
    # Create color palette
    colors = ['#1a535c', '#4ecdc4', '#ff6b6b', '#ffe66d', '#95d5b2', '#74c69d', '#52b788', '#40916c']
    
    fig, axes = plt.subplots(n_levels, 1, figsize=figsize, sharex=False)
    
    for i, (coeff, name, color) in enumerate(zip(coefficients, level_names, colors)):
        # Time axis scaled for each level
        t = np.arange(len(coeff)) / (sampling_rate / (2 ** (n_levels - 1 - i if i > 0 else n_levels - 1)))
        
        axes[i].plot(coeff, color=color, linewidth=0.7, alpha=0.9)
        axes[i].fill_between(range(len(coeff)), coeff, alpha=0.3, color=color)
        axes[i].set_ylabel(name, fontsize=9, rotation=0, ha='right', va='center')
        axes[i].yaxis.set_label_coords(-0.08, 0.5)
        axes[i].grid(True, alpha=0.2)
        axes[i].set_xlim(0, len(coeff))
    
    axes[-1].set_xlabel("Sample Index", fontsize=11)
    fig.suptitle(f"{title} ({wavelet.upper()})", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_brain_wave_bands(coefficients: List[np.ndarray],
                          band_names: List[str] = None,
                          title: str = "EEG Brain Wave Decomposition",
                          figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot DWT coefficients mapped to EEG brain wave frequency bands.
    
    Standard EEG bands:
    - Delta (δ): 0.5-4 Hz (deep sleep)
    - Theta (θ): 4-8 Hz (drowsiness, meditation)
    - Alpha (α): 8-13 Hz (relaxed, eyes closed)
    - Beta (β): 13-30 Hz (active thinking)
    - Gamma (γ): 30-100 Hz (cognitive processing)
    
    Args:
        coefficients: DWT coefficients [cA, cD5, cD4, cD3, cD2, cD1]
        band_names: Custom band names
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    if band_names is None:
        # Default EEG band names
        if len(coefficients) >= 6:
            band_names = ['δ (Delta)', 'θ (Theta)', 'α (Alpha)', 'β (Beta)', 'γ (Gamma)', 'HF Noise']
        else:
            band_names = ['Approx'] + [f'Detail {len(coefficients) - 1 - i}' for i in range(1, len(coefficients))]
    
    # EEG-inspired colors
    colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51', '#9b2226']
    
    n_bands = min(len(coefficients), len(band_names))
    fig, axes = plt.subplots(n_bands, 1, figsize=figsize)
    
    for i in range(n_bands):
        coeff = coefficients[i]
        color = colors[i % len(colors)]
        
        axes[i].plot(coeff, color=color, linewidth=0.6, alpha=0.9)
        axes[i].fill_between(range(len(coeff)), coeff, alpha=0.25, color=color)
        axes[i].set_ylabel(band_names[i], fontsize=10, fontweight='bold')
        axes[i].grid(True, alpha=0.2)
        axes[i].set_xlim(0, len(coeff))
        
        # Add energy annotation
        energy = np.sum(coeff ** 2)
        axes[i].annotate(f'E = {energy:.2f}', xy=(0.98, 0.85), 
                        xycoords='axes fraction', fontsize=8, ha='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[-1].set_xlabel("Sample Index", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

