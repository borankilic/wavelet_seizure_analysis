"""
Signal Plotting Functions
===========================
Time-domain signal visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union


def plot_signals(signals: Union[np.ndarray, List[np.ndarray]],
                 labels: Optional[List[str]] = None,
                 sampling_rate: float = 1.0,
                 title: str = "Signal Plot",
                 xlabel: str = "Time (s)",
                 ylabel: str = "Amplitude",
                 figsize: Tuple[int, int] = (12, 4),
                 colors: Optional[List[str]] = None,
                 alpha: float = 0.8,
                 grid: bool = True,
                 stacked: bool = True) -> plt.Figure:
    """
    Generalized time-signal plotter for single or multiple sequences.
    
    If multiple sequences are provided and stacked=True, creates vertically 
    stacked subplots. Otherwise, overlays them on the same axes.
    
    Args:
        signals: Single signal array or list of signal arrays
        labels: Labels for each signal
        sampling_rate: Sampling rate in Hz
        title: Main title for the plot
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        colors: Colors for each signal
        alpha: Transparency
        grid: Show grid
        stacked: Stack multiple signals vertically
    
    Returns:
        matplotlib Figure object
    """
    # Handle single signal input
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]
    elif isinstance(signals, np.ndarray) and signals.ndim == 2:
        signals = [signals[i] for i in range(signals.shape[0])]
    
    n_signals = len(signals)
    
    # Default colors
    if colors is None:
        cmap = plt.cm.Set2
        colors = [cmap(i / max(n_signals - 1, 1)) for i in range(n_signals)]
    
    # Default labels
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(n_signals)]
    
    if stacked and n_signals > 1:
        # Stacked subplots
        fig, axes = plt.subplots(n_signals, 1, figsize=(figsize[0], figsize[1] * n_signals),
                                  sharex=True)
        if n_signals == 1:
            axes = [axes]
        
        for i, (signal, label, color) in enumerate(zip(signals, labels, colors)):
            t = np.arange(len(signal)) / sampling_rate
            axes[i].plot(t, signal, color=color, alpha=alpha, linewidth=2)
            axes[i].set_ylabel(label, fontsize=18)
            if grid:
                axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(t[0], t[-1])
        
        axes[-1].set_xlabel(xlabel, fontsize=18)
        fig.suptitle(title, fontsize=24, fontweight='bold')
        plt.tight_layout()
    else:
        # Overlay on same axes
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        for signal, label, color in zip(signals, labels, colors):
            t = np.arange(len(signal)) / sampling_rate
            ax.plot(t, signal, color=color, alpha=alpha, label=label, linewidth=2)
        
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=24, fontweight='bold')
        if grid:
            ax.grid(True, alpha=0.3)
        if n_signals > 1:
            ax.legend(loc='upper right', fontsize=20)
        ax.set_xlim(t[0], t[-1])
        plt.tight_layout()
    
    return fig


def plot_signal_comparison(original: np.ndarray, processed: np.ndarray,
                           sampling_rate: float = 1.0,
                           labels: List[str] = None,
                           title: str = "Signal Comparison") -> plt.Figure:
    """
    Plot original and processed signals for comparison.
    
    Args:
        original: Original signal
        processed: Processed signal
        sampling_rate: Sampling rate
        labels: Labels for [original, processed]
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    if labels is None:
        labels = ["Original", "Processed"]
    
    return plot_signals(
        [original, processed],
        labels=labels,
        sampling_rate=sampling_rate,
        title=title,
        colors=['#2E86AB', '#E94F37'],
        stacked=True
    )
