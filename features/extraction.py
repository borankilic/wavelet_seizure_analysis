"""
Feature Extraction Module
==========================
Extract statistical features from DWT coefficients for EEG classification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import dwt


# =============================================================================
# EEG FREQUENCY BANDS
# =============================================================================

# Standard EEG frequency bands (at 173.61 Hz sampling rate)
# Level mapping for 5-level decomposition at ~173 Hz:
#   D1: 43.4-86.8 Hz (Gamma + High-freq noise)
#   D2: 21.7-43.4 Hz (Beta + Low Gamma)
#   D3: 10.8-21.7 Hz (Alpha + Low Beta)
#   D4: 5.4-10.8 Hz  (Theta + Low Alpha)
#   D5: 2.7-5.4 Hz   (Delta + Low Theta)
#   A5: 0-2.7 Hz     (Sub-Delta)

EEG_BANDS = {
    'delta': (0.5, 4),      # Deep sleep
    'theta': (4, 8),        # Drowsiness, meditation
    'alpha': (8, 13),       # Relaxed, eyes closed
    'beta': (13, 30),       # Active thinking
    'gamma': (30, 100)      # Cognitive processing
}

BAND_LEVEL_MAPPING = {
    # For 5-level decomposition at 173.61 Hz
    'delta': 'A5',
    'theta': 'D5',
    'alpha': 'D4',
    'beta': 'D3',
    'gamma': 'D2'
}


# =============================================================================
# STATISTICAL FEATURES
# =============================================================================

def compute_energy(coefficients: np.ndarray) -> float:
    """
    Compute energy of wavelet coefficients.
    
    Energy = sum of squared coefficients
    
    Args:
        coefficients: Wavelet coefficients
    
    Returns:
        Energy value
    """
    return np.sum(coefficients ** 2)


def compute_normalized_energy(coefficients: np.ndarray, total_energy: float = None) -> float:
    """
    Compute normalized (relative) energy.
    
    Args:
        coefficients: Wavelet coefficients
        total_energy: Total energy for normalization
    
    Returns:
        Normalized energy (0-1)
    """
    energy = compute_energy(coefficients)
    if total_energy is None or total_energy == 0:
        return 0.0
    return energy / total_energy


def compute_entropy(coefficients: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute Shannon entropy of wavelet coefficients.
    
    Uses normalized squared coefficients as probability distribution.
    
    Args:
        coefficients: Wavelet coefficients
        eps: Small value to avoid log(0)
    
    Returns:
        Shannon entropy
    """
    # Normalize to get probability distribution
    energy = coefficients ** 2
    total = np.sum(energy)
    
    if total < eps:
        return 0.0
    
    prob = energy / total
    prob = prob[prob > eps]  # Remove zeros
    
    return -np.sum(prob * np.log2(prob))


def compute_log_energy_entropy(coefficients: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute log energy entropy.
    
    Log energy entropy = sum of log(coefficients^2)
    
    Args:
        coefficients: Wavelet coefficients
        eps: Small value to avoid log(0)
    
    Returns:
        Log energy entropy
    """
    energy = coefficients ** 2
    energy = energy[energy > eps]
    return np.sum(np.log(energy))


def compute_statistical_features(coefficients: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistical features from coefficients.
    
    Args:
        coefficients: Wavelet coefficients
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(coefficients)
    features['std'] = np.std(coefficients)
    features['var'] = np.var(coefficients)
    features['min'] = np.min(coefficients)
    features['max'] = np.max(coefficients)
    features['range'] = features['max'] - features['min']
    
    # Higher-order statistics
    features['skewness'] = compute_skewness(coefficients)
    features['kurtosis'] = compute_kurtosis(coefficients)
    
    # Energy-based
    features['energy'] = compute_energy(coefficients)
    features['rms'] = np.sqrt(np.mean(coefficients ** 2))
    
    # Entropy-based
    features['entropy'] = compute_entropy(coefficients)
    features['log_energy_entropy'] = compute_log_energy_entropy(coefficients)
    
    # Percentiles
    features['median'] = np.median(coefficients)
    features['iqr'] = np.percentile(coefficients, 75) - np.percentile(coefficients, 25)
    
    # Zero crossings
    features['zero_crossings'] = np.sum(np.diff(np.sign(coefficients)) != 0)
    
    return features


def compute_skewness(x: np.ndarray) -> float:
    """Compute skewness of data."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.sum(((x - mean) / std) ** 3) / n


def compute_kurtosis(x: np.ndarray) -> float:
    """Compute kurtosis of data."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.sum(((x - mean) / std) ** 4) / n - 3


# =============================================================================
# DWT FEATURE EXTRACTION
# =============================================================================

def extract_dwt_features(signal: np.ndarray,
                         wavelet: str = 'db4',
                         level: int = 5,
                         feature_set: str = 'standard') -> np.ndarray:
    """
    Extract features from DWT decomposition of EEG signal.
    
    Decomposes the signal into frequency bands corresponding to
    brain wave rhythms (delta, theta, alpha, beta, gamma).
    
    Args:
        signal: Input EEG signal
        wavelet: Wavelet to use
        level: Decomposition level (5 for EEG)
        feature_set: 'standard' (energy, entropy, std) or 'full' (all stats)
    
    Returns:
        Feature vector
    """
    # Perform DWT decomposition
    coeffs = dwt(signal, wavelet, level)
    
    # Compute total energy for normalization
    total_energy = sum(compute_energy(c) for c in coeffs)
    
    features = []
    
    for i, coeff in enumerate(coeffs):
        if feature_set == 'standard':
            # Standard features: energy, entropy, std
            features.extend([
                compute_normalized_energy(coeff, total_energy),
                compute_entropy(coeff),
                np.std(coeff)
            ])
        elif feature_set == 'full':
            # Full statistical features
            stats = compute_statistical_features(coeff)
            features.extend([
                stats['energy'] / (total_energy + 1e-10),  # Normalized energy
                stats['entropy'],
                stats['std'],
                stats['mean'],
                stats['skewness'],
                stats['kurtosis'],
                stats['rms']
            ])
        else:
            # Minimal: just energy and entropy
            features.extend([
                compute_normalized_energy(coeff, total_energy),
                compute_entropy(coeff)
            ])
    
    return np.array(features)


def extract_features_batch(signals: np.ndarray,
                           wavelet: str = 'db4',
                           level: int = 5,
                           feature_set: str = 'standard',
                           verbose: bool = False) -> np.ndarray:
    """
    Extract DWT features from multiple signals.
    
    Args:
        signals: 2D array of signals (n_samples, signal_length)
        wavelet: Wavelet to use
        level: Decomposition level
        feature_set: Feature set type
        verbose: Print progress
    
    Returns:
        Feature matrix (n_samples, n_features)
    """
    n_samples = len(signals)
    features_list = []
    
    for i, signal in enumerate(signals):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processing signal {i+1}/{n_samples}")
        
        features = extract_dwt_features(signal, wavelet, level, feature_set)
        features_list.append(features)
    
    return np.array(features_list)


def get_feature_names(coefficients: List[np.ndarray] = None,
                     level: int = None,
                     feature_set: str = 'standard') -> List[str]:
    """
    Get names for extracted features.
    
    Automatically determines the level from coefficients if provided.
    If coefficients are not provided, uses the level parameter.
    
    Args:
        coefficients: List of DWT coefficients (first is approximation, rest are details)
                     If provided, level will be inferred from this.
        level: Decomposition level (used only if coefficients not provided)
        feature_set: Feature set type ('standard', 'full', or 'minimal')
    
    Returns:
        List of feature names
    
    Examples:
        >>> coeffs = dwt(signal, 'db4', level=5)
        >>> names = get_feature_names(coefficients=coeffs, feature_set='standard')
        >>> # Or with explicit level (backward compatibility):
        >>> names = get_feature_names(level=5, feature_set='standard')
    """
    # Infer level from coefficients if provided
    if coefficients is not None:
        # Number of coefficients = level + 1 (1 approximation + level details)
        # So level = len(coefficients) - 1
        level = len(coefficients) - 1
    elif level is None:
        # Default to 5 if neither provided
        level = 5
    
    # Generate band names: A_level, D_level, D_level-1, ..., D1
    band_names = ['A' + str(level)] + ['D' + str(level - i) for i in range(level)]
    
    # Determine stat names based on feature_set
    if feature_set == 'standard':
        stat_names = ['energy', 'entropy', 'std']
    elif feature_set == 'full':
        stat_names = ['energy', 'entropy', 'std', 'mean', 'skew', 'kurt', 'rms']
    else:  # 'minimal'
        stat_names = ['energy', 'entropy']
    
    # Generate feature names: band_stat
    names = []
    for band in band_names:
        for stat in stat_names:
            names.append(f"{band}_{stat}")
    
    return names


# =============================================================================
# BAND-SPECIFIC FEATURES
# =============================================================================

def extract_band_energies(signal: np.ndarray,
                          wavelet: str = 'db4',
                          level: int = 5) -> Dict[str, float]:
    """
    Extract energy in each frequency band.
    
    Maps DWT levels to EEG bands:
    - Delta: A5 (0-2.7 Hz)
    - Theta: D5 (2.7-5.4 Hz)
    - Alpha: D4 (5.4-10.8 Hz)
    - Beta: D3 (10.8-21.7 Hz)
    - Gamma: D2 + D1 (21.7-86.8 Hz)
    
    Args:
        signal: Input signal
        wavelet: Wavelet to use
        level: Decomposition level
    
    Returns:
        Dictionary of band energies
    """
    coeffs = dwt(signal, wavelet, level)
    total_energy = sum(compute_energy(c) for c in coeffs)
    
    # Map coefficients to bands (assuming 5-level decomposition)
    band_energies = {
        'delta': compute_normalized_energy(coeffs[0], total_energy),  # A5
        'theta': compute_normalized_energy(coeffs[1], total_energy),  # D5
        'alpha': compute_normalized_energy(coeffs[2], total_energy),  # D4
        'beta': compute_normalized_energy(coeffs[3], total_energy),   # D3
        'gamma': compute_normalized_energy(coeffs[4], total_energy) + 
                 (compute_normalized_energy(coeffs[5], total_energy) if len(coeffs) > 5 else 0)  # D2 + D1
    }
    
    return band_energies


def compute_band_ratios(band_energies: Dict[str, float]) -> Dict[str, float]:
    """
    Compute ratios between frequency bands.
    
    Common ratios used in EEG analysis for seizure detection.
    
    Args:
        band_energies: Dictionary of band energies
    
    Returns:
        Dictionary of band ratios
    """
    eps = 1e-10
    
    ratios = {
        'theta_alpha': band_energies['theta'] / (band_energies['alpha'] + eps),
        'delta_alpha': band_energies['delta'] / (band_energies['alpha'] + eps),
        'delta_beta': band_energies['delta'] / (band_energies['beta'] + eps),
        'theta_beta': band_energies['theta'] / (band_energies['beta'] + eps),
        'alpha_beta': band_energies['alpha'] / (band_energies['beta'] + eps),
        'slow_fast': (band_energies['delta'] + band_energies['theta']) / 
                     (band_energies['alpha'] + band_energies['beta'] + band_energies['gamma'] + eps)
    }
    
    return ratios

