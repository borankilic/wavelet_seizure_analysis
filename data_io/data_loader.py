"""
IO Module: Data Loading and Preprocessing
===========================================
Functions to load and preprocess the UCI Epileptic Seizure Recognition dataset.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


# =============================================================================
# DATASET INFORMATION
# =============================================================================

"""
UCI Epileptic Seizure Recognition Dataset
==========================================

The dataset contains 11,500 EEG recordings, each 23.6 seconds long sampled at 173.61 Hz
(resulting in 4097 data points, stored as 178 samples after downsampling).

Original 5 Classes:
    1 - Seizure activity (eyes open)
    2 - EEG from tumor area (eyes open)
    3 - EEG from healthy brain area (eyes open)
    4 - Eyes closed, tumor-free
    5 - Eyes open, tumor-free

For binary classification:
    - Class 1 → Seizure (positive class)
    - Classes 2-5 → Non-Seizure (negative class)

Each row has 179 columns:
    - Columns 1-178: EEG signal values (X1 to X178)
    - Column 179: Class label (y)
"""

SAMPLING_RATE = 173.61  # Hz
SIGNAL_LENGTH = 178     # samples per recording
DURATION = SIGNAL_LENGTH / SAMPLING_RATE  # ~1.02 seconds

CLASS_NAMES_ORIGINAL = {
    1: 'Seizure',
    2: 'Tumor area (eyes open)',
    3: 'Healthy area (eyes open)',
    4: 'Eyes closed',
    5: 'Eyes open'
}

CLASS_NAMES_BINARY = {
    0: 'Non-Seizure',
    1: 'Seizure'
}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_epilepsy_data(filepath: str,
                       binary: bool = True,
                       normalize: bool = True,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI Epileptic Seizure Recognition dataset.
    
    Args:
        filepath: Path to the CSV file
        binary: If True, convert to binary classification (Seizure vs Non-Seizure)
        normalize: If True, normalize each signal to zero mean and unit variance
        verbose: Print dataset information
    
    Returns:
        X: Signal data (n_samples, signal_length)
        y: Labels (n_samples,)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    
    if verbose:
        print(f"Loading dataset from: {filepath}")
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Handle different column naming conventions
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Extract signal columns (X1 to X178 or first 178 columns)
    signal_cols = [col for col in df.columns if col.startswith('X') or col.isdigit()]
    
    if len(signal_cols) == 0:
        # Assume all columns except last are signals
        signal_cols = df.columns[:-1].tolist()
    
    # Get signals and labels
    X = df[signal_cols[:SIGNAL_LENGTH]].values.astype(np.float64)
    
    # Get labels (last column or 'y' column)
    if 'y' in df.columns:
        y = df['y'].values
    else:
        y = df.iloc[:, -1].values
    
    # Convert to binary if requested
    if binary:
        # Class 1 = Seizure, Classes 2-5 = Non-Seizure
        y = (y == 1).astype(int)
    
    # Normalize signals if requested
    if normalize:
        X = normalize_signals(X)
    
    if verbose:
        print(f"  Loaded {len(X)} samples with {X.shape[1]} features each")
        print(f"  Sampling rate: {SAMPLING_RATE} Hz")
        print(f"  Signal duration: {DURATION:.3f} seconds")
        
        if binary:
            n_seizure = np.sum(y == 1)
            n_non_seizure = np.sum(y == 0)
            print(f"  Binary labels: {n_seizure} seizure, {n_non_seizure} non-seizure")
            print(f"  Class ratio: {n_seizure/len(y)*100:.1f}% seizure")
        else:
            unique, counts = np.unique(y, return_counts=True)
            print(f"  Class distribution:")
            for cls, cnt in zip(unique, counts):
                print(f"    Class {cls}: {cnt} ({cnt/len(y)*100:.1f}%)")
    
    return X, y


def normalize_signals(X: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize signals.
    
    Args:
        X: Signal data (n_samples, signal_length)
        method: Normalization method ('zscore', 'minmax', 'robust')
    
    Returns:
        Normalized signals
    """
    X_norm = X.copy()
    
    if method == 'zscore':
        # Z-score normalization (per signal)
        mean = np.mean(X_norm, axis=1, keepdims=True)
        std = np.std(X_norm, axis=1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        X_norm = (X_norm - mean) / std
        
    elif method == 'minmax':
        # Min-max normalization (per signal)
        min_val = np.min(X_norm, axis=1, keepdims=True)
        max_val = np.max(X_norm, axis=1, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        X_norm = (X_norm - min_val) / range_val
        
    elif method == 'robust':
        # Robust normalization using median and IQR
        median = np.median(X_norm, axis=1, keepdims=True)
        q1 = np.percentile(X_norm, 25, axis=1, keepdims=True)
        q3 = np.percentile(X_norm, 75, axis=1, keepdims=True)
        iqr = q3 - q1
        iqr[iqr == 0] = 1
        X_norm = (X_norm - median) / iqr
    
    return X_norm


def train_test_split(X: np.ndarray, 
                     y: np.ndarray,
                     test_size: float = 0.2,
                     stratify: bool = True,
                     random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Proportion for test set
        stratify: Maintain class distribution
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if stratify:
        # Stratified split
        train_indices = []
        test_indices = []
        
        for cls in np.unique(y):
            cls_indices = indices[y == cls]
            np.random.shuffle(cls_indices)
            n_test = int(len(cls_indices) * test_size)
            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
    else:
        # Random split
        np.random.shuffle(indices)
        n_test = int(n_samples * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def get_sample_signals(X: np.ndarray, 
                       y: np.ndarray,
                       n_per_class: int = 5,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get sample signals from each class.
    
    Args:
        X: Signal data
        y: Labels
        n_per_class: Number of samples per class
        random_state: Random seed
    
    Returns:
        Sample signals and their labels
    """
    np.random.seed(random_state)
    
    samples_X = []
    samples_y = []
    
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        selected = np.random.choice(cls_indices, size=min(n_per_class, len(cls_indices)), replace=False)
        samples_X.extend(X[selected])
        samples_y.extend(y[selected])
    
    return np.array(samples_X), np.array(samples_y)


# =============================================================================
# DATA INFORMATION
# =============================================================================

def get_dataset_info() -> dict:
    """
    Get information about the dataset.
    
    Returns:
        Dictionary with dataset metadata
    """
    return {
        'name': 'UCI Epileptic Seizure Recognition',
        'sampling_rate': SAMPLING_RATE,
        'signal_length': SIGNAL_LENGTH,
        'duration': DURATION,
        'n_samples': 11500,
        'original_classes': CLASS_NAMES_ORIGINAL,
        'binary_classes': CLASS_NAMES_BINARY,
        'source': 'https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition'
    }


def print_dataset_info():
    """Print formatted dataset information."""
    info = get_dataset_info()
    
    print("=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Name: {info['name']}")
    print(f"Sampling Rate: {info['sampling_rate']} Hz")
    print(f"Signal Length: {info['signal_length']} samples")
    print(f"Duration: {info['duration']:.3f} seconds")
    print(f"Total Samples: {info['n_samples']}")
    print()
    print("Original Classes:")
    for cls, name in info['original_classes'].items():
        print(f"  {cls}: {name}")
    print()
    print("Binary Classification:")
    for cls, name in info['binary_classes'].items():
        print(f"  {cls}: {name}")
    print("=" * 60)

