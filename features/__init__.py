"""
Features Module
================
Feature extraction and dimensionality reduction.
"""

from .extraction import (
    # Feature extraction
    extract_dwt_features,
    extract_features_batch,
    get_feature_names,
    extract_band_energies,
    compute_band_ratios,
    
    # Statistical features
    compute_energy,
    compute_entropy,
    compute_statistical_features
)

from .projection import (
    # Dimensionality reduction
    DimensionalityReducer,
    reduce_dimensions,
    pca_reduce,
    tsne_reduce,
    umap_reduce,
    analyze_pca_components
)

__all__ = [
    # Extraction
    'extract_dwt_features', 'extract_features_batch', 'get_feature_names',
    'extract_band_energies', 'compute_band_ratios',
    'compute_energy', 'compute_entropy', 'compute_statistical_features',
    
    # Projection
    'DimensionalityReducer', 'reduce_dimensions',
    'pca_reduce', 'tsne_reduce', 'umap_reduce', 'analyze_pca_components'
]

