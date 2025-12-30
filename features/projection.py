"""
Projection Module: Dimensionality Reduction
=============================================
Unified interface for PCA, UMAP, and t-SNE projections.
"""

import numpy as np
from typing import Optional, Tuple, Union, Literal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sklearn
import inspect
import json

# Try importing UMAP (optional dependency)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")


# =============================================================================
# UNIFIED PROJECTION INTERFACE
# =============================================================================

class DimensionalityReducer:
    """
    Unified interface for dimensionality reduction.
    
    Supports PCA, UMAP, and t-SNE with consistent API.
    """
    
    def __init__(self, 
                 method: Literal['pca', 'umap', 'tsne'] = 'pca',
                 n_components: int = 2,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('pca', 'umap', 'tsne')
            n_components: Number of output dimensions
            random_state: Random seed for reproducibility
            **kwargs: Additional arguments for the specific method
        """
        self.method = method.lower()
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.scaler = StandardScaler()
        self.reducer = None
        self.is_fitted = False
        
        self._init_reducer()
    
    def _init_reducer(self):
        """Initialize the specific reducer."""
        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
                **self.kwargs
            )
        elif self.method == 'tsne':
            # #region agent log
            try:
                with open('/Users/borankilic/Desktop/BOUN 2025-2026 Fall/EE473/EE473_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A","location":"projection.py:66","message":"TSNE init - sklearn version","data":{"sklearn_version":sklearn.__version__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            except: pass
            # #endregion
            
            # t-SNE specific defaults
            # Note: In scikit-learn 1.2+, n_iter was renamed to max_iter
            perplexity = self.kwargs.pop('perplexity', 30)
            max_iter = self.kwargs.pop('n_iter', self.kwargs.pop('max_iter', 1000))  # Support both for compatibility
            learning_rate = self.kwargs.pop('learning_rate', 'auto')
            
            # #region agent log
            try:
                with open('/Users/borankilic/Desktop/BOUN 2025-2026 Fall/EE473/EE473_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A","location":"projection.py:75","message":"Before TSNE init - parameters","data":{"max_iter":max_iter,"learning_rate":learning_rate,"perplexity":perplexity,"n_components":self.n_components},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            except: pass
            # #endregion
            
            self.reducer = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
                perplexity=perplexity,
                max_iter=max_iter,  # Fixed: use max_iter instead of n_iter
                learning_rate=learning_rate,
                **self.kwargs
            )
            
            # #region agent log
            try:
                with open('/Users/borankilic/Desktop/BOUN 2025-2026 Fall/EE473/EE473_project/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"post-fix","hypothesisId":"A","location":"projection.py:88","message":"TSNE init successful","data":{"status":"success"},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            except: pass
            # #endregion
        elif self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            
            # UMAP specific defaults
            n_neighbors = self.kwargs.pop('n_neighbors', 15)
            min_dist = self.kwargs.pop('min_dist', 0.1)
            metric = self.kwargs.pop('metric', 'euclidean')
            
            self.reducer = umap.UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. "
                           f"Choose from: pca, umap, tsne")
    
    def fit(self, X: np.ndarray) -> 'DimensionalityReducer':
        """
        Fit the reducer to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            self
        """
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'tsne':
            # t-SNE doesn't support separate fit/transform
            pass
        else:
            self.reducer.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted reducer.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Reduced features (n_samples, n_components)
        """
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'tsne':
            # t-SNE needs fit_transform
            return self.reducer.fit_transform(X_scaled)
        else:
            return self.reducer.transform(X_scaled)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            Reduced features (n_samples, n_components)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return self.reducer.fit_transform(X_scaled)
    
    def get_explained_variance(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio (PCA only).
        
        Returns:
            Explained variance ratio per component or None
        """
        if self.method == 'pca' and self.is_fitted:
            return self.reducer.explained_variance_ratio_
        return None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def reduce_dimensions(features: np.ndarray,
                      method: str = 'pca',
                      n_components: int = 2,
                      random_state: int = 42,
                      **kwargs) -> Tuple[np.ndarray, DimensionalityReducer]:
    """
    Reduce dimensionality of feature matrix.
    
    Convenience function for one-time dimensionality reduction.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        method: Reduction method ('pca', 'umap', 'tsne')
        n_components: Number of output dimensions
        random_state: Random seed
        **kwargs: Additional method-specific arguments
    
    Returns:
        reduced: Reduced features
        reducer: Fitted reducer object
    """
    reducer = DimensionalityReducer(
        method=method,
        n_components=n_components,
        random_state=random_state,
        **kwargs
    )
    
    reduced = reducer.fit_transform(features)
    
    return reduced, reducer


def pca_reduce(features: np.ndarray, 
               n_components: int = 2,
               return_variance: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Reduce dimensions using PCA.
    
    Args:
        features: Feature matrix
        n_components: Number of components
        return_variance: Also return explained variance
    
    Returns:
        Reduced features, optionally with explained variance
    """
    reduced, reducer = reduce_dimensions(features, method='pca', n_components=n_components)
    
    if return_variance:
        variance = reducer.get_explained_variance()
        return reduced, variance
    return reduced


def tsne_reduce(features: np.ndarray,
                n_components: int = 2,
                perplexity: float = 30,
                max_iter: int = 1000) -> np.ndarray:
    """
    Reduce dimensions using t-SNE.
    
    Args:
        features: Feature matrix
        n_components: Number of components (usually 2)
        perplexity: t-SNE perplexity parameter
        max_iter: Maximum number of iterations (was n_iter in older sklearn versions)
    
    Returns:
        Reduced features
    """
    reduced, _ = reduce_dimensions(
        features,
        method='tsne',
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter
    )
    return reduced


def umap_reduce(features: np.ndarray,
                n_components: int = 2,
                n_neighbors: int = 15,
                min_dist: float = 0.1) -> np.ndarray:
    """
    Reduce dimensions using UMAP.
    
    Args:
        features: Feature matrix
        n_components: Number of components
        n_neighbors: Number of neighbors for manifold approximation
        min_dist: Minimum distance between points
    
    Returns:
        Reduced features
    """
    reduced, _ = reduce_dimensions(
        features,
        method='umap',
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist
    )
    return reduced


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================

def analyze_pca_components(features: np.ndarray,
                           feature_names: list = None,
                           n_components: int = None) -> dict:
    """
    Analyze PCA components and their contribution.
    
    Args:
        features: Feature matrix
        feature_names: Names of features
        n_components: Number of components to analyze
    
    Returns:
        Dictionary with analysis results
    """
    if n_components is None:
        n_components = min(features.shape)
    
    # Fit PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Get results
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    components = pca.components_
    
    # Find number of components for 95% variance
    n_for_95 = np.argmax(cumulative_var >= 0.95) + 1
    
    results = {
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var,
        'components': components,
        'n_components_95pct': n_for_95,
        'feature_names': feature_names
    }
    
    # Feature importance (sum of absolute loadings)
    if feature_names is not None:
        importance = np.sum(np.abs(components), axis=0)
        importance = importance / np.sum(importance)
        results['feature_importance'] = dict(zip(feature_names, importance))
    
    return results

