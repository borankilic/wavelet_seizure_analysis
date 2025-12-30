"""
Feature and Classification Result Plotting
===========================================
Visualization of feature projections and classification metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


def plot_feature_projection(features: np.ndarray,
                            labels: np.ndarray,
                            method_name: str = "Projection",
                            class_names: List[str] = None,
                            title: str = None,
                            figsize: Tuple[int, int] = (10, 8),
                            alpha: float = 0.6,
                            s: int = 30) -> plt.Figure:
    """
    Plot 2D feature projection with class labels.
    
    Args:
        features: 2D projected features (n_samples, 2)
        labels: Class labels
        method_name: Name of projection method (e.g., 'PCA', 'UMAP', 't-SNE')
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        alpha: Point transparency
        s: Point size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    
    if class_names is None:
        class_names = [f"Class {l}" for l in unique_labels]
    
    # Colorblind-friendly colors
    colors = ['#0077B6', '#E63946', '#2A9D8F', '#F4A261', '#9B2226']
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(features[mask, 0], features[mask, 1],
                  c=colors[i % len(colors)], label=class_names[i],
                  alpha=alpha, s=s, edgecolors='white', linewidth=0.3)
    
    ax.set_xlabel(f"{method_name} Component 1", fontsize=11)
    ax.set_ylabel(f"{method_name} Component 2", fontsize=11)
    
    if title is None:
        title = f"{method_name} Feature Projection"
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str] = None,
                          title: str = "Confusion Matrix",
                          cmap: str = 'Blues',
                          figsize: Tuple[int, int] = (8, 6),
                          normalize: bool = False) -> plt.Figure:
    """
    Plot confusion matrix with annotations.
    
    Args:
        cm: Confusion matrix array
        class_names: Names for each class
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        normalize: Normalize to percentages
    
    Returns:
        matplotlib Figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2%'
    else:
        fmt = 'd'
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                text = f'{cm[i, j]:.2%}'
            else:
                text = f'{cm[i, j]}'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black', fontsize=12)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    return fig

