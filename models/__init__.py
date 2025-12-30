"""
Models Module
==============
Classification models and training.
"""

from .classifiers import (
    # Classifier wrapper
    EEGClassifier,
    
    # Training and evaluation
    train_and_evaluate,
    compare_classifiers,
    hyperparameter_search
)

__all__ = [
    'EEGClassifier',
    'train_and_evaluate', 'compare_classifiers', 'hyperparameter_search'
]

