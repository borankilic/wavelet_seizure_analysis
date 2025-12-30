"""
Models Module: Classification Models
======================================
SVM and XGBoost classifiers with training and evaluation wrappers.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal, List
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    # Check if it's an OpenMP/library loading issue
    if 'libomp' in str(e).lower() or 'openmp' in str(e).lower() or 'xgb' in str(e).lower():
        print("Warning: XGBoost installed but cannot load (OpenMP runtime missing).")
        print("  macOS users: Run 'brew install libomp' to install OpenMP runtime.")
    else:
        print("Warning: XGBoost not available. Install with: pip install xgboost")


# =============================================================================
# CLASSIFIER WRAPPER
# =============================================================================

class EEGClassifier:
    """
    Unified classifier wrapper for EEG seizure detection.
    
    Supports SVM and XGBoost with consistent API.
    """
    
    def __init__(self,
                 model_type: Literal['svm', 'xgboost'] = 'svm',
                 scale_features: bool = True,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of classifier ('svm' or 'xgboost')
            scale_features: Whether to standardize features
            random_state: Random seed
            **kwargs: Additional model parameters
        """
        self.model_type = model_type.lower()
        self.scale_features = scale_features
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.scaler = StandardScaler() if scale_features else None
        self.model = None
        self.is_fitted = False
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the specific model."""
        if self.model_type == 'svm':
            # Default SVM parameters for EEG classification
            C = self.kwargs.pop('C', 1.0)
            kernel = self.kwargs.pop('kernel', 'rbf')
            gamma = self.kwargs.pop('gamma', 'scale')
            
            self.model = SVC(
                C=C,
                kernel=kernel,
                gamma=gamma,
                random_state=self.random_state,
                probability=True,
                **self.kwargs
            )
            
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available. Install with: pip install xgboost")
            
            # Default XGBoost parameters
            n_estimators = self.kwargs.pop('n_estimators', 100)
            max_depth = self.kwargs.pop('max_depth', 6)
            learning_rate = self.kwargs.pop('learning_rate', 0.1)
            
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Choose from: svm, xgboost")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EEGClassifier':
        """
        Train the classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            self
        """
        if self.scale_features:
            X = self.scaler.fit_transform(X)
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if self.scale_features:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
        
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        if self.scale_features:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        return metrics
    
    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Confusion matrix
        """
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                       cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
        
        Returns:
            Cross-validation scores
        """
        if self.scale_features:
            X = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': scores,
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores)
        }


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_and_evaluate(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       model_type: str = 'svm',
                       verbose: bool = True,
                       **kwargs) -> Tuple[EEGClassifier, Dict]:
    """
    Train and evaluate a classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_type: Type of classifier
        verbose: Print results
        **kwargs: Model parameters
    
    Returns:
        Trained classifier and evaluation metrics
    """
    # Initialize and train
    classifier = EEGClassifier(model_type=model_type, **kwargs)
    
    if verbose:
        print(f"\nTraining {model_type.upper()} classifier...")
    
    classifier.fit(X_train, y_train)
    
    # Evaluate on training set
    train_metrics = classifier.evaluate(X_train, y_train)
    
    # Evaluate on test set
    test_metrics = classifier.evaluate(X_test, y_test)
    
    # Get confusion matrix
    cm = classifier.get_confusion_matrix(X_test, y_test)
    
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'confusion_matrix': cm
    }
    
    if verbose:
        print(f"\nTraining Results:")
        print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")
        
        print(f"\nTest Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1 Score: {test_metrics['f1']:.4f}")
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
    
    return classifier, results


def hyperparameter_search(X: np.ndarray,
                          y: np.ndarray,
                          model_type: str = 'svm',
                          cv: int = 5,
                          verbose: bool = True) -> Tuple[dict, float]:
    """
    Perform hyperparameter search using grid search.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of classifier
        cv: Number of CV folds
        verbose: Print results
    
    Returns:
        Best parameters and best score
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_type == 'svm':
        model = SVC(random_state=42)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        }
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if verbose:
        print(f"\nPerforming grid search for {model_type.upper()}...")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_scaled, y)
    
    if verbose:
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Best CV Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_score_


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_classifiers(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        classifiers: List[str] = None) -> Dict[str, Dict]:
    """
    Compare multiple classifiers.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        classifiers: List of classifier types to compare
    
    Returns:
        Dictionary of results for each classifier
    """
    if classifiers is None:
        classifiers = ['svm']
        if XGBOOST_AVAILABLE:
            classifiers.append('xgboost')
    
    results = {}
    
    for clf_type in classifiers:
        try:
            clf, metrics = train_and_evaluate(
                X_train, y_train, X_test, y_test,
                model_type=clf_type, verbose=False
            )
            results[clf_type] = {
                'classifier': clf,
                **metrics
            }
        except Exception as e:
            print(f"Error training {clf_type}: {e}")
            results[clf_type] = None
    
    # Print comparison
    print("\n" + "=" * 60)
    print("CLASSIFIER COMPARISON")
    print("=" * 60)
    print(f"{'Model':<12} {'Accuracy':<10} {'F1':<10} {'ROC-AUC':<10}")
    print("-" * 42)
    
    for name, result in results.items():
        if result is not None:
            metrics = result['test_metrics']
            print(f"{name.upper():<12} {metrics['accuracy']:.4f}     "
                  f"{metrics['f1']:.4f}     {metrics['roc_auc']:.4f}")
    
    print("=" * 60)
    
    return results

