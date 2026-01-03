"""
Models Module: Classification Models
======================================
SVM, XGBoost, Random Forest, and Naive Bayes classifiers with training and evaluation wrappers.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal, List
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost (optional)
XGBOOST_AVAILABLE = False
XGBOOST_ERROR_MSG = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR_MSG = str(e)
    # Check if it's an OpenMP/library loading issue
    error_str = str(e).lower()
    if 'libomp' in error_str or 'openmp' in error_str or 'xgb' in error_str or 'libxgboost' in error_str:
        print("Warning: XGBoost installed but cannot load (OpenMP runtime missing).")
        print("  macOS users: Run 'brew install libomp' to install OpenMP runtime.")
        print(f"  Error details: {str(e)[:200]}")
    else:
        print("Warning: XGBoost not available. Install with: pip install xgboost")


# =============================================================================
# CLASSIFIER WRAPPER
# =============================================================================

class EEGClassifier:
    """
    Unified classifier wrapper for EEG seizure detection.
    
    Supports SVM, XGBoost, Random Forest, and Naive Bayes with consistent API.
    """
    
    def __init__(self,
                 model_type: Literal['svm', 'xgboost', 'random_forest', 'naive_bayes'] = 'svm',
                 scale_features: bool = True,
                 random_state: int = 42,
                 **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type: Type of classifier ('svm', 'xgboost', 'random_forest', or 'naive_bayes')
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
                if XGBOOST_ERROR_MSG and ('libomp' in XGBOOST_ERROR_MSG.lower() or 'openmp' in XGBOOST_ERROR_MSG.lower()):
                    raise ImportError(
                        f"XGBoost cannot load due to missing OpenMP runtime.\n"
                        f"Error: {XGBOOST_ERROR_MSG[:300]}\n"
                        f"Solution: macOS users should run 'brew install libomp'"
                    )
                else:
                    raise ImportError(
                        f"XGBoost not available. Install with: pip install xgboost\n"
                        f"Error: {XGBOOST_ERROR_MSG if XGBOOST_ERROR_MSG else 'Import failed'}"
                    )
            
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
        
        elif self.model_type == 'random_forest':
            # Default Random Forest parameters
            n_estimators = self.kwargs.pop('n_estimators', 100)
            max_depth = self.kwargs.pop('max_depth', None)
            min_samples_split = self.kwargs.pop('min_samples_split', 2)
            min_samples_leaf = self.kwargs.pop('min_samples_leaf', 1)
            
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
                **self.kwargs
            )
        
        elif self.model_type == 'naive_bayes':
            # Gaussian Naive Bayes (suitable for continuous features)
            var_smoothing = self.kwargs.pop('var_smoothing', 1e-9)
            
            self.model = GaussianNB(
                var_smoothing=var_smoothing,
                **self.kwargs
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Choose from: svm, xgboost, random_forest, naive_bayes")
    
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


def cross_validate_classifier(X: np.ndarray,
                              y: np.ndarray,
                              model_type: str = 'svm',
                              cv_folds: int = 5,
                              verbose: bool = True,
                              **kwargs) -> Dict:
    """
    Perform k-fold cross-validation with comprehensive metrics.
    
    This function performs stratified k-fold cross-validation and computes
    all evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) for
    each fold, then returns mean and standard deviation across folds.
    
    Args:
        X: Feature matrix (full dataset)
        y: Labels (full dataset)
        model_type: Type of classifier ('svm' or 'xgboost')
        cv_folds: Number of cross-validation folds (default: 5)
        verbose: Print detailed results
        **kwargs: Additional model parameters
    
    Returns:
        Dictionary containing:
            - fold_scores: List of metric dictionaries for each fold
            - mean_metrics: Mean of each metric across folds
            - std_metrics: Standard deviation of each metric across folds
            - all_metrics: Combined dictionary with all results
    """
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_scores = []
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Performing {cv_folds}-Fold Cross-Validation")
        print(f"Model: {model_type.upper()}")
        print(f"{'='*70}")
    
    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Split data
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Train classifier
        classifier = EEGClassifier(model_type=model_type, **kwargs)
        classifier.fit(X_train_fold, y_train_fold)
        
        # Evaluate on validation fold
        fold_metrics = classifier.evaluate(X_val_fold, y_val_fold)
        fold_scores.append(fold_metrics)
        
        if verbose:
            print(f"\nFold {fold_idx}/{cv_folds}:")
            print(f"  Accuracy: {fold_metrics['accuracy']:.4f}")
            print(f"  Precision: {fold_metrics['precision']:.4f}")
            print(f"  Recall: {fold_metrics['recall']:.4f}")
            print(f"  F1 Score: {fold_metrics['f1']:.4f}")
            print(f"  ROC-AUC: {fold_metrics['roc_auc']:.4f}")
    
    # Compute mean and std across folds
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    mean_metrics = {}
    std_metrics = {}
    
    for metric in metrics_list:
        values = [fold[metric] for fold in fold_scores]
        mean_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
    
    if verbose:
        print(f"\n{'='*70}")
        print("Cross-Validation Summary (Mean ± Std)")
        print(f"{'='*70}")
        print(f"Accuracy:  {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
        print(f"Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
        print(f"Recall:    {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
        print(f"F1 Score:  {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
        print(f"ROC-AUC:   {mean_metrics['roc_auc']:.4f} ± {std_metrics['roc_auc']:.4f}")
        print(f"{'='*70}\n")
    
    return {
        'fold_scores': fold_scores,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'all_metrics': {
            'mean': mean_metrics,
            'std': std_metrics,
            'folds': fold_scores
        }
    }


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
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'naive_bayes':
        model = GaussianNB()
        param_grid = {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: svm, xgboost, random_forest, naive_bayes")
    
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

