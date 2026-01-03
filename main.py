"""
EEG Epilepsy Classification System
====================================
Main execution pipeline for seizure detection using wavelet transforms.

Pipeline:
1. Load Data
2. Denoise (via DWT soft thresholding)
3. Extract Features (DWT sub-bands)
4. Train/Evaluate Classifier (using full features)
5. Visualize Results (with dimensionality reduction)

Author: EE473 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from data_io.data_loader import (
    load_epilepsy_data, train_test_split, get_sample_signals,
    print_dataset_info, SAMPLING_RATE
)
from core import dwt, cwt, denoise_signal, scales_to_frequencies
from features.extraction import extract_features_batch, get_feature_names
from features.projection import reduce_dimensions
from models.classifiers import train_and_evaluate, cross_validate_classifier
from viz import (
    plot_signals, plot_signal_comparison, plot_scalogram,
    plot_dwt_coefficients, plot_brain_wave_bands,
    plot_feature_projection, plot_confusion_matrix
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset path
DATA_PATH = Path("Epileptic Seizure Recognition.csv")

# Wavelet parameters
WAVELET = 'db4'           # Daubechies-4 wavelet
DWT_LEVEL = 5             # 5 levels for EEG bands
DENOISE_THRESHOLD = 'soft'

# Feature extraction
FEATURE_SET = 'full'  # 'standard', 'full', or 'minimal'

# Visualization (dimensionality reduction)
PROJECTION_METHOD = 'pca' # 'pca', 'umap', 'tsne'

# Classification
CLASSIFIER = 'svm'            # 'svm', 'xgboost', 'random_forest', or 'naive_bayes'
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(data_path: Path = DATA_PATH,
                 wavelet: str = None,
                 classifier: str = None,
                 projection_method: str = 'pca',
                 visualize: bool = True,
                 save_figures: bool = False,
                 verbose: bool = True,
                 cross_validate: bool = True,
                 cv_folds: int = 5) -> dict:
    """
    Run the complete EEG classification pipeline.
    
    Pipeline:
    1. Load Data
    2. Denoise (DWT soft thresholding)
    3. Extract Features (DWT sub-bands)
    4. Train/Evaluate Classifier (using FULL features - no dimensionality reduction)
    4b. Cross-Validation (5-fold, optional)
    5. Visualize (with PCA/UMAP for visualization only)
    
    Args:
        data_path: Path to dataset
        wavelet: Wavelet to use (default: WAVELET global)
        classifier: Classifier to use (default: CLASSIFIER global)
        visualize: Generate visualizations
        save_figures: Save figures to disk
        verbose: Print progress
        cross_validate: Perform k-fold cross-validation (default: True)
        cv_folds: Number of cross-validation folds (default: 5)
    
    Returns:
        Dictionary with all results
    """
    # Use parameters or fall back to globals
    if wavelet is None:
        wavelet = WAVELET
    if classifier is None:
        classifier = CLASSIFIER
    
    print("\n" + "="*70)
    print("EEG EPILEPSY CLASSIFICATION PIPELINE")
    print("="*70)
    
    results = {}
    save_path = "figures" if save_figures else None
    
    if save_figures:
        Path(save_path).mkdir(exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: LOADING DATA")
        print("="*70)
        print_dataset_info()
    
    X, y = load_epilepsy_data(data_path, binary=True, normalize=True, verbose=verbose)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=True, random_state=RANDOM_STATE
    )
    
    results['data'] = {
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    if verbose:
        print(f"\n  Training set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
    
    # Visualize sample signals
    if visualize:
        if verbose:
            print("\n  Generating signal visualizations...")
        
        # Get samples from each class
        seizure_samples = X_train[y_train == 1][:3]
        non_seizure_samples = X_train[y_train == 0][:3]
        
        # Plot seizure signals
        fig = plot_signals(
            seizure_samples,
            labels=[f"Seizure {i+1}" for i in range(3)],
            sampling_rate=SAMPLING_RATE,
            title="Seizure EEG Signals",
            stacked=True
        )
        if save_path:
            fig.savefig(f"{save_path}/seizure_signals.png", dpi=150)
        plt.close(fig)
        
        # DWT decomposition of a seizure signal
        seizure_idx = np.where(y_train == 1)[0][0]
        coeffs = dwt(X_train[seizure_idx], wavelet=wavelet, level=DWT_LEVEL, verbose=False)
        fig = plot_brain_wave_bands(
            coeffs,
            band_names=['δ (Delta)', 'θ (Theta)', 'α (Alpha)', 
                       'β (Beta)', 'γ (Gamma)', 'HF'][:len(coeffs)],
            title="EEG Brain Wave Decomposition"
        )
        if save_path:
            fig.savefig(f"{save_path}/dwt_decomposition.png", dpi=150)
        plt.close(fig)
        
        # CWT scalogram
        scales = np.geomspace(1, 64, num=128)
        cwt_coeffs = cwt(X_train[seizure_idx], scales, wavelet='morlet', sampling_rate=SAMPLING_RATE)
        frequencies = scales_to_frequencies(scales, 'morlet', SAMPLING_RATE)
        # Construct time axis from signal length and sampling rate
        time = np.arange(cwt_coeffs.shape[1]) / SAMPLING_RATE
        #ax = plot_scalogram(
        #    cwt_coeffs, scales=scales, time=time, frequencies=frequencies,
        #    yaxis='frequency',
        #    yscale='log',
        #    cscale='log',
        #    title="CWT Scalogram (Morlet Wavelet)"
        #)
        #fig = ax.figure
        #if save_path:
        #    fig.savefig(f"{save_path}/cwt_scalogram.png", dpi=150)
        #plt.close(fig)
    
    # =========================================================================
    # STEP 2: Denoise
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 2: DENOISING SIGNALS")
        print("="*70)
        print(f"  Wavelet: {wavelet}")
        print(f"  Method: Soft thresholding (VisuShrink)")
    
    X_train_denoised = np.zeros_like(X_train)
    X_test_denoised = np.zeros_like(X_test)
    
    for i in range(len(X_train)):
        if verbose and (i + 1) % 2000 == 0:
            print(f"  Processing training signal {i+1}/{len(X_train)}")
        X_train_denoised[i] = denoise_signal(X_train[i], wavelet=wavelet, 
                                             threshold_type='soft', verbose=False)
    
    for i in range(len(X_test)):
        X_test_denoised[i] = denoise_signal(X_test[i], wavelet=wavelet, 
                                            threshold_type='soft', verbose=False)
    
    if verbose:
        noise = X_train - X_train_denoised
        snr_after = np.mean(np.var(noise, axis=1))
        print(f"\n  Denoising complete!")
        print(f"  Average noise variance removed: {snr_after:.4f}")
    
    if visualize:
        fig = plot_signal_comparison(X_train[0], X_train_denoised[0], 
                                     sampling_rate=SAMPLING_RATE,
                                     labels=["Original", "Denoised"],
                                     title="DWT Denoising Effect")
        if save_path:
            fig.savefig(f"{save_path}/denoising_comparison.png", dpi=150)
        plt.close(fig)
    
    # =========================================================================
    # STEP 3: Extract Features
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 3: EXTRACTING FEATURES")
        print("="*70)
        print(f"  Wavelet: {wavelet}")
        print(f"  Decomposition level: {DWT_LEVEL}")
        print(f"  Feature set: {FEATURE_SET}")
    
    features_train = extract_features_batch(
        X_train_denoised, wavelet=wavelet, level=DWT_LEVEL,
        feature_set=FEATURE_SET, verbose=verbose
    )
    features_test = extract_features_batch(
        X_test_denoised, wavelet=wavelet, level=DWT_LEVEL,
        feature_set=FEATURE_SET, verbose=False
    )
    
    # Get feature names by extracting coefficients from a sample to determine actual level
    # This ensures we get the correct level even if the requested level was adjusted
    sample_coeffs = dwt(X_train_denoised[0], wavelet, DWT_LEVEL)
    feature_names = get_feature_names(coefficients=sample_coeffs, feature_set=FEATURE_SET)
    
    results['features'] = {
        'n_features': features_train.shape[1],
        'feature_names': feature_names
    }
    
    if verbose:
        print(f"\n  Extracted {features_train.shape[1]} features per sample")
        print(f"  Feature matrix shape: {features_train.shape}")
    
    # =========================================================================
    # STEP 4: Train and Evaluate Classifier (using FULL features)
    # =========================================================================
    if verbose:
        print("\n" + "="*70)
        print("STEP 4: TRAINING CLASSIFIER (USING FULL FEATURES)")
        print("="*70)
        print(f"  Classifier: {classifier.upper()}")
        print(f"  Using all {features_train.shape[1]} features (no dimensionality reduction)")
    
    clf_model, clf_results = train_and_evaluate(
        features_train, y_train, features_test, y_test,
        model_type=classifier, verbose=verbose
    )
    results['classification'] = clf_results
    
    # =========================================================================
    # STEP 4b: Cross-Validation (Verification)
    # =========================================================================
    if cross_validate:
        if verbose:
            print("\n" + "="*70)
            print(f"STEP 4b: {cv_folds}-FOLD CROSS-VALIDATION (VERIFICATION)")
            print("="*70)
            print(f"  Performing cross-validation on full training set")
            print(f"  This provides a more robust estimate of model performance")
        
        # Combine train and test for cross-validation
        X_full = np.vstack([features_train, features_test])
        y_full = np.hstack([y_train, y_test])
        
        cv_results = cross_validate_classifier(
            X_full, y_full,
            model_type=classifier,
            cv_folds=cv_folds,
            verbose=verbose
        )
        results['cross_validation'] = cv_results
        
        if verbose:
            print(f"\n  Cross-validation completed successfully!")
            print(f"  Mean CV Accuracy: {cv_results['mean_metrics']['accuracy']:.4f} ± {cv_results['std_metrics']['accuracy']:.4f}")
    
    if visualize:
        fig = plot_confusion_matrix(
            clf_results['confusion_matrix'],
            class_names=['Non-Seizure', 'Seizure'],
            title="Classification Results",
            normalize=True
        )
        if save_path:
            fig.savefig(f"{save_path}/confusion_matrix.png", dpi=150)
        plt.close(fig)
    
    # =========================================================================
    # STEP 5: Visualize Feature Space (with dimensionality reduction)
    # =========================================================================
    if visualize:
        if verbose:
            print("\n" + "="*70)
            print("STEP 5: VISUALIZING FEATURE SPACE")
            print("="*70)
            print(f"  Note: Dimensionality reduction used for VISUALIZATION ONLY")
            print(f"  Method: {PROJECTION_METHOD.upper()}")
        
        reduced_train, reducer = reduce_dimensions(
            features_train, method=PROJECTION_METHOD, n_components=2
        )
        
        if verbose and PROJECTION_METHOD == 'pca':
            var_explained = reducer.get_explained_variance()
            if var_explained is not None:
                print(f"  Explained variance: {var_explained[0]:.2%}, {var_explained[1]:.2%}")
                print(f"  Total: {sum(var_explained):.2%}")
        
        fig = plot_feature_projection(
            reduced_train, y_train,
            method_name=PROJECTION_METHOD.upper(),
            class_names=['Non-Seizure', 'Seizure'],
            title=f"EEG Features - {PROJECTION_METHOD.upper()} Projection (Visualization Only)"
        )
        if save_path:
            fig.savefig(f"{save_path}/{PROJECTION_METHOD}_projection.png", dpi=150)
        plt.close(fig)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nFinal Test Results (using {features_train.shape[1]} features):")
    print(f"  Accuracy:  {clf_results['test_metrics']['accuracy']:.4f}")
    print(f"  Precision: {clf_results['test_metrics']['precision']:.4f}")
    print(f"  Recall:    {clf_results['test_metrics']['recall']:.4f}")
    print(f"  F1 Score:  {clf_results['test_metrics']['f1']:.4f}")
    print(f"  ROC-AUC:   {clf_results['test_metrics']['roc_auc']:.4f}")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Epilepsy Classification Pipeline"
    )
    parser.add_argument(
        "--data", type=str, default=str(DATA_PATH),
        help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Disable visualizations"
    )
    parser.add_argument(
        "--save-figs", action="store_true",
        help="Save figures to disk"
    )
    parser.add_argument(
        "--wavelet", type=str, default=WAVELET,
        help="Wavelet to use (db4, db2, haar, etc.)"
    )
    parser.add_argument(
        "--classifier", type=str, default=CLASSIFIER,
        choices=['svm', 'xgboost', 'random_forest', 'naive_bayes'],
        help="Classifier to use"
    )
    
    args = parser.parse_args()
    
    # Update configuration
    DATA_PATH = Path(args.data)
    WAVELET = args.wavelet
    CLASSIFIER = args.classifier
    
    # Run pipeline
    results = run_pipeline(
        data_path=DATA_PATH,
        visualize=not args.no_viz,
        projection_method= PROJECTION_METHOD,
        cross_validate=True,
        cv_folds=5,
        save_figures=args.save_figs,
        verbose=True
    )
