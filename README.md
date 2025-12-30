# EEG Epilepsy Classification System

A modular Python-based DSP and Machine Learning pipeline for epileptic seizure detection using custom wavelet transform implementations.

## Project Structure

```
EE473_project/
├── core/                           # Custom DSP algorithms (organized by function)
│   ├── __init__.py
│   ├── filters.py                  # Wavelet filters and basic operations
│   ├── cwt.py                      # Continuous Wavelet Transform
│   ├── dwt_decomposition.py        # DWT decomposition (Mallat's algorithm)
│   ├── dwt_reconstruction.py       # DWT reconstruction (inverse Mallat)
│   └── denoising.py                # Wavelet denoising functions
│
├── data_io/                        # Data loading and preprocessing
│   ├── __init__.py
│   └── data_loader.py              # UCI Epileptic Seizure Recognition dataset
│
├── features/                       # Feature extraction and projection
│   ├── __init__.py
│   ├── extraction.py               # DWT feature extraction (energy, entropy, etc.)
│   └── projection.py               # Dimensionality reduction (PCA, UMAP, t-SNE)
│
├── models/                         # Classification models
│   ├── __init__.py
│   └── classifiers.py              # SVM and XGBoost wrappers
│
├── viz/                            # Visualization (organized by plot type)
│   ├── __init__.py
│   ├── signal_plots.py             # Time-domain signal plots
│   ├── scalogram_plots.py          # CWT/DWT coefficient visualizations
│   └── feature_plots.py            # Feature projections and confusion matrix
│
├── main.py                         # Main pipeline (no function definitions)
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Key Design Principles

### 1. Modular Organization
- **Functions grouped by tight category**, not one large file
- **Separate files for distinct operations** (CWT vs DWT vs denoising)
- **Simple, short functions** can stay in the same file they're used

### 2. Clean main.py
- **No function definitions** in main.py
- Uses **verbosity parameters** in library functions
- Just calls library functions and handles variables/printing

### 3. Classification with Full Features
- **Dimensionality reduction used ONLY for visualization**
- Classifier trained on **all extracted features** (no PCA/UMAP before training)
- Better performance by using full feature space

## Core DSP Module Organization

### `filters.py`
- `get_wavelet_filters()`: Get QMF coefficients (uses PyWavelets when available)
- `periodic_convolve()`: Circular convolution
- `morlet_wavelet()`: Morlet wavelet generation

### `cwt.py`
- `cwt()`: Continuous Wavelet Transform
- `scales_to_frequencies()`: Convert scales to frequencies

### `dwt_decomposition.py`
- `dwt_single_level()`: One level of decomposition
- `dwt()`: Multi-level DWT (Mallat's algorithm)

### `dwt_reconstruction.py`
- `idwt_single_level()`: One level of reconstruction
- `idwt()`: Multi-level inverse DWT

### `denoising.py`
- `soft_threshold()`: Soft thresholding
- `hard_threshold()`: Hard thresholding
- `estimate_noise_std()`: MAD estimator
- `denoise_signal()`: Complete denoising pipeline

## Visualization Module Organization

### `signal_plots.py`
- `plot_signals()`: General time-series plotter
- `plot_signal_comparison()`: Before/after comparison

### `scalogram_plots.py`
- `plot_scalogram()`: CWT time-frequency representation
- `plot_dwt_coefficients()`: DWT decomposition levels
- `plot_brain_wave_bands()`: EEG frequency bands

### `feature_plots.py`
- `plot_feature_projection()`: 2D PCA/UMAP/t-SNE plots
- `plot_confusion_matrix()`: Classification results

## Pipeline

```
1. Load Data
   ↓
2. Denoise (DWT soft thresholding)
   ↓
3. Extract Features (DWT sub-bands: energy, entropy, std)
   ↓
4. Train/Evaluate Classifier (using ALL features)
   ↓
5. Visualize (PCA/UMAP for visualization only)
```

## Usage

```bash
# Run full pipeline
python main.py --data "Epileptic Seizure Recognition.csv"

# With options
python main.py --wavelet db4 --classifier svm --save-figs

# Without visualizations (faster)
python main.py --no-viz
```

## Key Features

- ✅ **Custom DWT/IDWT** matching PyWavelets (reconstruction error < 1e-15)
- ✅ **Manual CWT** with Morlet wavelet
- ✅ **Modular organization** (functions grouped logically)
- ✅ **Clean main.py** (no function definitions)
- ✅ **Full-feature classification** (no dimensionality reduction before training)
- ✅ **Verbosity control** throughout pipeline
- ✅ **5-level DWT** for EEG brain wave bands (δ, θ, α, β, γ)

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
umap-learn>=0.5.0 (optional)
PyWavelets>=1.1.0 (for filter coefficients)
```

## Author

EE473 Project - BOUN 2025-2026 Fall

