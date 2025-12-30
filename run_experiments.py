"""
Experiment Runner
==================
Run pipeline with different wavelets and classifiers, saving results.
"""

import json
from pathlib import Path
import shutil
import numpy as np
from main import run_pipeline, DATA_PATH


def run_experiments(data_path: Path = DATA_PATH,
                    wavelets: list = None,
                    classifiers: list = None,
                    verbose: bool = True) -> dict:
    """
    Run pipeline with different wavelets and classifiers.
    
    Args:
        data_path: Path to dataset
        wavelets: List of wavelets to test (default: ['haar', 'db2', 'db4', 'db6'])
        classifiers: List of classifiers to test (default: ['svm', 'xgboost'])
        verbose: Print progress
    
    Returns:
        Dictionary with all experiment results
    """
    if wavelets is None:
        wavelets = ['haar', 'db2', 'db4', 'db6']
    
    if classifiers is None:
        classifiers = ['svm', 'xgboost']
    
    # Filter out xgboost if not available (check directly to avoid import issues)
    if 'xgboost' in classifiers:
        try:
            import xgboost as xgb
        except (ImportError, Exception) as e:
            if 'libomp' in str(e).lower() or 'openmp' in str(e).lower() or 'xgb' in str(e).lower():
                print("Warning: XGBoost cannot load (OpenMP runtime missing). Removing from classifiers.")
                print("  macOS users: Run 'brew install libomp' to install OpenMP runtime.")
                classifiers = [c for c in classifiers if c != 'xgboost']
            else:
                print("Warning: XGBoost not available. Removing from classifiers.")
                classifiers = [c for c in classifiers if c != 'xgboost']
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    print("\n" + "="*70)
    print("RUNNING EXPERIMENTS")
    print("="*70)
    print(f"Wavelets: {wavelets}")
    print(f"Classifiers: {classifiers}")
    print(f"Total combinations: {len(wavelets) * len(classifiers)}")
    print("="*70)
    
    for wavelet in wavelets:
        for classifier in classifiers:
            config_name = f"{wavelet}_{classifier}"
            print(f"\n{'='*70}")
            print(f"Running: Wavelet={wavelet}, Classifier={classifier}")
            print(f"{'='*70}")
            
            # Create folder for this configuration
            config_dir = results_dir / config_name
            config_dir.mkdir(exist_ok=True)
            figures_dir = config_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            # Run pipeline with specific wavelet and classifier
            try:
                results = run_pipeline(
                    data_path=data_path,
                    wavelet=wavelet,
                    classifier=classifier,
                    visualize=True,
                    save_figures=True,
                    verbose=verbose
                )
            except (ImportError, Exception) as e:
                if 'xgboost' in str(e).lower() or 'xgb' in str(e).lower():
                    print(f"\n⚠ Skipping {config_name}: XGBoost not available or failed to load")
                    print(f"  Error: {str(e)}")
                    print(f"  Tip: Install OpenMP with 'brew install libomp' on macOS")
                    continue
                else:
                    raise
            
            # Move figures to config folder
            if Path("figures").exists():
                for fig_file in Path("figures").glob("*.png"):
                    shutil.move(str(fig_file), str(figures_dir / fig_file.name))
                Path("figures").rmdir()
            
            # Save results to JSON
            results_file = config_dir / "results.json"
            # Convert numpy types to Python types for JSON serialization
            results_serializable = _make_serializable(results)
            with open(results_file, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            # Store results
            all_results[config_name] = {
                'wavelet': wavelet,
                'classifier': classifier,
                'metrics': results.get('classification', {}).get('test_metrics', {}),
                'results_file': str(results_file),
                'figures_dir': str(figures_dir)
            }
            
            if verbose:
                metrics = results.get('classification', {}).get('test_metrics', {})
                print(f"\n✓ Completed: {config_name}")
                print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
                print(f"  ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
    
    # Create summary comparison
    if all_results:  # Only create summary if we have results
        summary_file = results_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"{'Wavelet':<12} {'Classifier':<12} {'Accuracy':<10} {'F1':<10} {'ROC-AUC':<10}\n")
            f.write("-"*70 + "\n")
            
            for config_name, result in sorted(all_results.items()):
                metrics = result['metrics']
                f.write(f"{result['wavelet']:<12} {result['classifier']:<12} "
                       f"{metrics.get('accuracy', 0):<10.4f} "
                       f"{metrics.get('f1', 0):<10.4f} "
                       f"{metrics.get('roc_auc', 0):<10.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Best Results:\n")
            f.write("="*70 + "\n")
            
            # Find best by accuracy
            best_acc = max(all_results.items(), 
                          key=lambda x: x[1]['metrics'].get('accuracy', 0))
            f.write(f"Best Accuracy: {best_acc[0]} ({best_acc[1]['metrics'].get('accuracy', 0):.4f})\n")
            
            # Find best by F1
            best_f1 = max(all_results.items(), 
                         key=lambda x: x[1]['metrics'].get('f1', 0))
            f.write(f"Best F1 Score: {best_f1[0]} ({best_f1[1]['metrics'].get('f1', 0):.4f})\n")
            
            # Find best by ROC-AUC
            best_auc = max(all_results.items(), 
                          key=lambda x: x[1]['metrics'].get('roc_auc', 0))
            f.write(f"Best ROC-AUC: {best_auc[0]} ({best_auc[1]['metrics'].get('roc_auc', 0):.4f})\n")
        
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {results_dir}")
        print(f"Summary file: {summary_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"{'Wavelet':<12} {'Classifier':<12} {'Accuracy':<10} {'F1':<10} {'ROC-AUC':<10}")
        print("-"*70)
        for config_name, result in sorted(all_results.items()):
            metrics = result['metrics']
            print(f"{result['wavelet']:<12} {result['classifier']:<12} "
                  f"{metrics.get('accuracy', 0):<10.4f} "
                  f"{metrics.get('f1', 0):<10.4f} "
                  f"{metrics.get('roc_auc', 0):<10.4f}")
    
    return all_results


def _make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


if __name__ == "__main__":
    # Run experiments
    results = run_experiments(
        data_path=DATA_PATH,
        wavelets=['haar', 'db2', 'db4', 'db6', 'sym2', 'sym4', 'sym8', 'coif2', 'coif4', 'coif6', 'coif8'],
        classifiers=['xgboost','svm'],
        verbose=True
    )

