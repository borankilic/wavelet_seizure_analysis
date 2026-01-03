"""
Experiment Runner
==================
Run pipeline with different wavelets and classifiers, saving results.
"""

import json
import csv
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
        classifiers: List of classifiers to test 
                    (default: ['svm', 'xgboost', 'random_forest', 'naive_bayes'])
        verbose: Print progress
    
    Returns:
        Dictionary with all experiment results
    """
    if wavelets is None:
        wavelets = ['haar', 'db2', 'db4', 'db6']
    
    if classifiers is None:
        classifiers = ['svm', 'xgboost', 'random_forest', 'naive_bayes']
    
    # Filter out xgboost if not available (check directly to avoid import issues)
    if 'xgboost' in classifiers:
        try:
            import xgboost as xgb
            # Try to actually create a classifier to verify it works
            test_clf = xgb.XGBClassifier(n_estimators=1, use_label_encoder=False, eval_metric='logloss')
            del test_clf  # Clean up
        except (ImportError, Exception) as e:
            error_str = str(e).lower()
            if 'libomp' in error_str or 'openmp' in error_str or 'libxgboost' in error_str:
                print("Warning: XGBoost cannot load (OpenMP runtime missing). Removing from classifiers.")
                print("  macOS users: Run 'brew install libomp' to install OpenMP runtime.")
                print(f"  Error: {str(e)[:200]}")
            else:
                print("Warning: XGBoost not available. Removing from classifiers.")
                print(f"  Error: {str(e)[:200]}")
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
                    projection_method='pca',
                    cross_validate=True,
                    cv_folds=5,
                    visualize=True,
                    save_figures=True,
                    verbose=verbose
                )
            except (ImportError, Exception) as e:
                error_str = str(e).lower()
                if ('xgboost' in error_str or 'xgb' in error_str or 
                    'libomp' in error_str or 'openmp' in error_str or
                    classifier.lower() == 'xgboost'):
                    print(f"\n⚠ Skipping {config_name}: XGBoost not available or failed to load")
                    print(f"  Error: {str(e)[:300]}")
                    if 'libomp' in error_str or 'openmp' in error_str:
                        print(f"  Tip: Install OpenMP with 'brew install libomp' on macOS")
                    else:
                        print(f"  Tip: Install XGBoost with 'pip install xgboost'")
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
                'cv_metrics': results.get('cross_validation', {}).get('mean_metrics', {}),
                'cv_std': results.get('cross_validation', {}).get('std_metrics', {}),
                'results_file': str(results_file),
                'figures_dir': str(figures_dir)
            }
            
            if verbose:
                metrics = results.get('classification', {}).get('test_metrics', {})
                cv_metrics = results.get('cross_validation', {}).get('mean_metrics', {})
                print(f"\n✓ Completed: {config_name}")
                print(f"  Test Accuracy: {metrics.get('accuracy', 0):.4f}")
                print(f"  Test F1 Score: {metrics.get('f1', 0):.4f}")
                print(f"  Test ROC-AUC:  {metrics.get('roc_auc', 0):.4f}")
                if cv_metrics:
                    cv_std = results.get('cross_validation', {}).get('std_metrics', {})
                    print(f"  CV Accuracy:  {cv_metrics.get('accuracy', 0):.4f} ± {cv_std.get('accuracy', 0):.4f}")
                    print(f"  CV F1 Score:  {cv_metrics.get('f1', 0):.4f} ± {cv_std.get('f1', 0):.4f}")
    
    # Create summary comparison
    if all_results:  # Only create summary if we have results
        summary_file = results_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPERIMENT SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"{'Wavelet':<12} {'Classifier':<12} {'Test Acc':<10} {'Test F1':<10} {'CV Acc':<12} {'CV F1':<12}\n")
            f.write("-"*80 + "\n")
            
            for config_name, result in sorted(all_results.items()):
                metrics = result['metrics']
                cv_metrics = result.get('cv_metrics', {})
                cv_std = result.get('cv_std', {})
                f.write(f"{result['wavelet']:<12} {result['classifier']:<12} "
                       f"{metrics.get('accuracy', 0):<10.4f} "
                       f"{metrics.get('f1', 0):<10.4f} "
                       f"{cv_metrics.get('accuracy', 0):.4f}±{cv_std.get('accuracy', 0):.4f}  "
                       f"{cv_metrics.get('f1', 0):.4f}±{cv_std.get('f1', 0):.4f}\n")
            
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
        
        # Create CSV file with all results
        csv_file = results_dir / "results_summary.csv"
        _create_results_csv(all_results, csv_file, classifiers)
        
        print(f"\n{'='*70}")
        print("ALL EXPERIMENTS COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {results_dir}")
        print(f"Summary file: {summary_file}")
        print(f"CSV file: {csv_file}")
        
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


def _create_results_csv(all_results: dict, csv_file: Path, classifiers: list):
    """
    Create a CSV file with all experiment results.
    
    Rows: wavelets
    Columns: metrics for each classifier (test metrics + CV mean/std)
    
    Args:
        all_results: Dictionary of all experiment results
        csv_file: Path to save CSV file
        classifiers: List of classifiers used
    """
    # Get all unique wavelets and classifiers from results
    wavelets = sorted(set(r['wavelet'] for r in all_results.values()))
    classifiers_in_results = set(r['classifier'] for r in all_results.values())
    
    # Use provided classifiers list to maintain order, filter to only those with results
    if classifiers:
        classifiers_found = [c for c in classifiers if c in classifiers_in_results]
    else:
        classifiers_found = sorted(classifiers_in_results)
    
    # Define metric names
    test_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Create column headers
    headers = ['wavelet']
    
    for clf in classifiers_found:
        # Test metrics
        for metric in test_metrics:
            headers.append(f"{clf}_test_{metric}")
        
        # CV mean metrics
        for metric in cv_metric_names:
            headers.append(f"{clf}_cv_{metric}_mean")
        
        # CV std metrics
        for metric in cv_metric_names:
            headers.append(f"{clf}_cv_{metric}_std")
    
    # Create a dictionary to organize results by wavelet and classifier
    results_by_wavelet = {}
    for config_name, result in all_results.items():
        wavelet = result['wavelet']
        classifier = result['classifier']
        
        if wavelet not in results_by_wavelet:
            results_by_wavelet[wavelet] = {}
        
        results_by_wavelet[wavelet][classifier] = {
            'metrics': result.get('metrics', {}),
            'cv_metrics': result.get('cv_metrics', {}),
            'cv_std': result.get('cv_std', {})
        }
    
    # Write CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write headers
        writer.writerow(headers)
        
        # Write data rows
        for wavelet in wavelets:
            row = [wavelet]
            
            for clf in classifiers_found:
                if wavelet in results_by_wavelet and clf in results_by_wavelet[wavelet]:
                    result = results_by_wavelet[wavelet][clf]
                    metrics = result['metrics']
                    cv_metrics_dict = result['cv_metrics']
                    cv_std = result['cv_std']
                    
                    # Test metrics
                    for metric in test_metrics:
                        value = metrics.get(metric, np.nan)
                        row.append(f"{value:.6f}" if not np.isnan(value) else "")
                    
                    # CV mean metrics
                    for metric in cv_metric_names:
                        value = cv_metrics_dict.get(metric, np.nan)
                        row.append(f"{value:.6f}" if not np.isnan(value) else "")
                    
                    # CV std metrics
                    for metric in cv_metric_names:
                        value = cv_std.get(metric, np.nan)
                        row.append(f"{value:.6f}" if not np.isnan(value) else "")
                else:
                    # No results for this wavelet-classifier combination
                    # Fill with empty strings
                    num_cols = len(test_metrics) + 2 * len(cv_metric_names)
                    row.extend([""] * num_cols)
            
            writer.writerow(row)


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
        #wavelets=['haar', 'db2', 'db4', 'db8', 'db16', 'bior1.1', 'bior2.2','bior4.1', 'bior6.8', 'sym2', 'sym4', 'sym8', 'sym16', 'coif2', 'coif4', 'coif8', 'coif16'],
        wavelets=['bior4.4'],
        classifiers=['svm', 'xgboost', 'random_forest', 'naive_bayes'],
        verbose=True
    )

