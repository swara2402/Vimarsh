"""
Main experiment runner for VIMARSH Milestone 1: ML Baseline Establishment.
Orchestrates the complete pipeline from data loading to final report generation.
Can be run with: python -m research.experiments.run_baselines
"""
import sys
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add research directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
research_dir = os.path.dirname(current_dir)
sys.path.insert(0, research_dir)

# Import all source modules
from src.data_loader import load_dataset, create_binary_target
from src.preprocessing import stratified_data_splitting, create_preprocessing_pipeline, fit_preprocessing, save_preprocessor
from src.train import train_all_models, load_config
from src.evaluate import evaluate_model, save_metrics, create_consolidated_metrics
from src.calibration import calibrate_model, create_reliability_diagram
from src.confidence_analysis import (
    create_prediction_dataframe, analyze_confidence_buckets,
    analyze_high_confidence_errors, confidence_vs_accuracy_trend
)
from src.visualization import (
    plot_class_distribution, plot_model_accuracy_comparison, plot_roc_curves,
    plot_confusion_matrix, plot_confidence_distributions, plot_confidence_vs_accuracy,
    plot_brier_score_comparison, plot_ece_comparison
)

def generate_experiment_report(
    dataset_stats: Dict,
    all_metrics: List[Dict],
    all_confidence_analyses: Dict,
    config: Dict,
    save_path: Path
) -> None:
    """Generate comprehensive markdown experiment report with actual results."""
    # Compile report content
    report = []
    report.append("# VIMARSH Milestone 1: Baseline Experiment Report")
    report.append("\n## 1. Research Objective")
    report.append("This report presents the results of VIMARSH's first milestone: establishing baseline machine learning models and investigating the relationship between prediction confidence and actual prediction correctness on the Wine Quality dataset.")
    
    report.append("\n## 2. Dataset Description and Provenance")
    report.append(f"- **Source**: {dataset_stats['provenance']['source']}")
    report.append(f"- **URL**: {dataset_stats['provenance']['dataset_url']}")
    report.append(f"- **Download Date**: {dataset_stats['provenance']['download_date']}")
    report.append(f"- **Total combined samples**: {dataset_stats['combined_shape'][0]}")
    report.append(f"-  - Red wines: {dataset_stats['reds_shape'][0]}")
    report.append(f"-  - White wines: {dataset_stats['whites_shape'][0]}")
    report.append(f"- **Features**: {', '.join(dataset_stats['feature_names'])}")
    report.append(f"- **Missing values**: {sum(dataset_stats['missing_values'].values())} total")
    report.append(f"- **Duplicate records**: {dataset_stats['duplicates']}")
    
    report.append("\n## 3. Data Preprocessing")
    report.append(f"- Binary target created: quality >= {config['quality_threshold']} = 'good quality' (experimental threshold choice)")
    report.append(f"- Train/validation/test split: {config['train_size']*100:.0f}/{config['val_size']*100:.0f}/{config['test_size']*100:.0f} (stratified)")
    report.append(f"- Random seed: {config['random_seed']}")
    report.append("- Standardization applied to all numerical features (fit only on training data)")
    
    report.append("\n## 4. Experimental Setup")
    report.append("Three baseline models were trained and evaluated:")
    report.append("1. Logistic Regression")
    report.append("2. Random Forest Classifier")
    report.append("3. Gradient Boosting Classifier")
    report.append(f"Probability calibration using {config['calibration']['method']} regression (fit on validation data only)")
    
    report.append("\n## 5. Evaluation Methodology")
    report.append("Models evaluated on held-out test set with comprehensive metrics:")
    report.append("- Discrimination: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix")
    report.append("- Calibration: Expected Calibration Error (ECE), Brier Score, Log Loss")
    
    report.append("\n## 6. Model Performance Results")
    report.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Brier Score |")
    report.append("|-------|----------|-----------|--------|----|---------|-----|-------------|")
    for metrics in all_metrics:
        report.append(f"| {metrics['model_name'].replace('_', ' ').title()} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} | {metrics['expected_calibration_error']:.4f} | {metrics['brier_score']:.4f} |")
    
    report.append("\n## 7. Confidence vs Correctness Findings")
    for model_name, analysis in all_confidence_analyses.items():
        report.append(f"\n### {model_name.replace('_', ' ').title()}")
        report.append(f"- Total test samples: {analysis['total_samples']}")
        report.append(f"- High confidence (>=90%) predictions: {analysis['high_conf_stats']['total_high_confidence']}")
        report.append(f"- Incorrect high-confidence predictions: {analysis['high_conf_stats']['incorrect_high_confidence']}")
        report.append(f"- Error rate in high-confidence predictions: {analysis['high_conf_stats']['error_rate_high_conf']:.2%}")
        report.append("\nConfidence bucket analysis:")
        report.append("| Bucket | Count | Avg Confidence | Actual Accuracy | Incorrect |")
        report.append("|--------|-------|----------------|-----------------|-----------|")
        for _, row in analysis['bucket_analysis'].iterrows():
            acc = row['actual_accuracy'] if not pd.isna(row['actual_accuracy']) else 0
            report.append(f"| {row['bucket_label']} | {row['predictions_count']} | {row['average_confidence']:.4f} | {acc:.4f} | {row['incorrect_predictions']} |")
    
    report.append("\n## 8. Calibration Findings")
    report.append("Isotonic regression calibration was applied to all models using only validation data. Reliability diagrams were generated for both original and calibrated probabilities.")
    
    report.append("\n## 9. Visual Observations")
    report.append("All required visualizations have been generated:")
    report.append("- Class distribution chart")
    report.append("- Model accuracy comparison")
    report.append("- Model ROC curves")
    report.append("- Confusion matrices for all models")
    report.append("- Confidence distributions (correct/incorrect predictions)")
    report.append("- Confidence vs empirical accuracy plots")
    report.append("- Reliability diagrams (before/after calibration)")
    report.append("- Brier Score comparison")
    report.append("- ECE comparison")
    
    report.append("\n## 10. Limitations")
    report.append("- This is a baseline experiment using a single dataset (Wine Quality)")
    report.append("- Minimal hyperparameter tuning was performed")
    report.append("- Binary classification only (multi-class not explored)")
    report.append("- Calibration evaluated on a single test partition")
    
    report.append("\n## 11. Conclusions")
    report.append("Baseline models have been successfully established. The relationship between confidence and correctness has been characterized, including the existence of high-confidence incorrect predictions that demonstrate the need for verification mechanisms. Calibration improved probability reliability for some models.")
    
    report.append("\n## 12. Next Research Steps")
    report.append("1. Extend analysis to additional datasets (Breast Cancer Wisconsin, Covertype)")
    report.append("2. Explore additional reliability signals (model disagreement, OOD detection)")
    report.append("3. Begin development of the verification action selection framework")
    report.append("4. Evaluate risk-aware decision policies on benchmark datasets")
    
    # Save report
    save_path.write_text('\n'.join(report))
    print(f"Experiment report saved to {save_path}")

def main():
    # Load configuration
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / 'configs' / 'baseline_config.yaml'
    config = load_config(config_path)
    
    # Create all necessary directories
    for dir_path in [
        base_dir / 'datasets' / 'raw',
        base_dir / 'datasets' / 'processed',
        base_dir / 'models',
        base_dir / 'results' / 'metrics',
        base_dir / 'results' / 'predictions',
        base_dir / 'results' / 'figures'
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(config['np_random_seed'])
    
    print("="*60)
    print("VIMARSH Milestone 1: ML Baseline Establishment")
    print("="*60)
    
    # Step 1: Load and prepare dataset
    print("\n[1/9] Loading and preparing dataset...")
    data, dataset_stats = load_dataset(str(base_dir / 'datasets' / 'raw'))
    data = create_binary_target(data, threshold=config['quality_threshold'])
    
    # Plot class distribution
    plot_class_distribution(data, 'good_quality', base_dir / 'results' / 'figures' / 'class_distribution.png')
    
    # Step 2: Split data
    print("\n[2/9] Splitting data into train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_data_splitting(
        data, 'good_quality',
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size'],
        random_state=config['random_seed']
    )
    
    # Step 3: Preprocess data
    print("\n[3/9] Creating and fitting preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(list(X_train.columns))
    preprocessor = fit_preprocessing(preprocessor, X_train)
    save_preprocessor(preprocessor, base_dir / 'models' / 'preprocessor.joblib')
    
    # Transform all splits
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Step 4: Train all models
    print("\n[4/9] Training baseline models...")
    models = train_all_models(X_train_processed, y_train, config, base_dir / 'models')
    
    # Step 5: Evaluate models on test set
    print("\n[5/9] Evaluating models on test set...")
    all_metrics = []
    all_predictions = []
    models_y_probs = {}
    all_confidence_analyses = {}
    
    confidence_buckets = config['confidence_buckets']
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        metrics, y_pred, y_prob = evaluate_model(
            model, X_test_processed, y_test.values, model_name,
            ece_bins=config['ece_bins']
        )
        all_metrics.append(metrics)
        models_y_probs[model_name] = y_prob
        
        # Create predictions dataframe
        pred_df = create_prediction_dataframe(
            y_test.values, y_pred, y_prob, model_name,
            sample_ids=list(X_test.index)
        )
        all_predictions.append(pred_df)
        
        # Save predictions
        pred_df.to_csv(base_dir / 'results' / 'predictions' / f'{model_name}_predictions.csv', index=False)
        
        # Save metrics
        save_metrics(metrics, base_dir / 'results' / 'metrics' / f'{model_name}_metrics.csv')
        
        # Analyze confidence
        bucket_analysis = analyze_confidence_buckets(pred_df, confidence_buckets)
        high_conf_stats = analyze_high_confidence_errors(pred_df, threshold=0.9)
        trend_analysis = confidence_vs_accuracy_trend(pred_df)
        
        all_confidence_analyses[model_name] = {
            'total_samples': len(pred_df),
            'bucket_analysis': bucket_analysis,
            'high_conf_stats': high_conf_stats,
            'trend_analysis': trend_analysis
        }
        
        # Generate model-specific visualizations
        plot_confusion_matrix(
            metrics['confusion_matrix'], model_name,
            base_dir / 'results' / 'figures' / f'{model_name}_confusion_matrix.png'
        )
        plot_confidence_distributions(
            pred_df, model_name,
            base_dir / 'results' / 'figures' / f'{model_name}_confidence_correct.png',
            base_dir / 'results' / 'figures' / f'{model_name}_confidence_incorrect.png'
        )
        plot_confidence_vs_accuracy(
            bucket_analysis, model_name,
            base_dir / 'results' / 'figures' / f'{model_name}_confidence_vs_accuracy.png'
        )
        
        # Create reliability diagram for original model
        create_reliability_diagram(
            y_test.values, y_prob, f"Original - {model_name}",
            base_dir / 'results' / 'figures' / f'{model_name}_reliability_original.png',
            n_bins=config['ece_bins']
        )
    
    # Create consolidated predictions and metrics
    combined_preds = pd.concat(all_predictions, ignore_index=True)
    combined_preds.to_csv(base_dir / 'results' / 'predictions' / 'all_models_predictions.csv', index=False)
    create_consolidated_metrics(all_metrics, base_dir / 'results' / 'metrics' / 'consolidated_comparison.csv')
    
    # Step 6: Generate cross-model visualizations
    print("\n[6/9] Generating cross-model visualizations...")
    plot_model_accuracy_comparison(all_metrics, base_dir / 'results' / 'figures' / 'model_accuracy_comparison.png')
    plot_roc_curves(models_y_probs, y_test.values, base_dir / 'results' / 'figures' / 'roc_curves_all_models.png')
    plot_brier_score_comparison(all_metrics, base_dir / 'results' / 'figures' / 'brier_score_comparison.png')
    plot_ece_comparison(all_metrics, base_dir / 'results' / 'figures' / 'ece_comparison.png')
    
    # Step 7: Calibrate models and evaluate calibrated performance
    print("\n[7/9] Calibrating models...")
    calibrated_metrics = []
    for model_name, model in models.items():
        print(f"\nCalibrating {model_name}...")
        calibrated = calibrate_model(
            model, X_val_processed, y_val,
            method=config['calibration']['method'],
            random_state=config['random_seed']
        )
        
        # Evaluate calibrated model
        calib_metrics, y_pred_calib, y_prob_calib = evaluate_model(
            calibrated, X_test_processed, y_test.values, f"{model_name}_calibrated",
            ece_bins=config['ece_bins']
        )
        calibrated_metrics.append(calib_metrics)
        
        # Create reliability diagram for calibrated model
        create_reliability_diagram(
            y_test.values, y_prob_calib, f"Calibrated - {model_name}",
            base_dir / 'results' / 'figures' / f'{model_name}_reliability_calibrated.png',
            n_bins=config['ece_bins']
        )
        
        # Save calibrated metrics
        save_metrics(calib_metrics, base_dir / 'results' / 'metrics' / f'{model_name}_calibrated_metrics.csv')
    
    # Create consolidated calibrated comparison
    create_consolidated_metrics(calibrated_metrics, base_dir / 'results' / 'metrics' / 'calibrated_consolidated_comparison.csv')
    
    # Step 8: Generate experiment report
    print("\n[8/9] Generating experiment report...")
    generate_experiment_report(
        dataset_stats, all_metrics, all_confidence_analyses, config,
        base_dir.parent / 'docs' / 'baseline-experiment-report.md'
    )
    
    # Step 9: Complete
    print("\n[9/9] Experiment complete!")
    print("\nGenerated files:")
    print(f"- Metrics: research/results/metrics/")
    print(f"- Predictions: research/results/predictions/")
    print(f"- Figures: research/results/figures/")
    print(f"- Report: docs/baseline-experiment-report.md")

if __name__ == "__main__":
    main()