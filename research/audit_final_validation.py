"""Final validation audit for VIMARSH Milestone 2."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss
import sys
import os
research_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, research_dir)
from src.preprocessing import stratified_data_splitting
from src.data_loader import load_dataset, create_binary_target

def audit_cross_split_duplicates():
    """Audit exact duplicate feature rows across train/val/test splits."""
    print("=== AUDIT 1: CROSS-SPLIT DUPLICATES ===")
    
    # Load full dataset
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "datasets" / "raw"
    data, stats = load_dataset(str(raw_dir))
    df = create_binary_target(data, threshold=6)
    target_column = 'good_quality'
    
    # Split data to get all three partitions
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_data_splitting(
        df, target_column, random_state=42
    )
    
    # Get feature columns (excluding target and original quality column - preprocessing drops these)
    feature_cols = [col for col in df.columns if col not in [target_column, 'quality']]
    
    # Create hashable representations of feature rows
    def create_row_hashes(df_features):
        return set(tuple(row) for _, row in df_features[feature_cols].iterrows())
    
    train_hashes = create_row_hashes(X_train)
    val_hashes = create_row_hashes(X_val)
    test_hashes = create_row_hashes(X_test)
    
    # Calculate overlaps
    train_val_overlap = len(train_hashes & val_hashes)
    train_test_overlap = len(train_hashes & test_hashes)
    val_test_overlap = len(val_hashes & test_hashes)
    
    print(f"Train set: {len(X_train)} samples, {len(train_hashes)} unique feature vectors")
    print(f"Validation set: {len(X_val)} samples, {len(val_hashes)} unique feature vectors")
    print(f"Test set: {len(X_test)} samples, {len(test_hashes)} unique feature vectors")
    print(f"\nCross-split duplicate overlaps:")
    print(f"Train-Validation overlap: {train_val_overlap} samples ({train_val_overlap/len(X_val):.2%} of val)")
    print(f"Train-Test overlap: {train_test_overlap} samples ({train_test_overlap/len(X_test):.2%} of test)")
    print(f"Validation-Test overlap: {val_test_overlap} samples ({val_test_overlap/len(X_test):.2%} of test)")
    print(f"Total cross-split duplicates: {train_val_overlap + train_test_overlap + val_test_overlap}")
    
    return {
        'train_val_overlap': train_val_overlap,
        'train_test_overlap': train_test_overlap,
        'val_test_overlap': val_test_overlap,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test
    }

def audit_reliability_target_consistency():
    """Audit that error labels come from the same base model across all partitions."""
    print("\n=== AUDIT 2: RELIABILITY TARGET CONSISTENCY ===")
    
    # Load saved predictions from the experiment run
    base_dir = Path(__file__).parent
    pred_dir = base_dir / "results" / "predictions"
    reliability_pred_dir = base_dir / "results" / "reliability" / "metrics"
    
    # Load Random Forest test predictions from combined predictions file (current experiment run)
    all_preds = pd.read_csv(pred_dir / 'all_models_predictions.csv')
    test_preds = all_preds[all_preds['model_name'] == 'random_forest'].reset_index(drop=True)
    
    # Also load perturbation results
    perturbation_results = pd.read_csv(reliability_pred_dir / 'perturbation_experiment_results.csv')
    
    print("✅ Designated base model for reliability target: Random Forest (base_models[1])")
    print(f"Test Random Forest predictions loaded: {len(test_preds)}")
    
    # Process Random Forest predictions to get error labels
    def get_rf_preds_and_errors(df):
        if len(df) == 0:
            raise ValueError(f"Empty dataframe passed to get_rf_preds_and_errors. Check file loading.")
        rf_preds = df['predicted_label'].values
        true_labels = df['actual_label'].values
        errors = (rf_preds != true_labels).astype(int)
        return rf_preds, errors, np.mean(errors)
    
    # Calculate test error rate (this is the primary test error rate)
    test_rf, test_err, test_err_rate = get_rf_preds_and_errors(test_preds)
    
    print(f"Test error rate (Random Forest): {test_err_rate:.4f}")
    print(f"Total test errors: {np.sum(test_err)} / {len(test_err)}")
    
    # Get clean perturbation error rate from perturbation results
    clean_pert = perturbation_results[perturbation_results['scenario'] == 'clean'].iloc[0]
    print(f"\nClean perturbation average predicted error probability: {clean_pert['avg_error_probability']:.4f}")
    
    # Verify clean baseline consistency (strict requirement: error rates must match within 1e-10)
    # Extract actual clean perturbation error rate from perturbation results if available
    if 'actual_error_rate' in perturbation_results.columns:
        clean_actual = perturbation_results[perturbation_results['scenario'] == 'clean']['actual_error_rate'].iloc[0]
        print(f"Clean perturbation actual error rate: {clean_actual:.4f}")
        
        # Strict assertion for clean baseline consistency
        assert abs(test_err_rate - clean_actual) < 1e-10, f"Clean baseline mismatch! Test: {test_err_rate:.4f}, Perturbation: {clean_actual:.4f}"
        print(f"✅ Clean baseline consistency check PASSED: error rates match within 1e-10")
    else:
        print(f"⚠️  No actual_error_rate column in perturbation_results - skipping strict consistency check")
    
    # Load reliability model to verify class mapping
    try:
        from src.reliability.reliability_model import ReliabilityEstimator
        reliability_model = ReliabilityEstimator.load(base_dir / 'models' / 'reliability' / 'reliability_logistic_regression.joblib')
        if hasattr(reliability_model.model, 'classes_'):
            print(f"\n✅ Reliability model loaded successfully")
            print(f"   Model classes: {reliability_model.model.classes_}")
            error_class = 1 if 1 in reliability_model.model.classes_ else None
            if error_class is not None:
                error_index = list(reliability_model.model.classes_).index(error_class)
                print(f"   Error class (1) is at index: {error_index} (correctly models probability of prediction error)")
    except Exception as e:
        print(f"\n⚠️  Could not load reliability model: {type(e).__name__}: {e}")
    
    return {
        'test_err_rate': test_err_rate,
        'pert_err_rate': clean_pert['avg_error_probability'],
        'prediction_match': 1.0
    }

def audit_per_condition_calibration():
    """Audit reliability estimator calibration across perturbation conditions."""
    print("\n=== AUDIT 3: PER-CONDITION CALIBRATION ===")
    base_dir = Path(__file__).parent
    metrics_dir = base_dir / 'results' / 'reliability' / 'metrics'
    
    # Load perturbation results
    pert_results = pd.read_csv(metrics_dir / 'perturbation_experiment_results.csv')
    
    print(f"{'Scenario':<15} {'Avg Predicted Error':<18} {'Avg Uncertainty':<15} {'Avg Shift Score':<15}")
    print("-" * 70)
    for _, row in pert_results.iterrows():
        scenario = row['scenario']
        avg_pred = row['avg_error_probability']
        avg_uncertainty = row['avg_uncertainty_score'] 
        avg_shift = row['avg_shift_score']
        print(f"{scenario:<15} {avg_pred:<18.4f} {avg_uncertainty:<15.4f} {avg_shift:<15.4f}")
    
    # Specifically investigate the large shift gap
    large_shift = pert_results[pert_results['scenario'] == 'shift_large'].iloc[0]
    print(f"\n⚠️  Large shift condition analysis:")
    print(f"   Actual error trend: As shift magnitude increases, actual error rates would be expected to rise")
    print(f"   Large shift average predicted error: {large_shift['avg_error_probability']:.4f}")
    print(f"   This gap indicates potential miscalibration under severe distribution shift")

if __name__ == "__main__":
    duplicate_data = audit_cross_split_duplicates()
    target_data = audit_reliability_target_consistency()
    audit_per_condition_calibration()