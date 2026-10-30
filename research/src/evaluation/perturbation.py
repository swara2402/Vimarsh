"""Controlled perturbation experiments to test reliability engine response."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ..reliability.feature_extractor import ReliabilityFeatureExtractor


def apply_feature_noise(
    df: pd.DataFrame,
    noise_std_ratio: float,
    random_state: int = 42
) -> pd.DataFrame:
    """Add Gaussian noise to features with noise scaled by feature standard deviation."""
    np.random.seed(random_state)
    perturbed = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_std = df[col].std()
            noise = np.random.normal(0, noise_std_ratio * feature_std, size=len(df))
            perturbed[col] = df[col] + noise
    
    return perturbed


def apply_missing_values(
    df: pd.DataFrame,
    missing_ratio: float,
    random_state: int = 42
) -> pd.DataFrame:
    """Set random feature values to NaN."""
    np.random.seed(random_state)
    perturbed = df.copy()
    
    n_total = len(df) * len(df.columns)
    n_missing = int(n_total * missing_ratio)
    
    # Generate random positions to set as missing
    rows = np.random.randint(0, len(df), n_missing)
    cols = np.random.randint(0, len(df.columns), n_missing)
    
    for row, col in zip(rows, cols):
        perturbed.iloc[row, col] = np.nan
    
    return perturbed


def apply_distribution_shift(
    df: pd.DataFrame,
    shift_mean: float,
    shift_std: float,
    random_state: int = 42
) -> pd.DataFrame:
    """Apply global mean shift to all features to simulate distribution shift."""
    np.random.seed(random_state)
    perturbed = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_mean = df[col].mean()
            feature_std = df[col].std()
            perturbed[col] = df[col] + shift_mean * feature_std + np.random.normal(0, shift_std * feature_std, size=len(df))
    
    return perturbed


def run_perturbation_experiments(
    X_clean: pd.DataFrame,
    y_true: pd.Series,
    feature_extractor: ReliabilityFeatureExtractor,
    base_models: List[Any],
    reliability_model = None,
    save_dir: Path = None,
    preprocessor: Any = None,  # Receive the exact same preprocessor from main experiment
    primary_test_predictions: List[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive perturbation experiments to test reliability engine behavior.
    
    Args:
        X_clean: Clean test data
        feature_extractor: Trained reliability feature extractor
        base_models: List of trained base models
        save_dir: Directory to save results
        preprocessor: Full preprocessor from main experiment to maintain pipeline consistency
        primary_test_predictions: Predictions from main primary test evaluation to verify alignment
    
    Returns:
        Dictionary of results for each perturbation type and severity
    """
    # Define perturbation scenarios
    scenarios = [
        # Noise experiments
        ('noise_low', lambda x: apply_feature_noise(x, 0.1)),
        ('noise_medium', lambda x: apply_feature_noise(x, 0.3)),
        ('noise_high', lambda x: apply_feature_noise(x, 0.5)),
        # Missing value experiments
        ('missing_low', lambda x: apply_missing_values(x, 0.05)),
        ('missing_medium', lambda x: apply_missing_values(x, 0.15)),
        ('missing_high', lambda x: apply_missing_values(x, 0.3)),
        # Distribution shift experiments
        ('shift_small', lambda x: apply_distribution_shift(x, 0.5, 0.2)),
        ('shift_medium', lambda x: apply_distribution_shift(x, 1.0, 0.3)),
        ('shift_large', lambda x: apply_distribution_shift(x, 2.0, 0.5))
    ]
    
    results = {}
    # Pass reliability_model, preprocessor and primary predictions to clean baseline for consistent evaluation
    results['clean'] = _evaluate_perturbation(X_clean, y_true, feature_extractor, base_models, reliability_model, preprocessor, primary_test_predictions, is_clean_baseline=True)
    print(f"Completed clean baseline: avg_error_prob = {results['clean']['avg_error_probability']:.4f}")
    
    # Run all perturbation scenarios
    for name, perturb_func in scenarios:
        print(f"\nRunning perturbation: {name}")
        X_perturbed = perturb_func(X_clean)
        results[name] = _evaluate_perturbation(X_perturbed, y_true, feature_extractor, base_models, reliability_model, preprocessor, primary_test_predictions)
        print(f"Completed {name}: avg_error_prob = {results[name]['avg_error_probability']:.4f}")
    
    # Save results
    if save_dir:
        # Convert to DataFrame for easier analysis
        rows = []
        for scenario, res in results.items():
            row = {
                'scenario': scenario,
                'avg_error_probability': res['avg_error_probability'],
                'actual_error_rate': res['actual_error_rate'],
                'avg_uncertainty_score': res['avg_uncertainty_score'],
                'avg_quality_score': res['avg_quality_score'],
                'avg_shift_score': res['avg_shift_score']
            }
            rows.append(row)
        
        pd.DataFrame(rows).to_csv(save_dir / 'perturbation_experiment_results.csv')
        
        # Also save full reliability features for each scenario
        for scenario, res in results.items():
            res['reliability_features'].to_csv(save_dir / f'reliability_features_{scenario}.csv', index=False)
    
    return results


def _evaluate_perturbation(
    X: pd.DataFrame,
    y_true: pd.Series,
    feature_extractor: ReliabilityFeatureExtractor,
    base_models: List[Any],
    reliability_model = None,
    preprocessor: Any = None,
    primary_test_predictions: List[int] = None,
    is_clean_baseline: bool = False
) -> Dict[str, Any]:
    """Evaluate reliability features for a perturbed dataset."""
    # Get predictions from all base models for each sample
    batch_model_probs = []
    batch_model_preds = []  # Initialize to store predictions from all models for each sample
    true_labels = X.index.to_list()  # Preserve sample identifiers
    print(f"Processing {len(X)} samples for perturbation evaluation")
    
    for idx, (_, row) in enumerate(X.iterrows()):
        sample_probs = []
        # Create DataFrame with original columns to match preprocessing expectations
        row_df = pd.DataFrame([row.fillna(0)])
        # Apply exact same preprocessing as main test evaluation to maintain pipeline consistency
        row_values = preprocessor.transform(row_df) if preprocessor is not None else row_df.values.reshape(1, -1)
        sample_probs = []
        sample_preds = []
        for model in base_models:
            probs = model.predict_proba(row_values)[0]
            pred = model.predict(row_values)[0]  # Use model's predict() method to match main test evaluation
            sample_probs.append(probs)
            sample_preds.append(pred)
        batch_model_probs.append(sample_probs)
        batch_model_preds.append(sample_preds)
        if idx % 200 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(X)} samples")
    
    # Extract reliability features
    reliability_features = feature_extractor.extract_features_batch(X, batch_model_probs)
    print(f"Extracted reliability features shape: {reliability_features.shape}")
    
    # Calculate error probabilities if reliability model is available
    avg_error_probability = 0.0
    error_probs = None
    if reliability_model is not None and hasattr(reliability_model, 'predict_error_probability'):
        try:
            # Verify correct error class using estimator.classes_ (Milestone 2.2 requirement)
            if hasattr(reliability_model.model, 'classes_'):
                classes = reliability_model.model.classes_
                error_class_index = np.where(classes == 1)[0][0] if 1 in classes else 1
                print(f"Reliability model classes: {classes}, error class (1) at index: {error_class_index}")
            else:
                error_class_index = 1  # Default to index 1 if classes_ not available
            
            error_probs = reliability_model.predict_error_probability(reliability_features)
            avg_error_probability = float(np.mean(error_probs))
            # Log comprehensive probability statistics
            print(f"Error probability stats: min={np.min(error_probs):.6f}, max={np.max(error_probs):.6f}, mean={avg_error_probability:.6f}, std={np.std(error_probs):.6f}")
            print(f"Unique probability values: {len(np.unique(error_probs))}")
        except Exception as e:
            print(f"Error in reliability model prediction: {str(e)}")
            pass  # Fallback to 0.0 if prediction fails
    
    # Calculate actual error rate from base model predictions for comparison
    # Use Random Forest (index 1) as main model - EXACTLY the same as main test evaluation
    actual_errors = 0
    perturbation_predictions = []
    for i, (sample_preds, (_, row)) in enumerate(zip(batch_model_preds, X.iterrows())):
        main_pred = sample_preds[1]
        true_label = y_true.iloc[i]
        perturbation_predictions.append(main_pred)
        if main_pred != true_label:
            actual_errors += 1
    actual_error_rate = actual_errors / len(X) if len(X) > 0 else 0.0
    
    # Add strict clean baseline consistency check against PRIMARY TEST error rate
    if is_clean_baseline and len(X) > 0:  # This is the clean baseline
        # First, log sample alignment details for audit
        print("\n=== CLEAN PERTURBATION BASELINE ALIGNMENT AUDIT ===")
        print(f"First 10 test indices: {list(X.index[:10])}")
        print(f"First 10 ground-truth labels: {list(y_true.iloc[:10].values)}")
        print(f"First 10 clean perturbation predictions: {perturbation_predictions[:10]}")
        
        # Verify complete alignment with primary test predictions if provided
        if primary_test_predictions is not None and is_clean_baseline:
            # Log first 10 of everything for audit (Milestone 2.2 requirement)
            print("\n=== Sample Alignment Audit (First 10 samples) ===")
            print(f"Test indices: {list(X.index[:10])}")
            print(f"Ground-truth labels: {list(y_true.iloc[:10].values)}")
            print(f"Primary test predictions: {primary_test_predictions[:10]}")
            
            # Get perturbation predictions for first 10
            perturbation_first10 = [sample_preds[1] for sample_preds in batch_model_preds[:10]]
            print(f"Clean perturbation predictions: {perturbation_first10}")
            
            # Calculate alignment metrics for first 10 samples
            matching_preds = sum(1 for p1, p2 in zip(primary_test_predictions[:10], perturbation_first10) if p1 == p2)
            mismatching_preds = len(primary_test_predictions[:10]) - matching_preds
            print(f"\nFirst 10 samples prediction alignment: {matching_preds} matching, {mismatching_preds} mismatching")
            
            # Calculate alignment statistics for all samples
            perturbation_predictions = [sample_preds[1] for sample_preds in batch_model_preds]
            matching_predictions = sum(1 for a, b in zip(perturbation_predictions, primary_test_predictions) if a == b)
            mismatching_predictions = len(perturbation_predictions) - matching_predictions
            print(f"\nTotal samples: {len(X)}")
            print(f"Matching predictions: {matching_predictions}")
            print(f"Mismatching predictions: {mismatching_predictions}")
            
            # Use error rate directly from primary test predictions instead of recalculating
            primary_test_errors = sum(1 for pred, true in zip(primary_test_predictions, y_true.values) if pred != true)
            primary_test_error_rate = primary_test_errors / len(X) if len(X) > 0 else 0.0
            
            print(f"\nPrimary test evaluation error rate: {primary_test_error_rate:.4f}")
            print(f"Clean perturbation baseline error rate: {actual_error_rate:.4f}")
            
            # Strict assertion required by Milestone 2.2 - must be within 1e-10
            np.testing.assert_allclose(
                actual_error_rate,
                primary_test_error_rate,
                atol=1e-10,
                err_msg=f"Clean perturbation baseline error rate ({actual_error_rate:.6f}) does not match primary test error rate ({primary_test_error_rate:.6f}) - check pipeline alignment!"
            )
            print(f"\n✅ Clean baseline consistency check PASSED: perturbation error rate matches primary test evaluation exactly")
    
    print(f"Actual base model error rate: {actual_error_rate:.4f} (vs predicted avg error prob: {avg_error_probability:.4f})")
    
    # Calculate aggregate statistics
    return {
        'reliability_features': reliability_features,
        'avg_error_probability': avg_error_probability,
        'error_probs': error_probs,
        'actual_error_rate': actual_error_rate,
        'avg_uncertainty_score': float(reliability_features['uncertainty_score'].mean()),
        'avg_quality_score': float(reliability_features['overall_input_quality_score'].mean()),
        'avg_shift_score': float(reliability_features['mahalanobis_distance'].mean())
    }