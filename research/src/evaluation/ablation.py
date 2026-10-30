"""Ablation experiment implementation to test feature group contributions."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from ..reliability.reliability_model import ReliabilityEstimator
from .reliability_metrics import calculate_comprehensive_metrics


# Feature group definitions for ablation studies
ABLATION_GROUPS = {
    'A_uncertainty_only': [
        'max_probability', 'predictive_entropy', 'probability_margin', 'uncertainty_score'
    ],
    'B_plus_disagreement': [
        'max_probability', 'predictive_entropy', 'probability_margin', 'uncertainty_score',
        'disagreement_fraction', 'probability_variance', 'probability_std', 'ensemble_entropy'
    ],
    'C_plus_quality': [
        'max_probability', 'predictive_entropy', 'probability_margin', 'uncertainty_score',
        'disagreement_fraction', 'probability_variance', 'probability_std', 'ensemble_entropy',
        'missing_feature_ratio', 'overall_input_quality_score', 'isolation_forest_anomaly'
    ],
    'D_full_features': None  # Use all features
}


def run_ablation_study(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_type: str = 'random_forest',
    random_state: int = 42,
    save_dir: Path = None
) -> Dict[str, Dict[str, float]]:
    """
    Run complete ablation study to evaluate feature group contributions.
    
    Args:
        X_train: Training reliability features
        y_train: Training error labels
        X_val: Validation features
        y_val: Validation labels
        X_test: Test features
        y_test: Test labels
        model_type: Reliability model type to use
        random_state: Random seed
        save_dir: Directory to save results
    
    Returns:
        Dictionary of results for each ablation variant
    """
    results = {}
    
    for variant_name, feature_list in ABLATION_GROUPS.items():
        print(f"\nRunning ablation variant: {variant_name}")
        
        # Select features for this variant
        if feature_list is None:
            X_train_variant = X_train.copy()
            X_val_variant = X_val.copy()
            X_test_variant = X_test.copy()
        else:
            # Verify features exist
            missing = [f for f in feature_list if f not in X_train.columns]
            if missing:
                raise ValueError(f"Missing features for {variant_name}: {missing}")
            
            X_train_variant = X_train[feature_list]
            X_val_variant = X_val[feature_list]
            X_test_variant = X_test[feature_list]
        
        # Train and evaluate model
        model = ReliabilityEstimator(model_type=model_type, random_state=random_state)
        model.train(X_train_variant, y_train, X_val_variant, y_val)
        
        # Evaluate on test set
        test_pred_proba = model.predict_error_probability(X_test_variant)
        test_metrics = calculate_comprehensive_metrics(y_test, test_pred_proba)
        
        results[variant_name] = test_metrics
        print(f"Completed {variant_name}: ROC-AUC = {test_metrics['roc_auc']:.4f}, PR-AUC = {test_metrics['pr_auc']:.4f}")
        
        # Save model if directory is provided
        if save_dir:
            # Skip model saving to avoid pickling errors with nested wrapper classes
            # model.save(save_dir / f'reliability_model_{variant_name}.joblib')
            pass
    
    # Save consolidated results
    if save_dir:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.to_csv(save_dir / 'ablation_study_results.csv')
    
    return results