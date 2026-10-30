"""
Evaluation module for calculating comprehensive performance metrics.
Computes accuracy, precision, recall, F1, ROC-AUC, confusion matrix, plus calibration metrics.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.calibration import calculate_calibration_metrics

def calculate_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Calculate comprehensive set of performance metrics.
    Combines discrimination metrics and calibration metrics.
    """
    # Discrimination metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob[:, 1])
    cm = confusion_matrix(y_true, y_pred)
    
    # Calibration metrics
    calib_metrics = calculate_calibration_metrics(y_true, y_prob, n_bins)
    
    # Compile all metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        **calib_metrics
    }
    
    return metrics

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    ece_bins: int = 10
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    Returns all performance and calibration metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    metrics = calculate_performance_metrics(y_test, y_pred, y_prob, n_bins=ece_bins)
    metrics['model_name'] = model_name
    
    print(f"\nEvaluation complete for {model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  ECE: {metrics['expected_calibration_error']:.4f}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    
    return metrics, y_pred, y_prob

def save_metrics(metrics: Dict[str, Any], save_path: Path) -> None:
    """Save metrics to CSV file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten metrics for CSV
    flat_metrics = metrics.copy()
    cm = flat_metrics.pop('confusion_matrix')
    flat_metrics['tn'] = cm[0][0]
    flat_metrics['fp'] = cm[0][1]
    flat_metrics['fn'] = cm[1][0]
    flat_metrics['tp'] = cm[1][1]
    
    pd.DataFrame([flat_metrics]).to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")

def create_consolidated_metrics(metrics_list: list, save_path: Path) -> None:
    """Create consolidated comparison table for all models."""
    # Extract key metrics for comparison
    comparison_data = []
    for metrics in metrics_list:
        comparison_data.append({
            'model': metrics['model_name'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc'],
            'ece': metrics['expected_calibration_error'],
            'brier_score': metrics['brier_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(save_path, index=False)
    print(f"Consolidated metrics saved to {save_path}")
    print("\nModel comparison:")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    # Test evaluation
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Create test data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, 'random_forest')
    print("\nFull metrics dictionary:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f"  {k}: {v}")