"""
Calibration module for probability calibration and reliability metrics.
Implements Platt scaling and isotonic regression, plus ECE, Brier score, and reliability diagrams.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Expected Calibration Error (ECE).
    Uses uniform binning strategy over [0,1].
    ECE = sum_{b=1 to B} (n_b/n) * |acc_b - conf_b|
    """
    if len(y_prob.shape) > 1:
        # If probability matrix is provided, use positive class probabilities
        y_prob = y_prob[:, 1]
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    bin_stats = []
    total_samples = len(y_true)
    
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        bin_count = np.sum(mask)
        
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_prob[mask])
            bin_weight = bin_count / total_samples
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
            
            bin_stats.append({
                'bin': bin_idx,
                'bin_low': bins[bin_idx],
                'bin_high': bins[bin_idx+1],
                'count': bin_count,
                'accuracy': bin_accuracy,
                'confidence': bin_confidence,
                'weight': bin_weight
            })
        else:
            bin_stats.append({
                'bin': bin_idx,
                'bin_low': bins[bin_idx],
                'bin_high': bins[bin_idx+1],
                'count': 0,
                'accuracy': np.nan,
                'confidence': np.nan,
                'weight': 0.0
            })
    
    return ece, pd.DataFrame(bin_stats)

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattScaling


class CalibratedClassifierWrapper:
    """Module-level calibrated model wrapper that can be serialized with joblib."""

    def __init__(self, base_model: Any, calibrator: Any):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = np.asarray(getattr(base_model, "classes_", [0, 1]))

    def predict(self, X):
        return self.base_model.predict(X)

    def predict_proba(self, X):
        base_positive = self.base_model.predict_proba(X)[:, 1]
        if hasattr(self.calibrator, "predict_proba"):
            calibrated_positive = self.calibrator.predict_proba(
                base_positive.reshape(-1, 1)
            )[:, 1]
        else:
            calibrated_positive = self.calibrator.predict(base_positive)
        return np.column_stack([1 - calibrated_positive, calibrated_positive])

def calibrate_model(
    base_model: Any,
    X_val: np.ndarray,
    y_val: pd.Series,
    method: str = 'isotonic',
    random_state: int = 42
) -> Any:
    """
    Manually calibrate model probabilities using validation data only.
    Prevents leakage by never using test data for calibration.
    Implements isotonic regression or Platt scaling (sigmoid) manually.
    Returns a wrapped model with calibrated predict_proba.
    """
    # Get base model probabilities on validation data
    y_prob_val = base_model.predict_proba(X_val)[:, 1]  # Use positive class probabilities
    
    # Fit calibration model
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif method == 'sigmoid':
        calibrator = PlattScaling(random_state=random_state)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    calibrator.fit(y_prob_val, y_val)
    
    # Module-level wrapper keeps the calibrated model joblib-serializable.
    calibrated = CalibratedClassifierWrapper(base_model, calibrator)
    print(f"Model calibrated successfully using {method} calibration")
    return calibrated

def calculate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Calculate comprehensive calibration metrics."""
    ece, _ = calculate_ece(y_true, y_prob, n_bins)
    brier = brier_score_loss(y_true, y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob)
    ll = log_loss(y_true, y_prob)
    
    return {
        'expected_calibration_error': float(ece),
        'brier_score': float(brier),
        'log_loss': float(ll)
    }

def create_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: Path,
    n_bins: int = 10
) -> None:
    """Create and save reliability diagram."""
    ece, bin_stats = calculate_ece(y_true, y_prob, n_bins)
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(bin_stats['confidence'], bin_stats['accuracy'], s=100, alpha=0.7, label='Model calibration')
    plt.plot(bin_stats['confidence'].dropna(), bin_stats['accuracy'].dropna(), '-o')
    
    plt.xlabel('Mean predicted probability (confidence)')
    plt.ylabel('Fraction of positive samples (accuracy)')
    plt.title(f'{title}\nECE = {ece:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Reliability diagram saved to {save_path}")

if __name__ == "__main__":
    # Test calibration functions
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Create synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    ece, bin_stats = calculate_ece(y_test, y_prob, n_bins=10)
    print(f"ECE: {ece:.4f}")
    
    metrics = calculate_calibration_metrics(y_test, y_prob)
    print(f"Calibration metrics: {metrics}")