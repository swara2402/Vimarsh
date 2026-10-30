"""Core evaluation metrics for reliability engine."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, precision_recall_fscore_support, confusion_matrix


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Calculate all evaluation metrics for reliability estimator."""
    metrics = {}
    
    # Probabilistic metrics
    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
    metrics['pr_auc'] = float(average_precision_score(y_true, y_pred_proba))
    metrics['brier_score'] = float(brier_score_loss(y_true, y_pred_proba))
    metrics['log_loss'] = float(log_loss(y_true, y_pred_proba))
    
    # Classification metrics at threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1_score'] = float(f1)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Calculate ECE
    metrics['ece'] = calculate_ece(y_true, y_pred_proba)
    
    # Class balance
    metrics['error_fraction'] = float(np.mean(y_true))
    
    return metrics


def calculate_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    for bin_idx in range(n_bins):
        bin_mask = (y_pred_proba >= bin_edges[bin_idx]) & (y_pred_proba < bin_edges[bin_idx + 1])
        bin_count = np.sum(bin_mask)
        
        if bin_count > 0:
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_confidence = np.mean(y_pred_proba[bin_mask])
            ece += bin_count / len(y_true) * abs(bin_confidence - bin_accuracy)
    
    return ece


def calculate_high_confidence_error_detection(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    original_confidences: np.ndarray,
    confidence_threshold: float = 0.9
) -> Dict[str, float]:
    """Calculate error detection performance on high-confidence original predictions."""
    high_conf_mask = original_confidences >= confidence_threshold
    if np.sum(high_conf_mask) == 0:
        return {'high_conf_sample_count': 0}
    
    y_true_high = y_true[high_conf_mask]
    y_pred_proba_high = y_pred_proba[high_conf_mask]
    
    metrics = {
        'high_conf_sample_count': int(np.sum(high_conf_mask)),
        'high_conf_error_rate': float(np.mean(y_true_high)) if len(y_true_high) > 0 else 0.0
    }
    
    if len(np.unique(y_true_high)) > 1:  # Need both classes to calculate ROC-AUC
        metrics['high_conf_roc_auc'] = float(roc_auc_score(y_true_high, y_pred_proba_high))
        metrics['high_conf_pr_auc'] = float(average_precision_score(y_true_high, y_pred_proba_high))
    
    return metrics