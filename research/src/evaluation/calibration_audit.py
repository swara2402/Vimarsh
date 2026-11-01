"""Calibration and selective-risk diagnostics for binary classifiers."""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def _validated_inputs(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim == 2:
        if p.shape[1] != 2:
            raise ValueError("probabilities must have shape (n,) or (n, 2)")
        p = p[:, 1]
    if p.ndim != 1 or len(p) != len(y) or len(y) == 0:
        raise ValueError("y_true and probabilities must be non-empty and have matching lengths")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return y.astype(int), p


def _bin_error(confidence: np.ndarray, outcomes: np.ndarray, edges: np.ndarray) -> Tuple[float, float]:
    ids = np.digitize(confidence, edges[1:-1], right=False)
    errors, weights = [], []
    for idx in range(len(edges) - 1):
        mask = ids == idx
        if mask.any():
            errors.append(abs(float(outcomes[mask].mean()) - float(confidence[mask].mean())))
            weights.append(float(mask.mean()))
    if not errors:
        return 0.0, 0.0
    return float(np.dot(errors, weights)), float(max(errors))


def _adaptive_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def calibration_audit(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10,
                      high_confidence_threshold: float = 0.9) -> Dict[str, float]:
    """Compute probability, top-label, classwise and selective-risk diagnostics."""
    y, p1 = _validated_inputs(y_true, probabilities)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    if not 0.0 <= high_confidence_threshold <= 1.0:
        raise ValueError("high_confidence_threshold must be in [0, 1]")
    p = np.column_stack((1.0 - p1, p1))
    pred = (p1 >= 0.5).astype(int)
    confidence = np.maximum(p1, 1.0 - p1)
    correct = (pred == y).astype(float)
    ece, mce = _bin_error(p1, y.astype(float), np.linspace(0.0, 1.0, n_bins + 1))
    adaptive_ece, _ = _bin_error(p1, y.astype(float), _adaptive_edges(p1, n_bins))
    top_ece, top_mce = _bin_error(confidence, correct, np.linspace(0.5, 1.0, n_bins + 1))
    class0_ece, _ = _bin_error(p[:, 0], (y == 0).astype(float), np.linspace(0.0, 1.0, n_bins + 1))
    class1_ece, _ = _bin_error(p[:, 1], (y == 1).astype(float), np.linspace(0.0, 1.0, n_bins + 1))
    order = np.argsort(-confidence, kind="mergesort")
    cumulative_risk = np.cumsum(1.0 - correct[order]) / np.arange(1, len(y) + 1)
    coverage = np.arange(1, len(y) + 1, dtype=float) / len(y)
    aurc = float(np.trapz(cumulative_risk, coverage)) if len(y) > 1 else float(cumulative_risk[0])
    high = confidence >= high_confidence_threshold
    return {
        "n_samples": int(len(y)), "ece": ece, "mce": mce, "adaptive_ece": adaptive_ece,
        "top_label_ece": top_ece, "top_label_mce": top_mce,
        "classwise_ece_mean": float((class0_ece + class1_ece) / 2.0),
        "brier_score": float(brier_score_loss(y, p1)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "high_confidence_threshold": float(high_confidence_threshold),
        "high_confidence_count": int(high.sum()),
        "high_confidence_error_rate": float((1.0 - correct[high]).mean()) if high.any() else 0.0,
        "risk_coverage_auc": aurc,
    }
