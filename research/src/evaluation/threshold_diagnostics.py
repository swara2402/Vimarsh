"""Leakage-safe validation-selected coverage diagnostics."""

import numpy as np


def select_validation_threshold(validation_risk_scores, test_risk_scores, target_coverage=0.75):
    """Select a low-risk acceptance threshold on validation scores only.

    Lower risk scores are accepted. Tied test scores at the threshold are all
    accepted, so realized test coverage may exceed the nominal target.
    """
    validation_scores = np.asarray(validation_risk_scores, dtype=float)
    test_scores = np.asarray(test_risk_scores, dtype=float)
    if validation_scores.ndim != 1 or test_scores.ndim != 1:
        raise ValueError("Validation and test risk scores must be one-dimensional")
    if not len(validation_scores) or not len(test_scores):
        raise ValueError("Validation and test risk scores must be non-empty")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between zero and one")
    if not np.isfinite(validation_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("Validation and test risk scores must be finite")

    threshold = float(np.quantile(validation_scores, target_coverage, method="higher"))
    accepted = test_scores <= threshold
    return accepted, threshold


def validation_threshold_metrics(
    validation_risk_scores,
    test_risk_scores,
    test_predictions,
    test_labels,
    target_coverage=0.75,
):
    """Apply a validation-selected threshold, then score accepted test rows."""
    predictions = np.asarray(test_predictions)
    labels = np.asarray(test_labels)
    accepted, threshold = select_validation_threshold(
        validation_risk_scores, test_risk_scores, target_coverage
    )
    if predictions.ndim != 1 or labels.ndim != 1:
        raise ValueError("Test predictions and labels must be one-dimensional")
    if len(accepted) != len(predictions) or len(accepted) != len(labels):
        raise ValueError("Test scores, predictions, and labels must have equal lengths")

    accepted_count = int(accepted.sum())
    observed_error = (
        float(np.mean(predictions[accepted] != labels[accepted]))
        if accepted_count else None
    )
    return {
        "validation_selected_threshold": threshold,
        "target_coverage": float(target_coverage),
        "test_coverage": float(accepted.mean()),
        "accepted_test_observations": accepted_count,
        "observed_error_rate": observed_error,
    }
