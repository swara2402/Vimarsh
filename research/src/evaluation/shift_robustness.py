"""Controlled feature-shift stress tests for frozen selective policies.

The shift is a synthetic covariate perturbation, not an estimate of natural
production drift. Validation-selected thresholds are frozen before shifted
test labels are evaluated.
"""
import numpy as np
import pandas as pd

from src.evaluation.paired_bootstrap import paired_bootstrap_policy_comparisons


def make_mean_feature_shift(features, training_scale, severity, random_state=42):
    """Apply a deterministic signed mean offset plus small Gaussian perturbation.

    Severity is measured in training-set standard deviations. The alternating
    signed offset makes the perturbation reproducible and non-uniform across
    dimensions; added noise prevents it being only a rigid translation.
    """
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if not features.columns.is_unique:
        raise ValueError("feature columns must be unique")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("features must contain only finite numeric values")
    if not np.isfinite(severity) or severity < 0:
        raise ValueError("severity must be a finite non-negative number")
    if not isinstance(training_scale, pd.Series):
        training_scale = pd.Series(training_scale, index=features.columns)
    if list(training_scale.index) != list(features.columns):
        raise ValueError("training_scale columns must exactly match features")
    scale = training_scale.to_numpy(dtype=float)
    if not np.isfinite(scale).all() or np.any(scale < 0):
        raise ValueError("training_scale must contain finite non-negative values")
    scale = np.where(scale > 0, scale, 1.0)
    direction = np.where(np.arange(features.shape[1]) % 2 == 0, 1.0, -1.0)
    rng = np.random.default_rng(random_state)
    perturbation = severity * (direction + rng.normal(0.0, 0.1, size=features.shape))
    shifted = features.to_numpy(dtype=float) + perturbation * scale
    return pd.DataFrame(shifted, index=features.index, columns=features.columns)


def evaluate_frozen_policy_under_shift(
    validation_scores_by_policy,
    shifted_test_scores_by_policy,
    shifted_predictions,
    shifted_labels,
    groups=None,
    target_coverage=0.75,
    n_bootstrap=500,
    random_state=42,
):
    """Evaluate frozen validation thresholds on shifted test scores."""
    from src.evaluation.threshold_diagnostics import select_validation_threshold

    if set(validation_scores_by_policy) != set(shifted_test_scores_by_policy):
        raise ValueError("Validation and shifted-test policy names must match")
    if "confidence_only" not in validation_scores_by_policy:
        raise ValueError("confidence_only reference policy is required")
    predictions = np.asarray(shifted_predictions)
    labels = np.asarray(shifted_labels)
    if predictions.ndim != 1 or labels.ndim != 1 or len(predictions) != len(labels):
        raise ValueError("shifted predictions and labels must be aligned one-dimensional arrays")
    if not len(labels):
        raise ValueError("shifted test labels must not be empty")

    metrics = {}
    for name, validation_scores in validation_scores_by_policy.items():
        test_scores = shifted_test_scores_by_policy[name]
        accepted, threshold = select_validation_threshold(
            validation_scores, test_scores, target_coverage
        )
        if len(accepted) != len(labels):
            raise ValueError("shifted scores and labels must have equal lengths")
        metrics[name] = {
            "frozen_validation_threshold": float(threshold),
            "coverage": float(accepted.mean()),
            "accepted_observations": int(accepted.sum()),
            "accepted_error_rate": (
                float(np.mean(predictions[accepted] != labels[accepted]))
                if accepted.any() else None
            ),
            "overall_accuracy": float(np.mean(predictions == labels)),
        }

    paired = paired_bootstrap_policy_comparisons(
        validation_scores_by_policy,
        shifted_test_scores_by_policy,
        predictions,
        labels,
        reference_policy="confidence_only",
        target_coverage=target_coverage,
        groups=groups,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    return {
        "threshold_protocol": "Thresholds selected on original validation scores and frozen for shifted test evaluation; shifted labels are used only for final metrics.",
        "target_coverage": float(target_coverage),
        "policy_metrics": metrics,
        "paired_bootstrap_vs_confidence": paired,
    }



def _validate_training_reference(features, reference, name):
    if not isinstance(reference, pd.Series):
        reference = pd.Series(reference, index=features.columns)
    if list(reference.index) != list(features.columns):
        raise ValueError(f"{name} columns must exactly match features")
    values = reference.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def make_gaussian_feature_noise(features, training_scale, noise_ratio, random_state=42):
    """Add independent zero-mean Gaussian noise scaled by training feature SD.

    The noise ratio is relative to each feature's training-set standard
    deviation. It is a stress-test parameter, not an estimate of natural noise.
    """
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if not features.columns.is_unique:
        raise ValueError("feature columns must be unique")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite numeric values")
    if not np.isfinite(noise_ratio) or noise_ratio < 0:
        raise ValueError("noise_ratio must be finite and non-negative")
    scale = _validate_training_reference(features, training_scale, "training_scale")
    rng = np.random.default_rng(random_state)
    noisy = values + rng.normal(size=values.shape) * (noise_ratio * scale)
    return pd.DataFrame(noisy, index=features.index, columns=features.columns)


def make_training_median_mask(features, training_medians, mask_fraction, random_state=42):
    """Replace a random fraction of test cells with training-only feature medians.

    This models feature masking without introducing NaNs into estimators whose
    fitted preprocessing pipeline does not support missing values. The
    replacement values are computed from training data, never test data.
    """
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if not features.columns.is_unique:
        raise ValueError("feature columns must be unique")
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite numeric values")
    if not np.isfinite(mask_fraction) or not 0 <= mask_fraction <= 1:
        raise ValueError("mask_fraction must be finite and between 0 and 1")
    medians = _validate_training_reference(features, training_medians, "training_medians")
    result = values.copy()
    cell_count = result.size
    masked_count = int(np.floor(cell_count * mask_fraction))
    if masked_count:
        rng = np.random.default_rng(random_state)
        selected = rng.choice(cell_count, size=masked_count, replace=False)
        rows, cols = np.unravel_index(selected, result.shape)
        result[rows, cols] = medians[cols]
    return pd.DataFrame(result, index=features.index, columns=features.columns)
