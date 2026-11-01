import numpy as np
import pytest
from src.evaluation.calibration_audit import calibration_audit


def test_perfect_probabilities_have_zero_calibration_error():
    y = np.array([0, 0, 1, 1])
    result = calibration_audit(y, np.array([0.0, 0.0, 1.0, 1.0]), n_bins=4)
    assert result["ece"] == 0.0
    assert result["mce"] == 0.0
    assert result["top_label_ece"] == 0.0
    assert result["brier_score"] == 0.0


def test_two_column_probabilities_are_supported():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.3, 0.7]])
    result = calibration_audit(y, p)
    assert result["n_samples"] == 4
    assert 0.0 <= result["ece"] <= 1.0
    assert 0.0 <= result["risk_coverage_auc"] <= 1.0


@pytest.mark.parametrize("probabilities", [
    np.array([]), np.array([0.2, np.nan]), np.array([0.2, 1.2]),
    np.array([[0.2, 0.3, 0.5]]),
])
def test_invalid_probabilities_are_rejected(probabilities):
    labels = np.array([0]) if probabilities.ndim == 2 else np.array([0, 1])
    with pytest.raises(ValueError):
        calibration_audit(labels, probabilities)


def test_length_label_and_bin_validation():
    with pytest.raises(ValueError):
        calibration_audit(np.array([0, 1]), np.array([0.2]))
    with pytest.raises(ValueError):
        calibration_audit(np.array([0, 2]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError):
        calibration_audit(np.array([0, 1]), np.array([0.2, 0.8]), n_bins=0)


def test_probability_one_is_included_in_last_bin():
    result = calibration_audit(np.array([1, 0]), np.array([1.0, 0.0]), n_bins=2)
    assert result["ece"] == 0.0
    assert result["top_label_ece"] == 0.0
