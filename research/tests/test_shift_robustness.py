import numpy as np
import pandas as pd
import pytest

from src.evaluation.shift_robustness import (
    evaluate_frozen_policy_under_shift,
    make_mean_feature_shift,
)


def test_zero_severity_preserves_features_and_metadata():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=[8, 9])
    shifted = make_mean_feature_shift(frame, frame.std(), 0.0, random_state=2)
    pd.testing.assert_frame_equal(shifted, frame)


def test_shift_is_deterministic_and_scales_with_training_standard_deviation():
    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 4.0, 6.0]})
    scale = pd.Series({"a": 2.0, "b": 4.0})
    first = make_mean_feature_shift(frame, scale, 0.5, random_state=11)
    second = make_mean_feature_shift(frame, scale, 0.5, random_state=11)
    pd.testing.assert_frame_equal(first, second)
    assert np.mean(first["a"] - frame["a"]) > 0
    assert np.mean(first["b"] - frame["b"]) < 0


@pytest.mark.parametrize("severity", [-0.1, np.nan, np.inf])
def test_invalid_shift_severity_is_rejected(severity):
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        make_mean_feature_shift(frame, frame.std(), severity)


def test_frozen_validation_thresholds_and_paired_comparisons_are_reported():
    validation = {
        "confidence_only": np.array([0.1, 0.2, 0.3, 0.4]),
        "reliability_driven": np.array([0.4, 0.3, 0.2, 0.1]),
    }
    shifted_scores = {
        "confidence_only": np.array([0.1, 0.2, 0.7, 0.8, 0.1, 0.8]),
        "reliability_driven": np.array([0.7, 0.1, 0.2, 0.8, 0.1, 0.2]),
    }
    predictions = np.array([0, 1, 0, 1, 0, 1])
    labels = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    result = evaluate_frozen_policy_under_shift(
        validation, shifted_scores, predictions, labels,
        groups=groups, n_bootstrap=100, random_state=5,
    )
    assert result["target_coverage"] == 0.75
    assert result["policy_metrics"]["confidence_only"]["frozen_validation_threshold"] == 0.3
    assert set(result["paired_bootstrap_vs_confidence"]) == {"reliability_driven"}
    assert result["paired_bootstrap_vs_confidence"]["reliability_driven"]["bootstrap_replicates"] == 100
    assert "frozen" in result["threshold_protocol"]


def test_policy_name_mismatch_is_rejected():
    with pytest.raises(ValueError, match="policy names"):
        evaluate_frozen_policy_under_shift(
            {"confidence_only": [0.1, 0.2]},
            {"other": [0.1, 0.2]},
            [0, 1], [0, 1],
        )
