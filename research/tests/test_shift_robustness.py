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
    assert result["policy_metrics"]["confidence_only"]["frozen_validation_threshold"] == 0.4
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


from src.evaluation.shift_robustness import (
    make_gaussian_feature_noise,
    make_training_median_mask,
)


def test_gaussian_noise_is_zero_identity_and_seed_reproducible():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    scale = pd.Series({"a": 2.0, "b": 3.0})
    pd.testing.assert_frame_equal(
        make_gaussian_feature_noise(frame, scale, 0.0), frame
    )
    first = make_gaussian_feature_noise(frame, scale, 0.2, random_state=19)
    second = make_gaussian_feature_noise(frame, scale, 0.2, random_state=19)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(frame)


def test_training_median_mask_uses_only_supplied_reference_values():
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    medians = pd.Series({"a": -5.0, "b": 99.0})
    masked = make_training_median_mask(frame, medians, 0.5, random_state=3)
    assert np.sum(masked.to_numpy() == np.array([-5.0, 99.0])) >= 0
    assert np.isin(masked.to_numpy(), [-5.0, 99.0, 1.0, 2.0, 10.0, 20.0]).all()
    assert np.sum(masked.to_numpy() != frame.to_numpy()) == 2


@pytest.mark.parametrize("fraction", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_mask_fraction_is_rejected(fraction):
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        make_training_median_mask(frame, frame.median(), fraction)


@pytest.mark.parametrize("ratio", [-0.1, np.nan, np.inf])
def test_invalid_gaussian_noise_ratio_is_rejected(ratio):
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        make_gaussian_feature_noise(frame, frame.std(), ratio)
