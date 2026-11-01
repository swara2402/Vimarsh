import numpy as np
import pytest

from src.evaluation.paired_bootstrap import (
    paired_bootstrap_policy_comparisons,
    paired_bootstrap_selective_comparison,
)


def test_identical_policies_have_zero_paired_differences():
    accepted = np.array([True, True, False, True, False, True])
    predictions = np.array([0, 1, 0, 1, 1, 0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = paired_bootstrap_selective_comparison(
        accepted, accepted, predictions, labels,
        n_bootstrap=200, random_state=7,
    )
    assert result["coverage_difference"] == 0.0
    assert result["coverage_difference_ci"] == [0.0, 0.0]
    assert result["accepted_error_rate_difference"] == 0.0
    assert result["accepted_error_rate_difference_ci"] == [0.0, 0.0]


def test_group_bootstrap_is_reproducible_and_reports_cluster_unit():
    reference = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    candidate = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    predictions = np.array([0, 1, 1, 0, 1, 0])
    labels = np.array([0, 0, 1, 0, 0, 0])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    kwargs = dict(
        groups=groups, n_bootstrap=300, random_state=19,
    )
    first = paired_bootstrap_selective_comparison(
        reference, candidate, predictions, labels, **kwargs
    )
    second = paired_bootstrap_selective_comparison(
        reference, candidate, predictions, labels, **kwargs
    )
    assert first == second
    assert first["resampling_unit"] == "supplied_group"
    assert first["bootstrap_replicates"] == 300
    assert len(first["coverage_difference_ci"]) == 2


def test_policy_comparison_selects_thresholds_without_using_test_labels():
    validation = {
        "confidence_only": np.array([0.1, 0.2, 0.3, 0.4]),
        "reliability_driven": np.array([0.4, 0.3, 0.2, 0.1]),
    }
    test = {
        "confidence_only": np.array([0.1, 0.2, 0.3, 0.4]),
        "reliability_driven": np.array([0.4, 0.3, 0.2, 0.1]),
    }
    result = paired_bootstrap_policy_comparisons(
        validation, test,
        test_predictions=np.array([0, 1, 0, 1]),
        test_labels=np.array([0, 0, 1, 1]),
        target_coverage=0.5, n_bootstrap=200, random_state=3,
    )
    assert set(result) == {"reliability_driven"}
    assert result["reliability_driven"]["reference_policy"] == "confidence_only"
    assert result["reliability_driven"]["bootstrap_replicates"] == 200


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bootstrap": 99},
        {"confidence_level": 1.0},
        {"groups": np.array([1, 2])},
    ],
)
def test_invalid_bootstrap_configuration_is_rejected(kwargs):
    defaults = dict(
        reference_accepted=np.array([1, 0, 1, 0], dtype=bool),
        candidate_accepted=np.array([0, 1, 1, 0], dtype=bool),
        test_predictions=np.array([0, 1, 0, 1]),
        test_labels=np.array([0, 0, 1, 1]),
        n_bootstrap=200,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        paired_bootstrap_selective_comparison(**defaults)


def test_misaligned_policy_names_are_rejected():
    with pytest.raises(ValueError, match="policy names"):
        paired_bootstrap_policy_comparisons(
            {"confidence_only": [0.1, 0.2]},
            {"reliability_driven": [0.1, 0.2]},
            [0, 1], [0, 1],
        )
