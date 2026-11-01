import numpy as np
import pytest

from src.evaluation.threshold_diagnostics import (
    select_validation_threshold,
    validation_threshold_metrics,
)


def test_threshold_is_selected_from_validation_scores_and_applied_to_test():
    validation = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    test = np.array([1.0, 3.0, 3.0, 4.0])
    accepted, threshold = select_validation_threshold(validation, test, 0.6)
    assert threshold == 3.0
    assert accepted.tolist() == [True, True, True, False]


def test_test_labels_do_not_enter_threshold_selection():
    validation = [0.0, 1.0, 2.0, 3.0, 4.0]
    test = [0.5, 1.5, 2.5, 3.5]
    first_mask, first_threshold = select_validation_threshold(validation, test, 0.5)
    second_mask, second_threshold = select_validation_threshold(validation, test, 0.5)
    assert np.array_equal(first_mask, second_mask)
    assert first_threshold == second_threshold


def test_metric_scoring_uses_only_accepted_test_rows():
    result = validation_threshold_metrics(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        target_coverage=0.5,
    )
    assert result["accepted_test_observations"] == 2
    assert result["test_coverage"] == pytest.approx(0.5)
    assert result["observed_error_rate"] == pytest.approx(0.5)


def test_ties_at_validation_threshold_can_exceed_nominal_coverage():
    accepted, threshold = select_validation_threshold(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [4.0, 4.0, 3.0],
        target_coverage=0.2,
    )
    assert threshold == 4.0
    assert accepted.tolist() == [True, True, True]


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1, 1.1])
def test_invalid_target_coverage_is_rejected(target):
    with pytest.raises(ValueError):
        select_validation_threshold([0.0, 1.0], [0.0], target)


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [-np.inf]])
def test_non_finite_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        select_validation_threshold(scores, [0.0], 0.5)
    with pytest.raises(ValueError):
        select_validation_threshold([0.0, 1.0], scores, 0.5)


def test_empty_or_misaligned_metric_inputs_are_rejected():
    with pytest.raises(ValueError):
        select_validation_threshold([], [0.0], 0.5)
    with pytest.raises(ValueError):
        validation_threshold_metrics([0.0, 1.0], [0.0], [0, 1], [0])
