import pytest

from src.verification.outcome_audit import audit_prediction_changes
from experiments.run_measured_verification_outcomes import top_budget_mask


def test_audit_counts_corrections_regressions_and_cost():
    result = audit_prediction_changes(
        y_true=[1, 0, 1, 0],
        baseline_predictions=[0, 0, 1, 1],
        candidate_predictions=[1, 1, 1, 0],
        action_mask=[True, True, True, False],
        action_costs=[1.5, 1.5, 1.5, 1.5],
    )
    assert result["corrections"] == 1
    assert result["regressions"] == 1
    assert result["net_corrections"] == 0
    assert result["baseline_errors_in_actioned_cases"] == 1
    assert result["realized_action_cost"] == pytest.approx(4.5)
    assert result["cost_per_net_correction"] is None


def test_abstention_reduces_coverage_and_excludes_unissued_errors():
    result = audit_prediction_changes(
        y_true=[0, 1, 0],
        baseline_predictions=[1, 1, 0],
        candidate_predictions=[1, 1, 0],
        action_mask=[False, False, True],
        action_costs=[0.0, 0.0, 0.75],
        prediction_issued=[True, False, True],
    )
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["errors_among_issued_predictions"] == 1
    assert result["error_rate_among_issued_predictions"] == pytest.approx(0.5)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        audit_prediction_changes([0, 1], [0], [0, 1], [True, False], [1, 1])
    with pytest.raises(ValueError):
        audit_prediction_changes([0], [0], [0], [True], [-1])


def test_top_budget_mask_has_exact_action_count_without_labels():
    mask = top_budget_mask([0.1, 0.9, 0.4, 0.8], 2)
    assert mask.tolist() == [False, True, False, True]
