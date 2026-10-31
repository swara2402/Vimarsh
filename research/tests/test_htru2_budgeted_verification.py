"""Budget and leakage tests for the HTRU2 constrained verification experiment."""
import numpy as np
import pandas as pd
import pytest

from experiments.run_htru2_budgeted_verification import (
    class_harm_weights, select_under_budget, evaluate_selection,
)
from src.datasets.htru2 import parse_htru2_csv


def test_budgeted_selection_respects_hard_capacity_and_positive_benefit():
    scores = np.array([0.9, 0.8, 0.4, 0.01])
    harms = np.array([5.0, 1.0, 5.0, 1.0])
    chosen = select_under_budget(scores, harms, budget_count=2, seed=17)
    assert chosen.sum() <= 2
    assert chosen[0] and chosen[1]
    assert not chosen[3]


def test_selection_is_deterministic_and_does_not_accept_unprofitable_cases():
    scores = np.array([0.01, 0.02, 0.03])
    harms = np.ones(3)
    first = select_under_budget(scores, harms, budget_count=3, seed=42)
    second = select_under_budget(scores, harms, budget_count=3, seed=42)
    np.testing.assert_array_equal(first, second)
    assert not first.any()


def test_asymmetric_scenario_harm_depends_on_predicted_error_direction():
    np.testing.assert_array_equal(class_harm_weights(np.array([0, 1])), [5.0, 1.0])


def test_outcome_scoring_happens_after_selection_and_respects_budget():
    selected = np.array([True, False, False, True])
    predictions = np.array([0, 1, 0, 1])
    labels = np.array([1, 1, 0, 0])
    harms = np.array([5.0, 1.0, 5.0, 1.0])
    result = evaluate_selection("test", selected, predictions, labels, harms)
    assert result["reviewed_count"] == 2
    assert result["budget_respected"]
    assert result["weighted_error_loss_before_review"] == 6.0
    assert result["simulated_gross_loss_avoided"] == pytest.approx(4.8)
    assert result["review_cost_units"] == pytest.approx(0.5)
    assert result["simulated_net_benefit"] == pytest.approx(4.3)


def test_htru2_parser_validates_schema_and_target(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame([[1,2,3,4,5,6,7,8,0], [2,3,4,5,6,7,8,9,1]]).to_csv(path, header=False, index=False)
    X, y = parse_htru2_csv(path)
    assert X.shape == (2, 8)
    assert y.tolist() == [0, 1]
    bad = tmp_path / "bad.csv"
    pd.DataFrame([[1,2,3,4,5,6,7,8,2]]).to_csv(bad, header=False, index=False)
    with pytest.raises(ValueError):
        parse_htru2_csv(bad)
