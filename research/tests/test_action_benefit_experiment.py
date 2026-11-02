import numpy as np
import pytest
from experiments.run_action_benefit_experiment import build_policy_context, summarize_selection, top_budget_mask

def test_top_budget_mask_selects_exact_count_and_is_stable():
    mask = top_budget_mask([0.1, 0.9, 0.9, 0.2], 2)
    assert mask.tolist() == [False, True, True, False]

def test_top_budget_mask_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        top_budget_mask([0.1, float("nan")], 1)
    with pytest.raises(ValueError):
        top_budget_mask([0.1], 2)

def test_paired_summary_counts_corrections_and_regressions():
    result = summarize_selection(
        [True, True, False, True],
        baseline=[0, 1, 1, 0],
        alternate=[1, 0, 1, 0],
        truth=[1, 1, 0, 1],
    )
    assert result["selected_count"] == 3
    assert result["corrections"] == 1
    assert result["introduced_regressions"] == 1
    assert result["total_realized_benefit"] == 0.0

def test_empty_selection_is_well_defined():
    result = summarize_selection([False, False], [0, 1], [1, 1], [1, 1])
    assert result["selected_count"] == 0
    assert result["mean_realized_benefit_per_selected_case"] is None

def test_paired_summary_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="equal lengths"):
        summarize_selection([True], [0, 1], [1, 1], [1, 1])

def test_policy_context_does_not_depend_on_alternate_model_outputs():
    context = build_policy_context([[0.9, 0.1], [0.55, 0.45]], harm=2.0)
    assert context == [
        {"uncertainty": pytest.approx(0.1), "harm": 2.0},
        {"uncertainty": pytest.approx(0.45), "harm": 2.0},
    ]
    assert all("disagreement" not in row for row in context)

def test_policy_context_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        build_policy_context([[0.9, float("nan")]])
