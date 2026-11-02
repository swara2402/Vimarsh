import pytest
from src.verification.action_benefit_estimator import estimate_action_benefit

def rows(corrected=35, regressed=5, propensity=0.5):
    result = []
    for i in range(corrected + regressed):
        is_corrected = i < corrected
        result.append({
            "action": "alternate_model", "status": "completed",
            "propensity": propensity, "baseline_prediction": 0 if is_corrected else 1,
            "final_prediction": 1 if is_corrected else 0, "ground_truth": 1,
            "context": {"uncertainty": 0.4 + (i % 3) * 0.01, "harm": 1.0},
        })
    return result

def test_positive_paired_benefit_yields_candidate_recommendation():
    result = estimate_action_benefit(rows(), "alternate_model", {"uncertainty": 0.41, "harm": 1.0})
    assert result["status"] == "recommend_for_further_evaluation"
    assert result["propensity_adjusted"] is True
    assert result["context_conditioned"] is True
    assert result["lower_bound_diagnostic"] > 0

def test_regressions_or_uncertain_benefit_abstain():
    result = estimate_action_benefit(rows(corrected=5, regressed=35), "alternate_model",
                                     {"uncertainty": 0.41, "harm": 1.0})
    assert result["status"] == "abstain"

def test_missing_propensity_records_are_not_used():
    data = rows()
    for row in data:
        row["propensity"] = None
    result = estimate_action_benefit(data, "alternate_model", {"uncertainty": 0.41, "harm": 1.0})
    assert result["sample_count"] == 0
    assert result["status"] == "abstain"

def test_only_matching_action_and_completed_records_are_used():
    data = rows()
    data += [{"action": "human_review", "status": "completed", "propensity": 1.0}]
    data += [{"action": "alternate_model", "status": "pending", "propensity": 1.0}]
    result = estimate_action_benefit(data, "alternate_model", {"uncertainty": 0.41, "harm": 1.0})
    assert result["sample_count"] == 40

def test_insufficient_effective_sample_size_abstains():
    result = estimate_action_benefit(rows(corrected=4, regressed=1), "alternate_model",
                                     {"uncertainty": 0.41, "harm": 1.0}, min_effective_samples=10)
    assert result["reason"] == "insufficient_effective_sample_size"

def test_context_schema_mismatch_is_excluded():
    data = rows()
    data[0]["context"] = {"uncertainty": 0.4, "harm": 1.0, "unexpected": 3.0}
    result = estimate_action_benefit(data, "alternate_model", {"uncertainty": 0.41, "harm": 1.0})
    assert result["sample_count"] == 39

@pytest.mark.parametrize("context", [{"harm": 1.0, "true_label": 1}, {"uncertainty": 0.3}])
def test_label_leakage_and_missing_harm_are_rejected(context):
    with pytest.raises(ValueError):
        estimate_action_benefit(rows(), "alternate_model", context)

def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        estimate_action_benefit(rows(), "alternate_model", {"uncertainty": 0.4, "harm": 1.0}, confidence=1.0)
