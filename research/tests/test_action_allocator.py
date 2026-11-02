import pytest
from src.verification.action_allocator import allocate_verification_actions

def test_selects_maximum_benefit_within_shared_budget():
    candidates = [
        {"case_id": "a", "action": "alternate_model", "lower_bound_benefit": 10, "cost": 5, "available": True},
        {"case_id": "b", "action": "human_review", "lower_bound_benefit": 7, "cost": 3, "available": True},
        {"case_id": "c", "action": "additional_evidence", "lower_bound_benefit": 6, "cost": 2, "available": True},
    ]
    result = allocate_verification_actions(candidates, total_budget=5)
    assert result["status"] == "allocated"
    assert {x["case_id"] for x in result["selected"]} == {"b", "c"}
    assert result["total_cost"] == 5
    assert result["total_benefit"] == 13

def test_at_most_one_action_per_case():
    candidates = [
        {"case_id": "same", "action": "alternate_model", "lower_bound_benefit": 8, "cost": 2, "available": True},
        {"case_id": "same", "action": "human_review", "lower_bound_benefit": 9, "cost": 2, "available": True},
    ]
    result = allocate_verification_actions(candidates, total_budget=4)
    assert len(result["selected"]) == 1
    assert result["selected"][0]["action"] == "human_review"

def test_action_capacity_is_enforced():
    candidates = [
        {"case_id": "a", "action": "human_review", "lower_bound_benefit": 9, "cost": 1, "available": True},
        {"case_id": "b", "action": "human_review", "lower_bound_benefit": 8, "cost": 1, "available": True},
        {"case_id": "c", "action": "alternate_model", "lower_bound_benefit": 5, "cost": 1, "available": True},
    ]
    result = allocate_verification_actions(candidates, total_budget=3, action_capacities={"human_review": 1})
    assert sum(x["action"] == "human_review" for x in result["selected"]) == 1
    assert {x["case_id"] for x in result["selected"]} == {"a", "c"}

def test_unavailable_and_nonpositive_benefit_candidates_are_excluded():
    candidates = [
        {"case_id": "a", "action": "alternate_model", "lower_bound_benefit": 9, "cost": 1, "available": False},
        {"case_id": "b", "action": "human_review", "lower_bound_benefit": 0, "cost": 1, "available": True},
        {"case_id": "c", "action": "additional_evidence", "lower_bound_benefit": -2, "cost": 1, "available": True},
    ]
    result = allocate_verification_actions(candidates, total_budget=5)
    assert result["status"] == "abstain"
    assert result["selected"] == []

def test_zero_budget_abstains():
    result = allocate_verification_actions(
        [{"case_id": "a", "action": "alternate_model", "lower_bound_benefit": 3, "cost": 0, "available": True}],
        total_budget=0,
    )
    assert result["status"] == "abstain"

def test_duplicate_case_action_and_invalid_values_are_rejected():
    candidate = {"case_id": "a", "action": "alternate_model", "lower_bound_benefit": 2, "cost": 1, "available": True}
    with pytest.raises(ValueError, match="duplicate"):
        allocate_verification_actions([candidate, candidate], 2)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_verification_actions([], -1)
    with pytest.raises(ValueError, match="supported actions"):
        allocate_verification_actions([], 2, action_capacities={"unknown": 1})
