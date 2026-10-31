"""Tests for safety constraint engine."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from research.src.verification.actions import (
    VerificationActionExecutor, EvidenceProvider, SimulatedHumanReviewer, AlternativeModelRunner
)
from research.src.verification.cost_model import BudgetManager, ActionCosts
from research.src.verification.safety_constraints import SafetyConstraintEngine, ActionFeasibility
from research.src.verification.actions import ActionType


def test_budget_manager():
    """Test budget management functionality."""
    budget = BudgetManager(total_budget=100.0)
    
    assert budget.get_remaining_budget() == 100.0
    assert budget.can_afford(50.0) is True
    assert budget.spend(30.0) is True
    assert budget.get_remaining_budget() == 70.0
    assert budget.can_afford(80.0) is False
    assert budget.spend(80.0) is False  # Cannot overspend


def test_accept_always_available():
    """Test that ACCEPT is always available."""
    executor = VerificationActionExecutor()
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    is_available, reason = engine.check_action_availability(ActionType.ACCEPT)
    assert is_available
    assert "ACCEPT is always available" in reason


def test_abstain_always_available_fallback():
    """Test that ABSTAIN is always available as fallback."""
    executor = VerificationActionExecutor()
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    is_available, reason = engine.check_action_availability(ActionType.ABSTAIN)
    assert is_available
    assert "ABSTAIN is always available as fallback" in reason
    
    # ABSTAIN should always be in feasible actions
    feasible = engine.get_feasible_actions(baseline_risk=1.0)
    assert ActionType.ABSTAIN in feasible


def test_request_additional_evidence_unavailable_without_provider():
    """Test that REQUEST_ADDITIONAL_EVIDENCE is unavailable without a provider."""
    executor = VerificationActionExecutor()  # No evidence provider
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    is_available, reason = engine.check_action_availability(ActionType.REQUEST_ADDITIONAL_EVIDENCE)
    assert not is_available
    assert "no evidence provider configured" in reason


def test_request_additional_evidence_available_with_provider():
    """Test that REQUEST_ADDITIONAL_EVIDENCE is available with a configured provider."""
    evidence_provider = EvidenceProvider(name="test", available=True, cost=2.0)
    executor = VerificationActionExecutor(evidence_provider=evidence_provider)
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    is_available, reason = engine.check_action_availability(ActionType.REQUEST_ADDITIONAL_EVIDENCE)
    assert is_available
    assert "Evidence provider is available" in reason


def test_human_review_unavailable_without_reviewer():
    """Test that HUMAN_REVIEW is unavailable without a configured reviewer."""
    executor = VerificationActionExecutor()  # No human reviewer
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    is_available, reason = engine.check_action_availability(ActionType.HUMAN_REVIEW)
    assert not is_available
    assert "no human reviewer configured" in reason


def test_human_review_available_with_reviewer():
    """Test that HUMAN_REVIEW is available with a configured reviewer."""
    reviewer = SimulatedHumanReviewer(accuracy=0.9, cost=5.0, available=True)
    executor = VerificationActionExecutor(human_reviewer=reviewer)
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    is_available, reason = engine.check_action_availability(ActionType.HUMAN_REVIEW)
    assert is_available
    assert "Human reviewer is available" in reason


def test_high_risk_safety_constraints():
    """Test that high-risk samples enforce safety constraints."""
    # Create engine with human reviewer available
    reviewer = SimulatedHumanReviewer(accuracy=0.9, cost=5.0, available=True)
    executor = VerificationActionExecutor(human_reviewer=reviewer)
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    # High risk sample (>= 8.0)
    HIGH_RISK = 9.0
    
    # ACCEPT should not be safe for high risk
    is_safe, reason = engine.check_safety_constraints(ActionType.ACCEPT, HIGH_RISK)
    assert not is_safe
    assert "High risk" in reason
    
    # Only HUMAN_REVIEW or ABSTAIN should be safe
    is_safe, _ = engine.check_safety_constraints(ActionType.HUMAN_REVIEW, HIGH_RISK)
    assert is_safe
    
    is_safe, _ = engine.check_safety_constraints(ActionType.ABSTAIN, HIGH_RISK)
    assert is_safe


def test_medium_high_risk_cannot_accept():
    """Test that medium-high risk samples cannot be directly accepted."""
    reviewer = SimulatedHumanReviewer(accuracy=0.9, cost=5.0, available=True)
    executor = VerificationActionExecutor(human_reviewer=reviewer)
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    # Medium-high risk (>=4.0 but <8.0)
    MEDIUM_RISK = 5.0
    
    # ACCEPT should not be safe
    is_safe, reason = engine.check_safety_constraints(ActionType.ACCEPT, MEDIUM_RISK)
    assert not is_safe
    assert "cannot be directly ACCEPTED" in reason


def test_budget_constraints_enforced():
    """Test that actions cannot be selected if they exceed budget."""
    reviewer = SimulatedHumanReviewer(accuracy=0.9, cost=5.0, available=True)
    executor = VerificationActionExecutor(human_reviewer=reviewer)
    # Small budget that can't afford human review
    budget_manager = BudgetManager(total_budget=3.0)  # Human review costs 5.0
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    # HUMAN_REVIEW should fail budget check
    is_budget_feasible, reason = engine.check_budget_feasibility(ActionType.HUMAN_REVIEW)
    assert not is_budget_feasible
    assert "Insufficient budget" in reason
    
    # ACCEPT should still be budget feasible
    is_budget_feasible, _ = engine.check_budget_feasibility(ActionType.ACCEPT)
    assert is_budget_feasible


def test_evaluate_all_actions_returns_feasibility():
    """Test that evaluate_all_actions returns feasibility for all actions."""
    executor = VerificationActionExecutor()
    budget_manager = BudgetManager(100.0)
    action_costs = ActionCosts()
    engine = SafetyConstraintEngine(executor, budget_manager, action_costs)
    
    feasibility = engine.evaluate_all_actions(baseline_risk=2.0)
    
    # Should have feasibility for all action types
    assert len(feasibility) == 5
    for action_type in ActionType:
        assert action_type in feasibility
        assert isinstance(feasibility[action_type], ActionFeasibility)