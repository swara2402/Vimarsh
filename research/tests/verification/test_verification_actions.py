"""Tests for verification actions implementation."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from research.src.verification.actions import (
    ActionType, VerificationActionExecutor, EvidenceProvider, 
    AlternativeModelRunner, SimulatedHumanReviewer, ActionResult
)


def test_action_type_enum():
    """Test that all action types are properly defined."""
    actions = list(ActionType)
    assert len(actions) == 5
    assert ActionType.ACCEPT in actions
    assert ActionType.REQUEST_ADDITIONAL_EVIDENCE in actions
    assert ActionType.ALTERNATE_MODEL in actions
    assert ActionType.HUMAN_REVIEW in actions
    assert ActionType.ABSTAIN in actions


def test_evidence_provider_availability():
    """Test evidence provider availability checks."""
    # Available provider
    available_provider = EvidenceProvider(name="test", available=True, cost=2.0)
    assert available_provider.available
    assert available_provider.can_provide
    
    # Unavailable provider
    unavailable_provider = EvidenceProvider(name="test", available=False, cost=2.0)
    assert not unavailable_provider.available
    
    # Test requesting evidence from unavailable provider
    sample = pd.Series({"alcohol": 10.0, "pH": 3.3})
    enriched, cost = unavailable_provider.request_evidence(sample)
    assert enriched is None
    assert cost == 2.0


def test_human_reviewer_configuration():
    """Test simulated human reviewer configuration."""
    reviewer = SimulatedHumanReviewer(
        accuracy=0.9,
        cost=5.0,
        available=True,
        max_reviews=10
    )
    
    assert reviewer.accuracy == 0.9
    assert reviewer.cost == 5.0
    assert reviewer.available
    assert reviewer.max_reviews == 10
    assert reviewer.reviews_completed == 0


def test_action_executor_accept():
    """Test ACCEPT action execution."""
    executor = VerificationActionExecutor()
    result = executor.execute_accept()
    
    assert isinstance(result, ActionResult)
    assert result.action_type == ActionType.ACCEPT
    assert result.success
    assert result.cost_incurred == 0.0


def test_action_executor_abstain():
    """Test ABSTAIN action execution."""
    executor = VerificationActionExecutor()
    result = executor.execute_abstain()
    
    assert result.action_type == ActionType.ABSTAIN
    assert result.success
    assert result.cost_incurred == 0.5  # Operational cost
    assert result.additional_data["abstained"]


def test_unavailable_request_additional_evidence():
    """Test REQUEST_ADDITIONAL_EVIDENCE when provider is unavailable."""
    executor = VerificationActionExecutor()  # No evidence provider configured
    sample = pd.Series({"alcohol": 10.0})
    result = executor.execute_request_additional_evidence(sample)
    
    assert result.action_type == ActionType.REQUEST_ADDITIONAL_EVIDENCE
    assert not result.success
    assert "Evidence provider not available" in result.additional_data["reason"]


def test_available_evidence_provider():
    """Test REQUEST_ADDITIONAL_EVIDENCE with available provider."""
    evidence_provider = EvidenceProvider(name="test", available=True, cost=2.0)
    executor = VerificationActionExecutor(evidence_provider=evidence_provider)
    sample = pd.Series({"alcohol": 10.0})
    result = executor.execute_request_additional_evidence(sample)
    
    assert result.success
    assert result.cost_incurred == 2.0
    assert result.additional_data["enriched_sample"] is not None


def test_unavailable_human_review():
    """Test HUMAN_REVIEW when reviewer is unavailable."""
    executor = VerificationActionExecutor()  # No human reviewer configured
    sample = pd.Series({"alcohol": 10.0})
    result = executor.execute_human_review(sample, 1, 0.8)
    
    assert result.action_type == ActionType.HUMAN_REVIEW
    assert not result.success
    assert "Human reviewer not available" in result.additional_data["reason"]


def test_alternate_model_runner_loads_models():
    """Test that alternative model runner properly loads models."""
    # Create temporary model paths for testing
    base_path = Path(__file__).parent.parent.parent / "models"
    model_paths = {
        "logistic_regression": base_path / "logistic_regression.joblib",
        "random_forest": base_path / "random_forest.joblib"
    }
    preprocessor_path = base_path / "preprocessor.joblib"
    
    # Only test if files exist (they should in the real repo)
    if all(p.exists() for p in list(model_paths.values()) + [preprocessor_path]):
        runner = AlternativeModelRunner(model_paths, preprocessor_path)
        alternatives = runner.get_available_alternatives()
        assert len(alternatives) >= 1
        assert "logistic_regression" in alternatives or "random_forest" in alternatives