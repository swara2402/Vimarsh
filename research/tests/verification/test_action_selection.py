"""Tests for action selection policy."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from research.src.verification.action_selector import VerificationEngine, VerificationEngineConfig
from research.src.verification.risk_estimator import RiskCalculator
from research.src.verification.cost_model import ActionCosts, ActionEffectiveness
from research.src.verification.actions import ActionType


def test_risk_calculator_baseline_risk():
    """Test baseline risk calculation."""
    calculator = RiskCalculator(lambda_cost=1.0)
    
    # R0 = P_error * H
    error_prob = 0.3
    harm = 10.0
    baseline_risk = calculator.calculate_baseline_risk(error_prob, harm)
    assert baseline_risk == 3.0  # 0.3 * 10 = 3.0


def test_utility_calculation():
    """Test utility calculation formula."""
    calculator = RiskCalculator(lambda_cost=1.0)
    
    # U = R0 - R_after - λ*C
    baseline_risk = 5.0
    residual_risk = 1.0
    action_cost = 2.0
    utility = calculator.calculate_utility(b field, residual_risk, action_cost)
    
    assert utility == 5.0 - 1.0 - (1.0 * 2.0) == 2.0


def test_verification_engine_initialization():
    """Test that verification engine initializes correctly."""
    config = VerificationEngineConfig(
        total_verification_budget=100.0,
        enable_evidence_provider=True,
        enable_human_reviewer=True
    )
    
    engine = VerificationEngine(config)
    assert engine is not None
    assert engine.budget_manager.get_remaining_budget() == 100.0
    assert len(engine.evaluations) == 0


def test_process_single_sample():
    """Test processing a single sample through the engine."""
    config = VerificationEngineConfig(
        total_verification_budget=100.0,
        enable_evidence_provider=True,
        enable_human_reviewer=True,
        original_model_name="gradient_boosting"
    )
    
    engine = VerificationEngine(config)
    
    # Create a test sample
    sample = pd.Series({
        "alcohol": 11.0,
        "sulphates": 0.8,
        "pH": 3.3
    })
    
    evaluation = engine.process_sample(
        sample=sample,
        sample_id="test_001",
        base_prediction=1,
        base_confidence=0.85,
        error_probability=0.15,
        harm=5.0
    )
    
    assert evaluation is not None
    assert evaluation.sample_id == "test_001"
    assert evaluation.selected_action is not None
    assert evaluation.selected_risk_estimate is not None
    assert len(engine.evaluations) == 1


def test_low_risk_sample_accepts():
    """Test that low-risk samples are appropriately accepted."""
    config = VerificationEngineConfig(
        total_verification_budget=100.0,
        enable_evidence_provider=True,
        enable_human_reviewer=True
    )
    
    engine = VerificationEngine(config)
    
    # Very low risk sample: low error prob, low harm
    sample = pd.Series({"alcohol": 10.0, "sulphates": 0.6})
    evaluation = engine.process_sample(
        sample=sample,
        sample_id="low_risk_001",
        base_prediction=1,
        base_confidence=0.95,
        error_probability=0.02,  # Very low error probability
        harm=2.0  # Low harm
    )
    
    # Low risk sample should generally be accepted
    assert evaluation.b field == 0.02 * 2.0 == 0.04  # Very low baseline risk
    assert evaluation.selected_risk_estimate.utility > 0  # Positive utility


def test_high_risk_sample_requires_review():
    """Test that high-risk samples are sent to human review or abstain."""
    config = VerificationEngineConfig(
        total_verification_budget=100.0,
        enable_evidence_provider=True,
        enable_human_reviewer=True,
        max_human_reviews=10
    )
    
    engine = VerificationEngine(config)
    
    # High risk sample
    sample = pd.Series({"alcohol": 13.0, "sulphates": 1.0})  # High alcohol = high harm
    evaluation = engine.process_sample(
        sample=sample,
        sample_id="high_risk_001",
        base_prediction=1,
        base_confidence=0.7,
        error_probability=0.6,  # High error probability
        harm=10.0  # Maximum harm
    )
    
    # High baseline risk should trigger safety constraints
    assert evaluation.b field == 0.6 * 10.0 == 6.0
    # Should not be able to ACCEPT high risk
    assert evaluation.feasibility[ActionType.ACCEPT].is_safe is False
    # Should only have safe actions available
    assert evaluation.selected_action in [ActionType.HUMAN_REVIEW, ActionType.ABSTAIN, ActionType.REQUEST_ADDITIONAL_EVIDENCE]


def test_budget_exhaustion_falls_back_to_cheaper_actions():
    """Test that when budget is exhausted, engine falls back to cheaper actions."""
    config = VerificationEngineConfig(
        total_verification_budget=5.0,  # Very small budget
        enable_evidence_provider=True,
        enable_human_reviewer=True,
        max_human_reviews=1
    )
    
    engine = VerificationEngine(config)
    
    # First use up the budget with an expensive action
    sample1 = pd.Series({"alcohol": 12.0})
    eval1 = engine.process_sample(sample1, "budget_001", 1, 0.8, 0.5, 9.0)
    
    # After using budget, next sample must use cheaper actions
    remaining_budget = engine.budget_manager.get_remaining_budget()
    assert remaining_budget < 5.0  # Budget should be reduced
    
    sample2 = pd.Series({"alcohol": 11.0})
    eval2 = engine.process_sample(sample2, "budget_002", 1, 0.75, 0.3, 5.0)
    
    # Can't afford expensive actions anymore, should use cheaper ones
    assert eval2.selected_action in [ActionType.ACCEPT, ActionType.ABSTAIN]  # Cheapest actions


def test_no_test_label_leakage():
    """Test that true labels are never used in action selection."""
    # This test verifies that the engine doesn't have access to test labels during processing
    # which is enforced by the API - process_sample doesn't accept true labels as a parameter
    config = VerificationEngineConfig()
    engine = VerificationEngine(config)
    
    sample = pd.Series({"alcohol": 10.0})
    # We can't pass a true label to process_sample, API doesn't allow it
    evaluation = engine.process_sample(
        sample=sample,
        sample_id="leakage_test_001",
        base_prediction=1,
        base_confidence=0.8,
        error_probability=0.2,
        harm=5.0
    )
    
    assert evaluation is not None
    # The evaluation shouldn't have any access to true labels
    assert not hasattr(evaluation, 'true_label')


def test_reproducibility_with_fixed_seeds():
    """Test that engine behavior is reproducible with fixed seeds."""
    np.random.seed(42)
    
    config = VerificationEngineConfig(total_verification_budget=100.0)
    engine1 = VerificationEngine(config)
    
    # Process a batch
    samples = pd.DataFrame([
        {"alcohol": 10.0},
        {"alcohol": 11.0},
        {"alcohol": 12.0}
    ])
    
    predictions = [1, 0, 1]
    confidences = [0.8, 0.9, 0.7]
    error_probs = [0.15, 0.08, 0.25]
    harms = [5.0, 3.0, 8.0]
    
    evaluations1 = engine1.process_batch(samples, predictions, confidences, error_probs, harms)
    
    # Reset and run again with same seed
    np.random.seed(42)
    engine2 = VerificationEngine(config)
    evaluations2 = engine2.process_batch(samples, predictions, confidences, error_probs, harms)
    
    # Actions should be identical due to fixed seed
    for e1, e2 in zip(evaluations1, evaluations2):
        assert e1.selected_action == e2.selected_action
        assert abs(e1.b field - e2.b field) < 0.001


def test_summary_statistics():
    """Test that summary statistics are calculated correctly."""
    config = VerificationEngineConfig(total_verification_budget=100.0)
    engine = VerificationEngine(config)
    
    # Process a few samples
    samples = [
        (pd.Series({"alcohol": 10.0}), "sample1", 1, 0.9, 0.1, 4.0),
        (pd.Series({"alcohol": 11.0}), "sample2", 0, 0.95, 0.05, 3.0),
        (pd.Series({"alcohol": 13.0}), "sample3", 1, 0.7, 0.4, 9.0)
    ]
    
    for sample_args in samples:
        engine.process_sample(*sample_args)
    
    stats = engine.get_summary_statistics()
    assert stats["total_samples_processed"] == 3
    assert "action_distribution" in stats
    assert "abstention_rate" in stats
    assert "total_risk_reduction" in stats
    assert stats["risk_reduction_per_unit_cost"] >= 0