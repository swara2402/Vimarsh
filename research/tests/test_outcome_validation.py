"""
Milestone 3.2 outcome validation tests.
Tests core requirements: alternate model independence, no label leakage, abstention handling, etc.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.verification.outcome_validator import OutcomeValidator, SampleOutcome, ActionMetrics, PolicyComparison
from src.verification.actions import ActionType, VerificationActionExecutor, EvidenceProvider, AlternativeModelRunner, SimulatedHumanReviewer
from src.verification.action_evaluator import ActionEvaluation, RiskEstimate


def test_alternate_model_independence():
    """Test that alternate model is genuinely different from original model (Milestone 3.2 requirement)."""
    # Create mock action executor with alternate model runner
    mock_alt_runner = Mock()
    mock_alt_runner.get_available_alternatives.return_value = ["random_forest"]
    
    # Create mock models with measurable disagreement
    base_model = Mock()
    base_model.predict.return_value = np.array([0, 1, 0, 1, 0])
    alt_model1 = Mock()
    alt_model1.predict.return_value = np.array([0, 1, 1, 0, 0])  # 20% disagreement
    mock_alt_runner.available_models = {"random_forest": alt_model1}
    
    mock_executor = Mock()
    mock_executor.alternate_model_runner = mock_alt_runner
    
    validator = OutcomeValidator(mock_executor)
    
    # Create test data
    X_test = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })
    preprocessor = Mock()
    preprocessor.transform.return_value = X_test
    
    # Run validation
    results = validator.validate_alternate_model_independence(X_test, base_model, preprocessor)
    
    # Verify results
    assert "model_differences" in results
    assert len(results["model_differences"]) == 1
    model_diff = results["model_differences"][0]
    assert model_diff["prediction_disagreement_rate"] == 0.4  # 2/5 samples disagree (matches our mock data)
    assert model_diff["is_different_model"]  # >1% disagreement threshold - numpy boolean checks pass


def test_evidence_no_label_leakage():
    """Test that evidence provider code does not access test labels (Milestone 3.2 research integrity)."""
    # Create valid evidence provider that doesn't use labels
    valid_evidence = EvidenceProvider(name="test", available=True, cost=1.0)
    mock_executor = Mock()
    mock_executor.evidence_provider = valid_evidence
    
    validator = OutcomeValidator(mock_executor)
    results = validator.validate_evidence_no_label_leakage()
    
    # Should not detect any leakage
    assert results["potential_label_leakage_detected"] is False
    assert results["evidence_generations_is_valid"] is True


def test_abstention_not_counted_as_correct():
    """Test that abstentions are never counted as correct predictions (Milestone 3.2 requirement)."""
    mock_executor = Mock()
    validator = OutcomeValidator(mock_executor)
    
    # Add 10 samples: 8 abstain, 2 accept (1 correct, 1 incorrect)
    for i in range(10):
        if i < 8:
            # Abstained sample
            outcome = SampleOutcome(
                sample_id=f"test_{i}",
                true_label=1,
                base_prediction=1,
                base_correct=True,
                error_probability=0.5,
                harm=10.0,
                baseline_risk=5.0,
                selected_action=ActionType.ABSTAIN.value,
                final_prediction=None,
                final_correct=None,
                original_correct=True,
                is_covered=False
            )
        else:
            # Covered sample
            outcome = SampleOutcome(
                sample_id=f"test_{i}",
                true_label=1 if i == 8 else 0,
                base_prediction=1,
                base_correct=i == 8,
                error_probability=0.5,
                harm=10.0,
                baseline_risk=5.0,
                selected_action=ActionType.ACCEPT.value,
                final_prediction=1,
                final_correct=i == 8,
                original_correct=i == 8,
                is_covered=True
            )
        validator.sample_outcomes.append(outcome)
    
    # Calculate policy metrics
    policy = validator.calculate_policy_metrics("test_policy")
    
    # Verify abstention handling
    assert policy.abstention_rate == 0.8  # 8/10 abstained
    assert policy.coverage == 0.2  # 2/10 covered
    assert policy.observed_error_rate == 0.5  # 1 error out of 2 covered samples
    assert policy.total_observed_errors == 9  # 1 covered error + 8 abstentions


def test_corrected_prediction_detection():
    """Test that corrected predictions are properly detected (Milestone 3.2 requirement)."""
    mock_executor = Mock()
    validator = OutcomeValidator(mock_executor)
    
    # Add a sample that was corrected (incorrect -> correct)
    corrected_outcome = SampleOutcome(
        sample_id="corrected_test",
        true_label=1,
        base_prediction=0,  # Originally wrong
        base_correct=False,
        error_probability=0.9,
        harm=10.0,
        baseline_risk=9.0,
        selected_action=ActionType.HUMAN_REVIEW.value,
        final_prediction=1,  # After review, correct
        final_correct=True,
        corrected_prediction=True
    )
    validator.sample_outcomes.append(corrected_outcome)
    
    # Add a sample that had an error introduced (correct -> incorrect)
    error_outcome = SampleOutcome(
        sample_id="error_test",
        true_label=1,
        base_prediction=1,  # Originally correct
        base_correct=True,
        error_probability=0.1,
        harm=10.0,
        baseline_risk=1.0,
        selected_action=ActionType.ALTERNATE_MODEL.value,
        final_prediction=0,  # After alternate model, wrong
        final_correct=False,
        introduced_error=True
    )
    validator.sample_outcomes.append(error_outcome)
    
    # Calculate metrics
    action_metrics = validator.calculate_action_metrics()
    human_metrics = action_metrics[ActionType.HUMAN_REVIEW.value]
    alt_metrics = action_metrics[ActionType.ALTERNATE_MODEL.value]
    
    # Verify correction tracking
    assert human_metrics.corrected_predictions == 1
    assert human_metrics.correction_rate == 1.0  # 1/1 originally incorrect corrected
    assert alt_metrics.introduced_errors == 1
    assert alt_metrics.error_introduction_rate == 1.0  # 1/1 originally correct now incorrect


def test_simulated_human_review_labelling():
    """Test that human review is correctly labelled as simulated (Milestone 3.2 requirement)."""
    # Create properly configured simulated human reviewer
    human_reviewer = SimulatedHumanReviewer(accuracy=0.95, cost=3.0, available=True, max_reviews=50)
    mock_executor = Mock()
    mock_executor.human_reviewer = human_reviewer
    
    validator = OutcomeValidator(mock_executor)
    results = validator.validate_simulated_human_review()
    
    # Configured simulation probability is not an observed accuracy measurement.
    assert results["is_simulated_human_review"] is True
    assert results["configured_accuracy_probability"] == 0.95
    assert results["observed_review_accuracy"] is None
    assert results["observed_correct_reviews"] == 0
    assert results["observed_incorrect_reviews"] == 0
    assert results["completed_successful_reviews"] == 0
    assert "accuracy_interpretation" in results
    assert "human_review_labelled_correctly" not in results


def test_no_label_leakage_in_action_selection():
    """Test that test labels never influence action selection (Milestone 3.2 research integrity)."""
    # The core requirement is that in the outcome_validator, true_label is ONLY passed
    # to execute_human_review for simulation purposes and NEVER used during action selection
    # This test verifies our code implementation doesn't use labels in selection
    
    # Check that in outcome_validator.py's execute_and_evaluate_action, the true_label
    # is only used after action selection, to calculate final correctness, not to influence it
    import inspect
    from src.verification.outcome_validator import OutcomeValidator
    source = inspect.getsource(OutcomeValidator.execute_and_evaluate_action)
    
    # Verify true_label is NOT used in any action selection logic
    # It should only be used to calculate base_correct and final_correct, never to choose the action
    lines = source.split('\n')
    true_label_usages = [line.strip() for line in lines if 'true_label' in line]
    
    # True label should only be used for evaluation, not selection
    assert len(true_label_usages) >= 2  # Used to calculate base_correct and passed to human review
    # It should never be used in any condition that affects action execution
    assert not any('if true_label' in line for line in lines)
    assert not any('true_label in' in line and 'action' in line for line in lines)
    
    # Core integrity check passes: labels are only used for offline evaluation, not inferences

def test_human_review_capacity_exhaustion_is_not_successful():
    """A review beyond max_reviews must be reported as not executed."""
    reviewer = SimulatedHumanReviewer(
        accuracy=0.95, cost=3.0, available=True, max_reviews=1, reviews_completed=1
    )
    executor = VerificationActionExecutor(human_reviewer=reviewer)
    result = executor.execute_human_review(
        pd.Series({"feature": 1}), original_prediction=0,
        original_confidence=0.8, true_label=1
    )
    assert result.success is False
    assert result.cost_incurred == 0.0
    assert result.additional_data["reason"] == "Human review capacity exhausted"
    assert reviewer.reviews_completed == 1


def test_policy_cost_reconciles_to_sample_execution_costs(tmp_path, monkeypatch):
    """Policy total cost must equal the sum of recorded per-sample execution costs."""
    monkeypatch.chdir(tmp_path)
    validator = OutcomeValidator(Mock())
    validator.sample_outcomes.extend([
        SampleOutcome(
            sample_id="cost_1", true_label=1, base_prediction=1, base_correct=True,
            error_probability=0.1, harm=1.0, baseline_risk=0.1,
            selected_action=ActionType.ACCEPT.value, final_prediction=1,
            final_correct=True, original_correct=True, is_covered=True, cost_incurred=0.0
        ),
        SampleOutcome(
            sample_id="cost_2", true_label=0, base_prediction=1, base_correct=False,
            error_probability=0.2, harm=2.0, baseline_risk=0.4,
            selected_action=ActionType.ALTERNATE_MODEL.value, final_prediction=0,
            final_correct=True, original_correct=False, is_covered=True,
            corrected_prediction=True, action_result={"success": True, "cost_incurred": 4.0}, cost_incurred=4.0
        ),
    ])
    metrics = validator.calculate_policy_metrics("cost_reconciliation_test")
    assert metrics.total_verification_cost == pytest.approx(4.0)


def test_policy_metrics_reject_duplicate_sample_ids(tmp_path, monkeypatch):
    """Duplicate sample IDs must fail before an ambiguous evaluation table is exported."""
    monkeypatch.chdir(tmp_path)
    validator = OutcomeValidator(Mock())
    sample = SampleOutcome(
        sample_id="duplicate", true_label=1, base_prediction=1, base_correct=True,
        error_probability=0.1, harm=1.0, baseline_risk=0.1,
        selected_action=ActionType.ACCEPT.value, final_prediction=1, final_correct=True,
        original_correct=True, is_covered=True
    )
    validator.sample_outcomes.extend([sample, sample])
    with pytest.raises(AssertionError, match="duplicate sample IDs"):
        validator.calculate_policy_metrics("duplicate_id_test")

def test_policy_metrics_reject_action_cost_mismatch(tmp_path, monkeypatch):
    """Recorded sample cost must match the cost returned by its executed action."""
    monkeypatch.chdir(tmp_path)
    validator = OutcomeValidator(Mock())
    validator.sample_outcomes.append(SampleOutcome(
        sample_id="cost_mismatch", true_label=1, base_prediction=1, base_correct=True,
        error_probability=0.1, harm=1.0, baseline_risk=0.1,
        selected_action=ActionType.ACCEPT.value, final_prediction=1, final_correct=True,
        original_correct=True, is_covered=True,
        action_result={"success": True, "cost_incurred": 2.0},
        cost_incurred=1.0
    ))
    with pytest.raises(AssertionError, match="Sample execution cost mismatch"):
        validator.calculate_policy_metrics("cost_mismatch_test")

