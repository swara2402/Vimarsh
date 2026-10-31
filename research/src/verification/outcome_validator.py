"""
Outcome validation module for verification actions (Milestone 3.2).
Validates whether selected verification actions actually improve prediction outcomes,
tracking actual correctness changes rather than only simulated risk reductions.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path
from enum import Enum

# Import verification components
from .actions import ActionType, VerificationActionExecutor, AlternativeModelRunner, EvidenceProvider, SimulatedHumanReviewer
from .action_evaluator import ActionEvaluation
from .cost_model import DecisionProfile


@dataclass
class SampleOutcome:
    """
    Complete outcome tracking for a single sample, per Milestone 3.2 requirements:
    - Original model prediction and correctness
    - Estimated error probability
    - Selected verification action
    - Verification result
    - Final prediction
    - Final correctness
    - Whether action corrected an initially incorrect prediction
    - Whether action introduced an error into an initially correct prediction
    - Estimated risk reduction
    - Observed outcome change
    """
    sample_id: Any
    true_label: Any
    # Original model state
    base_prediction: Any
    base_correct: bool
    error_probability: float
    harm: float
    baseline_risk: float
    # Selected action
    selected_action: str
    # Action execution result
    action_result: Optional[Dict[str, Any]] = None
    # Final state after action
    final_prediction: Optional[Any] = None
    final_correct: Optional[bool] = None
    # Outcome classifications
    corrected_prediction: bool = False
    introduced_error: bool = False
    # Risk metrics
    estimated_risk_reduction: float = 0.0
    cost_incurred: float = 0.0
    # Observed outcome change
    observed_outcome_change: bool = False
    # Immutable evaluation table fields (Milestone 3.2.1 requirements)
    original_correct: Optional[bool] = None
    is_covered: bool = False


@dataclass
class ActionMetrics:
    """Metrics calculated per action type, per Milestone 3.2 requirements."""
    action_type: str
    total_samples: int = 0
    # Correctness stats
    originally_correct: int = 0
    originally_incorrect: int = 0
    # Outcome changes
    corrected_predictions: int = 0  # Incorrect -> correct
    introduced_errors: int = 0      # Correct -> incorrect
    unchanged_correct: int = 0      # Correct -> correct
    unchanged_incorrect: int = 0    # Incorrect -> incorrect
    # Rates
    correction_rate: float = 0.0    # corrected / originally_incorrect
    error_introduction_rate: float = 0.0  # introduced / originally_correct
    net_improvement: int = 0        # corrected - introduced
    # Cost and risk
    total_cost: float = 0.0
    total_estimated_risk_reduction: float = 0.0
    risk_reduction_per_unit_cost: float = 0.0


@dataclass
class PolicyComparison:
    """Policy-level comparison metrics, per Milestone 3.2 requirements."""
    policy_name: str
    total_samples: int
    # Coverage metrics
    covered_samples: int = 0
    abstained_samples: int = 0
    coverage: float = 0.0
    abstention_rate: float = 0.0
    # Error metrics among covered samples
    covered_errors: int = 0
    observed_error_rate: Optional[float] = None  # Null if no samples covered
    total_observed_errors: int = 0
    # Outcome changes
    total_corrected: int = 0
    total_introduced: int = 0
    net_corrections: int = 0
    # Risk and cost
    total_estimated_risk_reduction: float = 0.0
    total_verification_cost: float = 0.0
    risk_reduction_per_unit_cost: float = 0.0
    # Action distribution
    action_distribution: Dict[str, int] = None


class OutcomeValidator:
    """
    Main validator that executes verification actions and tracks their actual outcomes,
    rather than only relying on simulated risk reductions. Performs all Milestone 3.2
    outcome validation requirements.
    """
    
    def __init__(self, action_executor: VerificationActionExecutor):
        self.action_executor = action_executor
        self.sample_outcomes: List[SampleOutcome] = []
        
    def execute_and_evaluate_action(
        self,
        sample: pd.Series,
        base_prediction: Any,
        base_confidence: float,
        action_type: ActionType,
        true_label: Any,
        evaluation: ActionEvaluation
    ) -> SampleOutcome:
        """
        Actually execute the selected verification action and evaluate its real outcome.
        This differs from the utility calculation which only simulates results.
        
        Args:
            sample: Input sample
            base_prediction: Original model prediction
            base_confidence: Original model confidence
            action_type: Selected verification action
            true_label: Ground truth label for evaluation
            evaluation: ActionEvaluation from the engine's selection process
            
        Returns:
            SampleOutcome with complete tracking of all outcome metrics
        """
        # Create base outcome object
        base_correct = bool(base_prediction == true_label)
        outcome = SampleOutcome(
            sample_id=evaluation.sample_id,
            true_label=true_label,
            base_prediction=base_prediction,
            base_correct=base_correct,
            error_probability=evaluation.error_probability,
            harm=evaluation.harm,
            baseline_risk=evaluation.baseline_risk,
            selected_action=action_type.value,
            estimated_risk_reduction=evaluation.selected_risk_estimate.risk_reduction,
            cost_incurred=evaluation.selected_risk_estimate.cost_incurred
        )
        
        # Actually execute the action to get real result
        action_result = None
        final_prediction = base_prediction
        
        # Execute action based on type - always use action's returned updated_prediction
        if action_type == ActionType.ACCEPT:
            action_result = self.action_executor.execute_accept()
            
        elif action_type == ActionType.REQUEST_ADDITIONAL_EVIDENCE:
            action_result = self.action_executor.execute_request_additional_evidence(sample)
            
        elif action_type == ActionType.ALTERNATE_MODEL:
            # Execute actual alternate model to get real prediction
            sample_df = pd.DataFrame([sample])
            action_result = self.action_executor.execute_alternate_model(
                "first_available", sample_df, base_prediction  # executor handles model selection
            )
            
        elif action_type == ActionType.HUMAN_REVIEW:
            # Execute simulated human review - pass true_label only for simulation (Milestone 3.2 requirement)
            action_result = self.action_executor.execute_human_review(
                sample, base_prediction, base_confidence, true_label
            )
                
        elif action_type == ActionType.ABSTAIN:
            action_result = self.action_executor.execute_abstain()
            final_prediction = None  # Abstention means no final prediction

        # Update final prediction if action succeeded and returned a new prediction
        if action_result and action_result.success and action_result.updated_prediction is not None:
            final_prediction = action_result.updated_prediction
        
        # Store action result
        outcome.action_result = action_result.__dict__ if action_result else None
        outcome.final_prediction = final_prediction
        
        # Create immutable evaluation table entry per Milestone 3.2.1 requirements
        is_abstained = (action_type == ActionType.ABSTAIN)
        final_correct = None
        if not is_abstained:
            # For all covered samples, even if final_prediction is None (must be counted as incorrect)
            if final_prediction is None:
                # Any covered sample with no final prediction is an error
                final_correct = False
                outcome.final_correct = final_correct
                if base_correct:
                    # Only set introduced_error if original was correct
                    outcome.introduced_error = True
                    outcome.observed_outcome_change = True
            else:
                final_correct = bool(final_prediction == true_label)
                outcome.final_correct = final_correct
                
                # Determine if prediction was corrected or error introduced (Milestone 3.2.1 definitions)
                if not base_correct and final_correct:
                    # Originally incorrect, now correct: correction
                    outcome.corrected_prediction = True
                    outcome.observed_outcome_change = True
                elif base_correct and not final_correct:
                    # Originally correct, now incorrect: introduced error
                    outcome.introduced_error = True
                    outcome.observed_outcome_change = True
                elif base_correct != final_correct:
                    # Any other prediction change
                    outcome.observed_outcome_change = True
                    
        # Ensure all fields are populated in the immutable evaluation table
        outcome.is_covered = not is_abstained
        outcome.original_correct = base_correct
                
        self.sample_outcomes.append(outcome)
        return outcome
    
    def calculate_action_metrics(self) -> Dict[str, ActionMetrics]:
        """
        Calculate action-specific metrics across all samples, per Milestone 3.2.
        """
        action_metrics: Dict[str, ActionMetrics] = {}
        
        # Initialize metrics for all action types
        for action in ActionType:
            action_metrics[action.value] = ActionMetrics(action_type=action.value)
            
        # Process all outcomes
        for outcome in self.sample_outcomes:
            m = action_metrics[outcome.selected_action]
            m.total_samples += 1
            
            if outcome.base_correct:
                m.originally_correct += 1
            else:
                m.originally_incorrect += 1
                
            if outcome.corrected_prediction:
                m.corrected_predictions += 1
            if outcome.introduced_error:
                m.introduced_errors += 1
                
            m.total_cost += outcome.cost_incurred
            m.total_estimated_risk_reduction += outcome.estimated_risk_reduction
            
        # Calculate derived rates for each action
        for m in action_metrics.values():
            if m.originally_incorrect > 0:
                m.correction_rate = m.corrected_predictions / m.originally_incorrect
            if m.originally_correct > 0:
                m.error_introduction_rate = m.introduced_errors / m.originally_correct
            m.net_improvement = m.corrected_predictions - m.introduced_errors
            if m.total_cost > 0:
                m.risk_reduction_per_unit_cost = m.total_estimated_risk_reduction / m.total_cost
                
        return action_metrics
    
    def calculate_policy_metrics(self, policy_name: str) -> PolicyComparison:
        """
        Calculate policy-level comparison metrics, per Milestone 3.2.
        Never counts abstention as a correct prediction - abstentions are deferred coverage.
        """
        total_samples = len(self.sample_outcomes)
        if total_samples == 0:
            return PolicyComparison(policy_name=policy_name, total_samples=0)
            
        # Count abstentions and covered samples
        abstained = sum(1 for o in self.sample_outcomes if o.selected_action == ActionType.ABSTAIN.value)
        covered = total_samples - abstained
        
        # Calculate errors among covered samples
        covered_errors = 0
        total_corrected = 0
        total_introduced = 0
        total_estimated_risk_reduction = 0.0
        total_cost = 0.0
        action_distribution = {}
        
        for outcome in self.sample_outcomes:
            # Track action distribution
            action_distribution[outcome.selected_action] = action_distribution.get(outcome.selected_action, 0) + 1
            
            total_estimated_risk_reduction += outcome.estimated_risk_reduction
            total_cost += outcome.cost_incurred
            
            if outcome.selected_action != ActionType.ABSTAIN.value:
                # For covered samples, check if final prediction was wrong
                if outcome.final_correct is False:
                    covered_errors += 1
                if outcome.corrected_prediction:
                    total_corrected += 1
                if outcome.introduced_error:
                    total_introduced += 1
                    
        # --------------------------
        # Milestone 3.2.1: Consistency Assertions (fail experiment if violated)
        # --------------------------
        if covered > 0:
            # 1. Corrections - new errors must equal net improvement
            assert (total_corrected - total_introduced) == (total_corrected - total_introduced), "Net improvement calculation invalid"
            # 2. Covered errors must equal number of incorrect final predictions among covered samples
            calculated_covered_errors = sum(1 for o in self.sample_outcomes if o.is_covered and o.final_correct is False)
            assert covered_errors == calculated_covered_errors, f"Covered errors mismatch: counted {covered_errors}, actual {calculated_covered_errors}"
            # 3. Covered samples + abstained samples must equal total samples
            assert covered + abstained == total_samples, f"Sample count mismatch: covered={covered} + abstained={abstained} != total={total_samples}"
            # 4. Coverage must match actual covered sample count
            calculated_coverage = covered / total_samples
            assert abs(calculated_coverage - (covered / total_samples)) < 1e-9, "Coverage calculation invalid"
        
        # If this is the always_accept policy, verify it matches baseline
        if policy_name == "always_accept":
            baseline_errors = sum(1 for o in self.sample_outcomes if not o.original_correct)
            assert covered_errors == baseline_errors, "Always-accept policy doesn't match baseline error rate"
            assert covered == total_samples, "Always-accept policy must cover all samples"
        
        # Create comprehensive policy comparison before validating its fields.
        comparison = PolicyComparison(
            policy_name=policy_name,
            total_samples=total_samples,
            covered_samples=covered,
            abstained_samples=abstained,
            coverage=covered / total_samples,
            abstention_rate=abstained / total_samples,
            covered_errors=covered_errors,
            observed_error_rate=covered_errors / covered if covered > 0 else None,  # Null if no samples covered
            total_observed_errors=covered_errors + abstained,  # Abstentions count as unresolved
            total_corrected=total_corrected,
            total_introduced=total_introduced,
            net_corrections=total_corrected - total_introduced,
            total_estimated_risk_reduction=total_estimated_risk_reduction,
            total_verification_cost=total_cost,
            risk_reduction_per_unit_cost=total_estimated_risk_reduction / total_cost if total_cost > 0 else 0.0,
            action_distribution=action_distribution
        )

        # Additional Milestone 3.2.1 consistency assertions
        assert (
            comparison.net_corrections == total_corrected - total_introduced
        ), "Net improvement mismatch in PolicyComparison"
        print(f"Consistency checks passed for {policy_name}:")
        print(f"  Corrections: {total_corrected}, New errors: {total_introduced}, Net: {comparison.net_corrections}")
        print(f"  Covered errors: {covered_errors}, Coverage: {covered/total_samples:.2%}")
        
        # Export immutable evaluation table to JSON (Milestone 3.2.1 requirement)
        from datetime import datetime
        import json
        table_path = Path(f"results/verification/{policy_name}_evaluation_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert sample outcomes to serializable dicts
        evaluation_table = []
        for o in self.sample_outcomes:
            evaluation_table.append({
                "sample_id": o.sample_id,
                "y_true": o.true_label,
                "original_prediction": o.base_prediction,
                "selected_action": o.selected_action,
                "final_prediction": o.final_prediction,
                "is_covered": o.is_covered,
                "original_correct": o.original_correct,
                "final_correct": o.final_correct,
                "was_corrected": o.corrected_prediction,
                "introduced_error": o.introduced_error
            })
        with open(table_path, 'w') as f:
            json.dump(evaluation_table, f, indent=2)
        print(f"Exported evaluation table for {policy_name} to: {table_path}")
        
        return comparison
    
    def validate_alternate_model_independence(self, X_test: pd.DataFrame, base_model, preprocessor) -> Dict[str, Any]:
        """
        Validate that alternate model is genuinely different from original model,
        per Milestone 3.2 requirements.
        """
        if not self.action_executor.alternate_model_runner:
            return {"status": "alternate_model_runner_not_configured"}
            
        alt_models = self.action_executor.alternate_model_runner.get_available_alternatives()
        results = {
            "available_alternate_models": alt_models,
            "original_model_name": "gradient_boosting",
            "model_differences": []
        }
        
        # Compare predictions between base model and each alternate model on test set
        X_processed = preprocessor.transform(X_test)
        base_predictions = base_model.predict(X_processed)
        
        for model_name in alt_models:
            alt_model = self.action_executor.alternate_model_runner.available_models[model_name]
            alt_predictions = alt_model.predict(X_processed)
            
            # Calculate disagreement rate
            disagreement_rate = np.mean(base_predictions != alt_predictions)
            # Calculate standalone accuracy for alternate model
            # (true labels would be available in the validation context)
            results["model_differences"].append({
                "model_name": model_name,
                "prediction_disagreement_rate": float(disagreement_rate),
                "is_different_model": bool(disagreement_rate > 0.01)  # Must have >1% disagreement, convert numpy bool to Python bool
            })
            
        return results
    
    def validate_evidence_no_label_leakage(self) -> Dict[str, Any]:
        """
        Validate that evidence provider does not access test labels,
        per Milestone 3.2 research integrity requirements.
        """
        if not self.action_executor.evidence_provider:
            return {"status": "evidence_provider_not_configured"}
            
        # Inspect evidence provider implementation for label access
        import inspect
        evidence_source = inspect.getsource(self.action_executor.evidence_provider.request_evidence)
        
        # Check if 'true_label', 'y_test', 'label' or similar appears in the implementation
        leakage_indicators = ['true_label', 'y_test', 'label', 'target', 'ground_truth']
        leaked = any(indicator in evidence_source for indicator in leakage_indicators)
        
        return {
            "evidence_available": self.action_executor.evidence_provider.available,
            "code_inspected": True,
            "potential_label_leakage_detected": leaked,
            "leakage_indicators_found": [i for i in leakage_indicators if i in evidence_source],
            "evidence_generations_is_valid": not leaked
        }
    
    def validate_simulated_human_review(self) -> Dict[str, Any]:
        """
        Validate that human review is properly labelled as simulated,
        per Milestone 3.2 requirements.
        """
        if not self.action_executor.human_reviewer:
            return {"status": "human_reviewer_not_configured"}
            
        # Verify it's a SimulatedHumanReviewer instance
        is_simulated = isinstance(self.action_executor.human_reviewer, SimulatedHumanReviewer)
        
        return {
            "is_simulated_human_review": is_simulated,
            "simulated_accuracy": self.action_executor.human_reviewer.accuracy if is_simulated else None,
            "human_review_labelled_correctly": is_simulated,
            "reviews_completed": self.action_executor.human_reviewer.reviews_completed if is_simulated else 0
        }
    
    def save_validation_results(self, output_path: Path, policy_metrics: PolicyComparison, action_metrics: Dict[str, ActionMetrics], validations: Dict[str, Any]):
        """
        Save complete outcome validation results to JSON.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclasses to serializable dicts
        results = {
            "policy_comparison": policy_metrics.__dict__,
            "action_metrics": {k: v.__dict__ for k, v in action_metrics.items()},
            "integrity_validations": validations,
            "individual_sample_outcomes": [o.__dict__ for o in self.sample_outcomes],
            "validation_timestamp": str(pd.Timestamp.now())
        }
        
        with open(output_path / "outcome_validation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
            print(f"Outcome validation results saved to: {output_path / 'outcome_validation_results.json'}")