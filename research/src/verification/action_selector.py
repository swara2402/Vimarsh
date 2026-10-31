"""Main action selector that orchestrates the entire verification process."""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Import all verification components
from .actions import (
    ActionType, VerificationActionExecutor, EvidenceProvider, 
    AlternativeModelRunner, SimulatedHumanReviewer
)
from .risk_estimator import RiskCalculator
from .cost_model import ActionCosts, ActionEffectiveness, BudgetManager, DecisionProfile
from .safety_constraints import SafetyConstraintEngine
from .action_evaluator import ActionEvaluator, ActionEvaluation


@dataclass
class VerificationEngineConfig:
    """Configuration for the complete verification engine."""
    # Budget configuration
    total_verification_budget: float = 1000.0
    lambda_cost_weight: float = 1.0
    
    # Decision profile (trades off risk, cost, and coverage)
    decision_profile: DecisionProfile = DecisionProfile.BALANCED
    
    # Action costs (default values can be overridden - if None, created from profile)
    action_costs: Optional[ActionCosts] = None
    
    # Action effectiveness factors (default values can be overridden - if None, created from profile)
    action_effectiveness: Optional[ActionEffectiveness] = None
    
    # Evidence provider configuration
    enable_evidence_provider: bool = True  # Keep additional evidence unavailable unless explicitly configured
    evidence_cost: float = 2.0
    
    # Human reviewer configuration (explicitly simulated)
    enable_human_reviewer: bool = True
    human_reviewer_accuracy: float = 0.9  # Simulated accuracy - not real-world measured
    human_review_cost: float = 5.0
    max_human_reviews: Optional[int] = 100
    
    # Safety thresholds
    high_risk_threshold: float = 8.0
    medium_risk_threshold: float = 4.0
    
    # Path to alternative models
    base_model_path: Path = Path(__file__).parent.parent.parent / "models"
    original_model_name: str = "gradient_boosting"  # Exclude this from alternatives


class VerificationEngine:
    """Main verification action engine that processes predictions and selects actions."""
    
    def __init__(self, config: VerificationEngineConfig):
        """
        Initialize the complete verification engine.
        
        Args:
            config: Configuration object for the engine
        """
        self.config = config
        
        # Initialize action costs and effectiveness from decision profile if not provided
        if not self.config.action_costs:
            self.config.action_costs = ActionCosts.from_profile(self.config.decision_profile)
        if not self.config.action_effectiveness:
            self.config.action_effectiveness = ActionEffectiveness.from_profile(self.config.decision_profile)
            
        # Set lambda cost weight based on decision profile if not explicitly overridden
        if self.config.lambda_cost_weight == 1.0:  # Only adjust if using default
            if self.config.decision_profile == DecisionProfile.CONSERVATIVE:
                # Conservative profile prioritizes risk reduction, weight costs less
                self.config.lambda_cost_weight = 0.5
            elif self.config.decision_profile == DecisionProfile.COVERAGE_ORIENTED:
                # Coverage-oriented profile weights costs more to encourage using cheaper automated actions
                self.config.lambda_cost_weight = 1.5
            # BALANCED keeps default lambda of 1.0
            
        # Update risk calculator with profile-specific lambda
        self.profile_lambda = self.config.lambda_cost_weight
            
        # Initialize dependencies
        self._setup_components()
        
    def _setup_components(self):
        """Setup all dependent components."""
        # Setup evidence provider
        evidence_provider = EvidenceProvider(
            name="default_evidence_provider",
            available=self.config.enable_evidence_provider,
            cost=self.config.evidence_cost
        )
        
        # Setup alternative model runner
        model_paths = {}
        for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
            if model_name != self.config.original_model_name:
                model_path = self.config.base_model_path / f"{model_name}.joblib"
                if model_path.exists():
                    model_paths[model_name] = model_path
                    
        preprocessor_path = self.config.base_model_path / "preprocessor.joblib"
        alternate_model_runner = AlternativeModelRunner(
            model_paths=model_paths,
            preprocessor_path=preprocessor_path
        )
        
        # Setup human reviewer
        human_reviewer = SimulatedHumanReviewer(
            accuracy=self.config.human_reviewer_accuracy,
            cost=self.config.human_review_cost,
            available=self.config.enable_human_reviewer,
            max_reviews=self.config.max_human_reviews
        )
        
        # Create action executor
        self.action_executor = VerificationActionExecutor(
            evidence_provider=evidence_provider,
            alternate_model_runner=alternate_model_runner,
            human_reviewer=human_reviewer
        )
        
        # Setup budget manager
        self.budget_manager = BudgetManager(
            total_budget=self.config.total_verification_budget
        )
        
        # Setup risk calculator
        self.risk_calculator = RiskCalculator(
            lambda_cost=self.profile_lambda
        )
        
        # Setup safety constraint engine
        self.safety_engine = SafetyConstraintEngine(
            action_executor=self.action_executor,
            budget_manager=self.budget_manager,
            action_costs=self.config.action_costs
        )
        
        # Create action evaluator
        self.action_evaluator = ActionEvaluator(
            safety_engine=self.safety_engine,
            risk_calculator=self.risk_calculator,
            action_costs=self.config.action_costs,
            action_effectiveness=self.config.action_effectiveness
        )
        
        # Store evaluations
        self.evaluations: List[ActionEvaluation] = []
        
    def process_sample(
        self,
        sample: pd.Series,
        sample_id: Any,
        base_prediction: Any,
        base_confidence: float,
        error_probability: float,
        harm: float
    ) -> ActionEvaluation:
        """
        Process a single sample and select a verification action.
        
        Args:
            sample: Input sample
            sample_id: Unique identifier for the sample
            base_prediction: Original model's prediction
            base_confidence: Original model's confidence
            error_probability: Reliability engine's estimated error probability
            harm: Estimated harm/consequence of incorrect prediction
            
        Returns:
            Complete evaluation with selected action
        """
        evaluation = self.action_evaluator.evaluate_sample(
            sample, sample_id, base_prediction, base_confidence, error_probability, harm
        )

        self.evaluations.append(evaluation)
        return evaluation
    
    def process_batch(
        self,
        samples: pd.DataFrame,
        predictions: List[Any],
        confidences: List[float],
        error_probabilities: List[float],
        harms: List[float]
    ) -> List[ActionEvaluation]:
        """
        Process a batch of samples.
        
        Args:
            samples: DataFrame of input samples
            predictions: List of base model predictions
            confidences: List of base model confidences
            error_probabilities: List of reliability engine error probabilities
            harms: List of harm estimates for each sample
            
        Returns:
            List of action evaluations for all samples
        """
        input_lengths = {
            len(samples),
            len(predictions),
            len(confidences),
            len(error_probabilities),
            len(harms),
        }
        if len(input_lengths) != 1:
            raise ValueError(
                "All batch inputs must have the same length; received "
                f"samples={len(samples)}, predictions={len(predictions)}, "
                f"confidences={len(confidences)}, "
                f"error_probabilities={len(error_probabilities)}, harms={len(harms)}"
            )
            
        evaluations = []
        for i in range(len(samples)):
            eval_result = self.process_sample(
                samples.iloc[i],
                sample_id=i,
                base_prediction=predictions[i],
                base_confidence=confidences[i],
                error_probability=error_probabilities[i],
                harm=harms[i]
            )
            evaluations.append(eval_result)
            
        return evaluations
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of all processed samples.
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.evaluations:
            return {"status": "No samples processed"}
            
        action_counts = {}
        total_cost = 0.0
        total_risk_reduction = 0.0
        total_baseline_risk = 0.0
        abstentions = 0
        
        for eval in self.evaluations:
            action = eval.selected_action.value
            action_counts[action] = action_counts.get(action, 0) + 1
            
            if action == ActionType.ABSTAIN.value:
                abstentions += 1
                
            total_cost += eval.selected_risk_estimate.cost_incurred
            total_risk_reduction += eval.selected_risk_estimate.risk_reduction
            total_baseline_risk += eval.baseline_risk
            
        n_samples = len(self.evaluations)
        summary = {
            "total_samples_processed": n_samples,
            "action_distribution": action_counts,
            "abstention_rate": abstentions / n_samples,
            "total_cost_incurred": total_cost,
            "remaining_budget": self.budget_manager.get_remaining_budget(),
            "total_budget": self.config.total_verification_budget,
            "total_risk_reduction": total_risk_reduction,
            "average_baseline_risk": total_baseline_risk / n_samples,
            "average_risk_reduction_per_sample": total_risk_reduction / n_samples,
            "risk_reduction_per_unit_cost": total_risk_reduction / total_cost if total_cost > 0 else 0.0
        }
        
        return summary