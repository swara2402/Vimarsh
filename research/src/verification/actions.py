"""Definition of verification actions and their implementations."""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class ActionType(Enum):
    """Enumeration of all possible verification actions."""
    ACCEPT = "ACCEPT"
    REQUEST_ADDITIONAL_EVIDENCE = "REQUEST_ADDITIONAL_EVIDENCE"
    ALTERNATE_MODEL = "ALTERNATE_MODEL"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    ABSTAIN = "ABSTAIN"


@dataclass
class ActionResult:
    """Result from executing a verification action."""
    action_type: ActionType
    success: bool
    updated_prediction: Optional[Any] = None
    updated_confidence: Optional[float] = None
    additional_data: Dict[str, Any] = None
    cost_incurred: float = 0.0
    error_probability_updated: Optional[float] = None
    prediction_disagrees: bool = False


@dataclass
class EvidenceProvider:
    """Interface for providing additional evidence. Configurable to be available or not."""
    name: str
    available: bool
    cost: float
    can_provide: bool = True
    
    def request_evidence(self, sample: pd.Series) -> Tuple[Optional[pd.Series], float]:
        """
        Request additional evidence for a sample.
        
        Args:
            sample: Original sample
            
        Returns:
            Tuple of (enriched_sample, cost_incurred)
        """
        if not self.available or not self.can_provide:
            return None, self.cost
        
        # In real implementation, this would fetch additional data
        # For simulation, return the original sample with a flag
        enriched = sample.copy()
        enriched['has_additional_evidence'] = True
        return enriched, self.cost


@dataclass
class AlternativeModelRunner:
    """Runner for alternative models, reusing existing trained models."""
    model_paths: Dict[str, Path]
    preprocessor_path: Path
    available_models: Dict[str, Any] = None
    
    def __post_init__(self):
        """Load models on initialization."""
        self.available_models = {}
        self.preprocessor = joblib.load(self.preprocessor_path)
        
        for name, path in self.model_paths.items():
            if path.exists():
                self.available_models[name] = joblib.load(path)
    
    def run_alternate_model(self, model_name: str, sample: pd.DataFrame, base_prediction: Any) -> Tuple[Any, float, float, bool]:
        """
        Run an alternative model on a sample. Genuinely evaluates model disagreement
        rather than automatically assuming improvement.
        
        Args:
            model_name: Name of the alternative model to run
            sample: Input sample
            base_prediction: Original base model's prediction to check for disagreement
            
        Returns:
            Tuple of (prediction, confidence, cost_incurred, prediction_disagrees)
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
            
        model = self.available_models[model_name]
        X_processed = self.preprocessor.transform(sample)
        prediction = model.predict(X_processed)[0]
        confidence = model.predict_proba(X_processed)[0].max()
        
        # Check if alternative model disagrees with base model prediction
        prediction_disagrees = (prediction != base_prediction)
        
        # Cost is fixed for running an alternative model
        cost = 4.0  # Matches action cost in cost_model.py
        
        return prediction, confidence, cost, prediction_disagrees
    
    def get_available_alternatives(self) -> List[str]:
        """Get list of available alternative models."""
        return list(self.available_models.keys())


@dataclass
class SimulatedHumanReviewer:
    """Simulated human reviewer with configurable accuracy, cost, and availability."""
    accuracy: float  # Probability that reviewer corrects an error
    cost: float
    available: bool
    max_reviews: Optional[int] = None
    reviews_completed: int = 0
    
    def review_sample(self, sample: pd.Series, original_prediction: Any, original_confidence: float, true_label: Any = None) -> Tuple[Any, float, float]:
        """
        Simulate human review of a sample. Milestone 3.2: Implements realistic correction logic
        that can correct actual errors and measure true outcome changes.
        
        Args:
            sample: Input sample
            original_prediction: Original model prediction
            original_confidence: Original model confidence
            true_label: Ground truth label (only used for simulated correction, NOT for inferences)
            
        Returns:
            Tuple of (reviewed_prediction, confidence_after_review, cost_incurred)
        """
        if not self.available:
            return original_prediction, original_confidence, 0.0
            
        if self.max_reviews and self.reviews_completed >= self.max_reviews:
            return original_prediction, original_confidence, 0.0
            
        self.reviews_completed += 1
        reviewed_prediction = original_prediction
        
        # Simulate realistic review that can correct errors (true_label only used for simulation)
        if true_label is not None:
            # If original prediction was wrong, reviewer has 'accuracy' chance to correct it
            if original_prediction != true_label:
                if np.random.random() < self.accuracy:
                    reviewed_prediction = true_label
            # If original prediction was correct, small chance to introduce error
            else:
                if np.random.random() < (1 - self.accuracy):
                    # Flip the label to simulate an error
                    reviewed_prediction = 1 - original_prediction
        
        return reviewed_prediction, self.accuracy, self.cost


class VerificationActionExecutor:
    """Main executor for verification actions."""
    
    def __init__(
        self,
        evidence_provider: Optional[EvidenceProvider] = None,
        alternate_model_runner: Optional[AlternativeModelRunner] = None,
        human_reviewer: Optional[SimulatedHumanReviewer] = None
    ):
        self.evidence_provider = evidence_provider
        self.alternate_model_runner = alternate_model_runner
        self.human_reviewer = human_reviewer
    
    def execute_accept(self) -> ActionResult:
        """Execute ACCEPT action - accept original prediction with no cost."""
        return ActionResult(
            action_type=ActionType.ACCEPT,
            success=True,
            cost_incurred=0.0
        )
    
    def execute_request_additional_evidence(self, sample: pd.Series) -> ActionResult:
        """Execute REQUEST_ADDITIONAL_EVIDENCE action."""
        if not self.evidence_provider or not self.evidence_provider.available:
            return ActionResult(
                action_type=ActionType.REQUEST_ADDITIONAL_EVIDENCE,
                success=False,
                cost_incurred=0.0,
                additional_data={"reason": "Evidence provider not available"}
            )
        
        enriched_sample, cost = self.evidence_provider.request_evidence(sample)
        
        return ActionResult(
            action_type=ActionType.REQUEST_ADDITIONAL_EVIDENCE,
            success=enriched_sample is not None,
            updated_prediction=None,  # Would require re-running prediction
            cost_incurred=cost,
            additional_data={"enriched_sample": enriched_sample}
        )
    
    def execute_alternate_model(self, model_name: str, sample: pd.DataFrame, base_prediction: Any) -> ActionResult:
        """Execute ALTERNATE_MODEL action with genuine disagreement checking."""
        if not self.alternate_model_runner:
            return ActionResult(
                action_type=ActionType.ALTERNATE_MODEL,
                success=False,
                additional_data={"reason": "Alternative model runner not configured"}
            )
            
        try:
            # Handle first_available model selection
            if model_name == "first_available":
                alt_models = self.alternate_model_runner.get_available_alternatives()
                if not alt_models:
                    return ActionResult(
                        action_type=ActionType.ALTERNATE_MODEL,
                        success=False,
                        additional_data={"reason": "No alternative models available"}
                    )
                model_name = alt_models[0]
                
            prediction, confidence, cost, prediction_disagrees = self.alternate_model_runner.run_alternate_model(
                model_name, sample, base_prediction
            )
            return ActionResult(
                action_type=ActionType.ALTERNATE_MODEL,
                success=True,
                updated_prediction=prediction,
                updated_confidence=confidence,
                cost_incurred=cost,
                prediction_disagrees=prediction_disagrees,
                additional_data={
                    "model_used": model_name,
                    "prediction_disagrees": prediction_disagrees
                }
            )
        except Exception as e:
            return ActionResult(
                action_type=ActionType.ALTERNATE_MODEL,
                success=False,
                additional_data={"error": str(e)}
            )
    
    def execute_human_review(self, sample: pd.Series, original_prediction: Any, original_confidence: float, true_label: Any = None) -> ActionResult:
        """Execute HUMAN_REVIEW action. Milestone 3.2: supports simulated correction logic."""
        if not self.human_reviewer or not self.human_reviewer.available:
            return ActionResult(
                action_type=ActionType.HUMAN_REVIEW,
                success=False,
                additional_data={"reason": "Human reviewer not available"}
            )
            
        reviewed_prediction, updated_confidence, cost = self.human_reviewer.review_sample(
            sample, original_prediction, original_confidence, true_label
        )
        
        return ActionResult(
            action_type=ActionType.HUMAN_REVIEW,
            success=True,
            updated_prediction=reviewed_prediction,
            updated_confidence=updated_confidence,
            cost_incurred=cost,
            additional_data={"is_simulated_review": True}
        )
    
    def execute_abstain(self) -> ActionResult:
        """Execute ABSTAIN action - abstain from providing a prediction."""
        # Abstention has an operational cost
        abstention_cost = 0.5
        return ActionResult(
            action_type=ActionType.ABSTAIN,
            success=True,
            cost_incurred=abstention_cost,
            additional_data={"abstained": True}
        )