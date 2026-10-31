"""Risk estimation utilities for verification actions."""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from .actions import ActionType, ActionResult


@dataclass
class RiskEstimate:
    """Container for risk estimates."""
    baseline_risk: float  # R0(s) = P_error(s) * H(s)
    residual_risk: float  # R_after(s,a) after action
    risk_reduction: float
    cost_incurred: float
    utility: float


class RiskCalculator:
    """Calculates baseline and residual risks for verification actions."""
    
    def __init__(self, lambda_cost: float = 1.0):
        """
        Initialize risk calculator.
        
        Args:
            lambda_cost: Weighting factor for costs in utility calculation
        """
        self.lambda_cost = lambda_cost
    
    def calculate_baseline_risk(self, error_probability: float, harm: float) -> float:
        """
        Calculate baseline risk R0(s) = P_error(s) * H(s).
        
        Args:
            error_probability: Estimated probability of incorrect prediction
            harm: Estimated harm/consequence of incorrect prediction
            
        Returns:
            Baseline risk value
        """
        return error_probability * harm
    
    def estimate_residual_risk(
        self,
        action_type: ActionType,
        baseline_risk: float,
        action_result: ActionResult,
        action_effectiveness: Dict[ActionType, float]
    ) -> float:
        """
        Estimate residual risk after executing an action.
        
        Args:
            action_type: Type of action executed
            baseline_risk: Original baseline risk R0(s)
            action_result: Result from executing the action
            action_effectiveness: Dictionary mapping actions to their effectiveness (risk reduction factor)
            
        Returns:
            Estimated residual risk
        """
        if not action_result.success:
            return baseline_risk  # If action failed, residual risk remains baseline
            
        effectiveness = action_effectiveness.get(action_type, 0.0)
        
        if action_type == ActionType.ACCEPT:
            # Accepting keeps the original risk
            return baseline_risk
            
        elif action_type == ActionType.ABSTAIN:
            # Abstention has some residual operational risk, not zero
            # Use the effectiveness factor from configuration (1 - effectiveness = residual factor)
            return baseline_risk * (1 - effectiveness)
            
        elif action_type == ActionType.HUMAN_REVIEW:
            # Human review reduces risk based on reviewer accuracy
            return baseline_risk * (1 - effectiveness)
            
        elif action_type == ActionType.ALTERNATE_MODEL:
            # Alternative model only reduces risk if it actually disagrees with the base prediction
            # This prevents assuming automatic improvement - genuine evaluation of disagreement
            if action_result.prediction_disagrees:
                return baseline_risk * (1 - effectiveness)
            else:
                # If model agrees with base prediction, no risk reduction is achieved
                return baseline_risk
            
        elif action_type == ActionType.REQUEST_ADDITIONAL_EVIDENCE:
            # Additional evidence reduces risk based on evidence quality
            return baseline_risk * (1 - effectiveness)
            
        return baseline_risk
    
    def calculate_utility(
        self,
        baseline_risk: float,
        residual_risk: float,
        action_cost: float
    ) -> float:
        """
        Calculate utility U(s,a) = R0(s) - R_after(s,a) - λ*C(s,a).
        
        Args:
            baseline_risk: Original baseline risk R0(s)
            residual_risk: Residual risk after action R_after(s,a)
            action_cost: Cost of performing the action C(s,a)
            
        Returns:
            Utility value
        """
        return baseline_risk - residual_risk - (self.lambda_cost * action_cost)
    
    def compute_full_risk_estimate(
        self,
        error_probability: float,
        harm: float,
        action_type: ActionType,
        action_result: ActionResult,
        action_effectiveness: Dict[ActionType, float]
    ) -> RiskEstimate:
        """
        Compute complete risk estimate for an action.
        
        Args:
            error_probability: Estimated probability of error
            harm: Estimated harm of error
            action_type: Type of action
            action_result: Result from executing action
            action_effectiveness: Effectiveness factors for all actions
            
        Returns:
            Complete RiskEstimate with all metrics
        """
        baseline_risk = self.calculate_baseline_risk(error_probability, harm)
        residual_risk = self.estimate_residual_risk(
            action_type, baseline_risk, action_result, action_effectiveness
        )
        risk_reduction = baseline_risk - residual_risk
        utility = self.calculate_utility(baseline_risk, residual_risk, action_result.cost_incurred)
        
        return RiskEstimate(
            baseline_risk=baseline_risk,
            residual_risk=residual_risk,
            risk_reduction=risk_reduction,
            cost_incurred=action_result.cost_incurred,
            utility=utility
        )