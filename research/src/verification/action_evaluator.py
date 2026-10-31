"""Action evaluator that calculates utilities for all feasible actions."""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import pandas as pd
from .actions import ActionType, VerificationActionExecutor, ActionResult
from .risk_estimator import RiskCalculator, RiskEstimate
from .safety_constraints import SafetyConstraintEngine, ActionFeasibility
from .cost_model import ActionCosts, ActionEffectiveness


@dataclass
class ActionEvaluation:
    """Complete evaluation of all actions for a sample.
    
    Includes all required logging fields for Milestone 3.1:
    - Baseline risk
    - Feasible actions
    - Estimated residual risk for each action
    - Action cost for each action
    - Utility for each action
    - Selected action
    - Selection explanation
    """
    sample_id: Any
    base_prediction: Any
    error_probability: float
    harm: float
    baseline_risk: float  # Logged baseline risk
    feasibility: Dict[ActionType, ActionFeasibility]  # Action feasibility checks
    feasible_actions: List[ActionType]  # List of feasible actions (logged)
    residual_risks: Dict[ActionType, float]  # Estimated residual risk for each feasible action
    action_costs: Dict[ActionType, float]  # Cost for each feasible action
    utilities: Dict[ActionType, float]  # Utility for each feasible action
    selected_action: ActionType  # Selected action
    explanation: str  # Selection explanation
    selected_risk_estimate: RiskEstimate  # Full risk estimate for selected action


class ActionEvaluator:
    """Evaluates all feasible actions and selects the optimal one."""
    
    def __init__(
        self,
        safety_engine: SafetyConstraintEngine,
        risk_calculator: RiskCalculator,
        action_costs: ActionCosts,
        action_effectiveness: ActionEffectiveness
    ):
        self.safety_engine = safety_engine
        self.risk_calculator = risk_calculator
        self.action_costs = action_costs
        self.cost_dict = action_costs.to_dict()
        self.effectiveness_dict = action_effectiveness.to_dict()
    
    def evaluate_sample(
        self,
        sample: pd.Series,
        sample_id: Any,
        base_prediction: Any,
        base_confidence: float,
        error_probability: float,
        harm: float
    ) -> ActionEvaluation:
        """
        Evaluate all feasible actions for a single sample and select the best.
        
        Args:
            sample: Input sample
            sample_id: Unique identifier for the sample
            base_prediction: Original model prediction
            base_confidence: Original model confidence
            error_probability: Estimated error probability from reliability engine
            harm: Estimated harm/consequence of incorrect prediction
            
        Returns:
            Complete ActionEvaluation with selected action and all metrics
        """
        # Calculate baseline risk
        baseline_risk = self.risk_calculator.calculate_baseline_risk(error_probability, harm)
        
        # Get feasibility for all actions
        feasibility = self.safety_engine.evaluate_all_actions(baseline_risk)
        feasible_actions = [a for a, f in feasibility.items() if f.overall_feasible]
        
        # Calculate risk, cost, and utility for each feasible action (log all values)
        risk_estimates: Dict[ActionType, Optional[RiskEstimate]] = {}
        action_utilities: Dict[ActionType, float] = {}
        residual_risks: Dict[ActionType, float] = {}
        action_costs_log: Dict[ActionType, float] = {}
        
        # Create mock action results for utility calculation (explicitly simulated)
        for action_type in feasible_actions:
            # Simulate successful execution for utility calculation - all effectiveness values are simulated
            mock_result = ActionResult(
                action_type=action_type,
                success=True,
                cost_incurred=self.cost_dict[action_type],
                prediction_disagrees=action_type == ActionType.ALTERNATE_MODEL  # Only set for actual execution
            )
            
            # Calculate risk estimate
            risk_est = self.risk_calculator.compute_full_risk_estimate(
                error_probability, harm, action_type, mock_result, self.effectiveness_dict
            )
            risk_estimates[action_type] = risk_est
            action_utilities[action_type] = risk_est.utility
            residual_risks[action_type] = risk_est.residual_risk
            action_costs_log[action_type] = risk_est.cost_incurred
        
        # Select action with maximum utility
        if not action_utilities:
            # Fallback to abstain if no feasible actions have utilities
            selected_action = ActionType.ABSTAIN
            explanation = "Fallback: No actions had positive utility, selected ABSTAIN"
        else:
            # Find action with maximum utility
            selected_action = max(action_utilities.items(), key=lambda x: x[1])[0]
            max_utility = action_utilities[selected_action]
            
            # If all utilities are negative, select the safest fallback (ABSTAIN or HUMAN_REVIEW)
            if max_utility < 0:
                # Prefer HUMAN_REVIEW if available, else ABSTAIN
                if ActionType.HUMAN_REVIEW in feasible_actions:
                    selected_action = ActionType.HUMAN_REVIEW
                    explanation = "All actions had negative utility, selected HUMAN_REVIEW as safest available"
                else:
                    selected_action = ActionType.ABSTAIN
                    explanation = "All actions had negative utility, selected ABSTAIN as fallback"
            else:
                # Generate detailed explanation for selection with all metrics
                other_actions = []
                for a, u in action_utilities.items():
                    if a != selected_action:
                        other_actions.append(f"{a.value}: utility={u:.2f}, risk={residual_risks[a]:.2f}, cost=${action_costs_log[a]:.2f}")
                other_str = "; ".join(other_actions)
                explanation = (f"Selected {selected_action.value} with utility {max_utility:.2f}, "
                             f"residual risk={residual_risks[selected_action]:.2f}, cost=${action_costs_log[selected_action]:.2f}. "
                             f"Other feasible actions: {other_str}")
        
        # Get the risk estimate for the selected action
        selected_risk_estimate = risk_estimates.get(selected_action)
        if not selected_risk_estimate:
            # This should never happen as ABSTAIN is always feasible
            mock_result = ActionResult(action_type=ActionType.ABSTAIN, success=True, cost_incurred=self.cost_dict[ActionType.ABSTAIN])
            selected_risk_estimate = self.risk_calculator.compute_full_risk_estimate(
                error_probability, harm, ActionType.ABSTAIN, mock_result, self.effectiveness_dict
            )
        
        # Spend the budget for the selected action
        self.safety_engine.budget_manager.spend(selected_risk_estimate.cost_incurred)
        
        return ActionEvaluation(
            sample_id=sample_id,
            base_prediction=base_prediction,
            error_probability=error_probability,
            harm=harm,
            baseline_risk=baseline_risk,
            feasibility=feasibility,
            feasible_actions=feasible_actions,
            residual_risks=residual_risks,
            action_costs=action_costs_log,
            utilities=action_utilities,
            selected_action=selected_action,
            explanation=explanation,
            selected_risk_estimate=selected_risk_estimate
        )