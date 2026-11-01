"""Safety constraint engine to determine which actions are feasible."""
from dataclasses import dataclass
from typing import Dict, List, Set, Any
from .actions import ActionType, VerificationActionExecutor, EvidenceProvider, AlternativeModelRunner, SimulatedHumanReviewer
from .cost_model import BudgetManager, ActionCosts


@dataclass
class ActionFeasibility:
    """Feasibility status for each action."""
    action_type: ActionType
    is_available: bool
    is_budget_feasible: bool
    is_safe: bool
    overall_feasible: bool
    reason: str = ""


class SafetyConstraintEngine:
    """Enforces safety and availability constraints on verification actions."""
    
    def __init__(
        self,
        action_executor: VerificationActionExecutor,
        budget_manager: BudgetManager,
        action_costs: ActionCosts,
        high_risk_threshold: float = 8.0,
        medium_risk_threshold: float = 4.0,
    ):
        if not (0 <= medium_risk_threshold < high_risk_threshold):
            raise ValueError("Risk thresholds must satisfy 0 <= medium < high")
        self.action_executor = action_executor
        self.budget_manager = budget_manager
        self.action_costs = action_costs.to_dict()
        self.high_risk_threshold = float(high_risk_threshold)
        self.medium_risk_threshold = float(medium_risk_threshold)
    
    def check_action_availability(self, action_type: ActionType) -> tuple[bool, str]:
        """
        Check if an action is fundamentally available in the current setup.
        
        Args:
            action_type: Action to check
            
        Returns:
            Tuple of (is_available, reason)
        """
        if action_type == ActionType.ACCEPT:
            return True, "ACCEPT is always available"
            
        elif action_type == ActionType.ABSTAIN:
            return True, "ABSTAIN is always available as fallback"
            
        elif action_type == ActionType.REQUEST_ADDITIONAL_EVIDENCE:
            if (self.action_executor.evidence_provider and 
                self.action_executor.evidence_provider.available):
                return True, "Evidence provider is available"
            return False, "REQUEST_ADDITIONAL_EVIDENCE unavailable: no evidence provider configured"
            
        elif action_type == ActionType.ALTERNATE_MODEL:
            if (self.action_executor.alternate_model_runner and 
                len(self.action_executor.alternate_model_runner.get_available_alternatives()) > 0):
                return True, "Alternative models are available"
            return False, "ALTERNATE_MODEL unavailable: no alternative models configured"
            
        elif action_type == ActionType.HUMAN_REVIEW:
            if (self.action_executor.human_reviewer and 
                self.action_executor.human_reviewer.available):
                # Check if reviewer has capacity
                if (self.action_executor.human_reviewer.max_reviews and
                    self.action_executor.human_reviewer.reviews_completed >= self.action_executor.human_reviewer.max_reviews):
                    return False, "HUMAN_REVIEW unavailable: max reviews capacity reached"
                return True, "Human reviewer is available"
            return False, "HUMAN_REVIEW unavailable: no human reviewer configured"
            
        return False, f"Unknown action type: {action_type}"
    
    def check_budget_feasibility(self, action_type: ActionType) -> tuple[bool, str]:
        """
        Check if the current budget can support the action.
        
        Args:
            action_type: Action to check
            
        Returns:
            Tuple of (is_budget_feasible, reason)
        """
        action_cost = self.action_costs.get(action_type, 0.0)
        if self.budget_manager.can_afford(action_cost):
            return True, f"Budget is sufficient (cost: {action_cost}, remaining: {self.budget_manager.get_remaining_budget()})"
        return False, f"Insufficient budget for {action_type}: need {action_cost}, has {self.budget_manager.get_remaining_budget()}"
    
    def check_safety_constraints(self, action_type: ActionType, baseline_risk: float) -> tuple[bool, str]:
        """
        Check if action is safe to execute given the risk level.
        Implements tiered safety constraints:
        - High risk (>high_risk_threshold): Only HUMAN_REVIEW or ABSTAIN allowed
        - Medium risk (medium_risk_threshold < risk <= high_risk_threshold): ALTERNATE_MODEL, REQUEST_ADDITIONAL_EVIDENCE, HUMAN_REVIEW, ABSTAIN allowed (ACCEPT not allowed)
        - Low risk (<=medium_risk_threshold): All actions allowed
        
        Args:
            action_type: Action to check
            baseline_risk: Baseline risk of the sample
            
        Returns:
            Tuple of (is_safe, reason)
        """
        # ABSTAIN is always safe
        if action_type == ActionType.ABSTAIN:
            return True, "ABSTAIN is always safe"
            
        # For extremely high risk samples, only HUMAN_REVIEW or ABSTAIN are allowed.
        if baseline_risk >= self.high_risk_threshold:
            if action_type not in [ActionType.HUMAN_REVIEW, ActionType.ABSTAIN]:
                return False, f"High risk ({baseline_risk:.2f}) requires HUMAN_REVIEW or ABSTAIN"
            return True, "High risk action is allowed"
        # For medium-high risk, can't just accept - but allow automated verification actions
        elif baseline_risk >= self.medium_risk_threshold and action_type == ActionType.ACCEPT:
            return False, f"Medium-high risk ({baseline_risk:.2f}) cannot be directly ACCEPTED"
            
        return True, "Action is safe for current risk level"
    
    def evaluate_all_actions(self, baseline_risk: float) -> Dict[ActionType, ActionFeasibility]:
        """
        Evaluate feasibility for all possible actions.
        
        Args:
            baseline_risk: Baseline risk of the current sample
            
        Returns:
            Dictionary of feasibility status for all actions
        """
        feasibility = {}
        
        for action_type in ActionType:
            # Check all constraints
            is_available, avail_reason = self.check_action_availability(action_type)
            is_budget_feasible, budget_reason = self.check_budget_feasibility(action_type)
            is_safe, safety_reason = self.check_safety_constraints(action_type, baseline_risk)
            
            overall_feasible = is_available and is_budget_feasible and is_safe
            
            # Combine reasons
            all_reasons = []
            if not is_available:
                all_reasons.append(avail_reason)
            if not is_budget_feasible:
                all_reasons.append(budget_reason)
            if not is_safe:
                all_reasons.append(safety_reason)
            
            feasibility[action_type] = ActionFeasibility(
                action_type=action_type,
                is_available=is_available,
                is_budget_feasible=is_budget_feasible,
                is_safe=is_safe,
                overall_feasible=overall_feasible,
                reason="; ".join(all_reasons) if all_reasons else "All constraints satisfied"
            )
            
        return feasibility
    
    def get_feasible_actions(self, baseline_risk: float) -> List[ActionType]:
        """
        Get list of all feasible actions for current conditions.
        
        Args:
            baseline_risk: Baseline risk of the current sample
            
        Returns:
            List of feasible action types
        """
        feasibility = self.evaluate_all_actions(baseline_risk)
        feasible = [a for a, f in feasibility.items() if f.overall_feasible]
        
        # Ensure ABSTAIN is always in feasible actions as fallback
        if ActionType.ABSTAIN not in feasible:
            feasible.append(ActionType.ABSTAIN)
            
        return feasible