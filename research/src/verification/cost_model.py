"""Cost model configuration for verification actions."""
from dataclasses import dataclass
from typing import Dict, Optional
from .actions import ActionType
from enum import Enum


class DecisionProfile(str, Enum):
    """Decision profiles that trade off risk, cost, and coverage differently."""
    CONSERVATIVE = "conservative"  # Prioritize safety
    BALANCED = "balanced"          # Trade off risk reduction and cost
    COVERAGE_ORIENTED = "coverage"  # Prefer useful automated predictions when safe


@dataclass
class ActionCosts:
    """Standard costs for each verification action. Configurable based on decision profile.
    Costs are calibrated to be proportional to typical risk reduction values, so that
    actions that provide meaningful risk reduction have positive utility for high-risk samples."""
    accept: float = 0.0
    request_additional_evidence: float = 1.00  # Reduced to align with typical risk values
    alternate_model: float = 1.50              # Reduced to make automated actions viable
    human_review: float = 3.00                 # Reduced to make human review feasible for high-risk samples
    abstain: float = 0.75                      # Still penalize abstention appropriately
    
    @classmethod
    def from_profile(cls, profile: DecisionProfile) -> "ActionCosts":
        """Create action costs adjusted for a specific decision profile."""
        base = cls()
        if profile == DecisionProfile.CONSERVATIVE:
            # Lower relative cost for safety-focused actions
            base.human_review = 2.50  # Make human review more affordable for conservative profile
            base.abstain = 0.50       # Lower abstention cost to still keep it as safe fallback
        elif profile == DecisionProfile.COVERAGE_ORIENTED:
            # Lower cost for automated verification to increase coverage
            base.alternate_model = 1.00
            base.request_additional_evidence = 0.75
            base.abstain = 1.25       # Increase abstention cost to prefer taking verification actions
        # BALANCED uses base values
        return base
    
    def to_dict(self) -> Dict[ActionType, float]:
        """Convert to dictionary mapping ActionType to cost."""
        return {
            ActionType.ACCEPT: self.accept,
            ActionType.REQUEST_ADDITIONAL_EVIDENCE: self.request_additional_evidence,
            ActionType.ALTERNATE_MODEL: self.alternate_model,
            ActionType.HUMAN_REVIEW: self.human_review,
            ActionType.ABSTAIN: self.abstain
        }


@dataclass
class ActionEffectiveness:
    """Effectiveness factors (risk reduction potential) for each action.
    These are simulated values - in production, they would be empirically measured."""
    accept: float = 0.0  # No risk reduction
    request_additional_evidence: float = 0.5  # 50% risk reduction if successful (simulated)
    alternate_model: float = 0.6  # 60% risk reduction if successful (simulated - depends on model disagreement)
    human_review: float = 0.98  # 98% risk reduction if successful - human review is most effective
    abstain: float = 0.85  # 85% risk reduction through abstention, less thorough than human review
    
    @classmethod
    def from_profile(cls, profile: DecisionProfile) -> "ActionEffectiveness":
        """Create effectiveness assumptions adjusted for a specific decision profile."""
        base = cls()
        if profile == DecisionProfile.CONSERVATIVE:
            # More conservative risk reduction estimates
            base.human_review = 0.99  # Human review even more effective for conservative profile
            base.abstain = 0.80       # Abstention still less effective than human review
            base.alternate_model = 0.5  # Assume less risk reduction from alternative models
            base.request_additional_evidence = 0.4
        elif profile == DecisionProfile.COVERAGE_ORIENTED:
            # More optimistic (but still simulated) assumptions for automated actions
            base.alternate_model = 0.7
            base.request_additional_evidence = 0.6
            base.abstain = 0.75        # Abstention slightly less effective to encourage taking action
        return base
    
    def to_dict(self):
        """Convert to dictionary mapping ActionType to effectiveness."""
        from .actions import ActionType
        return {
            ActionType.ACCEPT: self.accept,
            ActionType.REQUEST_ADDITIONAL_EVIDENCE: self.request_additional_evidence,
            ActionType.ALTERNATE_MODEL: self.alternate_model,
            ActionType.HUMAN_REVIEW: self.human_review,
            ActionType.ABSTAIN: self.abstain
        }


@dataclass
class BudgetManager:
    """Manages verification budget across all samples."""
    total_budget: float
    remaining_budget: float = None
    
    def __post_init__(self):
        if self.remaining_budget is None:
            self.remaining_budget = self.total_budget
    
    def can_afford(self, action_cost: float) -> bool:
        """Check if the remaining budget can afford an action."""
        return self.remaining_budget >= action_cost
    
    def spend(self, amount: float) -> bool:
        """Spend from the remaining budget."""
        if self.can_afford(amount):
            self.remaining_budget -= amount
            return True
        return False
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return self.remaining_budget