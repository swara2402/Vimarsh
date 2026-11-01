"""Validated request and response contracts for the decision API."""
from enum import Enum
from typing import Dict, List
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateAction(str, Enum):
    ALTERNATE_MODEL = "alternate_model"
    EVIDENCE = "additional_evidence"
    HUMAN_REVIEW = "human_review"


class DecisionAction(str, Enum):
    ACCEPT = "accept"
    REQUEST_ALTERNATE_MODEL = "request_alternate_model"
    ABSTAIN = "abstain"


class DecisionRequest(BaseModel):
    """Inference-time information only; ground-truth labels are not accepted."""
    model_config = ConfigDict(extra="forbid")

    prediction: str = Field(min_length=1, max_length=256)
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_error_probability: float = Field(ge=0.0, le=1.0)
    harm: float = Field(ge=0.0, le=10.0)
    available_actions: List[CandidateAction] = Field(default_factory=list, max_length=3)
    estimated_action_costs: Dict[CandidateAction, float] = Field(default_factory=dict)
    remaining_budget: float = Field(default=0.0, ge=0.0)

    @field_validator("estimated_action_costs")
    @classmethod
    def costs_must_be_finite_nonnegative(cls, value):
        import math
        if any(not math.isfinite(cost) or cost < 0 for cost in value.values()):
            raise ValueError("action costs must be finite and non-negative")
        return value


class DecisionResponse(BaseModel):
    decision_id: UUID
    action: DecisionAction
    baseline_risk: float
    recommendation_only: bool = True
    execution_status: str = "not_executed"
    simulation: bool = False
    rationale: List[str]
    eligible_actions: List[str]


class ReliabilityDecisionRequest(BaseModel):
    """Raw model features and decision context; no labels or precomputed scores."""
    model_config = ConfigDict(extra="forbid")
    sample: Dict[str, float] = Field(min_length=1)
    harm: float = Field(ge=0.0, le=10.0)
    available_actions: List[CandidateAction] = Field(default_factory=list, max_length=3)
    estimated_action_costs: Dict[CandidateAction, float] = Field(default_factory=dict)
    remaining_budget: float = Field(default=0.0, ge=0.0)

    @field_validator("sample")
    @classmethod
    def sample_values_must_be_finite(cls, value):
        import math
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("sample values must be finite")
        return value

    @field_validator("estimated_action_costs")
    @classmethod
    def reliability_costs_must_be_finite_nonnegative(cls, value):
        import math
        if any(not math.isfinite(cost) or cost < 0 for cost in value.values()):
            raise ValueError("action costs must be finite and non-negative")
        return value


class ReliabilityDecisionResponse(BaseModel):
    decision_id: UUID
    prediction: str
    confidence: float
    estimated_error_probability: float
    baseline_risk: float
    action: DecisionAction
    recommendation_only: bool = True
    execution_status: str = "not_executed"
    rationale: List[str]
    eligible_actions: List[str]
    base_model_count: int
