"""Validated contracts for offline action-outcome research endpoints."""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

ResearchAction = Literal["alternate_model", "human_review", "additional_evidence"]

class AlternateModelExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1, max_length=256)
    sample: Dict[str, float] = Field(min_length=1)
    harm: float = Field(ge=0, le=10)
    estimated_action_cost: float = Field(ge=0)

    @field_validator("sample")
    @classmethod
    def alternate_sample_values_must_be_finite(cls, value):
        import math
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("sample values must be finite")
        return value


class OutcomeCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1, max_length=256)
    action: ResearchAction
    policy_version: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    baseline_prediction: Any
    status: Literal["pending", "completed", "failed", "not_applicable"]
    estimated_action_cost: float = Field(ge=0)
    propensity: Optional[float] = Field(default=None, gt=0, le=1)
    randomized_assignment: bool = False
    assignment_id: Optional[str] = None
    context: Dict[str, float] = Field(default_factory=dict)
    action_result: Any = None
    final_prediction: Any = None
    ground_truth: Any = None
    observed_cost: Optional[float] = Field(default=None, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    provider_version: Optional[str] = None
    event_id: Optional[str] = None
    occurred_at: Optional[str] = None

class BenefitEstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: ResearchAction
    query_context: Dict[str, float] = Field(min_length=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    min_effective_samples: float = Field(default=10, gt=0)
    bandwidth: float = Field(default=1, gt=0)

class AllocationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1, max_length=256)
    context: Dict[str, float] = Field(min_length=1)
    available_actions: List[ResearchAction] = Field(default_factory=list)
    estimated_action_costs: Dict[ResearchAction, float] = Field(default_factory=dict)

class AllocationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: List[AllocationCase] = Field(min_length=1, max_length=1000)
    total_budget: float = Field(ge=0)
    action_capacities: Dict[ResearchAction, int] = Field(default_factory=dict)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    min_effective_samples: float = Field(default=10, gt=0)
    bandwidth: float = Field(default=1, gt=0)
