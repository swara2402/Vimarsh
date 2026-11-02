"""Validated contracts for offline action-outcome research endpoints."""
import math
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ResearchAction = Literal["alternate_model", "human_review", "additional_evidence"]


class StrictResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AlternateModelExecutionRequest(StrictResearchModel):
    case_id: str = Field(min_length=1, max_length=256)
    sample: Dict[str, float] = Field(min_length=1, max_length=128)
    harm: float = Field(ge=0, le=10)
    estimated_action_cost: float = Field(ge=0)

    @field_validator("sample")
    @classmethod
    def alternate_sample_values_must_be_finite(cls, value):
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("sample values must be finite")
        return value


class OutcomeCreateRequest(StrictResearchModel):
    case_id: str = Field(min_length=1, max_length=256)
    action: ResearchAction
    policy_version: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    baseline_prediction: Any
    status: Literal["pending", "completed", "failed", "not_applicable"]
    estimated_action_cost: float = Field(ge=0)
    propensity: Optional[float] = Field(default=None, gt=0, le=1)
    randomized_assignment: bool = False
    assignment_id: Optional[str] = Field(default=None, max_length=256)
    context: Dict[str, float] = Field(default_factory=dict, max_length=128)
    action_result: Any = None
    final_prediction: Any = None
    ground_truth: Any = None
    observed_cost: Optional[float] = Field(default=None, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    provider_version: Optional[str] = Field(default=None, max_length=128)
    event_id: Optional[str] = Field(default=None, min_length=1, max_length=256)
    occurred_at: Optional[str] = Field(default=None, max_length=64)


class OutcomeCompletionRequest(StrictResearchModel):
    final_prediction: Any
    ground_truth: Any
    observed_cost: Optional[float] = Field(default=None, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    provider_version: Optional[str] = Field(default=None, max_length=128)

    @field_validator("final_prediction", "ground_truth")
    @classmethod
    def retrospective_values_must_be_finite_scalars(cls, value):
        if isinstance(value, bool) or isinstance(value, (str, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError("final_prediction and ground_truth must be finite JSON scalars")


class BenefitEstimateRequest(StrictResearchModel):
    action: ResearchAction
    query_context: Dict[str, float] = Field(min_length=1, max_length=128)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    min_effective_samples: float = Field(default=10, gt=0)
    bandwidth: float = Field(default=1, gt=0)


class AllocationCase(StrictResearchModel):
    case_id: str = Field(min_length=1, max_length=256)
    context: Dict[str, float] = Field(min_length=1, max_length=128)
    available_actions: List[ResearchAction] = Field(default_factory=list, max_length=3)
    estimated_action_costs: Dict[ResearchAction, float] = Field(default_factory=dict, max_length=3)

    @field_validator("available_actions")
    @classmethod
    def actions_must_be_unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("available_actions must not contain duplicates")
        return value

    @field_validator("estimated_action_costs")
    @classmethod
    def action_costs_must_be_nonnegative(cls, value):
        if any(not math.isfinite(cost) or cost < 0 for cost in value.values()):
            raise ValueError("estimated action costs must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def costs_must_match_available_actions(self):
        unavailable = set(self.estimated_action_costs) - set(self.available_actions)
        if unavailable:
            raise ValueError("estimated costs may only be supplied for available actions")
        return self


class AllocationRequest(StrictResearchModel):
    cases: List[AllocationCase] = Field(min_length=1, max_length=1000)
    total_budget: float = Field(ge=0)
    action_capacities: Dict[ResearchAction, int] = Field(default_factory=dict, max_length=3)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    min_effective_samples: float = Field(default=10, gt=0)
    bandwidth: float = Field(default=1, gt=0)

    @field_validator("action_capacities", mode="before")
    @classmethod
    def capacities_must_be_nonnegative_integers(cls, value):
        if not isinstance(value, dict):
            raise ValueError("action_capacities must be an object")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value.values()):
            raise ValueError("action capacities must be non-negative integers")
        return value

    @model_validator(mode="after")
    def case_ids_must_be_unique(self):
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("allocation case_id values must be unique")
        return self


class AlternateModelRunRequest(StrictResearchModel):
    sample: Dict[str, float] = Field(min_length=1, max_length=128)

    @field_validator("sample")
    @classmethod
    def model_run_values_must_be_finite(cls, value):
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("sample values must be finite")
        return value
