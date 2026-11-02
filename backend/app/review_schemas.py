"""Strict API contracts for the operator-managed review queue."""
from pydantic import BaseModel, ConfigDict, Field


class ReviewEnqueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    case_id: str = Field(min_length=1, max_length=128)
    prediction: str = Field(min_length=1, max_length=256)
    rationale: list[str] = Field(default_factory=list, max_length=20)
    context: dict[str, float] = Field(default_factory=dict, max_length=64)
    priority: int = Field(default=2, ge=0, le=5)


class ReviewClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer_id: str = Field(min_length=1, max_length=128)


class ReviewCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer_id: str = Field(min_length=1, max_length=128)
    reviewer_note: str = Field(min_length=1, max_length=4000)
    final_prediction: str = Field(min_length=1, max_length=256)
    ground_truth: str | None = Field(default=None, max_length=256)
