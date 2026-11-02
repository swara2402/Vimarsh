"""Validated contracts for operator-submitted evidence."""
from pydantic import BaseModel, ConfigDict, Field


class EvidenceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=1, max_length=128)
    source_name: str = Field(min_length=1, max_length=128)
    source_reference: str = Field(min_length=1, max_length=2048)
    evidence_text: str = Field(min_length=1, max_length=20000)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict, max_length=32)
