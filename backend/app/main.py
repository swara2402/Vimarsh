"""VIMARSH FastAPI decision recommendation service."""
import os
from uuid import uuid4

from fastapi import FastAPI

from .policy import PolicyConfig, select_decision
from .schemas import DecisionRequest, DecisionResponse

app = FastAPI(
    title="VIMARSH Decision API",
    version="0.1.0",
    description=(
        "Research-stage recommendation API. It applies a transparent baseline "
        "policy and does not execute verification actions or establish deployment safety."
    ),
)

policy_config = PolicyConfig(
    max_acceptable_risk=float(os.getenv("VIMARSH_MAX_ACCEPTABLE_RISK", "0.5"))
)


@app.get("/health")
def health():
    return {"status": "ok", "service": "vimarsh-api", "mode": "recommendation_only"}


@app.post("/v1/decisions", response_model=DecisionResponse)
def create_decision(request: DecisionRequest):
    action, baseline_risk, rationale, eligible = select_decision(request, policy_config)
    return DecisionResponse(
        decision_id=uuid4(),
        action=action,
        baseline_risk=round(baseline_risk, 8),
        rationale=rationale,
        eligible_actions=eligible,
    )
