"""VIMARSH FastAPI decision recommendation service."""
import os
from functools import lru_cache
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from .policy import PolicyConfig, select_decision
from .runtime import ReliabilityBundleRuntime
from .schemas import (
    DecisionRequest, DecisionResponse, ReliabilityDecisionRequest,
    ReliabilityDecisionResponse, DecisionAction,
)

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


@lru_cache(maxsize=1)
def get_reliability_runtime():
    bundle_path = os.getenv("VIMARSH_RELIABILITY_BUNDLE")
    if not bundle_path:
        raise FileNotFoundError("VIMARSH_RELIABILITY_BUNDLE is not configured")
    return ReliabilityBundleRuntime(bundle_path)


@app.post("/v1/reliability/decisions", response_model=ReliabilityDecisionResponse)
def create_reliability_decision(request: ReliabilityDecisionRequest):
    try:
        runtime = get_reliability_runtime()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Configured reliability bundle is unavailable or invalid") from exc

    try:
        scored = runtime.score_one(request.sample)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    decision_request = DecisionRequest(
        prediction=scored["prediction"],
        confidence=scored["confidence"],
        estimated_error_probability=scored["estimated_error_probability"],
        harm=request.harm,
        available_actions=request.available_actions,
        estimated_action_costs=request.estimated_action_costs,
        remaining_budget=request.remaining_budget,
    )
    action, baseline_risk, rationale, eligible = select_decision(decision_request, policy_config)
    return ReliabilityDecisionResponse(
        decision_id=uuid4(),
        prediction=scored["prediction"],
        confidence=scored["confidence"],
        estimated_error_probability=scored["estimated_error_probability"],
        baseline_risk=round(baseline_risk, 8),
        action=action,
        rationale=rationale,
        eligible_actions=eligible,
        base_model_count=scored["base_model_count"],
    )
