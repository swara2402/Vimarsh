"""VIMARSH FastAPI decision recommendation service."""
import os
from functools import lru_cache
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .policy import PolicyConfig, select_decision
from .runtime import ReliabilityBundleRuntime
from .schemas import (
    DecisionRequest, DecisionResponse, ReliabilityDecisionRequest,
    ReliabilityDecisionResponse, DecisionAction,
)
from .research_schemas import (
    OutcomeCreateRequest, BenefitEstimateRequest, AllocationRequest,
    AlternateModelExecutionRequest, AlternateModelRunRequest,
)
from .research_services import (
    require_research_token, append_outcome, outcome_summary,
    estimate_benefit, allocate_cases,
)

app = FastAPI(
    title="VIMARSH Decision API",
    version="0.2.0",
    description=(
        "Research-stage API: decision endpoints return recommendations, an explicit endpoint "
        "runs trusted local alternate models, and protected research endpoints support "
        "outcome logging and offline allocation. Human review and evidence actions are not executed."
    ),
)

policy_config = PolicyConfig(
    max_acceptable_risk=float(os.getenv("VIMARSH_MAX_ACCEPTABLE_RISK", "0.5"))
)

cors_origins = [
    origin.strip()
    for origin in os.getenv("VIMARSH_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
if "*" in cors_origins:
    raise RuntimeError("Wildcard CORS origins are not permitted")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Vimarsh-Research-Token"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "vimarsh-api",
        "mode": "research_preview",
        "decision_mode": "recommendation_only",
        "alternate_model_execution": "explicit_opt_in",
    }


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


@app.get("/v1/reliability/schema")
def reliability_schema():
    try:
        runtime = get_reliability_runtime()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Reliability bundle is not configured") from exc
    return {
        "feature_names": runtime.feature_names,
        "primary_model_index": runtime.primary_model_index,
        "base_model_count": len(runtime.base_models),
        "artifact_version": 2,
    }


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


@app.post("/v1/research/outcomes", dependencies=[Depends(require_research_token)])
def create_research_outcome(request: OutcomeCreateRequest):
    """Append a validated action outcome to the protected, hash-chained research ledger."""
    try:
        row = append_outcome(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "event_id": row["event_id"],
        "status": row["status"],
        "ledger": "append_only_hash_chained",
        "ground_truth_stored": row["ground_truth"] is not None,
    }


@app.get("/v1/research/outcomes/summary", dependencies=[Depends(require_research_token)])
def research_outcome_summary():
    """Return aggregate ledger status without exposing individual retrospective labels."""
    try:
        return outcome_summary()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="Outcome ledger integrity verification failed") from exc


@app.post("/v1/research/action-benefit", dependencies=[Depends(require_research_token)])
def research_action_benefit(request: BenefitEstimateRequest):
    """Estimate context-conditioned action benefit from completed propensity-logged outcomes."""
    try:
        return estimate_benefit(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/research/allocate", dependencies=[Depends(require_research_token)])
def research_allocate(request: AllocationRequest):
    """Recommend a batch allocation from observed action outcomes; never execute actions."""
    try:
        return allocate_cases(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/research/execute/alternate-model", dependencies=[Depends(require_research_token)])
def execute_alternate_model(request: AlternateModelExecutionRequest):
    """Explicitly run non-primary models from the trusted local bundle and log a pending outcome."""
    try:
        runtime = get_reliability_runtime()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Trusted reliability bundle is unavailable") from exc
    try:
        primary = runtime.score_one(request.sample)
        alternate_result = runtime.run_alternate_models(request.sample)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not alternate_result["executed"]:
        raise HTTPException(status_code=409, detail="No alternate model is present in the configured bundle")
    context = {
        "estimated_error_probability": primary["estimated_error_probability"],
        "confidence": primary["confidence"],
        "harm": request.harm,
    }
    try:
        pending = append_outcome({
            "case_id": request.case_id,
            "action": "alternate_model",
            "policy_version": "manual-alternate-model-execution-v1",
            "model_version": "trusted-bundle-artifact-v2",
            "baseline_prediction": primary["prediction"],
            "status": "pending",
            "estimated_action_cost": request.estimated_action_cost,
            "context": context,
            "action_result": alternate_result["alternate_models"],
            "latency_ms": alternate_result["total_latency_ms"],
            "provider_version": "trusted-local-model-bundle",
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "case_id": request.case_id,
        "event_id": pending["event_id"],
        "baseline_prediction": primary["prediction"],
        "confidence": primary["confidence"],
        "estimated_error_probability": primary["estimated_error_probability"],
        "context_for_outcome_completion": context,
        "alternate_models": alternate_result["alternate_models"],
        "total_latency_ms": alternate_result["total_latency_ms"],
        "execution_status": "executed",
        "ground_truth_status": "pending_retrospective_evaluation",
        "note": "Complete a separate outcome record after ground truth is genuinely observed; this execution alone is not eligible for benefit estimation.",
    }


@app.get("/v1/actions/capabilities")
def action_capabilities():
    """Describe which action paths are actually configured, without claiming placeholders are live."""
    alternate_available = False
    try:
        runtime = get_reliability_runtime()
        alternate_available = len(runtime.base_models) > 1
    except (FileNotFoundError, ValueError, OSError):
        alternate_available = False
    return {
        "actions": {
            "accept": {"available": True, "mode": "decision_only"},
            "alternate_model": {
                "available": alternate_available,
                "mode": "explicit_local_inference" if alternate_available else "unavailable",
                "execution_endpoint": "/v1/actions/alternate-model/execute" if alternate_available else None,
                "requires_research_token": False,
                "research_logging_requires_token": True,
            },
            "human_review": {
                "available": False,
                "mode": "not_configured",
                "reason": "No operational reviewer queue or provider is configured.",
            },
            "additional_evidence": {
                "available": False,
                "mode": "not_configured",
                "reason": "No operational evidence source or acquisition provider is configured.",
            },
            "abstain": {"available": True, "mode": "decision_only"},
        },
        "execution_boundary": "The explicit local alternate-model endpoint runs trusted bundle inference; the protected research variant also logs a pending outcome. Allocation recommendations never execute actions.",
    }


@app.post("/v1/actions/alternate-model/execute")
def run_alternate_model_action(request: AlternateModelRunRequest):
    """Run non-primary models on an explicitly submitted sample; no labels are accepted or logged."""
    try:
        runtime = get_reliability_runtime()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Trusted reliability bundle is unavailable") from exc
    try:
        result = runtime.run_alternate_models(request.sample)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not result["executed"]:
        raise HTTPException(status_code=409, detail="No alternate model is present in the configured bundle")
    return {
        **result,
        "execution_status": "executed",
        "purpose": "explicit_alternate_model_inference",
        "not_a_final_verification_decision": True,
    }
