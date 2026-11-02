"""VIMARSH FastAPI decision recommendation service."""
import os
from functools import lru_cache
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .policy import PolicyConfig, select_decision
from .runtime import ReliabilityBundleRuntime
from .schemas import (
    DecisionRequest, DecisionResponse, ReliabilityDecisionRequest,
    ReliabilityDecisionResponse, DecisionAction,
)
from .research_schemas import (
    OutcomeCreateRequest, OutcomeCompletionRequest, BenefitEstimateRequest, AllocationRequest,
    AlternateModelExecutionRequest, AlternateModelRunRequest,
)
from .research_services import (
    require_research_token, append_outcome, outcome_summary,
    estimate_benefit, allocate_cases, complete_pending_outcome,
)
from .review_queue import (
    require_reviewer_token, enqueue as enqueue_review, list_cases as list_review_cases,
    claim_case as claim_review_case, complete_case as complete_review_case, list_events as list_review_events,
)
from .review_schemas import ReviewEnqueueRequest, ReviewClaimRequest, ReviewCompletionRequest
from .evidence_register import require_evidence_token, add_evidence, list_evidence
from .evidence_schemas import EvidenceCreateRequest, EvidenceResolveDOIRequest
from .evidence_providers import resolve_crossref_doi

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
    allow_headers=["Content-Type", "X-Vimarsh-Research-Token", "X-Vimarsh-Reviewer-Token", "X-Vimarsh-Evidence-Token"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Apply conservative response headers to API and local operator workflows."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/v1/"):
        response.headers["Cache-Control"] = "no-store"
    return response



def _json_safe_validation_detail(value):
    """Make rejected non-finite inputs safe to include in a JSON 422 response."""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_validation_detail(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_validation_detail(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return consistent 422 responses even when clients submit non-standard NaN/Infinity JSON."""
    return JSONResponse(
        status_code=422,
        content={"detail": _json_safe_validation_detail(exc.errors())},
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


@app.post("/v1/research/outcomes/{event_id}/complete", dependencies=[Depends(require_research_token)])
def complete_research_outcome(event_id: str, request: OutcomeCompletionRequest):
    """Append a linked completed event without mutating its original pending execution."""
    try:
        row = complete_pending_outcome(event_id, request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "event_id": row["event_id"],
        "parent_event_id": row["parent_event_id"],
        "case_id": row["case_id"],
        "status": row["status"],
        "analysis_eligible": row["propensity"] is not None,
        "ledger": "append_only_hash_chained",
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
                "available": bool(os.getenv("VIMARSH_REVIEWER_API_TOKEN")),
                "mode": "operator_managed_queue" if os.getenv("VIMARSH_REVIEWER_API_TOKEN") else "not_configured",
                "endpoint": "/v1/reviews",
                "reason": None if os.getenv("VIMARSH_REVIEWER_API_TOKEN") else "Set VIMARSH_REVIEWER_API_TOKEN to enable the local operator queue.",
            },
            "additional_evidence": {
                "available": bool(os.getenv("VIMARSH_EVIDENCE_API_TOKEN")),
                "mode": "operator_submitted_evidence" if os.getenv("VIMARSH_EVIDENCE_API_TOKEN") else "not_configured",
                "endpoint": "/v1/evidence",
                "reason": None if os.getenv("VIMARSH_EVIDENCE_API_TOKEN") else "Set VIMARSH_EVIDENCE_API_TOKEN to enable manual evidence intake; no external source is fetched or verified.",
            },
            "abstain": {"available": True, "mode": "decision_only"},
        },
        "execution_boundary": "The explicit local alternate-model endpoint runs trusted bundle inference; the protected research variant also logs a pending outcome. Allocation recommendations never execute actions.",
    }



@app.post("/v1/reviews", dependencies=[Depends(require_reviewer_token)])
def create_review_case(request: ReviewEnqueueRequest):
    """Queue a case for a human operator; this does not imply a reviewer is online."""
    try:
        return enqueue_review(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.get("/v1/reviews", dependencies=[Depends(require_reviewer_token)])
def get_review_cases(status: str = "pending", limit: int = 50):
    """List queued cases without exposing the outcome research ledger."""
    if status not in {"pending", "claimed", "completed", "cancelled"}:
        raise HTTPException(status_code=422, detail="Unsupported review status")
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return {"cases": list_review_cases(status=status, limit=limit)}

@app.post("/v1/reviews/{review_id}/claim", dependencies=[Depends(require_reviewer_token)])
def claim_review(review_id: str, request: ReviewClaimRequest):
    """Atomically claim one pending case for a reviewer."""
    try:
        return claim_review_case(review_id, request.reviewer_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

@app.post("/v1/reviews/{review_id}/complete", dependencies=[Depends(require_reviewer_token)])
def complete_review(review_id: str, request: ReviewCompletionRequest):
    """Record a reviewer adjudication after the case has been claimed."""
    try:
        return complete_review_case(review_id, request.reviewer_id, request.reviewer_note, request.final_prediction, request.ground_truth)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/v1/reviews/{review_id}/events", dependencies=[Depends(require_reviewer_token)])
def get_review_events(review_id: str):
    """Return audit events after verifying the global hash chain."""
    try:
        return {"events": list_review_events(review_id), "integrity": "verified"}
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="Review audit integrity verification failed") from exc


@app.post("/v1/evidence", dependencies=[Depends(require_evidence_token)])
def submit_evidence(request: EvidenceCreateRequest):
    """Append operator-submitted evidence with source metadata and a content digest."""
    try:
        return add_evidence(request.model_dump())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="Evidence record is invalid") from exc

@app.post("/v1/evidence/resolve-doi", dependencies=[Depends(require_evidence_token)])
def resolve_doi_evidence(request: EvidenceResolveDOIRequest):
    """Retrieve public bibliographic metadata from Crossref and append it to the evidence register."""
    import json

    try:
        source = resolve_crossref_doi(request.doi)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    record = add_evidence({
        "case_id": request.case_id,
        "source_name": "Crossref DOI metadata",
        "source_reference": source["url"],
        "evidence_text": json.dumps(source, sort_keys=True, ensure_ascii=False),
        "metadata": {
            "provider": "crossref",
            "doi": source["doi"],
            "work_type": source["work_type"],
            "metadata_only": True,
        },
    })
    return {**record, "provider": "crossref", "claim_verification": "not_performed"}


@app.get("/v1/evidence/{case_id}", dependencies=[Depends(require_evidence_token)])
def get_case_evidence(case_id: str, limit: int = 50):
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return {"case_id": case_id, "evidence": list_evidence(case_id, limit)}

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
