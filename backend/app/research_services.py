"""Research-only bridge to the validated outcome estimator and allocator."""
from __future__ import annotations
import os
import secrets
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.src.verification.action_benefit_estimator import estimate_action_benefit
from research.src.verification.action_outcome_ledger import ActionOutcomeLedger, ActionOutcomeRecord
from research.src.verification.action_allocator import allocate_verification_actions

def require_research_token(x_vimarsh_research_token: str | None = Header(default=None)) -> None:
    expected = os.getenv("VIMARSH_RESEARCH_API_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Research endpoints are disabled until VIMARSH_RESEARCH_API_TOKEN is configured")
    if not x_vimarsh_research_token or not secrets.compare_digest(x_vimarsh_research_token, expected):
        raise HTTPException(status_code=401, detail="Invalid research API token")

def get_ledger() -> ActionOutcomeLedger:
    configured = os.getenv("VIMARSH_OUTCOME_LEDGER", "research/data/action_outcomes.jsonl")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return ActionOutcomeLedger(path)

def append_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    record = ActionOutcomeRecord(**payload)
    return get_ledger().append(record)

def outcome_summary() -> dict[str, Any]:
    rows = get_ledger().read_verified()
    return {
        "total_records": len(rows),
        "completed_records": sum(row["status"] == "completed" for row in rows),
        "analysis_ready_records": len(get_ledger().analysis_ready_records()),
        "by_action": dict(Counter(row["action"] for row in rows)),
        "ledger_integrity": "verified",
    }

def estimate_benefit(payload: dict[str, Any]) -> dict[str, Any]:
    return estimate_action_benefit(get_ledger().analysis_ready_records(), **payload)

def allocate_cases(payload: dict[str, Any]) -> dict[str, Any]:
    records = get_ledger().analysis_ready_records()
    candidates = []
    diagnostics = []
    for case in payload["cases"]:
        for action in case["available_actions"]:
            cost = case["estimated_action_costs"].get(action)
            if cost is None:
                diagnostics.append({"case_id": case["case_id"], "action": action,
                                    "status": "excluded", "reason": "missing_estimated_action_cost"})
                continue
            estimate = estimate_action_benefit(
                records, action, case["context"],
                confidence=payload["confidence"],
                min_effective_samples=payload["min_effective_samples"],
                bandwidth=payload["bandwidth"],
            )
            diagnostics.append({"case_id": case["case_id"], **estimate})
            positive = estimate.get("status") == "recommend_for_further_evaluation"
            candidates.append({
                "case_id": case["case_id"], "action": action,
                "lower_bound_benefit": estimate.get("lower_bound_diagnostic", 0.0),
                "cost": cost, "available": positive,
            })
    allocation = allocate_verification_actions(
        candidates, payload["total_budget"],
        action_capacities=payload["action_capacities"],
    )
    return {
        "mode": "offline_research_recommendation_only",
        "execution_status": "not_executed",
        "outcome_records_used": len(records),
        "diagnostics": diagnostics,
        "allocation": allocation,
        "limitations": [
            "Benefit lower bounds are diagnostics, not guaranteed confidence bounds or causal effects.",
            "Only completed outcomes with recorded propensities are eligible for estimation.",
            "Allocation recommends actions; it does not execute human review or evidence acquisition.",
        ],
    }
