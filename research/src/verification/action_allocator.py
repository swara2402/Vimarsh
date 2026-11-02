"""Batch-level constrained allocation of verification actions using benefit diagnostics."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence

ALLOWED_ACTIONS = {"alternate_model", "human_review", "additional_evidence"}

def allocate_verification_actions(
    candidates: Sequence[Mapping[str, Any]],
    total_budget: float,
    *,
    action_capacities: Mapping[str, int] | None = None,
    time_limit_seconds: float = 10.0,
) -> dict[str, Any]:
    """Solve a binary allocation problem with case, budget, and action-capacity constraints.

    Only available candidates with strictly positive conservative benefit diagnostics
    are eligible. If the optimizer fails, the safe fallback is to allocate nothing.
    """
    if isinstance(total_budget, bool) or not isinstance(total_budget, (int, float)) or not math.isfinite(float(total_budget)) or total_budget < 0:
        raise ValueError("total_budget must be finite and non-negative")
    if not math.isfinite(time_limit_seconds) or time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive and finite")
    capacities = dict(action_capacities or {})
    for action, capacity in capacities.items():
        if action not in ALLOWED_ACTIONS or isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise ValueError("action capacities must be non-negative integers for supported actions")

    eligible = []
    seen = set()
    for item in candidates:
        case_id, action = item.get("case_id"), item.get("action")
        if not isinstance(case_id, str) or not case_id.strip() or action not in ALLOWED_ACTIONS:
            raise ValueError("each candidate needs a case_id and supported action")
        key = (case_id, action)
        if key in seen:
            raise ValueError("duplicate case_id/action candidate")
        seen.add(key)
        benefit, cost = item.get("lower_bound_benefit"), item.get("cost")
        if isinstance(benefit, bool) or not isinstance(benefit, (int, float)) or not math.isfinite(float(benefit)):
            raise ValueError("lower_bound_benefit must be finite")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(float(cost)) or cost < 0:
            raise ValueError("candidate cost must be finite and non-negative")
        if item.get("available", False) and benefit > 0 and capacities.get(action, len(candidates)) > 0:
            eligible.append({**dict(item), "lower_bound_benefit": float(benefit), "cost": float(cost)})

    if not eligible or total_budget == 0:
        return {"status": "abstain", "reason": "no_positive_feasible_candidates", "selected": [],
                "total_cost": 0.0, "total_benefit": 0.0, "remaining_budget": float(total_budget)}

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise RuntimeError("scipy is required for constrained action allocation") from exc

    n = len(eligible)
    case_ids = sorted({row["case_id"] for row in eligible})
    actions = sorted({row["action"] for row in eligible})
    constraint_rows = []
    upper_bounds = []

    # At most one verification action may be selected for each case.
    for case_id in case_ids:
        constraint_rows.append([1.0 if row["case_id"] == case_id else 0.0 for row in eligible])
        upper_bounds.append(1.0)

    # Shared total budget.
    constraint_rows.append([row["cost"] for row in eligible])
    upper_bounds.append(float(total_budget))

    # Per-action capacity constraints.
    for action in actions:
        if action in capacities:
            constraint_rows.append([1.0 if row["action"] == action else 0.0 for row in eligible])
            upper_bounds.append(float(capacities[action]))

    matrix = np.asarray(constraint_rows, dtype=float)
    constraint = LinearConstraint(matrix, -np.inf, np.asarray(upper_bounds, dtype=float))
    objective = -np.asarray([row["lower_bound_benefit"] for row in eligible], dtype=float)
    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraint,
        options={"time_limit": float(time_limit_seconds)},
    )
    if not result.success or result.x is None:
        return {"status": "abstain", "reason": "optimizer_failed_safe_fallback",
                "solver_message": str(result.message), "selected": [], "total_cost": 0.0,
                "total_benefit": 0.0, "remaining_budget": float(total_budget)}

    selected = [eligible[i] for i, value in enumerate(result.x) if value > 0.5]
    total_cost_used = sum(row["cost"] for row in selected)
    total_benefit = sum(row["lower_bound_benefit"] for row in selected)
    return {
        "status": "allocated" if selected else "abstain",
        "reason": "maximum_positive_benefit_under_constraints" if selected else "no_positive_feasible_candidates",
        "selected": selected,
        "total_cost": round(total_cost_used, 8),
        "total_benefit": round(total_benefit, 8),
        "remaining_budget": round(float(total_budget) - total_cost_used, 8),
        "solver": "scipy.optimize.milp",
        "optimality_proven": bool(result.success),
    }
