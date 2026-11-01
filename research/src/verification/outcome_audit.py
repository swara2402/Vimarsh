"""Outcome accounting based on observed held-out labels and executed predictions.

This module deliberately does not consume configured ActionEffectiveness values.
It measures corrections, regressions, residual errors, coverage, and realized cost
after an action's prediction (or abstention) is known.
"""
from typing import Any, Dict, Optional
import numpy as np


def audit_prediction_changes(
    y_true: Any,
    baseline_predictions: Any,
    candidate_predictions: Any,
    action_mask: Any,
    action_costs: Any,
    prediction_issued: Optional[Any] = None,
) -> Dict[str, Any]:
    """Audit a paired action against baseline predictions.

    Labels are used only for retrospective scoring. action_mask must be
    produced without access to labels. prediction_issued supports abstention:
    false means no prediction was released for that row.
    """
    y = np.asarray(y_true)
    baseline = np.asarray(baseline_predictions)
    candidate = np.asarray(candidate_predictions)
    acted = np.asarray(action_mask, dtype=bool)
    costs = np.asarray(action_costs, dtype=float)
    arrays = (baseline, candidate, acted, costs)
    if y.ndim != 1 or any(a.ndim != 1 for a in arrays):
        raise ValueError("All inputs must be one-dimensional")
    if any(len(a) != len(y) for a in arrays):
        raise ValueError("All inputs must have equal lengths")
    if len(y) == 0:
        raise ValueError("At least one observation is required")
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Action costs must be finite and non-negative")
    if prediction_issued is None:
        issued = np.ones(len(y), dtype=bool)
    else:
        issued = np.asarray(prediction_issued, dtype=bool)
        if issued.ndim != 1 or len(issued) != len(y):
            raise ValueError("prediction_issued must match y_true")

    final = np.where(acted, candidate, baseline)
    baseline_wrong = baseline != y
    final_wrong = final != y
    corrections = int(np.sum(acted & baseline_wrong & ~final_wrong))
    regressions = int(np.sum(acted & ~baseline_wrong & final_wrong))
    acted_baseline_errors = int(np.sum(acted & baseline_wrong))
    acted_residual_errors = int(np.sum(acted & final_wrong))
    issued_count = int(issued.sum())
    issued_errors = int(np.sum(final_wrong & issued))
    total_cost = float(costs[acted].sum())
    net = corrections - regressions
    return {
        "sample_count": int(len(y)),
        "action_count": int(acted.sum()),
        "action_rate": float(acted.mean()),
        "corrections": corrections,
        "regressions": regressions,
        "net_corrections": int(net),
        "baseline_errors_in_actioned_cases": acted_baseline_errors,
        "residual_errors_in_actioned_cases": acted_residual_errors,
        "actioned_case_error_reduction": float(net / acted_baseline_errors) if acted_baseline_errors else None,
        "baseline_error_count": int(baseline_wrong.sum()),
        "final_error_count_before_abstention": int(final_wrong.sum()),
        "prediction_issued_count": issued_count,
        "coverage": float(issued.mean()),
        "errors_among_issued_predictions": issued_errors,
        "error_rate_among_issued_predictions": float(issued_errors / issued_count) if issued_count else None,
        "realized_action_cost": total_cost,
        "cost_per_net_correction": float(total_cost / net) if net > 0 else None,
    }
