"""Context-conditioned, propensity-adjusted estimates of observed action benefit.

This is an offline research estimator. Its normal-approximation lower bound is
a diagnostic, not a finite-sample guarantee or a causal effect estimate.
"""
from __future__ import annotations
from math import exp, isfinite, log, sqrt
from statistics import NormalDist
from typing import Any, Mapping, Sequence

FORBIDDEN_CONTEXT_KEYS = {"label", "target", "ground_truth", "true_label", "y_true"}

def estimate_action_benefit(
    records: Sequence[Mapping[str, Any]],
    action: str,
    query_context: Mapping[str, float],
    *,
    confidence: float = 0.95,
    min_effective_samples: float = 10.0,
    bandwidth: float = 1.0,
) -> dict[str, Any]:
    """Estimate paired risk improvement for one action near a query context.

    Benefit is (baseline error - post-action error) * harm. Positive values
    indicate a correction; negative values indicate a regression. Inverse
    selection propensity and a Gaussian context kernel define normalized
    weights. Records without completed outcomes or known propensities are
    excluded rather than silently treated as representative.
    """
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action must be a non-empty string")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not isfinite(min_effective_samples) or min_effective_samples <= 0:
        raise ValueError("min_effective_samples must be positive and finite")
    if not isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be positive and finite")
    if not query_context or any(str(k).lower() in FORBIDDEN_CONTEXT_KEYS for k in query_context):
        raise ValueError("query_context must contain policy-time features only")
    if "harm" not in query_context:
        raise ValueError("query_context must include harm")
    query = {str(k): float(v) for k, v in query_context.items()}
    if any(not isfinite(v) for v in query.values()) or query["harm"] <= 0:
        raise ValueError("query context values must be finite and harm must be positive")

    eligible = []
    for row in records:
        if row.get("action") != action or row.get("status") != "completed":
            continue
        propensity = row.get("propensity")
        if propensity is None or not isinstance(propensity, (int, float)) or isinstance(propensity, bool):
            continue
        if not isfinite(float(propensity)) or not 0 < float(propensity) <= 1:
            continue
        context = row.get("context") or {}
        if set(context) != set(query):
            continue
        if any(str(k).lower() in FORBIDDEN_CONTEXT_KEYS for k in context):
            continue
        try:
            values = {str(k): float(v) for k, v in context.items()}
            harm = values["harm"]
            if not isfinite(harm) or harm <= 0 or any(not isfinite(v) for v in values.values()):
                continue
            baseline = row["baseline_prediction"]
            final = row["final_prediction"]
            truth = row["ground_truth"]
        except (KeyError, TypeError, ValueError):
            continue
        benefit = (float(baseline != truth) - float(final != truth)) * harm
        eligible.append((row, values, benefit, float(propensity), harm))

    if not eligible:
        return {
            "action": action, "status": "abstain", "reason": "no_completed_propensity_observed_outcomes",
            "sample_count": 0, "effective_sample_size": 0.0,
        }

    keys = list(query)
    scales = {}
    for key in keys:
        values = [item[1][key] for item in eligible]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        scales[key] = max(sqrt(variance), 1e-8)

    log_weights = []
    for _, context, _, propensity, _ in eligible:
        distance_sq = sum(((context[k] - query[k]) / scales[k]) ** 2 for k in keys) / len(keys)
        log_weights.append(-0.5 * distance_sq / (bandwidth ** 2) - log(propensity))
    max_log_weight = max(log_weights)
    weights = [exp(value - max_log_weight) for value in log_weights]
    total_weight = sum(weights)
    sum_sq = sum(w * w for w in weights)
    effective_n = (total_weight * total_weight) / sum_sq if sum_sq else 0.0
    if effective_n < min_effective_samples:
        return {
            "action": action, "status": "abstain", "reason": "insufficient_effective_sample_size",
            "sample_count": len(eligible), "effective_sample_size": round(effective_n, 6),
        }

    benefits = [item[2] for item in eligible]
    mean_benefit = sum(w * b for w, b in zip(weights, benefits)) / total_weight
    variance = sum(w * (b - mean_benefit) ** 2 for w, b in zip(weights, benefits)) / total_weight
    standard_error = sqrt(max(variance, 0.0) / effective_n)
    z_value = NormalDist().inv_cdf(confidence)
    # A bounded-outcome penalty prevents zero observed variance from implying
    # zero uncertainty. This remains a diagnostic under adaptive/kernel weights.
    max_harm = max(item[4] for item in eligible)
    hoeffding_style_penalty = 2.0 * max_harm * sqrt(log(1.0 / (1.0 - confidence)) / (2.0 * effective_n))
    uncertainty_penalty = max(z_value * standard_error, hoeffding_style_penalty)
    lower_bound = mean_benefit - uncertainty_penalty
    return {
        "action": action,
        "status": "recommend_for_further_evaluation" if lower_bound > 0 else "abstain",
        "reason": "positive_lower_bound_diagnostic" if lower_bound > 0 else "benefit_not_conservatively_positive",
        "sample_count": len(eligible),
        "effective_sample_size": round(effective_n, 6),
        "weighted_mean_benefit": round(mean_benefit, 8),
        "standard_error": round(standard_error, 8),
        "uncertainty_penalty": round(uncertainty_penalty, 8),
        "lower_bound_diagnostic": round(lower_bound, 8),
        "confidence_level": confidence,
        "propensity_adjusted": True,
        "context_conditioned": True,
        "uncertainty_method": "normal_approximation_diagnostic_not_a_guaranteed_bound",
    }
