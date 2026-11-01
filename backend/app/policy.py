"""Transparent, deterministic baseline policy with hard feasibility filters."""
from dataclasses import dataclass

from .schemas import CandidateAction, DecisionAction, DecisionRequest


@dataclass(frozen=True)
class PolicyConfig:
    max_acceptable_risk: float = 0.5


def select_decision(request: DecisionRequest, config: PolicyConfig):
    baseline_risk = request.estimated_error_probability * request.harm
    rationale = [
        "Baseline risk is estimated_error_probability multiplied by the supplied harm score.",
        "Confidence is reported as context and is not treated as a guarantee of correctness.",
        "This endpoint recommends an action; it does not execute verification."
    ]

    if baseline_risk <= config.max_acceptable_risk:
        rationale.append("Estimated baseline risk is within the configured acceptance limit.")
        return DecisionAction.ACCEPT, baseline_risk, rationale, []

    eligible = []
    if CandidateAction.ALTERNATE_MODEL in request.available_actions:
        cost = request.estimated_action_costs.get(CandidateAction.ALTERNATE_MODEL)
        if cost is not None and cost <= request.remaining_budget:
            eligible.append(CandidateAction.ALTERNATE_MODEL.value)

    # Evidence acquisition and human review are deliberately not treated as
    # executable until operational providers and measured costs/outcomes exist.
    if CandidateAction.EVIDENCE in request.available_actions:
        rationale.append("Additional evidence was excluded: no operational evidence provider is configured.")
    if CandidateAction.HUMAN_REVIEW in request.available_actions:
        rationale.append("Human review was excluded: no operational reviewer workflow is configured.")

    if CandidateAction.ALTERNATE_MODEL in request.available_actions:
        cost = request.estimated_action_costs.get(CandidateAction.ALTERNATE_MODEL)
        if cost is None:
            rationale.append("Alternate-model action was excluded because its cost was not supplied.")
        elif cost > request.remaining_budget:
            rationale.append("Alternate-model action was excluded because it exceeds the remaining budget.")

    if eligible:
        rationale.append("Alternate-model execution is feasible under the supplied availability and budget metadata.")
        return DecisionAction.REQUEST_ALTERNATE_MODEL, baseline_risk, rationale, eligible

    rationale.append("No supported verification action passed the hard feasibility checks; abstention is the safe fallback.")
    return DecisionAction.ABSTAIN, baseline_risk, rationale, eligible
