# VIMARSH

**Question. Evaluate. Verify.**

VIMARSH is a research-oriented framework for assessing the reliability of machine-learning predictions and selecting an appropriate verification action before a prediction is accepted.

Its central idea is simple:

> A prediction should earn the right to be trusted.

Rather than relying on a model's confidence score alone, VIMARSH investigates multiple reliability signals, estimates the risk of an incorrect decision, and chooses among safe, feasible verification actions.

## Research direction

**Working research title**

*VIMARSH: An Evidence-Driven Framework for Risk-Aware Verification of Machine Learning Predictions*

**Research question**

Can a system learn to select an appropriate verification action for an uncertain prediction by estimating how much each action could reduce decision risk, while accounting for safety constraints and verification cost?

This is an evolving research question, not a claim of established novelty or patentability. Prior-art review and experimental validation are required.

## Proposed workflow

1. **Prediction:** An existing ML model produces a prediction and its available confidence information.
2. **Reliability analysis:** VIMARSH extracts signals such as calibrated uncertainty, model disagreement, input quality, and distribution shift.
3. **Risk estimation:** The framework estimates the probability and potential consequence of an incorrect prediction.
4. **Safety filtering:** Hard constraints remove verification actions that are unsafe, unavailable, or outside resource limits.
5. **Action evaluation:** A learned evaluator estimates the expected risk reduction and cost of each remaining action.
6. **Verification decision:** The framework selects a worthwhile safe action or abstains.
7. **Feedback:** Outcomes and observed costs are recorded for evaluation and future learning.

## Initial verification actions

- Accept the prediction.
- Request additional evidence, when that evidence is genuinely obtainable.
- Run an alternate model.
- Escalate for human review.
- Abstain from making an automated decision.

## Initial research scope

The first prototype will focus on tabular classification. Image classification and deep-learning models may be explored in a later phase.

Potential benchmark datasets include Wine Quality, Breast Cancer Wisconsin, and Covertype. These are research benchmarks only; results must not be interpreted as clinical or other real-world deployment validation.

## Planned technology

| Area | Initial tools |
|---|---|
| Core language | Python |
| ML | scikit-learn, XGBoost, PyTorch |
| Data | Pandas, NumPy |
| API | FastAPI |
| Storage | PostgreSQL |
| Frontend | React |
| Visualization | Plotly |
| Experiments | Jupyter, MLflow |
| Version control | Git and GitHub |

The stack may change as the experiments clarify what is necessary.

## Repository structure

```text
vimarsh/
├── backend/       # API and framework implementation
├── docs/          # Research, methodology, and architecture
├── frontend/      # Research and experiment dashboard
├── research/      # Datasets, baselines, and experiments
├── tests/         # Unit, integration, and evaluation tests
├── .gitignore
└── README.md
```

## Research principles

- Never use test labels to make inference-time verification decisions.
- Keep training, validation, and test data separated.
- Report risk and coverage together; abstaining on every case is not a useful solution.
- Compare against meaningful baselines, not just the proposed method.
- Distinguish simulated verification outcomes from real human review.
- Account for selection bias when evaluating actions chosen by a policy.
- Report limitations, uncertainty, and negative results honestly.
- Do not claim novelty, safety, or patentability without supporting evidence.

## Current status

<sub>Early research setup. The repository currently contains project documentation and planned implementation areas; the framework and experiments are not yet implemented.</sub>

## Getting started

Implementation instructions will be added as the backend, experiments, and frontend are developed. See [Research Plan](docs/research-plan.md) for the proposed sequence of work.

## License

No license has been selected yet. Until one is added, the default copyright rules apply.