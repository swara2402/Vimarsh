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

The repository contains implemented baseline classification, reliability-analysis, and verification-action research code. Milestone 3.2 adds outcome and execution-cost auditing. Milestone 4 adds a reliability-to-verification inference bridge, serializable model artifacts, duplicate-grouped train/validation/test and out-of-fold splitting, and an integrated policy-comparison runner.

The integrated verification-engine workflow remains research-stage and runs on Wine Quality, with separate grouped multi-seed policy-comparison tracks for MAGIC Gamma Telescope and HTRU2. These benchmarks use different scenario assumptions and evaluation designs, so their simulated net-benefit and weighted-loss values are reported separately rather than pooled. The experiments use simulated human review and assumed action-effectiveness factors; they do not establish causal real-world risk reduction or broad domain generalization. Wine and MAGIC now share a validation-selected nominal 75% acceptance diagnostic, while HTRU2 reports separate hard-cap, matched-capacity, and validation-selected review-rate diagnostics. Milestone 48 adds paired cluster-bootstrap 95% intervals for validation-selected coverage and accepted-error differences against confidence-only on Wine and MAGIC; these intervals are conditional on each fitted model and split. The API and React dashboard remain planned.

## Getting started

1. Install the dependencies listed in [research/requirements.txt](research/requirements.txt).
2. Obtain the Wine Quality dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) and place the source files under `research/datasets/raw/` using the filenames expected by the data loader.
3. From the `research/` directory, run `python experiments/run_reliability_experiments.py` to train the reliability estimator and write its inference bundle.
4. Run `python experiments/run_integrated_verification.py` to use estimated error probabilities for verification action selection and offline outcome evaluation.

See [Milestone 4 integration notes](research/docs/milestone4-1-integration.md), the [Cross-dataset evidence audit](research/docs/cross-dataset-evidence-audit.md), [Paired bootstrap uncertainty protocol](research/docs/paired-bootstrap-uncertainty.md), [Research Plan](docs/research-plan.md), and [Architecture](docs/architecture.md) for assumptions, limitations, and the proposed sequence of work.

## License

No license has been selected yet. Until one is added, the default copyright rules apply.