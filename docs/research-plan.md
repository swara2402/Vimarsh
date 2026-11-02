# VIMARSH Research Plan

## 1. Problem statement

Machine-learning systems can produce confident predictions that are nevertheless incorrect, poorly calibrated, based on degraded inputs, or made under distribution shift. A confidence threshold alone does not determine what should happen next.

VIMARSH investigates whether a system can combine reliability evidence, estimated decision impact, verification costs, and hard safety constraints to choose a suitable next action.

## 2. Research question

Can a system learn to select the best feasible verification action for a prediction by estimating the expected reduction in decision risk, while respecting explicit safety and resource constraints?

## 3. Working hypothesis

A hybrid policy combining a learned action evaluator with a hard safety-constraint layer may achieve a more useful risk-cost trade-off than fixed confidence thresholds or unconstrained learned action selection.

This is a hypothesis to test, not a result.

## 4. Proposed state representation

A preliminary state may be represented as:

```text
s = [uncertainty, disagreement, input_quality,
     distribution_shift, harm, resource_constraints]
```

The final feature set must be justified empirically and documented. Avoid using unavailable information at prediction time.

## 5. Candidate actions

- Accept: retain the original prediction.
- Request evidence: acquire an additional feature or observation only when it is actually obtainable.
- Alternate model: obtain an independent model prediction.
- Human review: route the case to a reviewer. Simulated review must be labelled as simulation.
- Abstain: do not issue an automated decision.

The action set may be extended only when the action has a well-defined execution mechanism and measurable cost/outcome.

## 6. Preliminary decision formulation

Let:

- (P_e(s)) be the estimated probability that the original prediction is incorrect.
- (H(s)) be the consequence associated with an incorrect decision.
- (hat R(s,a)) be estimated remaining risk after action (a).
- (hat C(s,a)) be estimated action cost.
- (lambda) be a cost penalty.
- (A_{safe}(s)) be the actions permitted by hard safety and resource constraints.

A preliminary baseline risk is:

```text
R0(s) = P_e(s) * H(s)
```

A candidate utility is:

```text
U(s, a) = R0(s) - R_hat(s, a) - lambda * C_hat(s, a)
```

The selected action is:

```text
a* = argmax U(s, a), for a in A_safe(s)
```

If no feasible action has positive worthwhile utility, the system should abstain or follow a predefined safe fallback.

This formulation is provisional. Risk units, cost normalization, action outcomes, and the meaning of (H) must be specified before experiments.

## 7. Initial datasets and models

Potential tabular benchmarks:

- Wine Quality
- Breast Cancer Wisconsin
- Covertype

Candidate models:

- Logistic Regression
- Random Forest
- Gradient Boosting or XGBoost

These datasets are for controlled research benchmarks only. The Breast Cancer Wisconsin dataset is not a basis for clinical claims or medical deployment.

## 8. Reliability signals to investigate

- Predictive confidence and calibrated uncertainty.
- Predictive entropy.
- Disagreement among independently trained models.
- Missing, corrupted, or low-quality input features.
- Out-of-distribution or distribution-shift indicators.
- Decision impact and resource constraints.

Not every signal will necessarily improve the system. Use ablations to test its contribution.

## 9. Experimental conditions

Introduce controlled perturbations such as:

- Missing features.
- Random feature noise.
- Corrupted values.
- Distribution shifts.
- Conflicting model predictions.
- Unfamiliar examples.

Document perturbation generation, severity, random seeds, and affected features.

## 10. Verification simulation

For the initial offline prototype:

- Additional evidence may reveal a genuinely withheld feature, if the experiment defines that feature as obtainable.
- An alternate-model action runs a separately trained model.
- Human review may be represented by a clearly labelled simulated adjudication mechanism before a real reviewer interface exists.
- Abstention produces no automated prediction.

Ground-truth labels are for offline evaluation only. They must not leak into the inference-time policy or action-selection features.

## 11. Baselines

Compare the proposed framework with:

- Fixed confidence threshold.
- Predictive entropy threshold.
- Ensemble disagreement.
- Out-of-distribution detector.
- Selective prediction.
- Fixed escalation rule.
- Learned-only action policy.
- Hybrid constrained policy.

An oracle policy may be reported as an analysis-only upper bound, clearly separated from deployable methods.

## 12. Metrics

Report multiple dimensions:

- Accuracy and selective accuracy.
- Coverage and abstention rate.
- Calibration: Expected Calibration Error (ECE) and Brier score.
- Risk before and after verification.
- Verification cost and action frequency.
- Risk reduction per unit cost.
- Action regret, where a defensible offline definition is available.

Always report risk together with coverage. A system that rejects every prediction must not appear successful merely because it avoids incorrect automated outputs.

## 13. Data protocol

- Create separate training, validation, and test partitions.
- Fit preprocessing and calibration only on permitted training/validation data.
- Keep test labels inaccessible to the policy and all inference-time decisions.
- Use consistent splits across baselines.
- Repeat experiments across multiple random seeds.
- Report confidence intervals and variability.
- Evaluate across corruption levels and, where feasible, across datasets and model families.

## 14. Evaluation risks

Verification outcomes are not automatically causal benefits. Cases sent to human review may be systematically harder than cases accepted automatically. Logged action propensities, inverse-propensity weighting, doubly robust estimators, or carefully designed randomized evaluation may be needed to address selection bias.

Do not claim that an action reduced risk merely because the reviewed subset had better observed outcomes.

## 15. Ablation plan

- V1: Reliability estimator only.
- V2: Reliability estimator plus learned action evaluator.
- V3: Reliability estimator plus hard safety constraints.
- V4: Full hybrid framework.

Compare all variants on the same data splits and conditions.

## 16. Phased implementation

1. Finalize the research question and conduct a literature/prior-art review.
2. Inspect datasets and train baseline classifiers.
3. Measure confidence versus actual correctness.
4. Implement reliability signals and calibration.
5. Define feasible verification actions and their measurable outcomes/costs.
6. Implement the constrained action evaluator.
7. Build offline evaluation and baseline comparisons.
8. Add the API and dashboard after the experimental protocol is stable.
9. Document limitations and assess novelty with qualified research/IP advice.

## 17. Research integrity

This plan is a starting specification. It does not establish novelty, superiority, safety, or patentability. Those require evidence, reproducible experiments, claim-specific prior-art analysis, and, for IP decisions, professional advice.

## 18. Milestone 56: prior-art and IP review

A preliminary prior-art landscape review is maintained in [milestone56-prior-art-ip-landscape.md](../research/docs/milestone56-prior-art-ip-landscape.md). The scan identifies substantial existing work and patent disclosures covering model-disagreement reliability estimation, selective prediction/abstention, learning to defer, verification-based model qualification, cost-aware model selection, and budgeted evidence-driven verification.

Accordingly, broad claims around "reliability-aware prediction verification under a budget" are not treated as novel. A narrower research hypothesis for future testing is outcome-grounded, action-conditioned verification allocation using paired observed action outcomes, conservative benefit estimates, hard resource constraints, and explicit abstention. This is not yet a novelty finding.

Before any patent filing, the project requires real operational action-outcome data, a precise algorithmic formulation, matched-budget comparisons against established baselines, external validation, and claim-focused professional patent review. This desk review is not legal advice, a patentability opinion, or a freedom-to-operate assessment.
