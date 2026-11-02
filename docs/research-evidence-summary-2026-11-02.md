# VIMARSH research evidence summary

**Evidence snapshot:** 2 November 2026  
**Reproducibility source:** GitHub Actions research run `41438973113`, commit `8101d4ec6098e6de1dc9946a69159f8a2178d4e5`. All nine research jobs in that run completed successfully. Calibration/safety audit run `41438973085` also completed successfully. The run produced downloadable artifacts for Wine, MAGIC, HTRU2, independent datasets, measured alternate-model outcomes, primary-model sensitivity, calibration, and paired action-benefit analysis.

This document summarizes benchmark evidence, including negative findings. The research run produced artifacts for Wine, MAGIC, HTRU2, independent datasets, measured alternate-model outcomes, primary-model sensitivity, and paired action-benefit analysis; the separate calibration audit produced its own calibration artifact. It is not a claim of clinical, financial, or production validation.

## Executive interpretation

VIMARSH has a working research pipeline for prediction reliability estimation, selective acceptance/abstention, action-outcome recording, contextual action-benefit estimation, and constrained allocation. The current serving API still uses a transparent baseline policy; the contextual estimator and allocator are not yet the live serving policy.

The experiments show that reliability signals can be useful, but **they do not show consistent superiority over confidence-only baselines**. Results vary by dataset, primary model, operating point, and scenario assumptions. The first contextual action-benefit experiment abstained on every held-out case, so it did not establish a useful selection policy.

## Main benchmark results

All values below are means across seeds 17, 42, and 73. Coverage/error comparisons use validation-selected operating points, with thresholds fixed before test labels are inspected.

### Wine Quality: nominal 75% acceptance

| Ranking policy | Test coverage | Error among accepted |
|---|---:|---:|
| Reliability-driven | 75.96% ± 1.37 pp | 17.75% ± 1.52 pp |
| Confidence-only | 75.17% ± 1.11 pp | 16.72% ± 2.28 pp |
| Predictive entropy | 75.17% ± 1.11 pp | 16.72% ± 2.28 pp |
| Ensemble disagreement | 83.28% ± 0.90 pp | 19.47% ± 1.30 pp |

At this operating point, reliability-driven selection did **not** reduce accepted-set error relative to confidence-only. Predictive entropy matches confidence-only in this binary setup because the rankings are closely related. The disagreement policy accepted more cases but also had a higher accepted error rate.

### MAGIC Gamma Telescope: nominal 75% acceptance

| Ranking policy | Test coverage | Error among accepted |
|---|---:|---:|
| Reliability-driven | 75.67% ± 1.06 pp | 7.24% ± 0.43 pp |
| Confidence-only | 74.53% ± 0.99 pp | 7.04% ± 0.36 pp |
| Predictive entropy | 74.53% ± 0.99 pp | 7.04% ± 0.36 pp |
| Ensemble disagreement | 82.62% ± 0.32 pp | 9.97% ± 0.41 pp |

The reliability-driven result is close to confidence-only but has a slightly higher accepted error. This does not demonstrate a clear error advantage. MAGIC has 19,020 rows and 10 features; the benchmark uses a constant 5/10 harm assumption because the dataset contains no decision-specific harm labels.

### HTRU2: budgeted verification scenario

HTRU2 contains 17,898 rows and 8 features, with approximately 9.15% positive pulsar cases. The experiment uses a hypothetical 10% review budget, asymmetric false-negative/false-positive harm weights, a fixed review cost, and an assumed review-effectiveness factor. These are scenario settings, not dataset-provided labels or measured reviewer outcomes.

At exact matched capacity (9.97% reviewed), mean simulated net-benefit units were:

| Ranking policy | Simulated net benefit |
|---|---:|
| Reliability-driven | 22.92 |
| Confidence-only | 49.85 |
| Predictive entropy | 47.98 |
| Ensemble disagreement | -19.75 |
| Random | -47.75 |

The reliability ranking did not outperform confidence or entropy under the declared scenario. The separate positive-net-benefit policy used fewer reviews on average, but its mean simulated net benefit (42.37) was also below confidence cost-sensitive selection (63.58). Validation-threshold diagnostics can exceed the nominal cap when scores tie; the exact matched-capacity results are the appropriate fixed-budget comparison.

## Measured alternate-model substitution benchmark

A separate held-out Wine Quality experiment executes a Random Forest alternate model on a fixed 15% action budget, with Logistic Regression as the primary model. Across three seeds, mean net corrections were:

| Routing policy | Mean corrections | Mean introduced regressions | Mean net corrections | Mean cost units per net correction |
|---|---:|---:|---:|---:|
| Model disagreement | 82.7 | 53.0 | 29.7 | 7.50 |
| Low primary confidence | 38.3 | 22.0 | 16.3 | 18.49 |
| Random matched budget | 14.0 | 7.0 | 7.0 | 35.28 |

Each policy selected 139 of 929 test cases per seed. Importantly, both models' predictions were generated for every test row before the routing masks were applied. This is therefore an offline paired substitution/counterfactual comparison, not a compute-saving routing implementation; alternate-model inference was not limited to the selected 15%. It is **not** a human-review experiment, a real-world causal estimate, or measured monetary/latency cost. It does not mean the alternate model is always better.

## Primary-model and independent-dataset sensitivity

The MAGIC primary-model sensitivity experiment evaluated Logistic Regression, Random Forest, and Gradient Boosting separately. Reliability-driven selection performed better than confidence-only for some model-family comparisons and slightly worse for others. For example, at the validation-selected operating point, the Random Forest reliability arm had 75.35% coverage and 7.19% accepted error, versus 74.53% and 7.04% for confidence-only. The Gradient Boosting reliability arm had 75.72% coverage and 6.62% accepted error, versus 74.56% and 5.65% for confidence-only.

Additional Banknote and Ionosphere benchmarks show that results are dataset-dependent. Banknote is close to ceiling accuracy for several model families, making accepted-error comparisons less discriminating. Ionosphere results vary by primary model: reliability selection improves some comparisons but not all. These experiments support continued cross-dataset testing, not a universal generalization claim.

## Calibration and uncertainty audit

On the Breast Cancer Wisconsin benchmark, calibration methods traded off different metrics. For Logistic Regression, isotonic calibration reduced mean ECE from 0.0344 (raw) to 0.0279, but Brier score increased from 0.0184 to 0.0217 and log loss increased from 0.0670 to 0.3553. For Random Forest, isotonic reduced ECE from 0.0531 to 0.0362, while Brier score and log loss also worsened. Sigmoid calibration was worse on ECE in this run.

Therefore calibration must be judged with multiple metrics and fresh validation, not ECE alone. These are public tabular benchmark measurements, not clinical validation.

## Contextual action-benefit experiment: negative result

The three-seed paired contextual action-benefit experiment selected **zero of 929 test cases in every seed**. The confidence-only and random matched-budget baselines also selected zero because their budget was matched to the estimator's selected count, so that comparison is non-informative.

The follow-up diagnostics identify the reason: all 2,787 test queries abstained because the conservative benefit diagnostic was not positive, **not** because effective sample support was too small. Mean effective sample sizes were approximately 441.0, 426.5, and 439.2 for seeds 17, 42, and 73, well above the configured minimum of 10. Mean weighted benefits were 0.0024, -0.0062, and 0.0011, while mean uncertainty penalties were 0.1166, 0.1186, and 0.1168. The resulting mean lower-bound diagnostics were -0.1142, -0.1247, and -0.1158. In short, observed paired benefit was close to zero relative to the uncertainty penalty.

This is a valid negative result and a current limitation. It is not evidence that the contextual estimator is superior. The correct next step is to improve the action-effect signal or collect more informative genuine paired outcomes, then evaluate any pre-specified estimator change on a fresh held-out test. Do not weaken the gate merely to force non-zero selections.

## What is implemented versus what is not

Implemented research/application components include grouped and out-of-fold evaluation, reliability estimation, calibration audits, selective-policy comparisons, measured offline alternate-model substitution, a hash-chained outcome ledger, a contextual benefit estimator, a constrained batch allocator, and a FastAPI/React research preview.

Still not operational:

- Human-review queue/provider and real reviewer outcome collection.
- Additional-evidence acquisition providers.
- Real-world randomized paired action-outcome data.
- A production policy integration that routes the live serving API through the contextual allocator.
- Production identity/roles, durable transactional multi-worker storage, deployment hardening, monitoring, and domain-specific safety validation.

## Safe conclusion for a presentation

VIMARSH demonstrates an end-to-end research prototype that connects model reliability signals with risk-aware, constraint-based verification recommendations and a feedback-data research pipeline. The evidence is mixed and dataset-dependent. The work establishes an implementable research direction, not universal superiority, causal real-world risk reduction, production readiness, or patentability.
