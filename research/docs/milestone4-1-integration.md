# Milestone 4.1: Reliability-to-Verification Integration

## Objective

Connect the fitted reliability feature extractor and error-probability estimator to the verification action engine. The engine receives estimated probabilities of base-model error rather than synthetic random values.

## Inference-time contract

The integration accepts a batch of input feature rows, primary-model predictions, probability matrices from the primary and comparison models, and experiment-defined harm estimates. It validates probability matrices, extracts uncertainty/disagreement/input-quality/distribution-shift features, aligns them to the estimator's feature order, estimates one error probability per row, and passes those estimates into the verification engine.

Ground-truth labels are absent from the inference-time integration API. They are reserved for offline evaluation after action selection.

## Artifact production and leakage controls

The reliability experiment writes a joblib bundle containing the fitted feature extractor, reliability estimator, uncalibrated base models, preprocessor, and feature metadata. Random Forest is the primary model (index 1). Ensemble disagreement still uses all base models.

Final train/validation/test partitions, out-of-fold folds, and reliability-estimator train/calibration holdouts use grouped stratification over exact feature rows. A separate grouped holdout is used to fit calibrated comparison models. Inference uses uncalibrated base models so probability features match the scale used to train the reliability estimator.

## Multi-seed five-policy evaluation

The GitHub Actions run on 2026-10-31 completed successfully for all three jobs: research tests, the three-seed policy comparison, and the Wine Quality end-to-end workflow. The multi-seed artifact contains independent runs for seeds 17, 42, and 73, each with a 929-observation grouped test partition.

The policies use the same test partition within each seed, harm estimates, configured budget, action costs, and executor settings. Only the error-probability ranking signal changes. Coverage, observed error rate among covered predictions, and net corrections are summarized as arithmetic means and population standard deviations across the three seeds.

| Policy | Mean coverage | Coverage SD | Mean covered error rate | Error-rate SD | Mean net corrections |
|---|---:|---:|---:|---:|---:|
| Reliability-driven | 28.31% | 0.00 pp | 15.34% | 1.53 pp | 0 |
| Confidence-only | 28.31% | 0.00 pp | 9.76% | 0.47 pp | 0 |
| Predictive entropy | 11.09% | 0.40 pp | 4.80% | 1.90 pp | 0 |
| Ensemble disagreement | 83.28% | 0.90 pp | 19.47% | 1.30 pp | 0 |
| Always accept | 100.00% | 0.00 pp | 23.75% | 0.94 pp | 0 |

### Interpretation

At matched mean coverage of 28.31%, confidence-only outperformed reliability-driven selection in all three seeds. The mean covered error rate was 9.76% for confidence-only and 15.34% for reliability-driven selection, a 5.58 percentage-point difference. The reliability-driven policy therefore has not demonstrated an advantage over the simple confidence baseline on this benchmark.

Predictive entropy selected a much smaller subset and had the lowest mean covered error rate (4.80%), illustrating a different coverage/error trade-off rather than a direct matched-coverage win. Ensemble disagreement retained high coverage but also a high covered error rate. No policy produced net corrections in any seed.

The first single-seed run and the three-seed results should both remain in the research record. The multi-seed findings are negative evidence about the current reliability score, not a reason to tune thresholds against the test results and report the same test set as validation. Any feature or model changes should be developed on training/validation data and assessed on a fresh held-out evaluation.

## Implementation boundary and limitations

This is a research experiment, not a production deployment. Exact duplicate feature rows are isolated across partitions, but the three seeds still use the same Wine Quality dataset and do not establish cross-domain generalization. Human review remains simulated; evidence-provider actions are disabled; and action-effectiveness factors are assumptions, not measured causal effects. The population standard deviations across three seeds are descriptive and are not confidence intervals. No real-world risk-reduction or correction claim is supported.

## Validation and artifacts

The successful GitHub Actions run is [run 41218257057](https://github.com/swara2402/Vimarsh/actions/runs/41218257057). All three jobs passed. The `vimarsh-multiseed-policy-results` artifact includes the aggregate summary, per-seed integrated outcome JSON files, and seed-specific inference bundles. The artifact is retained until 2026-11-30. The workflow also completed the separate Wine Quality end-to-end job and uploaded its experiment artifact.

## Next research steps

1. Add a second, non-Wine tabular benchmark with a documented source, license, target definition, and preprocessing protocol. Prefer a non-high-stakes dataset with a sufficiently large sample and a multi-class or binary target supported by the current model interface.
2. Re-run the same grouped split, leakage controls, and five-policy comparison without tuning on the final test set.
3. Diagnose why the learned reliability score loses to confidence-only at matched coverage using validation-only feature ablations and calibration diagnostics.
4. Measure real action effectiveness before making claims about practical risk reduction.


## Second-dataset benchmark: UCI MAGIC Gamma Telescope

The grouped multi-seed MAGIC benchmark completed successfully in GitHub Actions run [41224170086](https://github.com/swara2402/Vimarsh/actions/runs/41224170086). The workflow passed the research test suite, MAGIC grouped experiment tests, and the MAGIC policy-comparison job; the separate Wine end-to-end job also passed. The MAGIC artifact contains one result JSON per seed and the aggregate summary.

The benchmark uses 19,020 observations, ten continuous features, and a binary gamma/hadron target. Exact feature duplicates are kept within partitions. Each seed has 2,718 test observations, with class prevalence near 64.8% in each split.

| Policy | Mean coverage | Coverage SD | Mean observed error rate | Error-rate SD | Mean net corrections |
|---|---:|---:|---:|---:|---:|
| Reliability-driven | 78.27% | 0.54 pp | 7.76% | 0.32 pp | 0 |
| Confidence-only | 75.50% | 0.00 pp | 14.91% | 0.59 pp | 0 |
| Predictive entropy | 51.09% | 0.47 pp | 8.16% | 0.35 pp | 0 |
| Ensemble disagreement | 82.62% | 0.32 pp | 9.97% | 0.41 pp | 0 |
| Always accept | 100.00% | 0.00 pp | 14.02% | 0.48 pp | 0 |

On MAGIC, reliability-driven selection recorded a lower mean observed error rate and slightly higher mean coverage than confidence-only. This is encouraging but not a matched-coverage comparison, so it does not establish superior ranking quality at a common operating point. No policy produced net corrections. The MAGIC results are a second tabular benchmark, not broad cross-domain generalization.

The dataset has no decision-specific harm labels; a constant 5.0/10 harm score was used as a disclosed scenario assumption. Human review was simulated, evidence-provider actions were disabled, and action-effectiveness factors were assumptions rather than measured causal effects. These limitations prevent real-world harm-reduction claims.

Detailed protocol, results, and limitations: [MAGIC Gamma benchmark protocol](magic-gamma-benchmark-protocol.md). Artifact: [vimarsh-magic-gamma-policy-results](https://github.com/swara2402/Vimarsh/actions/runs/41224170086/artifacts/13051567045).

## Revised research priorities

1. Add a validation-only threshold-selection and matched-coverage comparison so ranking quality can be compared at a common operating point without test-set tuning.
2. Diagnose reliability calibration and feature contributions using validation-only ablations across both benchmarks.
3. Evaluate additional datasets or repeated temporal/domain splits before making broader generalization claims.
4. Measure real action effectiveness before making claims about practical risk reduction.


## Completed validation-selected matched-coverage diagnostic

The updated three-seed workflow [41225143359](https://github.com/swara2402/Vimarsh/actions/runs/41225143359) passed all four jobs: research tests, MAGIC three-seed policy comparison, Wine multi-seed policy comparison, and Wine end-to-end training. The refreshed MAGIC artifact is [vimarsh-magic-gamma-policy-results](https://github.com/swara2402/Vimarsh/actions/runs/41225143359/artifacts/13051212563).

A validation-only threshold was selected for each ranking policy at a nominal 75% coverage target, then applied unchanged to the test partition. Across seeds 17, 42, and 73:

| Ranking policy | Mean actual test coverage | Coverage SD | Mean test observed error | Error SD |
|---|---:|---:|---:|---:|
| Reliability-driven | 75.67% | 1.06 pp | 7.24% | 0.43 pp |
| Confidence-only | 74.53% | 0.99 pp | 7.04% | 0.36 pp |
| Predictive entropy | 74.53% | 0.99 pp | 7.04% | 0.36 pp |
| Ensemble disagreement | 82.62% | 0.32 pp | 9.97% | 0.41 pp |

Reliability-driven coverage was 1.14 percentage points higher than confidence-only, but its observed error rate was 0.20 percentage points higher. This is a close result and does not show an error-rate advantage for the learned reliability score at this nominal operating point. Confidence-only and predictive entropy selected identical accepted sets in these runs. Ensemble disagreement exceeded the nominal target due to ties at its validation-selected threshold. Actual coverage is therefore reported rather than represented as exactly 75%.

This validation-selected diagnostic is separate from the action-engine results and does not prove statistically significant or general superiority. No test-set threshold tuning was used. The action-engine simulation still produced zero net corrections for every policy. The earlier MAGIC caveats remain unchanged: human review is simulated, evidence provider disabled, action-effectiveness factors assumed, and the harm score constant rather than domain-derived.


## Validation-only reliability feature ablation

The follow-up workflow [41227625014](https://github.com/swara2402/Vimarsh/actions/runs/41227625014) passed all four jobs: research tests, MAGIC Gamma policy comparison, Wine multi-seed comparison, and Wine end-to-end training. Its MAGIC artifact is [vimarsh-magic-gamma-policy-results](https://github.com/swara2402/Vimarsh/actions/runs/41227625014/artifacts/13052312639).

A feature ablation was evaluated on the validation partition only for seeds 17, 42, and 73. The four variants add feature groups progressively: (A) uncertainty only, (B) uncertainty plus ensemble disagreement, (C) plus input-quality features, and (D) full features including distribution-shift features. The table reports arithmetic means and population standard deviations across the three seeds.

| Feature set | Features | Mean ROC AUC | ROC AUC SD | Mean PR AUC | PR AUC SD | Mean Brier | Mean log loss | Mean ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Uncertainty only | 4 | 0.6854 | 0.0257 | 0.2588 | 0.0285 | 0.1168 | 0.3902 | 0.0092 |
| B. + disagreement | 10 | 0.7589 | 0.0144 | 0.3585 | 0.0168 | 0.1081 | 0.3607 | 0.0272 |
| C. + input quality | 18 | 0.7771 | 0.0081 | 0.3872 | 0.0214 | 0.1052 | 0.3509 | 0.0270 |
| D. Full features | 21 | 0.7903 | 0.0097 | 0.4040 | 0.0220 | 0.1035 | 0.3454 | 0.0310 |

Across these three validation runs, adding disagreement, input-quality, and distribution-shift feature groups progressively improved mean ROC AUC and PR AUC and reduced mean Brier score and log loss. However, ECE increased from the uncertainty-only baseline, so better ranking and proper scoring did not translate into better expected calibration error under this metric. The full feature set had the strongest mean validation ranking scores, but also the highest mean ECE. These are validation-only model-selection diagnostics, not independent test-set confirmation; they do not establish causal feature importance, broad generalization, or a patentable contribution. The ablation should be followed by a pre-specified evaluation on a fresh held-out dataset or domain before making stronger claims.
