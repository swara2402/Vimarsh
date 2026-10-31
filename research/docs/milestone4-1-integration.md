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
