# Milestone 4.1: Reliability-to-Verification Integration

## Objective

Connect the existing fitted reliability feature extractor and error-probability estimator to the verification action engine. The action engine must receive estimated probabilities of base-model error rather than synthetic random values.

## Inference-time contract

The integration accepts:

- A batch of input feature rows.
- Predictions from the primary model.
- Probability matrices from the primary model and independently trained comparison models.
- Harm estimates defined by the experiment.

It then:

1. Validates probability matrix dimensions, finite values, bounds, and row sums.
2. Extracts uncertainty, disagreement, input-quality, and distribution-shift features.
3. Aligns extracted columns to the fitted reliability estimator's recorded feature order.
4. Estimates one error probability per input row.
5. Passes those estimates and primary-model confidence values into the existing verification engine.

Ground-truth labels are intentionally absent from the inference-time integration API. They remain reserved for offline evaluation after action selection.

## Artifact production and integrated runner

The reliability experiment now writes a joblib bundle containing the fitted feature extractor, reliability estimator, base models, and preprocessor. The base-model wrappers were moved to module scope so they can be serialized. The artifact also stores a relative model-directory path and feature metadata.

The saved base models used for inference are the uncalibrated models corresponding to the uncalibrated out-of-fold probability features used to train the reliability estimator. The primary prediction model is Random Forest (index 1), while the feature extractor retains the original model ordering used during reliability training. Out-of-fold model predictions, the reliability-estimator train/calibration holdout, and the final train/validation/test partitions now use grouped stratification over exact feature rows. A separate grouped holdout is used to fit calibrated comparison models, so calibration observations are not used to fit those comparison estimators.

After generating the bundle, run `python experiments/run_integrated_verification.py` from the research directory. The runner loads the same deterministic Wine Quality split, computes model probabilities, invokes this bridge, executes the selected verification actions, and evaluates outcomes only after selection.

## First grouped dataset-backed policy comparison

The GitHub Actions run on 2026-10-31 completed both the research test job and the Wine Quality end-to-end job successfully. The integrated test partition contained 929 observations. The grouped split audit identified 5,320 unique exact-feature groups and zero groups shared across train, validation, and test. Class prevalence was 63.3% in each partition.

The primary Random Forest's test error rate under the always-accept baseline was 23.04%. The five policies were evaluated on the same test rows with the same configured budget, action costs, harm estimates, and executor settings. Only the error-probability ranking signal changed.

| Policy | Coverage | Error rate among covered predictions | Net corrections |
|---|---:|---:|---:|
| Reliability-driven | 28.31% | 17.49% | 0 |
| Confidence-only | 28.31% | 10.27% | 0 |
| Predictive entropy | 10.98% | 2.94% | 0 |
| Ensemble disagreement | 82.99% | 17.64% | 0 |
| Always accept | 100.00% | 23.04% | 0 |

### Interpretation

This run does not demonstrate that the learned reliability score improves action selection over the confidence-only baseline. At matched coverage, confidence-only had a lower observed error rate among covered predictions (10.27%) than reliability-driven selection (17.49%). Predictive entropy selected a smaller subset with a lower covered error rate, while ensemble disagreement covered a much larger fraction of the test set. These policies therefore expose different coverage/error trade-offs; this single run does not establish one universally superior policy.

No policy produced net corrections in this run. The experiment supports the implementation and comparison protocol, not a claim of demonstrated risk reduction or superior reliability-driven performance. The result is one deterministic grouped split and should be repeated across multiple seeds before drawing stronger conclusions.

The GitHub Actions run uploaded the generated metrics, report, integrated outcome JSON, and inference bundle as the `vimarsh-milestone4-experiment-results` artifact (14-day retention).

## Current implementation boundary

This is an integrated research experiment, not a production deployment. Exact duplicate feature rows are kept within one partition, including during out-of-fold and reliability-estimator calibration splits. Human review remains simulated, the evidence provider is disabled because no real provider is configured, and action-effectiveness factors remain assumptions rather than measured causal effects. A single Wine Quality benchmark does not establish cross-domain generalization.

## Validation

The dataset-backed GitHub Actions run completed successfully on the current Milestone 4.2 pull-request head. Both jobs passed: the research test suite (including compilation and grouped-splitting, model-alignment, integration, calibration-serialization, outcome-validation, and batch-feature tests) and the end-to-end Wine Quality workflow. The workflow uploaded the generated result files as a downloadable artifact.

## Next work

- Repeat the complete training and policy comparison across multiple random seeds and report mean and dispersion for each policy.
- Validate on at least one additional tabular dataset.
- Measure real action effectiveness before making claims about practical risk reduction.
