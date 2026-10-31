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

## Current implementation boundary

This is an integrated research experiment, not a production deployment. Exact duplicate feature rows are kept within one partition, including during out-of-fold and reliability-estimator calibration splits. Human review remains simulated, the evidence provider is disabled because no real provider is configured, and action-effectiveness factors remain assumptions rather than measured causal effects. A single Wine Quality benchmark does not establish cross-domain generalization.

## Validation

The tests cover row alignment, probability validation, estimator output validation, feature-schema compatibility, primary-model confidence alignment, grouped split isolation, batch/single-sample feature equivalence, and joblib serialization of calibrated model wrappers. GitHub Actions also runs the dataset-backed artifact-training and integrated verification workflow and uploads the generated result files as a downloadable workflow artifact.

## Next work

- Complete and archive the dataset-backed experiment with grouped out-of-fold splits.
- Compare reliability-driven action selection against confidence-only, predictive-entropy, and ensemble-disagreement scoring under the same action costs and budget.
- Repeat the policy comparison across multiple random seeds and then validate on at least one additional tabular dataset.
- Measure real action effectiveness before making claims about practical risk reduction.
