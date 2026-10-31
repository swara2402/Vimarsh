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

The saved base models used for inference are the uncalibrated models corresponding to the uncalibrated out-of-fold probability features used to train the reliability estimator. The primary prediction model is Random Forest (index 1), while the feature extractor retains the original model ordering used during reliability training.

After generating the bundle, run `python experiments/run_integrated_verification.py` from the research directory. The runner loads the same deterministic Wine Quality split, computes model probabilities, invokes this bridge, executes the selected verification actions, and evaluates outcomes only after selection.

## Current implementation boundary

This is an integrated research experiment, not a production deployment. It still depends on the existing Wine Quality split and configured action-effectiveness assumptions. Human review remains simulated, and the evidence provider is disabled because no real provider is configured. Cross-split duplicate records remain a limitation.

## Validation

The tests cover row alignment, probability validation, estimator output validation, feature-schema compatibility, forwarding estimated error probabilities to the verification engine, primary-model confidence alignment, and joblib serialization of calibrated model wrappers.

## Next work

- Run the full artifact-training and integrated experiment against the actual dataset in a reproducible environment.
- Compare integrated policies against confidence-only, entropy, and always-accept baselines on a duplicate-grouped split.
- Report risk, coverage, action costs, calibration, and uncertainty across multiple seeds.
