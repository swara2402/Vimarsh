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

## Current implementation boundary

This milestone adds a reusable inference bridge. It does not, by itself, train or persist the reliability estimator, guarantee that a compatible fitted extractor/estimator pair exists, or replace every legacy experiment runner. The training workflow must supply artifacts fitted without test-label leakage. Simulated evidence and human review remain explicitly simulated.

## Validation

The integration tests cover row alignment, probability validation, estimator output validation, feature-schema compatibility, and forwarding the estimated error probabilities to the verification engine.

## Next work

- Build a reproducible artifact-training command that produces a compatible feature extractor and reliability estimator from out-of-fold training predictions.
- Add an experiment runner that loads those artifacts and uses this bridge for policy evaluation.
- Compare integrated policies against confidence-only, entropy, and always-accept baselines on a duplicate-grouped split.
- Report risk, coverage, action costs, calibration, and uncertainty across multiple seeds.
