# Milestone 49: controlled covariate-shift robustness

## Purpose

This milestone adds a controlled stress test to the MAGIC Gamma Telescope
benchmark. It asks how the frozen selective policies behave when held-out test
features are perturbed away from the training distribution. It does not claim
that the perturbation represents natural production drift.

## Protocol

For each existing seed (17, 42, 73):

1. Fit preprocessing, base classifiers, and the reliability estimator using
   the existing grouped training protocol.
2. Compute policy risk scores on the original validation partition.
3. Select each policy's nominal 75% acceptance threshold using validation
   scores only.
4. Keep the fitted models and thresholds fixed.
5. Perturb only held-out test features at severities 0.25, 0.50, and 1.00
   training-set standard deviations. The perturbation applies an alternating
   signed per-feature mean offset and small Gaussian noise.
6. Recompute predictions and risk scores on the shifted features.
7. Apply the original validation thresholds unchanged. Evaluate shifted test
   coverage, accepted-set error, and overall accuracy.
8. Report paired candidate-minus-confidence-only bootstrap intervals using
   the same resampled exact-feature groups for both policies.

The perturbation generator is deterministic for a fixed seed and severity.
Zero severity is an identity transform. The stress-test results are emitted
under controlled_shift_robustness in each MAGIC seed JSON artifact.

## Interpretation

- This is a synthetic covariate-shift stress test, not a deployment simulation.
- The labels are not modified by the feature perturbation. The experiment
  therefore measures model behavior when the input representation is altered
  while the benchmark target remains fixed.
- Thresholds are deliberately not recalibrated on shifted test data. Re-selecting
  them would leak information from the evaluation distribution and conceal
  degradation relative to the original operating point.
- Coverage can fall or rise under shift because frozen thresholds are applied
  to changed score distributions.
- A change in accepted-set error is conditional on this particular perturbation,
  fitted model, split, and benchmark. It is not a causal risk-reduction result.
- Bootstrap intervals quantify test-sample uncertainty conditional on the
  fitted model and fixed threshold; they do not quantify training-seed or
  perturbation-family uncertainty.
- The alternating offset is one perturbation family. It does not cover
  rotations, feature-specific missingness, label shift, concept drift, or
  adversarial manipulation.

## Boundaries and next work

Do not claim general robustness from these three severities or one synthetic
perturbation family. The next extension should add independently motivated
perturbation families, severity calibration from domain evidence, additional
independent datasets, and at least one distinct model family. Any such
extension must preserve a frozen validation protocol and report negative
results.
