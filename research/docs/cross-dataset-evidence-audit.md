# Milestone 47: Cross-dataset evidence and comparability audit

## Purpose

This audit records what the current Wine Quality, MAGIC Gamma Telescope, and
HTRU2 experiments do and do not establish. It deliberately separates
classification/selection metrics from simulated verification utility. Metrics
from different datasets or scenario assumptions must not be pooled into one
headline score.

## Evidence inventory

| Benchmark | Repeated seeds | Main evaluation | Operating-point diagnostic | Consequence assumptions |
|---|---:|---|---|---|
| Wine Quality (red + white) | 17, 42, 73 | Five-policy integrated verification engine; coverage, observed error, net corrections | Added in milestone 47: validation-selected nominal 75% acceptance coverage | Simulated human review; action effectiveness and generated harm estimates are scenario/model assumptions |
| MAGIC Gamma Telescope | 17, 42, 73 | Five-policy integrated verification engine | Validation-selected nominal 75% acceptance coverage for four ranking policies | Constant harm 5/10; simulated review; assumed action effectiveness |
| HTRU2 | 17, 42, 73 | Hard 10% review-cap policy, positive-net-benefit eligibility, and exact matched-capacity ranking | Validation-selected nominal 10% review threshold | Hypothetical false-negative/false-positive consequence weights, review cost, and review effectiveness |

## What is comparable

The shared validation-selected coverage diagnostic now uses one implementation
for Wine and MAGIC:

- A risk score is produced independently for each policy.
- The threshold is selected from validation risk scores only, using the
  `higher` quantile at a nominal 75% acceptance target.
- Lower-risk test observations at or below the fixed threshold are accepted.
- The threshold is applied unchanged to the held-out test scores.
- Test labels are used only after the accepted mask is fixed, to calculate the
  observed error rate on accepted observations.
- Actual test coverage is reported. Ties at the threshold and score-distribution
  shift can make it differ from 75%.

The same utility is shared by the MAGIC and Wine experiment runners to avoid
subtle implementation drift. HTRU2's 10% diagnostic has the opposite selection
direction: it selects high-risk cases for review, not low-risk cases for
acceptance. It therefore remains a separately named review-rate diagnostic.

## What must not be pooled

The three datasets are not interchangeable experimental units. They differ in
sample size, class prevalence, feature distributions, target meaning, primary
model context, and evaluation setup. More importantly, the simulated action
utility metrics do not share a common empirical cost scale:

- Wine's integrated engine uses generated harm estimates and configured action
  costs/effectiveness.
- MAGIC uses a constant 5/10 harm assumption because the dataset has no
  decision-specific harm labels.
- HTRU2 uses hypothetical asymmetric consequence weights, a fixed per-review
  cost, and a specified reviewer-effectiveness assumption.

Consequently, do not average observed error rates, net corrections, weighted
loss, or simulated net benefit across datasets. Do not present the three
benchmarks as a pooled estimate of risk reduction. Report each benchmark and
its scenario assumptions separately.

## Findings that must remain visible

- MAGIC's validation-selected 75% acceptance diagnostic reported mean test
  coverage of 75.67% and accepted-set error of 7.24% for reliability-driven
  selection, versus 74.53% and 7.04% for confidence-only. This is close and
  does not demonstrate a reliability-driven error advantage.
- HTRU2's exact matched-capacity diagnostic found confidence and entropy ahead
  of learned reliability under the declared scenario. Its positive-net-benefit
  operating policy also did not outperform confidence. This negative result is
  part of the evidence, not an implementation defect to conceal.
- Wine's prior action-engine metrics were not a validation-selected matched
  coverage comparison. Milestone 47 adds that diagnostic; the resulting
  per-seed metrics must be read from the CI artifact before updating any
  numeric conclusion in this document.
- Three seeds and population standard deviations provide descriptive
  repeatability evidence, not inferential confidence intervals or proof of
  statistical significance.
- Human review remains simulated, evidence-provider actions are disabled, and
  action-effectiveness settings are assumptions rather than measured causal
  effects.

## Leakage and reproducibility controls

The shared diagnostic rejects empty, non-finite, or non-one-dimensional score
arrays and invalid target coverage. It does not accept labels as an input to
threshold selection. The separate metric function validates prediction/label
alignment and accesses labels only after the accepted mask is fixed.

The CI workflow runs unit tests for threshold selection and includes the
diagnostic in the Wine end-to-end artifact. The MAGIC experiment calls the same
shared utility. HTRU2 retains its separately tested high-risk review threshold
and hard-cap selection helpers.

## Milestone 48: paired uncertainty intervals

Wine and MAGIC now report paired bootstrap comparisons of each ranking policy
against confidence-only at the validation-selected nominal 75% acceptance
operating point. The bootstrap resamples exact-duplicate feature groups and
uses the same resampled groups for both policies. It reports candidate-minus-
reference differences in coverage and accepted-set error rate with percentile
95% intervals from 1,000 replicates per seed. Thresholds are fixed using
validation scores before test labels are used for scoring.

These intervals are conditional on the fitted model, threshold, and test split.
They quantify test-sample uncertainty, not variation from retraining. The
three-seed summaries remain descriptive. Do not pool interval estimates across
Wine and MAGIC or interpret an interval spanning zero as proof of equivalence.
See the [paired bootstrap protocol](paired-bootstrap-uncertainty.md) for
implementation and interpretation details.

## Interpretation boundary and next evidence

The current evidence supports a careful, benchmark-specific account of
reliability-aware verification, including negative comparative findings. It
does not yet support universal superiority, causal harm reduction, deployment
safety, cross-domain generalization beyond these tabular datasets, or
patentability.

The next high-value experiments are paired uncertainty intervals for
pre-specified per-dataset contrasts, broader independent datasets/model
families, controlled shift tests, and measured review outcomes or randomized
review evaluation. Any inferential analysis should preserve per-seed pairing
and should not treat repeated rows or seeds as independent external datasets.


## Milestone 49: controlled covariate-shift stress test

The MAGIC experiment now evaluates frozen policies on synthetic feature-shifted
test inputs at 0.25, 0.50, and 1.00 training-standard-deviation severities.
The perturbation is an alternating signed feature-wise mean offset plus small
Gaussian noise. Models and thresholds selected on the original validation
partition remain fixed. Per-policy coverage, accepted-set error, overall
accuracy, and paired candidate-versus-confidence-only bootstrap intervals are
reported for each seed and severity. Bootstrap clusters are inherited from the
original unshifted test features, preserving exact-duplicate membership despite
independent row-wise perturbation noise.

This is one synthetic perturbation family, not a model of natural deployment
drift. Labels remain unchanged, and results are conditional on the selected
perturbation, model, split, and fixed thresholds. The experiment does not
establish broad robustness or real-world safety. See
[controlled-shift-robustness.md](controlled-shift-robustness.md).
