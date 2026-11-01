# Milestone 48: Paired bootstrap uncertainty for selective-policy comparisons

## Purpose

Milestone 48 adds paired uncertainty intervals to the validation-selected
nominal 75% acceptance comparison on Wine Quality and MAGIC Gamma. It addresses
a limitation of reporting only point estimates and standard deviations across
three training seeds.

## Protocol

For each seed and dataset:

1. Each policy's low-risk acceptance threshold is selected from validation
   risk scores only, using the existing shared threshold diagnostic.
2. The selected threshold is applied unchanged to the test scores.
3. Test labels are accessed only after all acceptance masks are fixed.
4. The confidence-only policy is the predeclared reference.
5. The bootstrap resamples exact-duplicate feature groups, not individual
   duplicate rows. The same sampled groups are used for the reference and
   candidate policy, preserving the paired comparison.
6. Candidate-minus-reference differences are calculated for test coverage and
   accepted-set error rate.
7. Percentile 95% intervals are computed from 1,000 deterministic bootstrap
   replicates per comparison and seed.

The result is stored in each seed's experiment JSON under
`paired_bootstrap_vs_confidence`. Wine's multi-seed report retains the
per-seed interval objects in `per_seed_results`; MAGIC retains them in each
seed run within its multi-seed summary.

## Interpretation

- A positive coverage difference means the candidate accepts more test
  observations than confidence-only.
- A positive accepted-error difference means the candidate's accepted subset
  has a higher observed error rate than confidence-only.
- An interval spanning zero does not establish a directional difference at
  the stated nominal level.
- These intervals are conditional on the fitted models, validation-selected
  thresholds, and observed train/validation/test split. They quantify
  test-sample uncertainty, not model-training uncertainty.
- The separate variation across seeds remains descriptive. Three seeds are
  not enough to make strong inferential claims about training randomness.
- Cluster bootstrap handles exact duplicate feature rows as a resampling unit;
  it does not guarantee independence across all other possible sources of
  dependence.

## Boundaries

The intervals do not repair scenario assumptions in the simulated action
engine. Human review remains simulated, action-effectiveness factors remain
assumptions, and MAGIC's constant harm score remains a neutral scenario
assumption rather than an empirical harm label.

Do not pool interval estimates or policy effects across Wine and MAGIC. Their
targets, sample populations, and simulated utility assumptions differ. Do not
describe a non-significant result as equivalence, and do not claim universal
superiority from an interval that excludes zero on these two benchmarks.

The bootstrap implementation is shared, deterministic under a fixed seed,
and covered by tests for paired identity, cluster resampling, invalid inputs,
and validation-threshold integration.
