# MAGIC Gamma Telescope second-dataset experiment

## Status and scope

The MAGIC adapter and parser tests are complemented by an executable grouped,
multi-seed reliability and five-policy evaluation. The benchmark uses the UCI
MAGIC Gamma Telescope dataset (UCI dataset 159), with ten continuous numeric
features and the binary target mapping gamma = 1, hadron = 0. The dataset is
distributed by the UCI Machine Learning Repository under CC BY 4.0.

Source: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

## Experimental protocol

- Seeds: 17, 42, and 73.
- Train/validation/test partitions use stratified grouping by exact feature
  vectors. Duplicate feature vectors remain in one partition, and all rows are
  retained, including groups whose repeated rows may have differing labels.
- Five-fold grouped out-of-fold predictions train the reliability estimator.
- Logistic Regression, Random Forest, and Gradient Boosting are trained in
  each fold. Random Forest (model index 1) is the primary prediction model and
  its OOF errors are the reliability-estimator target.
- The reliability estimator's own train/calibration partition is also grouped
  by exact feature vectors.
- The final grouped test set compares reliability-driven, confidence-only,
  predictive-entropy, ensemble-disagreement, and always-accept policies using
  the same verification budget, action costs, and executor configuration.
- Test labels are used only after action selection to simulate offline outcomes.
  They are not inputs to the policy scores or action-selection step.
- The script writes one JSON result per seed and an aggregate summary to
  `research/results/magic_gamma/`.

## Consequence and simulation assumptions

MAGIC is a telescope-event classification benchmark and does not provide
decision-specific harm labels. The experiment therefore uses a constant harm
score of 5.0/10 for every test observation. This is a common scenario parameter,
not a learned or empirically grounded risk estimate. Human review is simulated;
the configured reviewer accuracy and action-effectiveness factors are
assumptions, not measured causal effects. Evidence-provider actions are
disabled.

## Interpretation boundary

A completed run would provide a second tabular benchmark under the stated
protocol. It would not establish broad cross-domain generalization, production
suitability, real human-review effectiveness, or measured real-world harm
reduction. Results must be read alongside the per-seed metrics and limitations.
No thresholds are selected using final test outcomes.


## Completed results (seeds 17, 42, 73)

GitHub Actions run [41224170086](https://github.com/swara2402/Vimarsh/actions/runs/41224170086) completed the MAGIC three-seed evaluation successfully and uploaded artifact
[vimarsh-magic-gamma-policy-results](https://github.com/swara2402/Vimarsh/actions/runs/41224170086/artifacts/13051567045).
Each seed used 2,718 test observations (8,154 test observations total across the three independent runs). The grouped split retained all 19,020 observations, with train/validation/test sizes of approximately 13,584/2,718/2,718 and closely matched class prevalence (about 64.8%).

| Policy | Mean coverage | Coverage SD | Mean observed error rate | Error-rate SD | Mean net corrections |
|---|---:|---:|---:|---:|---:|
| Reliability-driven | 78.27% | 0.54 pp | 7.76% | 0.32 pp | 0 |
| Confidence-only | 75.50% | 0.00 pp | 14.91% | 0.59 pp | 0 |
| Predictive entropy | 51.09% | 0.47 pp | 8.16% | 0.35 pp | 0 |
| Ensemble disagreement | 82.62% | 0.32 pp | 9.97% | 0.41 pp | 0 |
| Always accept | 100.00% | 0.00 pp | 14.02% | 0.48 pp | 0 |

### Interpretation and limits

On this benchmark, reliability-driven selection achieved a higher mean coverage than confidence-only (78.27% versus 75.50%) while also recording a lower mean observed error rate (7.76% versus 14.91%). This is a promising result across the three specified seeds, but it is not a matched-coverage comparison: the policies select different coverage levels, so the difference does not isolate ranking quality at a common operating point. A validation-only threshold-selection protocol and matched-coverage test comparison should be added before making a stronger comparative claim.

Predictive entropy operated at substantially lower coverage (51.09%) with an 8.16% observed error rate. Ensemble disagreement had the highest nontrivial coverage (82.62%) and a 9.97% observed error rate. Always-accept coverage was 100% with a 14.02% observed error rate. No policy produced net corrections in any seed.

These results are not directly comparable with Wine Quality results: the datasets, sample sizes, class distributions, and task difficulty differ. The MAGIC benchmark provides a second tabular evaluation, not broad cross-domain generalization. The constant harm score is a scenario assumption, not a learned harm estimate; human review is simulated; action-effectiveness settings are assumptions; and evidence-provider actions are disabled. Three seeds provide descriptive stability evidence, not confidence intervals. No final-test threshold tuning was performed.


## Validation-selected matched-coverage diagnostic

The experiment now also evaluates the four ranking policies at a common nominal
75% coverage target. For each policy, its score threshold is selected from the
validation partition's score distribution using the 75th percentile (higher
quantile). That fixed threshold is then applied to the untouched test partition.
Test labels are read only after the accepted subset has been determined, to
calculate its observed error rate. Actual test coverage is reported because
distribution differences and tied scores can move it away from exactly 75%.

This diagnostic isolates ranking behavior more directly than the verification
engine's cost-sensitive action metrics, which may produce different coverage
for each policy. It is reported separately and does not replace the five-policy
action-engine comparison. The always-accept baseline is not part of the
four-policy matched-coverage ranking diagnostic.

Regression tests verify that the threshold is derived from validation scores,
that test labels do not affect threshold selection, and that non-finite scores
are rejected. The updated three-seed CI run must complete before matched-
coverage performance results are added here.
