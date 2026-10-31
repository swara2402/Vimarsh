# HTRU2 hard-budget, cost-sensitive verification experiment

## Purpose

This is a fresh-dataset experiment of an explicitly constrained verification policy. It evaluates whether ranking prediction errors by an estimated error probability weighted by a predeclared consequence scenario can allocate a limited review budget more effectively than confidence, predictive entropy, ensemble disagreement, or random selection.

Dataset: [UCI HTRU2 (dataset 372)](https://archive.ics.uci.edu/dataset/372/htru2), eight continuous features and binary target (pulsar = 1, non-pulsar = 0). The experiment uses grouped stratified train/validation/test splits, grouped five-fold OOF model predictions, and a Random Forest primary model. Exact duplicate feature vectors stay within one partition.

## Predeclared policy and constraints

- Hard review capacity: at most 10% of test observations.
- Per-case review cost: 0.25 hypothetical cost units.
- Simulated review effectiveness: 0.80.
- Scenario consequence weight: false negative = 5.0 units; false positive = 1.0 unit.
- Expected net benefit eligibility: estimated error probability × predicted-error-direction consequence weight × review effectiveness must exceed per-case review cost.
- Eligible cases are ordered by expected net benefit; the fixed capacity is never exceeded.
- Baselines use the same predicted class consequence weights, review cost, effectiveness assumption, and hard capacity. The random baseline receives the same capacity.

These numbers are scenario assumptions selected to make the constrained decision rule testable. UCI HTRU2 provides no human review costs, action effectiveness, or consequence labels. The asymmetric consequence weights are not astrophysical ground truth and must not be presented as domain-validated costs.

## Leakage controls and evaluation

For each seed (17, 42, 73), the primary RF prediction-error labels are generated from grouped out-of-fold training predictions. The reliability estimator is trained and calibrated using disjoint grouped subsets. All policies are selected from test-time scores under a fixed predeclared budget. Test labels are used only after the selection masks are fixed, to calculate offline outcome metrics.

The report includes review count/fraction, cost spent, weighted error loss before and after simulated review, gross loss avoided, and net benefit. A no-review reference is included. Results are descriptive across three seeds, not confidence intervals or evidence of causal risk reduction.

## Interpretation boundary

A positive simulated net benefit is conditional on the assumed consequence weights, review cost, and reviewer effectiveness. It does not establish real review efficacy, real-world harm reduction, broad domain generalization, or patent novelty. The key comparison is whether reliability ranking adds value over simpler scores under the exact same constrained budget and scenario assumptions.
