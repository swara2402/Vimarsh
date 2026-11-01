# HTRU2 hard-budget, cost-sensitive verification experiment

## Purpose

This is a fresh-dataset experiment of an explicitly constrained verification policy. It evaluates whether ranking prediction errors by an estimated error probability weighted by a predeclared consequence scenario can allocate a limited review budget more effectively than confidence, predictive entropy, ensemble disagreement, or random selection.

Dataset: [UCI HTRU2 (dataset 372)](https://archive.ics.uci.edu/dataset/372/htru2), eight continuous features and binary target (pulsar = 1, non-pulsar = 0). The dataset contains 17,898 rows (16,259 non-pulsars and 1,639 pulsars). The experiment uses grouped stratified train/validation/test splits, grouped five-fold OOF model predictions, and a Random Forest primary model. Exact duplicate feature vectors stay within one partition.

## Predeclared policy and constraints

- Hard review capacity: at most 10% of test observations (255 rows in each 2,557-row test split).
- Per-case review cost: 0.25 hypothetical cost units.
- Simulated review effectiveness: 0.80.
- Scenario consequence weight: false negative = 5.0 units; false positive = 1.0 unit.
- Expected net benefit eligibility: estimated error probability × predicted-error-direction consequence weight × review effectiveness must exceed per-case review cost.
- Eligible cases are ordered by expected net benefit; the fixed capacity is never exceeded.
- Baselines use the same predicted-class consequence weights, review cost, effectiveness assumption, and hard capacity. They are not forced to use the same number of reviews: each cost-sensitive policy may stop below the cap when no further case has positive estimated net benefit. The random reference uses the full capacity.

These numbers are scenario assumptions selected to make the constrained decision rule testable. UCI HTRU2 provides no human review costs, action effectiveness, or consequence labels. The asymmetric consequence weights are not astrophysical ground truth and must not be presented as domain-validated costs.

## Leakage controls and evaluation

For each seed (17, 42, 73), the primary RF prediction-error labels are generated from grouped out-of-fold training predictions. The reliability estimator is trained and calibrated using disjoint grouped subsets. Test-time scores determine selection under the fixed predeclared budget. Test labels are used only after the selection masks are fixed, to calculate offline outcome metrics.

The CI benchmark completed successfully for all three seeds. Each split contained 12,784 training, 2,557 validation, and 2,557 test rows. Aggregate values below are means across the three seeds; variability is population standard deviation (SD), not a confidence interval.

| Policy | Mean reviewed fraction | Mean simulated net benefit | Mean post-review weighted error loss |
|---|---:|---:|---:|
| Reliability cost-sensitive | 1.38% | 42.37 (SD 6.56) | 148.80 (SD 9.65) |
| Confidence cost-sensitive | 4.45% | 63.58 (SD 5.05) | 108.00 (SD 4.09) |
| Entropy cost-sensitive | 9.97% | 47.98 (SD 4.25) | 88.27 (SD 4.43) |
| Disagreement cost-sensitive | 1.19% | 21.48 (SD 4.85) | 170.93 (SD 4.46) |
| Random budgeted | 9.97% | -47.75 (SD 6.23) | 184.00 (SD 10.68) |
| No review | 0.00% | 0.00 (SD 0.00) | 200.00 (SD 4.90) |

The reliability policy selected 33, 35, and 38 cases across the three seeds (1.29% to 1.49% of test rows), substantially below the 10% cap because only cases with positive estimated net benefit were eligible. Its mean simulated net benefit was positive, but it did **not** outperform the simpler confidence policy: confidence produced a higher mean net benefit (63.58 versus 42.37) while reviewing more cases. Entropy produced the lowest mean post-review weighted loss, but used nearly the entire review budget and had lower net benefit than confidence after review costs. Random full-capacity review had negative mean simulated net benefit under the chosen assumptions.

The results therefore do not support a claim that the learned reliability score is superior to confidence on HTRU2. They do show that the cost-sensitive eligibility rule can leave budget unused and that policy rankings depend on the assumed review cost, effectiveness, and asymmetric consequences. The comparison is under a common capacity ceiling and scenario, **not** at matched review counts or matched utilization.

## Matched-capacity ranking diagnostic

A second, explicitly separate diagnostic now evaluates ranking quality when every signal receives the same review count. For each seed, the diagnostic selects exactly the predeclared 10% test capacity (255 observations) using each score multiplied by the predicted-class scenario consequence weight and the fixed review-effectiveness factor. Because per-case review cost is constant across cases, subtracting that constant does not change the ranking. The compared signals are learned reliability, confidence risk, predictive entropy, ensemble disagreement, and random selection.

Unlike the positive-net-benefit policy above, this diagnostic intentionally fills the budget even when the selected cases have negative estimated net benefit. It therefore isolates ranking performance at common utilization and must not be interpreted as the recommended cost-sensitive operating policy. Selection uses model scores and predicted classes only; test labels are accessed afterward for offline outcome scoring. The diagnostic is reported separately from the positive-net-benefit results, preserving the distinction between ranking quality and budget-utilization/eligibility behavior.

The matched-capacity CI benchmark completed successfully for all three seeds. Each policy reviewed exactly 255 of 2,557 test rows (9.97%). Aggregate values are means across seeds; SD is population standard deviation.

| Matched-capacity ranking | Mean reviewed fraction | Mean simulated net benefit | Mean post-review weighted error loss |
|---|---:|---:|---:|
| Reliability | 9.97% | 22.92 (SD 11.69) | 113.33 (SD 14.00) |
| Confidence | 9.97% | 49.85 (SD 5.99) | 86.40 (SD 6.37) |
| Predictive entropy | 9.97% | 47.98 (SD 4.25) | 88.27 (SD 4.43) |
| Ensemble disagreement | 9.97% | -19.75 (SD 9.26) | 156.00 (SD 6.52) |
| Random | 9.97% | -47.75 (SD 6.23) | 184.00 (SD 10.68) |

At equal review counts, confidence achieved the highest mean simulated net benefit and lowest mean post-review weighted error loss. Entropy was close behind. Reliability did not outperform either confidence or entropy, and disagreement performed worse than random under this scenario. The ordering was consistent across all three seeds: confidence/entropy exceeded reliability in simulated net benefit for every seed. This strengthens the negative comparative finding for learned reliability on HTRU2; it does not establish that confidence is universally superior.

This diagnostic fills the full capacity regardless of whether individual cases have positive estimated net benefit. Its simulated net benefit therefore includes the consequences of reviewing lower-value cases and must be kept distinct from the positive-net-benefit operating policy above. The ranking calculation uses scores and predicted classes only; test labels are used only for subsequent offline outcome scoring.

## Interpretation boundary

A positive simulated net benefit is conditional on the assumed consequence weights, review cost, and reviewer effectiveness. It does not establish real review efficacy, real-world harm reduction, broad domain generalization, or patent novelty. Three seeds provide descriptive stability evidence only. The present HTRU2 result is a negative comparative finding for the learned reliability score versus confidence, and should be reported as such rather than hidden.
