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
