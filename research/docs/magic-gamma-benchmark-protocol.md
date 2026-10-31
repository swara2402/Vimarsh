# Independent benchmark protocol: UCI MAGIC Gamma Telescope

## Dataset selection

The second benchmark is the UCI MAGIC Gamma Telescope dataset (UCI dataset 159):
https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope

It contains ten continuous numerical measurements from simulated air showers and a binary class label. The experiment maps gamma (g) to 1 and hadron (h) to 0. UCI lists the dataset under CC BY 4.0. It is selected as a non-high-stakes binary classification benchmark with a numerical feature schema compatible with the current reliability feature analyzers.

## Data handling

The adapter is implemented in src/datasets/magic_gamma.py. It downloads the official raw CSV only when no local copy exists, validates the ten-column numerical feature schema, rejects unknown target labels and non-finite values, and returns features and labels separately. Dataset metadata records the source, dataset identifier, license, target mapping, row count, and class counts.

Unit tests validate target mapping, feature order, numeric parsing, and rejection of invalid labels and non-finite values. The loader tests do not download data during CI.

## Evaluation status

This commit adds the dataset adapter and its parser tests only. It does not yet constitute a second-dataset reliability or policy evaluation. No MAGIC model metrics, policy comparisons, or cross-dataset generalization claims should be reported until the full leakage-controlled training, reliability-estimator fitting, inference, and five-policy verification experiment has completed.

## Required protocol before reporting results

1. Create grouped stratified train/validation/test partitions using exact feature vectors as groups, preserving repeated rows and any conflicting labels within a group.
2. Generate grouped out-of-fold probabilities for Logistic Regression, Random Forest, and Gradient Boosting. Keep Random Forest as the primary model to preserve the current feature-extractor contract.
3. Fit the reliability estimator using out-of-fold primary-model error labels, with a grouped calibration holdout that is disjoint by feature group.
4. Evaluate the untouched test partition using the same five policies as the Wine experiment: reliability-driven, confidence-only, predictive entropy, ensemble disagreement, and always accept.
5. Use identical test observations, action budget, costs, executor settings, and documented harm assumptions across policies. Ground truth may only be used after action selection for offline outcome evaluation.
6. Repeat across multiple independent seeds and report per-seed values, means, and descriptive population standard deviations. Do not tune on the final test set.
7. Document that human review and action effectiveness remain simulated unless independently measured.

The MAGIC benchmark differs in source domain and input measurements from Wine Quality, but both are tabular classification tasks. A successful result would provide a second-domain benchmark, not proof of broad real-world generalization.
