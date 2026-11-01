# Milestone 52: Independent-dataset generalization

## Research question

Does the reliability-aware ranking pipeline remain usable when evaluated on independent tabular classification benchmarks with different sample sizes, feature counts, and feature semantics?

This milestone adds two UCI tasks:

- **Banknote Authentication** (UCI dataset 267): four continuous statistical features; authentic versus forged banknotes.
- **Ionosphere** (UCI dataset 52): 34 radar-return features; good versus bad radar returns.

These are independent offline benchmark tasks. They are not deployment data and do not represent financial authentication or operational radar systems.

## Experimental design

Each dataset is evaluated independently for seeds 17, 42, and 73. For each dataset/seed pair:

1. Exact duplicate feature vectors are kept within one split using grouped stratified holdouts. Train, validation, and test groups are disjoint.
2. Standardization is fitted on training data only.
3. Logistic Regression, Random Forest, and Gradient Boosting are trained on the same training partition.
4. Five-fold grouped out-of-fold predictions are generated within training data. Prediction-error labels from those OOF predictions train a separate reliability estimator for each primary-model family.
5. A grouped calibration holdout is reserved inside the training partition.
6. Reliability-driven, confidence-only, predictive-entropy, and ensemble-disagreement scores are compared at a nominal 75% acceptance operating point. Thresholds are selected on validation scores only and frozen before test evaluation.
7. Test labels are used only after the accepted set is fixed, to calculate observed error among accepted cases.

## Comparability and reporting

Results are reported per dataset, seed, primary model, and ranking policy. The summary reports descriptive means and population standard deviations over three seeds. It does not pool rows, model families, or metrics across datasets.

The two new datasets have different feature spaces. This is **independent within-dataset evaluation**, not zero-shot transfer of a model trained on one dataset to another. No claim of cross-dataset model transfer is made.

## Interpretation safeguards

- No test labels are used in preprocessing, model fitting, reliability training, calibration, or threshold selection.
- Duplicate feature groups are not allowed to cross split boundaries.
- The three seeds provide descriptive repeatability evidence, not inferential proof or statistical significance.
- A lower accepted-set error at a different realized coverage is not a fair ranking advantage; both nominal target and actual test coverage are reported.
- Reliability ranking is not assumed to outperform confidence, entropy, or disagreement. Negative or inconclusive findings must be retained.
- Human review, evidence-provider actions, action effectiveness, causal harm reduction, and deployment safety are not measured by this experiment.

## Reproduction

From the `research` directory:

```bash
python -m pytest -q tests/test_independent_benchmarks.py
python experiments/run_independent_generalization.py --seeds 17 42 73
```

The runner downloads the official UCI archives on first use and caches the source files under `research/datasets/raw/`. Outputs are written to `research/results/independent_generalization/`.

## Dataset provenance

- Banknote Authentication: https://archive.ics.uci.edu/dataset/267/banknote+authentication
- Ionosphere: https://archive.ics.uci.edu/dataset/52/ionosphere

Both dataset adapters record the UCI source, dataset identifier, feature count, class counts, and stated CC BY 4.0 license metadata.
