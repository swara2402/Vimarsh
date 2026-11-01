# Milestone 51: Primary-model family sensitivity

## Research question

Does the reliability-aware ranking signal retain useful error-detection and
selective-prediction behavior when the primary classifier changes, or are the
previous results specific to the Random Forest primary model?

## Design

The MAGIC Gamma Telescope benchmark is evaluated with the same grouped
train/validation/test split for each seed (17, 42, 73). The three primary
classifiers are Logistic Regression, Random Forest, and Gradient Boosting.
Each classifier is evaluated independently as the model whose prediction error
VIMARSH is trying to estimate.

For each seed:

1. Generate out-of-fold probabilities for all three classifiers using grouped
   five-fold training. Each fold fits its own preprocessing pipeline.
2. Fit the final three classifiers on the common training partition.
3. For each primary classifier, construct reliability features with that
   classifier designated as the primary model.
4. Create error labels only from that primary model's out-of-fold predictions.
5. Train a separate Random Forest reliability estimator on the corresponding
   out-of-fold error labels. Use a grouped calibration holdout drawn from the
   training partition only.
6. Compare reliability-driven ranking with confidence-only, predictive
   entropy, and ensemble disagreement.
7. Select the nominal 75% acceptance threshold from validation scores only,
   apply it unchanged to test scores, and access test labels only after the
   accepted set is fixed.
8. Calculate paired cluster-bootstrap intervals against confidence-only using
   exact-duplicate feature groups and shared resamples.

The split, fitted base models, reliability estimators, and thresholds are kept
separate by seed and primary-model family. The reliability estimator family
is intentionally held fixed to Random Forest so this experiment isolates
sensitivity to the primary classifier, rather than changing two model families
at once.

## Outputs

The runner writes one JSON result per seed and a descriptive aggregate summary
under `research/results/primary_model_sensitivity/`. Each model family reports
test classification performance, validation and test error-probability
metrics, validation-selected test coverage and accepted-set error for each
ranking policy, and paired bootstrap comparisons against confidence-only.

The summary reports arithmetic means and population standard deviations
across the three seeds separately for each primary classifier. It does not
pool observations or results across classifier families or datasets.

## Interpretation limits

- MAGIC is one benchmark, not evidence of broad domain generalization.
- Three seeds provide descriptive repeatability, not strong inferential
  evidence about training randomness.
- The reliability estimator remains a learned Random Forest in all arms.
- The experiment does not measure real human review, causal risk reduction,
  or deployment safety.
- A model-family difference is evidence about sensitivity under this protocol,
  not a universal ranking of classifier quality.
- No model, feature set, or threshold may be selected using the final test
  results. Any follow-up tuning requires a fresh held-out evaluation.

## Completion criteria

The experiment is complete only when the research tests pass, the three-seed
workflow produces all three model-family records for every seed, the artifact
is inspected for missing or non-finite metrics, and the pull request is merged
after all required checks pass.
