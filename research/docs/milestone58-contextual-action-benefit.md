# Milestone 58: Context-conditioned action-benefit estimator

## Purpose
Estimate the paired, observed change in decision loss for one verification action near a requested policy-time context. The estimator uses completed ledger records, inverse selection propensity weights, and a Gaussian context-similarity kernel.

For each observed case:

`benefit = (baseline_error - post_action_error) * harm`

Positive values are corrections; negative values are regressions.

## Method
- Restrict to the requested action and completed records.
- Exclude records without a valid non-zero selection propensity.
- Require an identical numeric context feature schema, including harm.
- Apply Gaussian context similarity and inverse propensity weighting.
- Compute a normalized weighted mean, effective sample size, weighted standard error, and a one-sided normal-approximation lower-bound diagnostic.
- Recommend further evaluation only when the diagnostic is positive and effective sample size meets the configured minimum; otherwise abstain.

## Critical interpretation limits
The lower-bound diagnostic is **not** a guaranteed confidence bound. Kernel weighting, estimated propensities, adaptive action assignment, unmeasured confounding, and non-random missing outcomes can invalidate nominal coverage. Propensity adjustment alone does not establish causality. This estimator is an offline research baseline and must not be connected to an automatic production action selector until validated against randomized or otherwise defensible evaluation data.

Human review and evidence acquisition need real provider outcomes. Synthetic tests validate implementation behavior only and are not empirical performance evidence.
