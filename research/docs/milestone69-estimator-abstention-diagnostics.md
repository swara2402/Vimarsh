# Milestone 69: Contextual-estimator abstention diagnostics

## Delivered implementation

- The paired action-benefit experiment now records why every test query was selected or abstained.
- It reports counts by decision status and abstention reason.
- It summarizes the observed sample count, effective sample size, weighted mean benefit, standard error, uncertainty penalty, and lower-bound diagnostic distribution where available.
- It explicitly marks matched-budget baseline comparisons as non-informative when the proposed estimator selects zero actions, because both baselines then receive a zero-action budget.
- The action-selection rule is unchanged. This milestone improves observability and does not relax the conservative decision threshold.
- Added deterministic helper tests for reason counts and numeric distribution summaries.

## Interpretation

An all-abstain result must be treated as a limitation, not as evidence of superiority. The diagnostics distinguish inadequate effective sample support from a non-positive conservative benefit diagnostic. If the weighted benefit is small relative to the uncertainty penalty, the policy will continue to abstain. Any future threshold or estimator change must be justified by a pre-specified sensitivity analysis and matched-budget evaluation, not by selecting a favorable test result.
