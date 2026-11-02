# Milestone 60: Paired contextual action-benefit experiment

## Experimental design
- Train a primary Logistic Regression model and an alternate Random Forest model on the training partition only.
- On the validation partition, randomly assign alternate-model execution with a recorded known propensity.
- Record completed paired baseline/alternate outcomes in the action-outcome ledger. The ledger is used only to fit the contextual benefit estimator.
- Query the estimator using primary-model uncertainty and harm only. The alternate-model output is excluded from policy-time context because it is the action being evaluated. Test labels are withheld until retrospective evaluation.
- Compare the proposed selection mask against confidence-only and random baselines at the same realized action count.

## Metrics
For selected test cases, report action count/rate, total and mean paired benefit, corrections, introduced regressions, and unchanged outcomes. Results are retained per seed.

## Boundaries
This is an offline benchmark of alternate-model substitution. The alternate model is evaluated on all test rows to provide retrospective paired outcomes, even though the policy selects only a subset. This is a counterfactual benchmark audit, not a claim that all actions would be executed in deployment. The experiment does not measure operational human review, additional evidence, causal deployment impact, or real monetary/latency costs. Validation exploration propensity is known by construction; it does not cure all distribution shift or transportability problems.
