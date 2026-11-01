# Milestone 53: measured verification outcomes

## Research question

When an alternate model is actually executed on a fixed number of held-out cases,
how often does replacing the primary prediction correct an error, and how often
does it introduce a new error? This milestone replaces configured action
effectiveness percentages with observed paired outcomes for the specific
alternate-model action.

## Protocol

- Dataset: UCI Wine Quality red and white records, using the existing experimental
  binary target (quality >= 6).
- Seeds: 17, 42, and 73.
- Split: grouped stratified train/validation/test partition; exact duplicate
  feature rows remain within one partition.
- Primary model: Logistic Regression.
- Executed alternate model: Random Forest.
- Routing policies: model disagreement, low primary-model confidence, and
  random selection as a matched-budget reference.
- Action budget: exactly 15% of the test partition per seed (floor rounded,
  with at least one selected case).
- Both models and all preprocessing are fitted on training data only.
- Routing masks use predictions/probabilities only. Test labels are supplied
  only to the retrospective outcome auditor.

## Reported outcomes

For every seed and policy, the JSON records action count/rate, corrections,
regressions, net corrections, baseline errors in actioned cases, residual
errors in actioned cases, total realized configured action cost, cost per net
correction when positive, coverage, and error rate among issued predictions.

A correction is an actioned case where the primary prediction was wrong and the
alternate prediction is right. A regression is an actioned case where the
primary prediction was right and the alternate prediction was wrong. Net
corrections are corrections minus regressions. This accounting makes harmful
model substitutions visible rather than counting only successful changes.

## Interpretation limits

This is an offline, paired benchmark of an executed alternate classifier. The
cost of 1.5 is a declared experiment unit, not measured currency or a validated
operational cost. The random matched-budget policy is a reference, not a
deployable strategy. The experiment does not measure human review quality,
additional-evidence value, reviewer latency, causal deployment impact, or
safety. The existing ActionEffectiveness configuration remains a simulation
parameter in the legacy utility engine and must not be described as empirically
validated by this experiment.

## Reproduction

From the research directory:

    python experiments/run_measured_verification_outcomes.py --seeds 17 42 73

Outputs are written to results/measured_verification/. Each seed is retained
separately; do not pool different seeds as independent datasets.
