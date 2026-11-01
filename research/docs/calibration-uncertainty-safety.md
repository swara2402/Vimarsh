# Milestone 54: Calibration, uncertainty and safety audit

## Objective

Measure whether predicted probabilities and confidence rankings correspond to observed correctness, and whether selective prediction reduces error at the cost of coverage. Calibration is a property of predictions on a specified population and split, not a permanent guarantee.

## Protocol

- Dataset: scikit-learn Breast Cancer Wisconsin diagnostic dataset, used only as a public tabular benchmark.
- Seeds: 17, 42 and 73.
- Per seed: stratified 60% training, 20% calibration and 20% test partitions.
- Models: standardized Logistic Regression and Random Forest.
- Calibration: raw probabilities, sigmoid/Platt scaling and isotonic regression.
- Base models are fitted on training data only. Calibration mappings use calibration labels only. Test labels are used only for final retrospective scoring.
- Metrics: uniform-bin ECE and MCE, adaptive-bin ECE, top-label ECE/MCE, classwise ECE, Brier score, log loss, high-confidence error rate and risk-coverage AUC.
- Report each seed and aggregate mean/standard deviation. Never select the best method using test outcomes.

## Interpretation and safeguards

- ECE depends on bin count and binning strategy; report multiple variants rather than treating one ECE value as definitive.
- Isotonic regression can overfit small calibration sets. A lower ECE on one test split is not proof of general calibration.
- Brier score and log loss combine calibration and discrimination effects.
- Risk-coverage AUC measures the ranking quality of confidence for selective prediction, not the safety of a deployed policy.
- The benchmark is not clinical validation, and no medical use is implied by the dataset name.
- These are offline retrospective measurements. They do not establish causal risk reduction, human-review effectiveness, distribution-shift guarantees or deployment safety.

## Reproduction

Run from the research directory:

    python experiments/run_calibration_safety_audit.py --seeds 17 42 73

Outputs are written to research/results/calibration_safety/.
