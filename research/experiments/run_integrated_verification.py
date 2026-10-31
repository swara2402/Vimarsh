"""Run Milestone 4 integrated verification using a fitted reliability bundle.

First run research/experiments/run_reliability_experiments.py to create the
inference bundle. This script then uses its fitted reliability estimator to
score predictions before the verification engine selects actions.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from src.data_loader import load_dataset, create_binary_target
from src.preprocessing import grouped_stratified_data_splitting
from src.verification.action_selector import VerificationEngine, VerificationEngineConfig
from src.verification.cost_model import DecisionProfile
from src.verification.outcome_validator import OutcomeValidator
from src.verification.reliability_integration import ReliabilityVerificationPipeline
from src.verification.simulation import generate_harm_estimates


def run_integrated_experiment(budget: float = 500.0, output_path: Path = None):
    """Run action selection with estimated error probabilities, then evaluate outcomes."""
    base_dir = RESEARCH_DIR
    bundle_path = (
        base_dir / "models" / "reliability" / "wine_quality_reliability_bundle.joblib"
    )
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Reliability bundle not found at {bundle_path}. "
            "Run experiments/run_reliability_experiments.py first."
        )

    bundle = joblib.load(bundle_path)
    required = {
        "feature_extractor",
        "reliability_estimator",
        "base_models",
        "preprocessor",
        "base_model_dir",
        "primary_model_index",
        "target_threshold",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError("Reliability bundle is missing fields: " + ", ".join(missing))

    data, dataset_stats = load_dataset(str(base_dir / "datasets" / "raw"))
    data = create_binary_target(data, threshold=bundle["target_threshold"])
    _, _, X_test, _, _, y_test = grouped_stratified_data_splitting(
        data,
        target_column="good_quality",
        test_size=0.15,
        val_size=0.15,
        random_state=bundle.get("random_seed", 42),
    )

    preprocessor = bundle["preprocessor"]
    processed_test = preprocessor.transform(X_test)
    base_models = bundle["base_models"]
    model_probabilities = [
        np.asarray(model.predict_proba(processed_test), dtype=float)
        for model in base_models
    ]
    primary_index = int(bundle["primary_model_index"])
    if primary_index < 0 or primary_index >= len(base_models):
        raise ValueError("primary_model_index is outside the stored base-model list")

    primary_model = base_models[primary_index]
    primary_predictions = primary_model.predict(processed_test).tolist()
    harms = generate_harm_estimates(X_test)

    pipeline = ReliabilityVerificationPipeline(
        bundle["feature_extractor"], bundle["reliability_estimator"]
    )
    reliability_features, reliability_error_probabilities = pipeline.score(
        X_test, model_probabilities
    )
    primary_probabilities = model_probabilities[primary_index]
    primary_confidences = primary_probabilities.max(axis=1)

    # Baselines use the same test samples, harm estimates, budget, and executor
    # configuration. Only the error-probability scoring signal changes.
    class_count = primary_probabilities.shape[1]
    entropy_baseline = reliability_features["predictive_entropy"].to_numpy() / np.log(class_count)
    policy_scores = {
        "reliability_driven": reliability_error_probabilities,
        "confidence_only": 1.0 - primary_confidences,
        "predictive_entropy": np.clip(entropy_baseline, 0.0, 1.0),
        "ensemble_disagreement": reliability_features["disagreement_fraction"].to_numpy(),
    }

    policy_comparisons = {}
    reliability_validator = None
    reliability_engine = None
    for policy_name, error_scores in policy_scores.items():
        policy_config = VerificationEngineConfig(
            total_verification_budget=budget,
            decision_profile=DecisionProfile.BALANCED,
            base_model_path=base_dir / Path(bundle["base_model_dir"]),
            original_model_name=bundle["base_model_names"][primary_index],
            enable_evidence_provider=False,
            enable_human_reviewer=True,
            max_human_reviews=50,
        )
        policy_engine = VerificationEngine(policy_config)
        policy_evaluations = policy_engine.process_batch(
            samples=X_test,
            predictions=primary_predictions,
            confidences=primary_confidences.tolist(),
            error_probabilities=np.asarray(error_scores, dtype=float).tolist(),
            harms=harms,
        )

        # Ground truth is introduced only after action selection, for offline
        # simulation of outcomes. Each policy gets a fresh executor and budget.
        np.random.seed(42)
        validator = OutcomeValidator(policy_engine.action_executor)
        for index, evaluation in enumerate(policy_evaluations):
            validator.execute_and_evaluate_action(
                sample=X_test.iloc[index],
                base_prediction=primary_predictions[index],
                base_confidence=float(primary_confidences[index]),
                action_type=evaluation.selected_action,
                true_label=y_test.iloc[index],
                evaluation=evaluation,
            )
        policy_metrics = validator.calculate_policy_metrics(policy_name)
        policy_comparisons[policy_name] = policy_metrics.__dict__
        if policy_name == "reliability_driven":
            reliability_validator = validator
            reliability_engine = policy_engine

    policy_metrics = policy_comparisons["reliability_driven"]
    alternate_checks = reliability_validator.validate_alternate_model_independence(
        X_test, primary_model, preprocessor
    )
    evidence_checks = reliability_validator.validate_evidence_no_label_leakage()
    human_checks = reliability_validator.validate_simulated_human_review()

    result = {
        "milestone": "4.2 - Integrated Reliability-Driven Verification",
        "test_set_size": int(len(X_test)),
        "reliability_probability_source": "fitted reliability estimator from out-of-fold training features",
        "split_strategy": bundle.get("split_strategy", "legacy_random_stratified"),
        "duplicate_group_isolation": bundle.get("split_strategy") == "stratified_grouped_exact_feature_rows",
        "oof_split_strategy": "stratified_grouped_exact_feature_rows",
        "reliability_calibration_split_strategy": "stratified_grouped_exact_feature_rows",
        "error_probability_summary": {
            "minimum": float(np.min(reliability_error_probabilities)),
            "mean": float(np.mean(reliability_error_probabilities)),
            "maximum": float(np.max(reliability_error_probabilities)),
        },
        "policy_metrics": policy_metrics,
        "policy_comparisons": policy_comparisons,
        "policy_comparison_protocol": "Same grouped test set, harm estimates, budget, action costs, and executor settings; only error-probability scoring signal differs.",
        "integrity_checks": {
            "alternate_model": alternate_checks,
            "evidence_provider": evidence_checks,
            "human_review": human_checks,
        },
        "limitations": [
            "Human review remains simulated and uses ground truth only to simulate post-selection review outcomes.",
            "Additional evidence remains disabled because no real evidence provider is configured.",
            "Grouped splitting isolates exact duplicate feature rows, but this single Wine Quality benchmark is not evidence of cross-domain generalization.",
            "The action engine's configured risk-reduction factors remain assumptions and are not replaced by observed causal action effects.",
        ],
    }

    if output_path is None:
        output_path = base_dir / "results" / "integrated_verification"
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "integrated_outcome_validation.json"
    with results_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)

    print("Integrated reliability-driven verification complete.")
    print(f"Samples: {len(X_test)}")
    print(f"Mean estimated error probability: {result['error_probability_summary']['mean']:.4f}")
    for name, metrics in policy_comparisons.items():
        error_rate = metrics["observed_error_rate"]
        error_text = "n/a" if error_rate is None else f"{error_rate:.2%}"
        print(
            f"{name}: coverage={metrics['coverage']:.2%}, "
            f"covered_error={error_text}, net_corrections={metrics['net_corrections']}"
        )
    print(f"Results: {results_file}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=500.0)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()
    run_integrated_experiment(budget=arguments.budget, output_path=arguments.output)
