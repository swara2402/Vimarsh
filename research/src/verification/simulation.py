"""Simulation utilities for running verification engine experiments."""
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split

# Import verification components
from .action_selector import VerificationEngine, VerificationEngineConfig
from .actions import ActionType, VerificationActionExecutor
from .outcome_validator import OutcomeValidator, PolicyComparison
from .cost_model import DecisionProfile
from src.data_loader import load_dataset, create_binary_target
from src.preprocessing import stratified_data_splitting, create_preprocessing_pipeline


def load_wine_quality_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess wine quality dataset."""
    # Load the red wine quality dataset
    df = pd.read_csv(Path(__file__).parent.parent.parent / "data/raw/winequality-red.csv", sep=';')
    
    # Create binary classification task: quality >= 7 is good (1), else bad (0)
    y = (df['quality'] >= 7).astype(int)
    X = df.drop('quality', axis=1)
    
    return X, y


def generate_harm_estimates(X: pd.DataFrame) -> List[float]:
    """
    Generate realistic harm estimates for each sample. In a real system, this would
    be based on domain knowledge or a harm model. For simulation, we derive harm from
    feature values that indicate higher-stakes cases.
    
    Args:
        X: Input features DataFrame
        
    Returns:
        List of harm values (scale: 1-10)
    """
    harms = []
    for _, row in X.iterrows():
        # Higher alcohol content might indicate premium wines with higher business impact
        # Higher sulphates might indicate more complex wines with higher value
        base_harm = 5.0
        alcohol_factor = (row['alcohol'] - 9.0) / 4.0  # normalize alcohol range
        sulphates_factor = row['sulphates'] / 2.0
        harm = base_harm + (alcohol_factor * 2) + (sulphates_factor * 1)
        # Clamp to 1-10 range
        harm = max(1.0, min(10.0, harm))
        harms.append(harm)
    return harms


def run_baseline_policy(policy_name: str, samples: pd.DataFrame, error_probabilities: List[float], 
                      harms: List[float], config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Run a baseline policy for comparison.
    
    Args:
        policy_name: Name of the baseline policy
        samples: Input samples
        error_probabilities: Reliability engine's error probabilities
        harms: Harm estimates for each sample
        config: Optional configuration overrides
        
    Returns:
        Dictionary of baseline results
    """
    config = config or {}
    baseline_results = {
        "policy": policy_name,
        "total_risk": 0.0,
        "total_cost": 0.0,
        "action_distribution": {},
        "abstentions": 0
    }
    
    for p_error, harm in zip(error_probabilities, harms):
        baseline_risk = p_error * harm
        action = None
        
        if policy_name == "always_accept":
            action = "ACCEPT"
            baseline_results["total_risk"] += baseline_risk
            
        elif policy_name == "confidence_threshold":
            threshold = config.get("threshold", 0.3)
            if p_error > threshold:
                action = "HUMAN_REVIEW"
                baseline_results["total_cost"] += 5.0
                baseline_results["total_risk"] += baseline_risk * 0.1  # 90% risk reduction
            else:
                action = "ACCEPT"
                baseline_results["total_risk"] += baseline_risk
                
        elif policy_name == "fixed_human_review":
            # Always send 10% of samples to human review
            if np.random.random() < 0.1:
                action = "HUMAN_REVIEW"
                baseline_results["total_cost"] += 5.0
                baseline_results["total_risk"] += baseline_risk * 0.1
            else:
                action = "ACCEPT"
                baseline_results["total_risk"] += baseline_risk
                
        elif policy_name == "reliability_threshold":
            threshold = config.get("threshold", 0.5)
            if p_error > threshold:
                if harm > 5.0:  # High harm cases go to human review
                    action = "HUMAN_REVIEW"
                    baseline_results["total_cost"] += 5.0
                    baseline_results["total_risk"] += baseline_risk * 0.1
                else:
                    action = "ABSTAIN"
                    baseline_results["total_cost"] += 0.5
                    baseline_results["total_risk"] += baseline_risk * 0.1
                    baseline_results["abstentions"] += 1
            else:
                action = "ACCEPT"
                baseline_results["total_risk"] += baseline_risk
                
        # Track action distribution
        baseline_results["action_distribution"][action] = baseline_results["action_distribution"].get(action, 0) + 1
    
    n_samples = len(samples)
    baseline_results["abstention_rate"] = baseline_results["abstentions"] / n_samples
    baseline_results["average_risk_per_sample"] = baseline_results["total_risk"] / n_samples
    baseline_results["average_cost_per_sample"] = baseline_results["total_cost"] / n_samples
    
    return baseline_results


def run_full_verification_experiment(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run complete verification engine experiment with all baselines.
    
    Args:
        output_path: Optional path to save results
        
    Returns:
        Dictionary with all experiment results
    """
    # Load data using official data_loader
    raw_data_dir = Path(__file__).parent.parent.parent / "datasets/raw"
    df, dataset_stats = load_dataset(str(raw_data_dir))
    df = create_binary_target(df, threshold=6)
    
    # Create consistent train/val/test split (returns X_train, X_val, X_test, y_train, y_val, y_test)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_data_splitting(df, target_column='good_quality', test_size=0.15, val_size=0.15, random_state=42)
    
    # Load the reliability model to get error probabilities for test set
    reliability_model_path = Path(__file__).parent.parent.parent / "models/reliability/reliability_logistic_regression.joblib"
    if reliability_model_path.exists():
        # In a real implementation, we would extract reliability features first
        # For this experiment, we'll simulate the error probabilities to demonstrate the engine
        print("✓ Reliability model found, using simulated error probabilities for demonstration")
        np.random.seed(42)
        error_probabilities = np.random.uniform(0.01, 0.8, size=len(X_test))
    else:
        # Fallback: if reliability model not found, simulate error probabilities
        print("⚠️ Reliability model not found, simulating error probabilities")
        np.random.seed(42)
        error_probabilities = np.random.uniform(0.01, 0.8, size=len(X_test))
    
    # Generate harm estimates
    harms = generate_harm_estimates(X_test)
    
    # Load base model predictions and confidences
    base_model_path = Path(__file__).parent.parent.parent / "models/gradient_boosting.joblib"
    preprocessor_path = Path(__file__).parent.parent.parent / "models/preprocessor.joblib"
    base_model = joblib.load(base_model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    X_test_processed = preprocessor.transform(X_test)
    base_predictions = base_model.predict(X_test_processed)
    base_confidences = base_model.predict_proba(X_test_processed).max(axis=1)
    
    # Initialize verification engine with full constraints
    full_config = VerificationEngineConfig(
        total_verification_budget=500.0,
        lambda_cost_weight=1.0,
        enable_evidence_provider=True,
        enable_human_reviewer=True,
        max_human_reviews=50,
        original_model_name="gradient_boosting"
    )
    
    engine = VerificationEngine(full_config)
    
    # Process test batch
    evaluations = engine.process_batch(
        X_test, 
        base_predictions.tolist(), 
        base_confidences.tolist(), 
        error_probabilities.tolist(), 
        harms
    )
    
    # Get engine statistics
    engine_stats = engine.get_summary_statistics()
    
    # Run baseline policies for comparison
    baselines = {}
    baselines["always_accept"] = run_baseline_policy("always_accept", X_test, error_probabilities, harms)
    baselines["confidence_threshold"] = run_baseline_policy("confidence_threshold", X_test, error_probabilities, harms, {"threshold": 0.3})
    baselines["fixed_human_review"] = run_baseline_policy("fixed_human_review", X_test, error_probabilities, harms)
    baselines["reliability_threshold"] = run_baseline_policy("reliability_threshold", X_test, error_probabilities, harms, {"threshold": 0.5})
    
    # Compile all results
    experiment_results = {
        "engine_results": engine_stats,
        "baseline_results": baselines,
        "test_set_size": len(X_test),
        "config": str(full_config),
        "individual_evaluations": [
            {
                "sample_id": e.sample_id,
                "selected_action": e.selected_action.value,
                "baseline_risk": e.baseline_risk,
                "residual_risk": e.selected_risk_estimate.residual_risk,
                "risk_reduction": e.selected_risk_estimate.risk_reduction,
                "cost": e.selected_risk_estimate.cost_incurred,
                "utility": e.selected_risk_estimate.utility
            } for e in evaluations
        ]
    }
    
    # Save results if output path provided
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "verification_experiment_results.json", 'w') as f:
            import json
            json.dump(experiment_results, f, indent=2)
    
    return experiment_results


def run_profile_experiments(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run verification engine experiments with all three decision profiles for comparison.
    Uses the exact same test split as the baseline experiment for fair comparison.
    
    Args:
        output_path: Optional path to save results
        
    Returns:
        Dictionary with all profile experiment results
    """
    # Load data using official data_loader - same split as baseline for consistency
    raw_data_dir = Path(__file__).parent.parent.parent / "datasets/raw"
    df, dataset_stats = load_dataset(str(raw_data_dir))
    df = create_binary_target(df, threshold=6)
    
    # Use EXACT same split as baseline (70%/15%/15%) with same random seed for consistency
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_data_splitting(
        df, target_column='good_quality', test_size=0.15, val_size=0.15, random_state=42
    )
    
    # Load base model predictions and confidences - same as baseline
    base_model_path = Path(__file__).parent.parent.parent / "models/gradient_boosting.joblib"
    preprocessor_path = Path(__file__).parent.parent.parent / "models/preprocessor.joblib"
    base_model = joblib.load(base_model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    X_test_processed = preprocessor.transform(X_test)
    base_predictions = base_model.predict(X_test_processed)
    base_confidences = base_model.predict_proba(X_test_processed).max(axis=1)
    
    # Use EXACT same error probabilities and harm estimates as baseline for fair comparison
    np.random.seed(42)  # Same seed as baseline
    error_probabilities = np.random.uniform(0.01, 0.8, size=len(X_test))
    harms = generate_harm_estimates(X_test)
    
    # Run experiments with all three decision profiles
    profile_results = {}
    
    # 1. Conservative profile (prioritize safety)
    print("\n=== Running CONSERVATIVE profile experiment ===")
    conservative_config = VerificationEngineConfig(
        total_verification_budget=500.0,
        lambda_cost_weight=1.0,
        decision_profile=DecisionProfile.CONSERVATIVE,
        enable_evidence_provider=False,  # Additional evidence unavailable per requirements
        enable_human_reviewer=True,      # Explicitly simulated human review
        max_human_reviews=50,
        original_model_name="gradient_boosting"
    )
    conservative_engine = VerificationEngine(conservative_config)
    conservative_evaluations = conservative_engine.process_batch(
        X_test, 
        base_predictions.tolist(), 
        base_confidences.tolist(), 
        error_probabilities.tolist(), 
        harms
    )
    profile_results["conservative"] = conservative_engine.get_summary_statistics()
    profile_results["conservative"]["decision_profile"] = "conservative"
    profile_results["conservative"]["individual_evaluations"] = [
        {
            "sample_id": e.sample_id,
            "selected_action": e.selected_action.value,
            "baseline_risk": e.baseline_risk,
            "residual_risk": e.selected_risk_estimate.residual_risk,
            "risk_reduction": e.selected_risk_estimate.risk_reduction,
            "cost": e.selected_risk_estimate.cost_incurred,
            "utility": e.selected_risk_estimate.utility,
            "explanation": e.explanation
        } for e in conservative_evaluations
    ]
    
    # 2. Balanced profile (trade off risk and cost - default)
    print("\n=== Running BALANCED profile experiment ===")
    balanced_config = VerificationEngineConfig(
        total_verification_budget=500.0,
        lambda_cost_weight=1.0,
        decision_profile=DecisionProfile.BALANCED,
        enable_evidence_provider=False,  # Additional evidence unavailable per requirements
        enable_human_reviewer=True,      # Explicitly simulated human review
        max_human_reviews=50,
        original_model_name="gradient_boosting"
    )
    balanced_engine = VerificationEngine(balanced_config)
    balanced_evaluations = balanced_engine.process_batch(
        X_test, 
        base_predictions.tolist(), 
        base_confidences.tolist(), 
        error_probabilities.tolist(), 
        harms
    )
    profile_results["balanced"] = balanced_engine.get_summary_statistics()
    profile_results["balanced"]["decision_profile"] = "balanced"
    profile_results["balanced"]["individual_evaluations"] = [
        {
            "sample_id": e.sample_id,
            "selected_action": e.selected_action.value,
            "baseline_risk": e.baseline_risk,
            "residual_risk": e.selected_risk_estimate.residual_risk,
            "risk_reduction": e.selected_risk_estimate.risk_reduction,
            "cost": e.selected_risk_estimate.cost_incurred,
            "utility": e.selected_risk_estimate.utility,
            "explanation": e.explanation
        } for e in balanced_evaluations
    ]
    
    # 3. Coverage-oriented profile (prefer automated predictions when safe)
    print("\n=== Running COVERAGE-ORIENTED profile experiment ===")
    coverage_config = VerificationEngineConfig(
        total_verification_budget=500.0,
        lambda_cost_weight=1.0,
        decision_profile=DecisionProfile.COVERAGE_ORIENTED,
        enable_evidence_provider=True,   # Enable additional evidence for coverage-oriented profile to test REQUEST_ADDITIONAL_EVIDENCE
        enable_human_reviewer=True,      # Explicitly simulated human review
        max_human_reviews=50,
        original_model_name="gradient_boosting"
    )
    coverage_engine = VerificationEngine(coverage_config)
    coverage_evaluations = coverage_engine.process_batch(
        X_test, 
        base_predictions.tolist(), 
        base_confidences.tolist(), 
        error_probabilities.tolist(), 
        harms
    )
    profile_results["coverage_oriented"] = coverage_engine.get_summary_statistics()
    profile_results["coverage_oriented"]["decision_profile"] = "coverage"
    profile_results["coverage_oriented"]["individual_evaluations"] = [
        {
            "sample_id": e.sample_id,
            "selected_action": e.selected_action.value,
            "baseline_risk": e.baseline_risk,
            "residual_risk": e.selected_risk_estimate.residual_risk,
            "risk_reduction": e.selected_risk_estimate.risk_reduction,
            "cost": e.selected_risk_estimate.cost_incurred,
            "utility": e.selected_risk_estimate.utility,
            "explanation": e.explanation
        } for e in coverage_evaluations
    ]
    
    # Compile comprehensive comparison results
    all_results = {
        "profile_comparison": profile_results,
        "test_set_size": len(X_test),
        "simulation_assumptions": """
        Key simulation assumptions (explicitly documented per Milestone 3.1 requirements):
        1. All action effectiveness values are simulated - not real-world measured improvements
        2. Human review is simulated with configurable accuracy (default 90%) - not real human reviewers
        3. Additional evidence collection is disabled by default (unavailable without real provider)
        4. Alternative models are genuinely evaluated for prediction disagreement - risk reduction only occurs if models disagree
        5. Error probabilities are simulated for demonstration - in production would come from reliability engine
        6. All experiments use identical test split and input values for fair comparison between profiles
        """,
        "comparison_metrics": {
            profile: {
                "total_risk_reduction": stats["total_risk_reduction"],
                "total_cost": stats["total_cost_incurred"],
                "coverage": 1 - stats["abstention_rate"],
                "abstention_rate": stats["abstention_rate"],
                "action_distribution": stats["action_distribution"],
                "average_baseline_risk": stats["average_baseline_risk"],
                "average_risk_reduction_per_sample": stats["average_risk_reduction_per_sample"],
                "risk_reduction_per_unit_cost": stats["risk_reduction_per_unit_cost"]
            } for profile, stats in profile_results.items()
        }
    }
    
    # Save comprehensive profile results
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "profile_comparison_results.json", 'w') as f:
            import json
            json.dump(all_results, f, indent=2)
        print(f"\nAll profile comparison results saved to: {output_path / 'profile_comparison_results.json'}")
    
    return all_results


def run_outcome_validation_experiments(output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Run Milestone 3.2 outcome validation experiments for all decision profiles.
    Actually executes verification actions and measures their real impact on prediction correctness,
    rather than relying solely on simulated residual risk reductions.
    """
    from pathlib import Path
    # Convert string path to Path object if needed
    if output_path is not None and isinstance(output_path, str):
        output_path = Path(output_path)
    # Load data using official data_loader - same split as all previous experiments for consistency
    raw_data_dir = Path(__file__).parent.parent.parent / "datasets/raw"
    df, dataset_stats = load_dataset(str(raw_data_dir))
    df = create_binary_target(df, threshold=6)
    
    # Use EXACT same split as all prior experiments for fair comparison
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_data_splitting(
        df, target_column='good_quality', test_size=0.15, val_size=0.15, random_state=42
    )
    
    # Load base model predictions and confidences - same as baseline
    base_model_path = Path(__file__).parent.parent.parent / "models/gradient_boosting.joblib"
    preprocessor_path = Path(__file__).parent.parent.parent / "models/preprocessor.joblib"
    base_model = joblib.load(base_model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    X_test_processed = preprocessor.transform(X_test)
    base_predictions = base_model.predict(X_test_processed)
    base_confidences = base_model.predict_proba(X_test_processed).max(axis=1)
    
    # Use EXACT same error probabilities and harm estimates as all prior experiments
    np.random.seed(42)
    error_probabilities = np.random.uniform(0.01, 0.8, size=len(X_test))
    harms = generate_harm_estimates(X_test)
    
    # Run outcome validation for each profile
    validation_results = {}
    
    # Define all three profiles to validate
    profile_configs = {
        "conservative": VerificationEngineConfig(
            total_verification_budget=500.0,
            lambda_cost_weight=1.0,
            decision_profile=DecisionProfile.CONSERVATIVE,
            enable_evidence_provider=False,
            enable_human_reviewer=True,
            max_human_reviews=50,
            original_model_name="gradient_boosting"
        ),
        "balanced": VerificationEngineConfig(
            total_verification_budget=500.0,
            lambda_cost_weight=1.0,
            decision_profile=DecisionProfile.BALANCED,
            enable_evidence_provider=False,
            enable_human_reviewer=True,
            max_human_reviews=50,
            original_model_name="gradient_boosting"
        ),
        "coverage_oriented": VerificationEngineConfig(
            total_verification_budget=500.0,
            lambda_cost_weight=1.0,
            decision_profile=DecisionProfile.COVERAGE_ORIENTED,
            enable_evidence_provider=True,
            enable_human_reviewer=True,
            max_human_reviews=50,
            original_model_name="gradient_boosting"
        )
    }
    
    # Process each profile with outcome validation
    for profile_name, config in profile_configs.items():
        print(f"\n=== Running OUTCOME VALIDATION for {profile_name.upper()} profile ===")
        
        # Create engine and process batch to get action selections
        engine = VerificationEngine(config)
        evaluations = engine.process_batch(
            X_test, 
            base_predictions.tolist(), 
            base_confidences.tolist(), 
            error_probabilities.tolist(), 
            harms
        )
        
        # Create outcome validator with the engine's action executor
        validator = OutcomeValidator(engine.action_executor)
        
        # Actually execute each selected action and measure outcomes
        for i, eval_result in enumerate(evaluations):
            sample = X_test.iloc[i]
            true_label = y_test.iloc[i]
            base_pred = base_predictions[i]
            base_conf = base_confidences[i]
            
            # Execute action and track actual outcome
            validator.execute_and_evaluate_action(
                sample=sample,
                base_prediction=base_pred,
                base_confidence=base_conf,
                action_type=eval_result.selected_action,
                true_label=true_label,
                evaluation=eval_result
            )
        
        # Run all Milestone 3.2 integrity validations
        model_validations = validator.validate_alternate_model_independence(X_test, base_model, preprocessor)
        evidence_validations = validator.validate_evidence_no_label_leakage()
        human_validations = validator.validate_simulated_human_review()
        
        # Calculate metrics
        action_metrics = validator.calculate_action_metrics()
        policy_metrics = validator.calculate_policy_metrics(profile_name)
        
        # Store all results for this profile
        actual_execution_cost = float(policy_metrics.total_verification_cost)
        planned_budget_remaining = float(engine.budget_manager.get_remaining_budget())
        planned_budget_spent = float(sum(
            evaluation.selected_risk_estimate.cost_incurred for evaluation in evaluations
        ))
        planned_budget_overrun = max(0.0, planned_budget_spent - config.total_verification_budget)
        actual_budget_overrun = max(0.0, actual_execution_cost - config.total_verification_budget)

        validation_results[profile_name] = {
            "policy_metrics": policy_metrics.__dict__,
            "action_metrics": {k: v.__dict__ for k, v in action_metrics.items()},
            "budget_audit": {
                "budget_limit": float(config.total_verification_budget),
                "planned_budget_spent": planned_budget_spent,
                "planned_budget_remaining": planned_budget_remaining,
                "planned_budget_exceeded": planned_budget_spent > config.total_verification_budget,
                "planned_budget_overrun": planned_budget_overrun,
                "actual_execution_cost": actual_execution_cost,
                "actual_cost_exceeds_budget": actual_execution_cost > config.total_verification_budget,
                "actual_cost_overrun": actual_budget_overrun,
                "budget_interpretation": "The selector budget constrains planned action costs. Executor-reported operational costs are audited separately and can exceed the planning limit when configured planning costs differ from execution costs."
            },
            "integrity_checks": {
                "alternate_model_validations": model_validations,
                "evidence_validations": evidence_validations,
                "human_review_validations": human_validations
            },
            "profile_config": str(config)
        }
        
        # Print key results
        print(f"  {profile_name} coverage: {policy_metrics.coverage:.2%}")
        print(f"  Observed error rate (covered samples): {policy_metrics.observed_error_rate:.2%}")
        print(f"  Total predictions corrected: {policy_metrics.total_corrected}")
        print(f"  New errors introduced: {policy_metrics.total_introduced}")
        print(f"  Net improvement: {policy_metrics.net_corrections}")
    
    # Add always-accept baseline for comparison
    print("\n=== Running OUTCOME VALIDATION for ALWAYS_ACCEPT baseline ===")
    always_accept_outcomes = []
    base_errors = sum(1 for p, t in zip(base_predictions, y_test) if p != t)
    base_error_rate = base_errors / len(y_test)
    
    validation_results["always_accept"] = {
        "policy_metrics": {
            "policy_name": "always_accept",
            "coverage": 1.0,
            "abstention_rate": 0.0,
            "observed_error_rate": base_error_rate,
            "total_observed_errors": base_errors,
            "total_corrected": 0,
            "total_introduced": 0,
            "net_corrections": 0
        }
    }
    print(f"  Always-accept baseline error rate: {base_error_rate:.2%}")
    
    # Compile comprehensive comparison
    all_validation_results = {
        "profile_validations": validation_results,
        "test_set_size": len(X_test),
        "baseline_error_rate": base_error_rate,
        "validation_timestamp": str(pd.Timestamp.now()),
        "milestone": "3.2 - Verification Outcome Validation",
        "simulation_assumptions": [
            "Error probabilities are generated synthetically for this experiment, not produced by a calibrated reliability model.",
            "Human review is simulated at the configured accuracy and receives the true label solely to simulate post-selection review outcomes; it is not a real human review workflow.",
            "The evidence provider is a synthetic placeholder that adds a flag to the sample; it does not retrieve external evidence or update the model prediction.",
            "Alternate-model predictions are produced by separately trained models, but disagreement does not guarantee correctness.",
            "Reported action costs are execution costs returned by the action executor; risk reduction remains an estimate from the action-selection stage."
        ],
        "dataset_validation_caveats": [
            "The experiment uses the existing stratified random split with random_state=42.",
            "The Wine Quality dataset contains duplicate rows, including duplicates across train, validation, and test partitions; test metrics may therefore be optimistic and should be re-evaluated with a duplicate-grouped split before making generalization claims."
        ]
    }
    
    # Save results if output path provided
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        results_file = output_path / "outcome_validation_comprehensive_results.json"
        with open(results_file, 'w') as f:
            import json
            json.dump(all_validation_results, f, indent=2)
        print(f"\nAll Milestone 3.2 outcome validation results saved to: {results_file}")
    
    return all_validation_results


if __name__ == "__main__":
    results = run_full_verification_experiment()
    print("\n=== Verification Engine Experiment Complete ===")
    print(f"Total samples processed: {results['engine_results']['total_samples_processed']}")
    print(f"Action distribution: {results['engine_results']['action_distribution']}")
    print(f"Abstention rate: {results['engine_results']['abstention_rate']:.2%}")
    print(f"Total risk reduction: {results['engine_results']['total_risk_reduction']:.2f}")
    print(f"Risk reduction per unit cost: {results['engine_results']['risk_reduction_per_unit_cost']:.3f}")