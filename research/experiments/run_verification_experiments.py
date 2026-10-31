"""Run full verification engine experiments with all baselines and decision profiles."""
import sys
import os
import json
from pathlib import Path

# Add research directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
research_dir = os.path.dirname(current_dir)
sys.path.insert(0, research_dir)

from src.verification.simulation import run_full_verification_experiment, run_profile_experiments, run_outcome_validation_experiments
from src.verification.cost_model import DecisionProfile


def main():
    """Main experiment execution function that runs all decision profiles."""
    print("=== VIMARSH Verification Action Engine Experiment ===")
    print("=== Milestone 3.1: Running all decision profiles to compare policies ===\n")
    
    # Create output directory
    output_path = Path(research_dir) / "results/verification"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First run the original baseline experiment to preserve results
    print("Running original baseline verification experiment...")
    baseline_results = run_full_verification_experiment(output_path=output_path)
    
    # Save baseline results separately to preserve them
    baseline_output = output_path / "baseline_verification_results.json"
    with open(baseline_output, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print(f"Original baseline results saved to: {baseline_output}\n")
    
    # Run all three decision profiles for Milestone 3.1 comparison
    print("Running all three decision profiles (Conservative, Balanced, Coverage-oriented)...")
    profile_results = run_profile_experiments(output_path=output_path)
    
    # Print comprehensive comparison of all profiles
    print("\n=== Profile Comparison Results ===")
    for profile_name, results in profile_results['profile_comparison'].items():
        print(f"\n{profile_name.upper()} PROFILE:")
        print(f"  Action distribution: {results['action_distribution']}")
        print(f"  Abstention rate: {results['abstention_rate']:.2%}")
        print(f"  Total cost: ${results['total_cost_incurred']:.2f}")
        print(f"  Total risk reduction: {results['total_risk_reduction']:.2f}")
        print(f"  Risk reduction per cost: {results['risk_reduction_per_unit_cost']:.3f}")
    
    # Save all profile results
    profile_output = output_path / "decision_profile_comparison_results.json"
    with open(profile_output, 'w') as f:
        json.dump(profile_results, f, indent=2)
    print(f"\nFull profile comparison results saved to: {profile_output}")
    
    # Print baseline comparison for reference
    print("\n=== Original Baseline Results (Preserved) ===")
    baseline_engine = baseline_results['engine_results']
    print(f"  Action distribution: {baseline_engine['action_distribution']}")
    print(f"  Abstention rate: {baseline_engine['abstention_rate']:.2%}")
    print(f"  Total cost: ${baseline_engine['total_cost_incurred']:.2f}")
    print(f"  Total risk reduction: {baseline_engine['total_risk_reduction']:.2f}")
    
    # Run Milestone 3.2 outcome validation experiments
    print("\n=== Running Milestone 3.2: Verification Outcome Validation ===")
    outcome_validation_results = run_outcome_validation_experiments(output_path=output_path)
    
    # Print comprehensive policy comparison for Milestone 3.2
    print("\n=== Milestone 3.2: Outcome Validation Policy Comparison ===")
    for profile_name, results in outcome_validation_results['profile_validations'].items():
        if 'policy_metrics' in results:
            pm = results['policy_metrics']
            print(f"\n{profile_name.upper()}:")
            print(f"  Coverage: {pm['coverage']:.2%}")
            if pm['observed_error_rate'] is not None:
                print(f"  Observed error rate (covered): {pm['observed_error_rate']:.2%}")
            else:
                print(f"  Observed error rate (covered): null (no covered samples)")
            print(f"  Net prediction improvements: {pm['net_corrections']}")
            if 'total_verification_cost' in pm:
                print(f"  Total verification cost: ${pm['total_verification_cost']:.2f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())