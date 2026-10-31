#!/usr/bin/env python3
"""Standalone runner for Milestone 3.2 outcome validation only"""
import os
from src.verification.simulation import run_outcome_validation_experiments

# Create output directory if it doesn't exist
output_path = "results/verification"
os.makedirs(output_path, exist_ok=True)

print("\n=== STANDALONE: Milestone 3.2 Outcome Validation ===\n")
# Run ONLY the outcome validation (no baseline/profile experiments)
outcome_validation_results = run_outcome_validation_experiments(output_path=output_path)

print("\n=== SUCCESS: All Milestone 3.2 validation complete ===\n")
# Print the policy comparison summary
for profile_name, results in outcome_validation_results['profile_validations'].items():
    if 'policy_metrics' in results:
        pm = results['policy_metrics']
        print(f"\n{profile_name.upper()}:")
        print(f"  Coverage: {pm['coverage']:.2%}")
        print(f"  Observed error rate (covered): {pm['observed_error_rate']:.2%}")
        print(f"  Net prediction changes: {pm['net_corrections']:+d}")
        if 'total_verification_cost' in pm:
            print(f"  Total verification cost: ${pm['total_verification_cost']:.2f}")
        # Handle null observed_error_rate if no covered samples
        if pm['observed_error_rate'] is not None:
            print(f"  Observed error rate (covered): {pm['observed_error_rate']:.2%}")
        else:
            print(f"  Observed error rate (covered): null (no covered samples)")