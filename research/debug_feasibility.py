"""Debug script to check why ALTERNATE_MODEL and HUMAN_REVIEW are not being selected."""
import sys
import os
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.verification.action_selector import VerificationEngine, VerificationEngineConfig
from src.verification.cost_model import DecisionProfile
from src.verification.actions import ActionType

# Create a simple test sample
sample = pd.Series({
    "alcohol": 13.0,
    "sulphates": 1.0,
    "pH": 3.3,
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978
})

# Initialize engine with full configuration
config = VerificationEngineConfig(
    total_verification_budget=500.0,
    lambda_cost_weight=1.0,
    decision_profile=DecisionProfile.CONSERVATIVE,
    enable_evidence_provider=False,
    enable_human_reviewer=True,
    max_human_reviews=50,
    original_model_name="gradient_boosting"
)

engine = VerificationEngine(config)

# Print out configuration values
print("=== ENGINE CONFIGURATION VALUES ===")
print(f"Action costs: {engine.config.action_costs}")
print(f"Action effectiveness: {engine.config.action_effectiveness}")
print(f"Lambda cost weight: {engine.config.lambda_cost_weight}")

# Check what alternative models are available
print("\n=== AVAILABILITY CHECK ===")
print("Checking alternative models...")
print(f"Alternate model runner exists: {engine.action_executor.alternate_model_runner is not None}")
if engine.action_executor.alternate_model_runner:
    print(f"Available alternatives: {engine.action_executor.alternate_model_runner.get_available_alternatives()}")

# Check human reviewer
print("\nChecking human reviewer...")
print(f"Human reviewer exists: {engine.action_executor.human_reviewer is not None}")
if engine.action_executor.human_reviewer:
    print(f"Human reviewer available: {engine.action_executor.human_reviewer.available}")
    print(f"Max reviews: {engine.action_executor.human_reviewer.max_reviews}")
    print(f"Reviews completed: {engine.action_executor.human_reviewer.reviews_completed}")

# Check feasibility for a high-risk sample
print("\n=== FEASIBILITY CHECK ===")
print("Checking feasibility for high-risk sample (baseline_risk=10.0)...")
baseline_risk = 10.0  # Very high risk
feasibility = engine.safety_engine.evaluate_all_actions(baseline_risk)
for action_type, feas in feasibility.items():
    print(f"\n{action_type.value}:")
    print(f"  Overall feasible: {feas.overall_feasible}")
    print(f"  Available: {feas.is_available}")
    print(f"  Budget feasible: {feas.is_budget_feasible}")
    print(f"  Safe: {feas.is_safe}")
    print(f"  Reason: {feas.reason}")

# Calculate utilities manually for HUMAN_REVIEW and ABSTAIN
print("\n=== MANUAL UTILITY CALCULATION ===")
R0 = 10.0
# HUMAN_REVIEW
hr_eff = engine.config.action_effectiveness.human_review
hr_r_hat = R0 * (1 - hr_eff)
hr_c_hat = engine.config.action_costs.human_review
hr_lambda = engine.config.lambda_cost_weight
hr_utility = R0 - hr_r_hat - hr_lambda * hr_c_hat
print(f"HUMAN_REVIEW: residual_risk={hr_r_hat:.2f}, cost=${hr_c_hat:.2f}, lambda={hr_lambda}, utility={hr_utility:.2f}")

# ABSTAIN
abs_eff = engine.config.action_effectiveness.abstain
abs_r_hat = R0 * (1 - abs_eff)
abs_c_hat = engine.config.action_costs.abstain
abs_utility = R0 - abs_r_hat - hr_lambda * abs_c_hat
print(f"ABSTAIN: residual_risk={abs_r_hat:.2f}, cost=${abs_c_hat:.2f}, lambda={hr_lambda}, utility={abs_utility:.2f}")

# ALTERNATE_MODEL
alt_eff = engine.config.action_effectiveness.alternate_model
alt_r_hat = R0 * (1 - alt_eff)
alt_c_hat = engine.config.action_costs.alternate_model
alt_utility = R0 - alt_r_hat - hr_lambda * alt_c_hat
print(f"ALTERNATE_MODEL: residual_risk={alt_r_hat:.2f}, cost=${alt_c_hat:.2f}, lambda={hr_lambda}, utility={alt_utility:.2f}")

# Now process this high-risk sample and see what action is selected
print("\n=== ENGINE PROCESSING RESULT ===")
print("Processing high-risk sample...")
evaluation = engine.process_sample(
    sample=sample,
    sample_id="debug_high_risk",
    base_prediction=1,
    base_confidence=0.7,
    error_probability=0.8,  # High error probability
    harm=10.0  # Maximum harm
)

print(f"\nSelected action: {evaluation.selected_action.value}")
print(f"Explanation: {evaluation.explanation}")
print(f"Feasible actions: {[a.value for a in evaluation.feasible_actions]}")