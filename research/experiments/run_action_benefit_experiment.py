"""Cross-fitted offline experiment for contextual alternate-model benefit estimation.

The alternate model is trained only on the training partition. Validation outcomes
fit the action-benefit estimator; test labels are used only by the final auditor.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from src.data_loader import load_dataset, create_binary_target
from src.preprocessing import grouped_stratified_data_splitting
from src.verification.action_outcome_ledger import ActionOutcomeLedger, ActionOutcomeRecord
from src.verification.action_benefit_estimator import estimate_action_benefit


def top_budget_mask(scores, count):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be finite one-dimensional values")
    if not 0 <= count <= len(scores):
        raise ValueError("count is outside score array")
    selected = np.zeros(len(scores), dtype=bool)
    if count:
        selected[np.argsort(-scores, kind="stable")[:count]] = True
    return selected


def summarize_selection(mask, baseline, alternate, truth):
    mask = np.asarray(mask, dtype=bool)
    if not (len(mask) == len(baseline) == len(alternate) == len(truth)):
        raise ValueError("paired arrays must have equal lengths")
    baseline_error = np.asarray(baseline) != np.asarray(truth)
    alternate_error = np.asarray(alternate) != np.asarray(truth)
    benefit = (baseline_error.astype(float) - alternate_error.astype(float))
    count = int(mask.sum())
    return {
        "selected_count": count,
        "action_rate": float(count / len(mask)) if len(mask) else 0.0,
        "total_realized_benefit": float(benefit[mask].sum()),
        "mean_realized_benefit_per_selected_case": float(benefit[mask].mean()) if count else None,
        "corrections": int((baseline_error & ~alternate_error & mask).sum()),
        "introduced_regressions": int((~baseline_error & alternate_error & mask).sum()),
        "unchanged_outcomes": int((baseline_error == alternate_error)[mask].sum()),
    }


def run(seed: int, action_fraction: float = 0.5, min_effective_samples: float = 10.0):
    if not 0 < action_fraction <= 1:
        raise ValueError("action_fraction must be in (0, 1]")
    data_dir = RESEARCH_DIR / "datasets" / "raw"
    frame, _ = load_dataset(str(data_dir))
    frame = create_binary_target(frame, threshold=6)
    X_train, X_val, X_test, y_train, y_val, y_test = grouped_stratified_data_splitting(
        frame, target_column="good_quality", test_size=0.15, val_size=0.15,
        random_state=seed,
    )

    primary = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    alternate = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    )
    primary.fit(X_train, y_train)
    alternate.fit(X_train, y_train)

    val_primary_prob = primary.predict_proba(X_val)
    val_primary = primary.predict(X_val)
    val_alternate = alternate.predict(X_val)
    test_primary_prob = primary.predict_proba(X_test)
    test_primary = primary.predict(X_test)
    test_alternate = alternate.predict(X_test)

    val_contexts = [
        {"uncertainty": float(1.0 - p.max()), "disagreement": float(a != p_label), "harm": 1.0}
        for p, p_label, a in zip(val_primary_prob, val_primary, val_alternate)
    ]
    test_contexts = [
        {"uncertainty": float(1.0 - p.max()), "disagreement": float(a != p_label), "harm": 1.0}
        for p, p_label, a in zip(test_primary_prob, test_primary, test_alternate)
    ]

    rng = np.random.default_rng(seed + 100_003)
    assigned = rng.random(len(X_val)) < action_fraction
    assignment_id = f"wine-validation-randomized-exploration-seed-{seed}"

    with tempfile.TemporaryDirectory(prefix="vimarsh-outcomes-") as temp_dir:
        ledger = ActionOutcomeLedger(Path(temp_dir) / "action_outcomes.jsonl")
        for index in np.flatnonzero(assigned):
            ledger.append(ActionOutcomeRecord(
                case_id=f"seed-{seed}-validation-{int(index)}",
                action="alternate_model",
                policy_version="randomized-validation-exploration-v1",
                model_version="random-forest-200",
                baseline_prediction=int(val_primary[index]),
                status="completed",
                estimated_action_cost=1.0,
                propensity=float(action_fraction),
                randomized_assignment=True,
                assignment_id=assignment_id,
                context=val_contexts[index],
                action_result={"executed": True, "model": "random-forest-200"},
                final_prediction=int(val_alternate[index]),
                ground_truth=int(y_val.iloc[index]),
                provider_version="offline-benchmark-runner-v1",
            ))
        training_records = ledger.analysis_ready_records()
        estimates = [
            estimate_action_benefit(
                training_records, "alternate_model", context,
                confidence=0.95, min_effective_samples=min_effective_samples, bandwidth=1.5,
            )
            for context in test_contexts
        ]

    proposed_mask = np.asarray([
        item.get("status") == "recommend_for_further_evaluation" for item in estimates
    ], dtype=bool)
    selected_count = int(proposed_mask.sum())
    confidence_scores = 1.0 - test_primary_prob.max(axis=1)
    confidence_mask = top_budget_mask(confidence_scores, selected_count)
    random_mask = top_budget_mask(rng.random(len(X_test)), selected_count)

    y_test_values = y_test.to_numpy()
    return {
        "milestone": "60 - paired contextual action-benefit experiment",
        "dataset": "UCI Wine Quality (red and white; binary target quality >= 6)",
        "seed": seed,
        "validation_exploration": {
            "assignment_id": assignment_id,
            "known_action_propensity": action_fraction,
            "assigned_and_completed": int(assigned.sum()),
            "eligible_ledger_records": len(training_records),
        },
        "test": {
            "size": int(len(X_test)),
            "contextual_estimator_eligible": selected_count,
            "contextual_estimator_abstentions": int(len(X_test) - selected_count),
            "contextual_estimator": summarize_selection(proposed_mask, test_primary, test_alternate, y_test_values),
            "confidence_only_matched_budget": summarize_selection(confidence_mask, test_primary, test_alternate, y_test_values),
            "random_matched_budget": summarize_selection(random_mask, test_primary, test_alternate, y_test_values),
        },
        "interpretation_boundary": (
            "Offline held-out paired alternate-model benchmark. Validation outcomes use "
            "randomized action assignment with a known propensity. The alternate model is "
            "also evaluated counterfactually on all test rows for retrospective comparison. "
            "This is not operational human review, evidence acquisition, causal deployment "
            "evidence, or measured monetary/latency cost."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--action-fraction", type=float, default=0.5)
    parser.add_argument("--min-effective-samples", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DIR / "results" / "action_benefit")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run(seed, args.action_fraction, args.min_effective_samples)
        for seed in args.seeds
    ]
    payload = {
        "milestone": "60 - paired contextual action-benefit experiment",
        "seed_count": len(results),
        "seeds": args.seeds,
        "per_seed_results": results,
        "note": "Per-seed held-out benchmark results are retained; no causal deployment claim is made.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
