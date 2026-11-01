"""Measure realized alternate-model corrections and regressions on held-out Wine data.

Routing policies are label-blind and receive an identical per-seed action
budget. Ground-truth labels are passed only to the final retrospective auditor.
This measures model substitution, not human review or evidence acquisition.
"""
import argparse
import json
import sys
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
from src.verification.outcome_audit import audit_prediction_changes


def top_budget_mask(scores, budget_count):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Routing scores must be finite one-dimensional values")
    if not 0 <= budget_count <= len(scores):
        raise ValueError("budget_count is outside the score array")
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:budget_count]] = True
    return mask


def run(seed: int, output_dir: Path, action_fraction: float = 0.15):
    data_dir = RESEARCH_DIR / "datasets" / "raw"
    frame, _ = load_dataset(str(data_dir))
    frame = create_binary_target(frame, threshold=6)
    X_train, X_val, X_test, y_train, y_val, y_test = grouped_stratified_data_splitting(
        frame, target_column="good_quality", test_size=0.15, val_size=0.15,
        random_state=seed,
    )

    primary = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
    )
    alternate = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    )
    primary.fit(X_train, y_train)
    alternate.fit(X_train, y_train)

    primary_prob = primary.predict_proba(X_test)
    primary_pred = primary.predict(X_test)
    alternate_pred = alternate.predict(X_test)
    uncertainty = 1.0 - primary_prob.max(axis=1)
    disagreement = (primary_pred != alternate_pred).astype(float)
    count = max(1, int(np.floor(len(X_test) * action_fraction)))
    rng = np.random.default_rng(seed)
    random_scores = rng.random(len(X_test))

    policies = {
        "model_disagreement": top_budget_mask(disagreement, count),
        "low_primary_confidence": top_budget_mask(uncertainty, count),
        "random_matched_budget": top_budget_mask(random_scores, count),
    }
    results = {}
    for name, mask in policies.items():
        results[name] = audit_prediction_changes(
            y_true=y_test.to_numpy(),
            baseline_predictions=primary_pred,
            candidate_predictions=alternate_pred,
            action_mask=mask,
            action_costs=np.full(len(X_test), 1.5, dtype=float),
        )
        results[name]["routing_policy"] = name
        results[name]["nominal_action_fraction"] = action_fraction
        results[name]["alternate_model"] = "RandomForestClassifier"
        results[name]["primary_model"] = "LogisticRegression"

    return {
        "milestone": "53 - measured verification outcomes",
        "dataset": "UCI Wine Quality (red and white; binary target quality >= 6)",
        "seed": seed,
        "split_sizes": {
            "train": int(len(X_train)), "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "test_class_prevalence": float(y_test.mean()),
        "action_budget_count": count,
        "action_fraction": action_fraction,
        "policies": results,
        "interpretation_boundary": (
            "Offline held-out comparison of an executed alternate model. "
            "It does not measure human review, evidence-provider value, causal "
            "deployment impact, or real-world monetary costs."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--action-fraction", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, default=RESEARCH_DIR / "results" / "measured_verification")
    args = parser.parse_args()
    if not 0 < args.action_fraction <= 1:
        raise ValueError("--action-fraction must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        result = run(seed, args.output_dir, args.action_fraction)
        path = args.output_dir / f"wine_measured_outcomes_seed_{seed}.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        summaries.append(result)
    summary = {
        "milestone": "53 - measured verification outcomes",
        "seed_count": len(summaries),
        "seeds": args.seeds,
        "per_seed_results": summaries,
        "note": "Per-seed outcomes are retained; no pooled causal-effect claim is made.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
