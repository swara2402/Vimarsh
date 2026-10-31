"""Evaluate a hard-budget, cost-sensitive verification policy on UCI HTRU2.

The policy ranks cases by estimated prediction-error probability multiplied by
an explicitly hypothetical class-dependent consequence weight. It selects only
positive expected-net-benefit reviews, subject to a hard 10% review-capacity
budget. Review cost, reviewer effectiveness, and asymmetric consequence weights
are scenario assumptions, not UCI-provided labels or measured quantities.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.htru2 import load_htru2_dataset
from src.preprocessing import (
    create_preprocessing_pipeline,
    grouped_stratified_holdout_indices,
    grouped_stratified_kfold_indices,
)
from src.train import train_logistic_regression, train_random_forest, train_gradient_boosting
from src.reliability.feature_extractor import ReliabilityFeatureExtractor
from src.reliability.reliability_model import ReliabilityEstimator
from experiments.run_magic_gamma_policy_comparison import grouped_magic_split

# Scenario parameters are fixed before evaluation; none are fitted to test labels.
REVIEW_BUDGET_FRACTION = 0.10
REVIEW_COST_UNITS = 0.25
REVIEW_EFFECTIVENESS = 0.80
FALSE_NEGATIVE_HARM = 5.0
FALSE_POSITIVE_HARM = 1.0
POLICIES = ("reliability_cost_sensitive", "confidence_cost_sensitive",
            "entropy_cost_sensitive", "disagreement_cost_sensitive", "random_budgeted")


def class_harm_weights(predictions):
    """Scenario consequence weight for the error direction implied by prediction."""
    predictions = np.asarray(predictions)
    return np.where(predictions == 0, FALSE_NEGATIVE_HARM, FALSE_POSITIVE_HARM).astype(float)


def select_under_budget(scores, harm_weights, budget_count, seed):
    """Select positive-net-benefit cases under a fixed integer review capacity."""
    scores = np.asarray(scores, dtype=float)
    harms = np.asarray(harm_weights, dtype=float)
    if scores.ndim != 1 or harms.ndim != 1 or len(scores) != len(harms):
        raise ValueError("Scores and harm weights must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isfinite(harms).all():
        raise ValueError("Scores and harm weights must be finite")
    if np.any(scores < 0) or np.any(harms < 0) or budget_count < 0:
        raise ValueError("Scores, harm weights, and budget must be non-negative")
    expected_gross_benefit = scores * harms * REVIEW_EFFECTIVENESS
    net_benefit = expected_gross_benefit - REVIEW_COST_UNITS
    eligible = np.flatnonzero(net_benefit > 0)
    if seed is not None:
        # Stable tie-breaking is deterministic and does not inspect outcomes.
        rng = np.random.default_rng(seed)
        tie_break = rng.random(len(scores))
        order = eligible[np.lexsort((tie_break[eligible], -net_benefit[eligible]))]
    else:
        order = eligible[np.argsort(-net_benefit[eligible], kind="stable")]
    chosen = order[:budget_count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    return mask


def select_top_budget(scores, harm_weights, budget_count):
    """Select exactly the available capacity by risk ranking, independent of net-benefit eligibility."""
    scores = np.asarray(scores, dtype=float)
    harms = np.asarray(harm_weights, dtype=float)
    if scores.ndim != 1 or harms.ndim != 1 or len(scores) != len(harms):
        raise ValueError("Scores and harm weights must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isfinite(harms).all():
        raise ValueError("Scores and harm weights must be finite")
    if np.any(scores < 0) or np.any(harms < 0) or budget_count < 0 or budget_count > len(scores):
        raise ValueError("Scores, harm weights, and budget must be non-negative and within capacity")
    # Review cost is constant per case, so ranking by gross expected benefit is
    # equivalent to ranking by net benefit. This diagnostic deliberately fills
    # the common capacity to isolate ranking quality from eligibility/utilization.
    ranking_score = scores * harms * REVIEW_EFFECTIVENESS
    order = np.argsort(-ranking_score, kind="stable")
    chosen = order[:budget_count]
    mask = np.zeros(len(scores), dtype=bool)
    mask[chosen] = True
    return mask


def evaluate_selection(name, selected, predictions, labels, harm_weights):
    """Use labels only after selection to calculate offline scenario outcomes."""
    selected = np.asarray(selected, dtype=bool)
    errors = np.asarray(predictions) != np.asarray(labels)
    weighted_losses = errors.astype(float) * np.asarray(harm_weights, dtype=float)
    before = float(weighted_losses.sum())
    selected_loss = float(weighted_losses[selected].sum())
    review_cost = float(selected.sum()) * REVIEW_COST_UNITS
    avoided = selected_loss * REVIEW_EFFECTIVENESS
    return {
        "policy": name,
        "reviewed_count": int(selected.sum()),
        "reviewed_fraction": float(selected.mean()) if len(selected) else 0.0,
        "budget_capacity": int(np.floor(len(selected) * REVIEW_BUDGET_FRACTION)),
        "review_cost_units": review_cost,
        "weighted_error_loss_before_review": before,
        "weighted_error_loss_after_simulated_review": before - avoided,
        "simulated_gross_loss_avoided": avoided,
        "simulated_net_benefit": avoided - review_cost,
        "reviewed_error_rate": float(errors[selected].mean()) if selected.any() else None,
        "budget_respected": bool(selected.sum() <= np.floor(len(selected) * REVIEW_BUDGET_FRACTION)),
    }


def run_seed(seed, raw_dir, output_dir):
    X, y, metadata = load_htru2_dataset(raw_dir)
    train_idx, val_idx, test_idx = grouped_magic_split(X, y, seed)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    oof = [None] * len(X_train)
    for fold, (fit_idx, hold_idx) in enumerate(
        grouped_stratified_kfold_indices(X_train, y_train, n_splits=5, random_state=seed)
    ):
        pre = create_preprocessing_pipeline(list(X_train.columns))
        fit_x = pre.fit_transform(X_train.iloc[fit_idx])
        hold_x = pre.transform(X_train.iloc[hold_idx])
        cfg = yaml.safe_load((ROOT / "configs" / "baseline_config.yaml").read_text())
        for model_cfg in cfg["models"].values():
            model_cfg["random_state"] = seed + fold
        models = [
            train_logistic_regression(fit_x, y_train.iloc[fit_idx], cfg),
            train_random_forest(fit_x, y_train.iloc[fit_idx], cfg),
            train_gradient_boosting(fit_x, y_train.iloc[fit_idx], cfg),
        ]
        for pos, original in enumerate(hold_idx):
            oof[original] = [m.predict_proba(hold_x[pos:pos+1])[0] for m in models]
    if any(item is None for item in oof):
        raise AssertionError("Grouped OOF predictions did not cover every training row")

    pre = create_preprocessing_pipeline(list(X_train.columns))
    train_x = pre.fit_transform(X_train)
    cfg = yaml.safe_load((ROOT / "configs" / "baseline_config.yaml").read_text())
    for model_cfg in cfg["models"].values():
        model_cfg["random_state"] = seed
    models = [
        train_logistic_regression(train_x, y_train, cfg),
        train_random_forest(train_x, y_train, cfg),
        train_gradient_boosting(train_x, y_train, cfg),
    ]
    extractor = ReliabilityFeatureExtractor(X_train, primary_model_index=1)
    oof_features = extractor.extract_features_batch(X_train, oof)
    oof_errors = np.asarray([int(np.argmax(p[1]) != int(label)) for p, label in zip(oof, y_train)])
    if len(np.unique(oof_errors)) != 2:
        raise ValueError("Reliability target has only one class")
    rel_train, rel_cal = grouped_stratified_holdout_indices(
        X_train, pd.Series(oof_errors, index=X_train.index), test_size=0.2, random_state=seed+2
    )
    reliability = ReliabilityEstimator(random_state=seed)
    reliability.train(oof_features.iloc[rel_train], oof_errors[rel_train],
                      oof_features.iloc[rel_cal], oof_errors[rel_cal])

    def scores_for(frame):
        transformed = pre.transform(frame)
        probs = [np.asarray(m.predict_proba(transformed), dtype=float) for m in models]
        rows = [[probs[m][i] for m in range(len(models))] for i in range(len(frame))]
        features = extractor.extract_features_batch(frame, rows)
        primary = probs[1]
        pred = np.asarray(models[1].predict(transformed))
        p_error = reliability.predict_error_probability(features)
        confidence_risk = 1.0 - primary.max(axis=1)
        entropy = -np.sum(np.clip(primary, 1e-12, 1) * np.log(np.clip(primary, 1e-12, 1)), axis=1) / np.log(primary.shape[1])
        disagreement = features["disagreement_fraction"].to_numpy()
        harms = class_harm_weights(pred)
        return pred, p_error, confidence_risk, entropy, disagreement, harms

    # Validation is used for diagnostics only; policy parameters are pre-registered.
    val_pred, val_rel, val_conf, val_ent, val_dis, val_harm = scores_for(X_val)
    test_pred, rel, conf, ent, dis, harms = scores_for(X_test)
    budget_count = int(np.floor(len(X_test) * REVIEW_BUDGET_FRACTION))
    ranked = {
        "reliability_cost_sensitive": rel,
        "confidence_cost_sensitive": conf,
        "entropy_cost_sensitive": ent,
        "disagreement_cost_sensitive": dis,
    }
    selections = {
        name: select_under_budget(score, harms, budget_count, seed)
        for name, score in ranked.items()
    }
    random_order = np.random.default_rng(seed).permutation(len(X_test))
    random_mask = np.zeros(len(X_test), dtype=bool)
    random_mask[random_order[:budget_count]] = True
    selections["random_budgeted"] = random_mask
    results = {
        name: evaluate_selection(name, mask, test_pred, y_test.to_numpy(), harms)
        for name, mask in selections.items()
    }

    # Matched-capacity diagnostic: all ranking signals review exactly the same
    # 10% test capacity. Selection uses scores and predicted-class scenario
    # weights only; test labels are read solely by evaluate_selection afterward.
    matched_rankings = {
        "matched_capacity_reliability": rel,
        "matched_capacity_confidence": conf,
        "matched_capacity_entropy": ent,
        "matched_capacity_disagreement": dis,
    }
    for name, score in matched_rankings.items():
        mask = select_top_budget(score, harms, budget_count)
        results[name] = evaluate_selection(name, mask, test_pred, y_test.to_numpy(), harms)
    results["matched_capacity_random"] = evaluate_selection(
        "matched_capacity_random", random_mask, test_pred, y_test.to_numpy(), harms
    )
    # Include no-review reference for context.
    results["no_review"] = evaluate_selection(
        "no_review", np.zeros(len(X_test), dtype=bool), test_pred, y_test.to_numpy(), harms
    )
    result = {
        "dataset": metadata, "seed": seed,
        "split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "split_prevalence": {"train": float(y_train.mean()), "validation": float(y_val.mean()), "test": float(y_test.mean())},
        "primary_model": "random_forest",
        "policy": "rank by P(error) proxy times predicted-error-direction scenario harm; select positive expected-net-benefit reviews under a hard 10% cap",
        "scenario_assumptions": {
            "review_budget_fraction": REVIEW_BUDGET_FRACTION,
            "review_cost_units_per_case": REVIEW_COST_UNITS,
            "simulated_review_effectiveness": REVIEW_EFFECTIVENESS,
            "false_negative_harm_units": FALSE_NEGATIVE_HARM,
            "false_positive_harm_units": FALSE_POSITIVE_HARM,
            "note": "All cost, consequence, and effectiveness values are hypothetical scenario settings; HTRU2 contains no decision-harm or review-cost labels."
        },
        "results": results,
        "limitations": [
            "Offline test labels are used only after policy selection to score outcomes.",
            "The class-asymmetric consequence weights, review cost, and reviewer effectiveness are scenario assumptions, not empirical domain costs.",
            "A second astronomy dataset is not evidence of broad domain generalization or real-world risk reduction.",
            "The random policy is a budget-matched reference; three seeds provide descriptive, not inferential, stability evidence."
        ]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"htru2_seed_{seed}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v["simulated_net_benefit"] for k, v in results.items()}, sort_keys=True))
    return result


def main(seeds):
    raw_dir = ROOT / "datasets" / "raw"
    out_dir = ROOT / "results" / "htru2_budgeted"
    runs = [run_seed(seed, raw_dir, out_dir) for seed in seeds]
    summary = {"dataset": runs[0]["dataset"], "seeds": seeds, "per_seed": runs, "aggregate": {}}
    names = list(runs[0]["results"])
    for name in names:
        fields = ("reviewed_fraction", "review_cost_units", "simulated_net_benefit",
                  "weighted_error_loss_before_review", "weighted_error_loss_after_simulated_review")
        summary["aggregate"][name] = {}
        for field in fields:
            values = [float(run["results"][name][field]) for run in runs]
            summary["aggregate"][name][field + "_mean"] = float(np.mean(values))
            summary["aggregate"][name][field + "_population_sd"] = float(np.std(values))
    (out_dir / "htru2_budgeted_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least one unique seed")
    main(args.seeds)
