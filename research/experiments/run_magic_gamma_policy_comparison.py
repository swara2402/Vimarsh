"""Grouped, multi-seed reliability and policy evaluation on UCI MAGIC.

This is a second-dataset benchmark, not a claim of cross-domain generalization.
Human review is simulated; harm is a constant scenario assumption because MAGIC
has no supplied decision-cost or consequence labels.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

RESEARCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESEARCH_DIR))

from src.datasets.magic_gamma import load_magic_dataset
from src.preprocessing import (
    create_preprocessing_pipeline,
    grouped_stratified_holdout_indices,
    grouped_stratified_kfold_indices,
)
from src.train import (
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
)
from src.reliability.feature_extractor import ReliabilityFeatureExtractor
from src.reliability.reliability_model import ReliabilityEstimator
from src.verification.action_selector import VerificationEngine, VerificationEngineConfig
from src.verification.cost_model import DecisionProfile
from src.verification.outcome_validator import OutcomeValidator


def _groups(X):
    return pd.MultiIndex.from_frame(X).factorize(sort=False)[0]


def grouped_magic_split(X, y, seed):
    """Grouped stratified 70/15/15 split retaining every row and isolating duplicates."""
    groups = _groups(X)

    def best_holdout(indices, target_fraction, folds, random_state):
        local_X = X.iloc[indices]
        local_y = y.iloc[indices]
        local_groups = groups[indices]
        splitter = StratifiedGroupKFold(
            n_splits=folds, shuffle=True, random_state=random_state
        )
        candidates = []
        for _, held in splitter.split(local_X, local_y, local_groups):
            prevalence_gap = abs(float(local_y.iloc[held].mean()) - float(local_y.mean()))
            size_gap = abs(len(held) / len(indices) - target_fraction)
            candidates.append((size_gap + 0.5 * prevalence_gap, indices[held]))
        return min(candidates, key=lambda item: item[0])[1]

    all_idx = np.arange(len(X))
    if len(np.unique(groups)) < 7:
        raise ValueError("MAGIC requires at least seven unique feature groups")
    test_idx = best_holdout(all_idx, 0.15, 7, seed)
    remaining = np.setdiff1d(all_idx, test_idx, assume_unique=False)
    val_fraction = 0.15 / 0.85
    val_idx = best_holdout(remaining, val_fraction, 6, seed + 1)
    train_idx = np.setdiff1d(remaining, val_idx, assume_unique=False)

    group_sets = [set(groups[idx]) for idx in (train_idx, val_idx, test_idx)]
    if (group_sets[0] & group_sets[1]) or (group_sets[0] & group_sets[2]) or (group_sets[1] & group_sets[2]):
        raise AssertionError("Exact feature duplicates crossed MAGIC data partitions")
    return train_idx, val_idx, test_idx


def validation_threshold_metrics(validation_scores, test_scores, test_predictions, test_labels, target_coverage=0.75):
    """Select a risk-score threshold on validation data, then evaluate untouched test rows."""
    validation_scores = np.asarray(validation_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    test_predictions = np.asarray(test_predictions)
    test_labels = np.asarray(test_labels)
    if validation_scores.ndim != 1 or test_scores.ndim != 1:
        raise ValueError("Risk scores must be one-dimensional")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between zero and one")
    if len(test_scores) != len(test_predictions) or len(test_scores) != len(test_labels):
        raise ValueError("Test scores, predictions, and labels must have equal lengths")
    if not np.isfinite(validation_scores).all() or not np.isfinite(test_scores).all():
        raise ValueError("Risk scores must be finite")
    threshold = float(np.quantile(validation_scores, target_coverage, method="higher"))
    accepted = test_scores <= threshold
    accepted_count = int(accepted.sum())
    observed_error = (
        float(np.mean(test_predictions[accepted] != test_labels[accepted]))
        if accepted_count else None
    )
    return {
        "validation_selected_threshold": threshold,
        "target_coverage": float(target_coverage),
        "test_coverage": float(accepted.mean()),
        "accepted_test_observations": accepted_count,
        "observed_error_rate": observed_error,
    }



def reliability_ablation_metrics(
    oof_features, oof_error, train_indices, calibration_indices,
    validation_features, validation_error, random_state,
):
    """Fit feature-group ablations on OOF training data; score validation only."""
    from sklearn.metrics import (
        average_precision_score, brier_score_loss, log_loss, roc_auc_score,
    )

    groups = {
        "uncertainty": [
            "max_probability", "predictive_entropy", "probability_margin",
            "uncertainty_score",
        ],
        "disagreement": [
            "count_class0", "count_class1", "disagreement_fraction",
            "probability_variance", "probability_std", "ensemble_entropy",
        ],
        "input_quality": [
            "missing_feature_count", "missing_feature_ratio",
            "invalid_value_count", "out_of_range_count", "z_score_anomaly_count",
            "isolation_forest_anomaly", "isolation_forest_score",
            "overall_input_quality_score",
        ],
        "distribution_shift": [
            "mahalanobis_distance", "mahalanobis_percentile", "is_extreme_distance",
        ],
    }
    variants = {
        "A_uncertainty_only": groups["uncertainty"],
        "B_uncertainty_disagreement": groups["uncertainty"] + groups["disagreement"],
        "C_plus_input_quality": (
            groups["uncertainty"] + groups["disagreement"] + groups["input_quality"]
        ),
        "D_full_features": sum(groups.values(), []),
    }
    results = {}
    for name, columns in variants.items():
        estimator = ReliabilityEstimator(model_type="random_forest", random_state=random_state)
        estimator.train(
            oof_features.iloc[train_indices][columns],
            np.asarray(oof_error)[train_indices],
            oof_features.iloc[calibration_indices][columns],
            np.asarray(oof_error)[calibration_indices],
        )
        probabilities = estimator.predict_error_probability(validation_features[columns])
        results[name] = {
            "validation_roc_auc": float(roc_auc_score(validation_error, probabilities)),
            "validation_pr_auc": float(average_precision_score(validation_error, probabilities)),
            "validation_brier_score": float(brier_score_loss(validation_error, probabilities)),
            "validation_log_loss": float(log_loss(validation_error, probabilities, labels=[0, 1])),
            "validation_ece": float(_expected_calibration_error(validation_error, probabilities)),
            "feature_count": len(columns),
        }
    return results


def _expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities < edges[index + 1] if index < bins - 1 else probabilities <= edges[index + 1]
        )
        if np.any(mask):
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return error


def _fit_models(X, y, config, seed):
    cfg = yaml.safe_load((RESEARCH_DIR / "configs" / "baseline_config.yaml").read_text())
    cfg["random_seed"] = seed
    cfg["np_random_seed"] = seed
    for model_cfg in cfg["models"].values():
        model_cfg["random_state"] = seed
    preprocessor = create_preprocessing_pipeline(list(X.columns))
    Xp = preprocessor.fit_transform(X)
    models = [
        train_logistic_regression(Xp, y, cfg),
        train_random_forest(Xp, y, cfg),
        train_gradient_boosting(Xp, y, cfg),
    ]
    return preprocessor, models


def run_seed(seed, output_dir):
    X, y, metadata = load_magic_dataset(RESEARCH_DIR / "datasets" / "raw")
    train_idx, val_idx, test_idx = grouped_magic_split(X, y, seed)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    oof_probabilities = [None] * len(X_train)
    for fold, (fit_idx, hold_idx) in enumerate(
        grouped_stratified_kfold_indices(X_train, y_train, n_splits=5, random_state=seed)
    ):
        pre = create_preprocessing_pipeline(list(X_train.columns))
        X_fit = pre.fit_transform(X_train.iloc[fit_idx])
        X_hold = pre.transform(X_train.iloc[hold_idx])
        cfg = yaml.safe_load((RESEARCH_DIR / "configs" / "baseline_config.yaml").read_text())
        for model_cfg in cfg["models"].values():
            model_cfg["random_state"] = seed + fold
        models = [
            train_logistic_regression(X_fit, y_train.iloc[fit_idx], cfg),
            train_random_forest(X_fit, y_train.iloc[fit_idx], cfg),
            train_gradient_boosting(X_fit, y_train.iloc[fit_idx], cfg),
        ]
        for local_pos, original_pos in enumerate(hold_idx):
            oof_probabilities[original_pos] = [
                model.predict_proba(X_hold[local_pos:local_pos + 1])[0]
                for model in models
            ]

    if any(item is None for item in oof_probabilities):
        raise AssertionError("OOF predictions do not cover every MAGIC training row")
    final_preprocessor, final_models = _fit_models(X_train, y_train, {}, seed)
    extractor = ReliabilityFeatureExtractor(X_train, primary_model_index=1)
    oof_features = extractor.extract_features_batch(X_train, oof_probabilities)
    oof_error = np.asarray([
        int(np.argmax(prob_set[1]) != int(label))
        for prob_set, label in zip(oof_probabilities, y_train.to_numpy())
    ], dtype=int)
    if len(np.unique(oof_error)) != 2:
        raise ValueError("OOF error labels contain only one class; reliability model cannot be trained")

    rel_train_idx, rel_cal_idx = grouped_stratified_holdout_indices(
        X_train, pd.Series(oof_error, index=X_train.index), test_size=0.2,
        random_state=seed + 2,
    )
    estimator = ReliabilityEstimator(model_type="random_forest", random_state=seed)
    estimator.train(
        oof_features.iloc[rel_train_idx], oof_error[rel_train_idx],
        oof_features.iloc[rel_cal_idx], oof_error[rel_cal_idx],
    )

    processed_val = final_preprocessor.transform(X_val)
    val_probs = [
        np.asarray(model.predict_proba(processed_val), dtype=float)
        for model in final_models
    ]
    val_features = extractor.extract_features_batch(
        X_val,
        [[val_probs[m][i] for m in range(len(final_models))] for i in range(len(X_val))],
    )
    val_reliability_scores = estimator.predict_error_probability(val_features)

    validation_primary_predictions = np.asarray(final_models[1].predict(processed_val))
    validation_error_labels = (
        validation_primary_predictions != y_val.to_numpy()
    ).astype(int)
    validation_ablation_results = reliability_ablation_metrics(
        oof_features, oof_error, rel_train_idx, rel_cal_idx,
        val_features, validation_error_labels, seed,
    )
    val_primary_probs = val_probs[1]
    val_entropy = -np.sum(
        np.clip(val_primary_probs, 1e-12, 1.0) * np.log(np.clip(val_primary_probs, 1e-12, 1.0)),
        axis=1,
    ) / np.log(val_primary_probs.shape[1])
    validation_policy_scores = {
        "reliability_driven": val_reliability_scores,
        "confidence_only": 1.0 - val_primary_probs.max(axis=1),
        "predictive_entropy": val_entropy,
        "ensemble_disagreement": val_features["disagreement_fraction"].to_numpy(),
    }

    processed_test = final_preprocessor.transform(X_test)
    model_probs = [
        np.asarray(model.predict_proba(processed_test), dtype=float)
        for model in final_models
    ]
    test_features = extractor.extract_features_batch(
        X_test,
        [[model_probs[m][i] for m in range(len(final_models))] for i in range(len(X_test))],
    )
    reliability_scores = estimator.predict_error_probability(test_features)
    primary_probs = model_probs[1]
    predictions = np.asarray(final_models[1].predict(processed_test))
    confidences = primary_probs.max(axis=1)

    # MAGIC supplies no harm labels. A constant 5/10 consequence score is used
    # for every row so policies differ only by their ranking signal.
    harms = [5.0] * len(X_test)
    class_entropy = -np.sum(
        np.clip(primary_probs, 1e-12, 1.0) * np.log(np.clip(primary_probs, 1e-12, 1.0)),
        axis=1,
    ) / np.log(primary_probs.shape[1])
    policy_scores = {
        "reliability_driven": reliability_scores,
        "confidence_only": 1.0 - confidences,
        "predictive_entropy": class_entropy,
        "ensemble_disagreement": test_features["disagreement_fraction"].to_numpy(),
        "always_accept": np.zeros(len(X_test)),
    }

    validation_matched_coverage = {
        name: validation_threshold_metrics(
            validation_policy_scores[name], policy_scores[name],
            predictions, y_test.to_numpy(), target_coverage=0.75,
        )
        for name in validation_policy_scores
    }

    comparisons = {}
    with tempfile.TemporaryDirectory(prefix=f"vimarsh-magic-{seed}-") as temp:
        model_dir = Path(temp)
        joblib.dump(final_preprocessor, model_dir / "preprocessor.joblib")
        for name, model in zip(
            ("logistic_regression", "random_forest", "gradient_boosting"), final_models
        ):
            joblib.dump(model, model_dir / f"{name}.joblib")

        for policy_name, scores in policy_scores.items():
            engine = VerificationEngine(VerificationEngineConfig(
                total_verification_budget=500.0,
                decision_profile=DecisionProfile.BALANCED,
                base_model_path=model_dir,
                original_model_name="random_forest",
                enable_evidence_provider=False,
                enable_human_reviewer=True,
                max_human_reviews=50,
            ))
            evaluations = engine.process_batch(
                X_test, predictions, confidences.tolist(),
                np.asarray(scores, dtype=float).tolist(), harms,
            )
            np.random.seed(seed)
            validator = OutcomeValidator(engine.action_executor)
            for i, evaluation in enumerate(evaluations):
                validator.execute_and_evaluate_action(
                    sample=X_test.iloc[i],
                    base_prediction=predictions[i],
                    base_confidence=float(confidences[i]),
                    action_type=evaluation.selected_action,
                    true_label=int(y_test.iloc[i]),
                    evaluation=evaluation,
                )
            comparisons[policy_name] = validator.calculate_policy_metrics(policy_name).__dict__

    result = {
        "dataset": metadata,
        "seed": int(seed),
        "split_sizes": {
            "train": int(len(train_idx)), "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_class_prevalence": {
            "train": float(y_train.mean()), "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "unique_feature_groups": int(len(_groups(X))),
        "cross_partition_duplicate_groups": 0,
        "primary_model": "random_forest",
        "reliability_training": "five-fold grouped OOF predictions; primary RF errors; grouped reliability train/calibration holdout",
        "matched_coverage_protocol": "For each ranking policy, select a 75% risk-score quantile threshold using validation scores only, then evaluate coverage and observed error on the untouched test partition. Test labels are used only for the final metric.",
        "policy_comparisons": comparisons,
        "validation_threshold_matched_coverage": validation_matched_coverage,
        "validation_feature_ablation": validation_ablation_results,
        "harm_assumption": "Constant 5.0/10 for every test observation; MAGIC provides no domain harm labels.",
        "limitations": [
            "Human review is simulated and ground truth is used only after action selection for offline outcome simulation.",
            "Evidence provider is disabled.",
            "Action-effectiveness factors are configured assumptions, not measured causal effects.",
            "A constant harm score is a neutral benchmark assumption, not a domain risk model.",
            "This benchmark is a second tabular dataset, not evidence of broad cross-domain or real-world generalization.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"magic_gamma_seed_{seed}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Saved {path}")
    for name, metrics in comparisons.items():
        rate = metrics["observed_error_rate"]
        print(f"{name}: coverage={metrics['coverage']:.3%}, covered_error={rate}, net={metrics['net_corrections']}")
    return result


def main(seeds):
    output_dir = RESEARCH_DIR / "results" / "magic_gamma"
    results = [run_seed(seed, output_dir) for seed in seeds]
    policy_names = list(results[0]["policy_comparisons"])
    summary = {"dataset": results[0]["dataset"], "seeds": seeds, "runs": results,
               "aggregate": {}}
    for name in policy_names:
        rows = [run["policy_comparisons"][name] for run in results]
        metrics = {}
        for field in ("coverage", "observed_error_rate", "net_corrections"):
            values = np.asarray([
                np.nan if row[field] is None else float(row[field]) for row in rows
            ], dtype=float)
            metrics[field + "_mean"] = float(np.nanmean(values))
            metrics[field + "_population_sd"] = float(np.nanstd(values, ddof=0))
            metrics[field + "_per_seed"] = [
                None if row[field] is None else float(row[field]) for row in rows
            ]
        summary["aggregate"][name] = metrics
    out = output_dir / "magic_gamma_multiseed_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Aggregate results: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    args = parser.parse_args()
    main(args.seeds)
