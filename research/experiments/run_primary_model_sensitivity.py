"""Compare reliability-aware error ranking across primary model families on MAGIC.

All primary models use identical grouped train/validation/test partitions per seed.
Reliability estimators are trained from out-of-fold errors for the corresponding
primary model. Thresholds are selected on validation scores and frozen on test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from src.evaluation.threshold_diagnostics import validation_threshold_metrics
from src.evaluation.paired_bootstrap import paired_bootstrap_policy_comparisons
from experiments.run_magic_gamma_policy_comparison import grouped_magic_split, _expected_calibration_error

MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")
OUTPUT_DIR = ROOT / "results" / "primary_model_sensitivity"


def _config(seed):
    cfg = yaml.safe_load((ROOT / "configs" / "baseline_config.yaml").read_text())
    cfg["random_seed"] = seed
    cfg["np_random_seed"] = seed
    for model_cfg in cfg["models"].values():
        model_cfg["random_state"] = seed
    return cfg


def _fit_models(X, y, seed):
    preprocessor = create_preprocessing_pipeline(list(X.columns))
    X_processed = preprocessor.fit_transform(X)
    cfg = _config(seed)
    models = [
        train_logistic_regression(X_processed, y, cfg),
        train_random_forest(X_processed, y, cfg),
        train_gradient_boosting(X_processed, y, cfg),
    ]
    return preprocessor, models


def _probability_entropy(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])


def _metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "ece": float(_expected_calibration_error(labels, probabilities)),
    }


def run_seed(seed):
    X, y, metadata = load_magic_dataset(ROOT / "datasets" / "raw")
    train_idx, val_idx, test_idx = grouped_magic_split(X, y, seed)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Generate one set of out-of-fold predictions from all three base models.
    oof_probabilities = [None] * len(X_train)
    for fold, (fit_idx, hold_idx) in enumerate(
        grouped_stratified_kfold_indices(
            X_train, y_train, n_splits=5, random_state=seed
        )
    ):
        preprocessor = create_preprocessing_pipeline(list(X_train.columns))
        X_fit = preprocessor.fit_transform(X_train.iloc[fit_idx])
        X_hold = preprocessor.transform(X_train.iloc[hold_idx])
        cfg = _config(seed + fold)
        fold_models = [
            train_logistic_regression(X_fit, y_train.iloc[fit_idx], cfg),
            train_random_forest(X_fit, y_train.iloc[fit_idx], cfg),
            train_gradient_boosting(X_fit, y_train.iloc[fit_idx], cfg),
        ]
        for local_pos, original_pos in enumerate(hold_idx):
            oof_probabilities[original_pos] = [
                model.predict_proba(X_hold[local_pos:local_pos + 1])[0]
                for model in fold_models
            ]
    if any(item is None for item in oof_probabilities):
        raise AssertionError("OOF predictions do not cover every training row")

    final_preprocessor, final_models = _fit_models(X_train, y_train, seed)
    processed_val = final_preprocessor.transform(X_val)
    processed_test = final_preprocessor.transform(X_test)
    val_probabilities = [
        np.asarray(model.predict_proba(processed_val), dtype=float)
        for model in final_models
    ]
    test_probabilities = [
        np.asarray(model.predict_proba(processed_test), dtype=float)
        for model in final_models
    ]

    # One grouped calibration holdout, shared by all primary-model comparisons.
    rel_train_idx, rel_cal_idx = grouped_stratified_holdout_indices(
        X_train, y_train, test_size=0.2, random_state=seed + 20
    )
    per_model = {}
    for primary_index, model_name in enumerate(MODEL_NAMES):
        extractor = ReliabilityFeatureExtractor(
            X_train, primary_model_index=primary_index
        )
        oof_features = extractor.extract_features_batch(
            X_train, oof_probabilities
        )
        oof_errors = np.asarray([
            int(np.argmax(prob_set[primary_index]) != int(label))
            for prob_set, label in zip(oof_probabilities, y_train.to_numpy())
        ], dtype=int)
        if len(np.unique(oof_errors[rel_train_idx])) != 2:
            raise ValueError(
                f"{model_name} training error labels contain one class for seed {seed}"
            )
        if len(np.unique(oof_errors[rel_cal_idx])) != 2:
            raise ValueError(
                f"{model_name} calibration error labels contain one class for seed {seed}"
            )

        estimator = ReliabilityEstimator(
            model_type="random_forest", random_state=seed + primary_index
        )
        estimator.train(
            oof_features.iloc[rel_train_idx], oof_errors[rel_train_idx],
            oof_features.iloc[rel_cal_idx], oof_errors[rel_cal_idx],
        )

        val_features = extractor.extract_features_batch(
            X_val,
            [[val_probabilities[m][i] for m in range(len(MODEL_NAMES))]
             for i in range(len(X_val))],
        )
        test_features = extractor.extract_features_batch(
            X_test,
            [[test_probabilities[m][i] for m in range(len(MODEL_NAMES))]
             for i in range(len(X_test))],
        )
        val_predictions = np.asarray(final_models[primary_index].predict(processed_val))
        test_predictions = np.asarray(final_models[primary_index].predict(processed_test))
        val_error = (val_predictions != y_val.to_numpy()).astype(int)
        test_error = (test_predictions != y_test.to_numpy()).astype(int)

        val_primary = val_probabilities[primary_index]
        test_primary = test_probabilities[primary_index]
        val_scores = {
            "reliability_driven": estimator.predict_error_probability(val_features),
            "confidence_only": 1.0 - val_primary.max(axis=1),
            "predictive_entropy": _probability_entropy(val_primary),
            "ensemble_disagreement": val_features["disagreement_fraction"].to_numpy(),
        }
        test_scores = {
            "reliability_driven": estimator.predict_error_probability(test_features),
            "confidence_only": 1.0 - test_primary.max(axis=1),
            "predictive_entropy": _probability_entropy(test_primary),
            "ensemble_disagreement": test_features["disagreement_fraction"].to_numpy(),
        }

        matched = {
            name: validation_threshold_metrics(
                val_scores[name], test_scores[name], test_predictions,
                y_test.to_numpy(), target_coverage=0.75,
            )
            for name in val_scores
        }
        paired = paired_bootstrap_policy_comparisons(
            val_scores, test_scores, test_predictions, y_test.to_numpy(),
            reference_policy="confidence_only", target_coverage=0.75,
            groups=pd.MultiIndex.from_frame(X_test).factorize(sort=False)[0],
            n_bootstrap=1000, random_state=seed + primary_index,
        )
        val_reliability = val_scores["reliability_driven"]
        test_reliability = test_scores["reliability_driven"]
        per_model[model_name] = {
            "primary_model_index": primary_index,
            "test_accuracy": float(np.mean(test_predictions == y_test.to_numpy())),
            "test_error_rate": float(test_error.mean()),
            "validation_reliability_metrics": _metrics(val_error, val_reliability),
            "test_reliability_metrics": _metrics(test_error, test_reliability),
            "validation_selected_nominal_75pct_coverage": matched,
            "paired_bootstrap_vs_confidence_only": paired,
            "reliability_estimator": "random_forest",
            "reliability_target": f"{model_name}_out_of_fold_prediction_error",
            "thresholds_selected_on_validation_only": True,
            "test_labels_used_after_selection_only": True,
        }

    return {
        "dataset": metadata,
        "seed": int(seed),
        "split_sizes": {
            "train": len(X_train), "validation": len(X_val), "test": len(X_test)
        },
        "primary_model_results": per_model,
    }


def summarize_seed_results(seed_results):
    """Return descriptive across-seed means/SDs without pooling test observations."""
    summary = {}
    for model_name in MODEL_NAMES:
        rows = [item["primary_model_results"][model_name] for item in seed_results]
        summary[model_name] = {}
        for metric in ("test_accuracy", "test_error_rate"):
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[model_name][metric] = {
                "mean": float(values.mean()),
                "population_sd": float(values.std(ddof=0)),
            }
        for policy in ("reliability_driven", "confidence_only", "predictive_entropy", "ensemble_disagreement"):
            values_coverage = np.asarray([
                row["validation_selected_nominal_75pct_coverage"][policy]["coverage"]
                for row in rows
            ], dtype=float)
            values_error = np.asarray([
                row["validation_selected_nominal_75pct_coverage"][policy]["accepted_error_rate"]
                for row in rows
            ], dtype=float)
            summary[model_name].setdefault("matched_coverage_policies", {})[policy] = {
                "coverage_mean": float(values_coverage.mean()),
                "coverage_population_sd": float(values_coverage.std(ddof=0)),
                "accepted_error_mean": float(values_error.mean()),
                "accepted_error_population_sd": float(values_error.std(ddof=0)),
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [run_seed(seed) for seed in args.seeds]
    for result in results:
        path = args.output_dir / f"primary_model_sensitivity_seed_{result['seed']}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {
        "seeds": args.seeds,
        "primary_model_summary": summarize_seed_results(results),
        "interpretation": (
            "Descriptive means and population SDs across seeds. Each dataset split "
            "is evaluated independently; no metrics are pooled across models or datasets."
        ),
    }
    (args.output_dir / "primary_model_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
