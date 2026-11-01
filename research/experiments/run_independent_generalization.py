"""Evaluate VIMARSH independently on two external UCI domains.

Each dataset gets its own preprocessing, primary models, OOF error targets,
reliability estimator, validation threshold and held-out test evaluation.
Nothing is trained on one dataset and transferred to another feature space.
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

from src.datasets.independent_benchmarks import (
    load_banknote_dataset, load_ionosphere_dataset,
)
from src.preprocessing import (
    create_preprocessing_pipeline,
    grouped_stratified_holdout_indices,
    grouped_stratified_kfold_indices,
)
from src.train import (
    train_logistic_regression, train_random_forest, train_gradient_boosting,
)
from src.reliability.feature_extractor import ReliabilityFeatureExtractor
from src.reliability.reliability_model import ReliabilityEstimator
from src.evaluation.threshold_diagnostics import validation_threshold_metrics

MODEL_NAMES = ("logistic_regression", "random_forest", "gradient_boosting")
DATASETS = {
    "banknote": load_banknote_dataset,
    "ionosphere": load_ionosphere_dataset,
}


def _config(seed):
    config = yaml.safe_load((ROOT / "configs" / "baseline_config.yaml").read_text())
    config["random_seed"] = int(seed)
    config["np_random_seed"] = int(seed)
    for model_config in config["models"].values():
        model_config["random_state"] = int(seed)
    return config


def _fit_models(X, y, seed):
    preprocessor = create_preprocessing_pipeline(list(X.columns))
    transformed = preprocessor.fit_transform(X)
    config = _config(seed)
    models = [
        train_logistic_regression(transformed, y, config),
        train_random_forest(transformed, y, config),
        train_gradient_boosting(transformed, y, config),
    ]
    return preprocessor, models


def _split(X, y, seed):
    train_val_idx, test_idx = grouped_stratified_holdout_indices(
        X, y, test_size=0.20, random_state=seed, n_splits=5,
    )
    X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]
    train_idx_local, val_idx_local = grouped_stratified_holdout_indices(
        X_train_val, y_train_val, test_size=0.20, random_state=seed + 1, n_splits=5,
    )
    train_idx = train_val_idx[train_idx_local]
    val_idx = train_val_idx[val_idx_local]
    return train_idx, val_idx, test_idx


def _entropy(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1) / np.log(p.shape[1])


def _batch_probabilities(models, transformed):
    return [np.asarray(model.predict_proba(transformed), dtype=float) for model in models]


def run_dataset_seed(dataset_name, loader, seed, raw_dir):
    X, y, metadata = loader(raw_dir)
    train_idx, val_idx, test_idx = _split(X, y, seed)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # OOF predictions are generated only within the training partition.
    oof = [None] * len(X_train)
    for fold, (fit_idx, hold_idx) in enumerate(
        grouped_stratified_kfold_indices(
            X_train, y_train, n_splits=5, random_state=seed,
        )
    ):
        fold_pre = create_preprocessing_pipeline(list(X_train.columns))
        fit_x = fold_pre.fit_transform(X_train.iloc[fit_idx])
        hold_x = fold_pre.transform(X_train.iloc[hold_idx])
        config = _config(seed + fold)
        fold_models = [
            train_logistic_regression(fit_x, y_train.iloc[fit_idx], config),
            train_random_forest(fit_x, y_train.iloc[fit_idx], config),
            train_gradient_boosting(fit_x, y_train.iloc[fit_idx], config),
        ]
        fold_probs = _batch_probabilities(fold_models, hold_x)
        for local_pos, original_pos in enumerate(hold_idx):
            oof[original_pos] = [probs[local_pos] for probs in fold_probs]
    if any(item is None for item in oof):
        raise AssertionError("OOF predictions did not cover every training observation")

    preprocessor, models = _fit_models(X_train, y_train, seed)
    val_probs = _batch_probabilities(models, preprocessor.transform(X_val))
    test_probs = _batch_probabilities(models, preprocessor.transform(X_test))

    # One grouped calibration holdout for the reliability estimator.
    per_model = {}
    for primary_index, model_name in enumerate(MODEL_NAMES):
        extractor = ReliabilityFeatureExtractor(
            X_train, primary_model_index=primary_index,
        )
        oof_features = extractor.extract_features_batch(X_train, oof)
        oof_errors = np.asarray([
            int(np.argmax(prob_set[primary_index]) != int(label))
            for prob_set, label in zip(oof, y_train.to_numpy())
        ], dtype=int)
        # Stratify the reliability fit/calibration split on OOF error labels,
        # not the original task labels, so rare error cases are represented.
        rel_fit_idx, rel_cal_idx = grouped_stratified_holdout_indices(
            X_train, pd.Series(oof_errors, index=X_train.index),
            test_size=0.25, random_state=seed + 20 + primary_index, n_splits=4,
        )
        if len(np.unique(oof_errors[rel_fit_idx])) != 2:
            raise ValueError(f"{dataset_name}/{seed}/{model_name}: reliability fit split has one error class")
        if len(np.unique(oof_errors[rel_cal_idx])) != 2:
            raise ValueError(f"{dataset_name}/{seed}/{model_name}: calibration split has one error class")

        estimator = ReliabilityEstimator(
            model_type="random_forest", random_state=seed + primary_index,
        )
        estimator.train(
            oof_features.iloc[rel_fit_idx], oof_errors[rel_fit_idx],
            oof_features.iloc[rel_cal_idx], oof_errors[rel_cal_idx],
        )

        val_rows = [
            [val_probs[m][i] for m in range(len(MODEL_NAMES))]
            for i in range(len(X_val))
        ]
        test_rows = [
            [test_probs[m][i] for m in range(len(MODEL_NAMES))]
            for i in range(len(X_test))
        ]
        val_features = extractor.extract_features_batch(X_val, val_rows)
        test_features = extractor.extract_features_batch(X_test, test_rows)

        val_pred = np.asarray(models[primary_index].predict(
            preprocessor.transform(X_val)
        ))
        test_pred = np.asarray(models[primary_index].predict(
            preprocessor.transform(X_test)
        ))
        val_primary = val_probs[primary_index]
        test_primary = test_probs[primary_index]

        val_scores = {
            "reliability_driven": estimator.predict_error_probability(val_features),
            "confidence_only": 1.0 - val_primary.max(axis=1),
            "predictive_entropy": _entropy(val_primary),
            "ensemble_disagreement": val_features["disagreement_fraction"].to_numpy(),
        }
        test_scores = {
            "reliability_driven": estimator.predict_error_probability(test_features),
            "confidence_only": 1.0 - test_primary.max(axis=1),
            "predictive_entropy": _entropy(test_primary),
            "ensemble_disagreement": test_features["disagreement_fraction"].to_numpy(),
        }
        policy_results = {
            name: validation_threshold_metrics(
                val_scores[name], test_scores[name], test_pred, y_test.to_numpy(),
                target_coverage=0.75,
            )
            for name in val_scores
        }
        per_model[model_name] = {
            "test_accuracy": float(np.mean(test_pred == y_test.to_numpy())),
            "test_error_rate": float(np.mean(test_pred != y_test.to_numpy())),
            "validation_selected_nominal_75pct_coverage": policy_results,
            "thresholds_selected_on_validation_only": True,
            "test_labels_used_after_selection_only": True,
            "reliability_target": f"{model_name}_out_of_fold_prediction_error",
        }

    return {
        "dataset": metadata,
        "seed": int(seed),
        "split_sizes": {
            "train": int(len(train_idx)), "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "split_prevalence": {
            "train": float(y_train.mean()), "validation": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "primary_model_results": per_model,
        "duplicate_feature_groups_disjoint_across_splits": True,
    }


def summarize(results):
    summary = {}
    for dataset_name in DATASETS:
        rows = [item for item in results if item["dataset_key"] == dataset_name]
        summary[dataset_name] = {}
        for model_name in MODEL_NAMES:
            model_rows = [row["primary_model_results"][model_name] for row in rows]
            summary[dataset_name][model_name] = {
                metric: {
                    "mean": float(np.mean([row[metric] for row in model_rows])),
                    "population_sd": float(np.std([row[metric] for row in model_rows])),
                }
                for metric in ("test_accuracy", "test_error_rate")
            }
            summary[dataset_name][model_name]["policies"] = {}
            for policy in (
                "reliability_driven", "confidence_only",
                "predictive_entropy", "ensemble_disagreement",
            ):
                coverages = [
                    row["validation_selected_nominal_75pct_coverage"][policy]["test_coverage"]
                    for row in model_rows
                ]
                errors = [
                    row["validation_selected_nominal_75pct_coverage"][policy]["observed_error_rate"]
                    for row in model_rows
                ]
                summary[dataset_name][model_name]["policies"][policy] = {
                    "test_coverage_mean": float(np.mean(coverages)),
                    "test_coverage_population_sd": float(np.std(coverages)),
                    "accepted_error_mean": float(np.mean(errors)),
                    "accepted_error_population_sd": float(np.std(errors)),
                }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "independent_generalization")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide at least one unique seed")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset_name, loader in DATASETS.items():
        for seed in args.seeds:
            result = run_dataset_seed(dataset_name, loader, seed, ROOT / "datasets" / "raw")
            result["dataset_key"] = dataset_name
            all_results.append(result)
            (args.output_dir / f"{dataset_name}_seed_{seed}.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
    summary = {
        "seeds": args.seeds,
        "dataset_summaries": summarize(all_results),
        "interpretation": (
            "Independent per-dataset experiments. Each benchmark is trained and evaluated "
            "within its own feature space; metrics are never pooled across datasets. "
            "Means and population SDs across three seeds are descriptive only."
        ),
        "limitations": [
            "The two UCI benchmarks are independent offline tabular tasks, not production domains.",
            "Reliability estimators are trained from grouped out-of-fold prediction errors per dataset.",
            "Human review, action effectiveness, and causal harm reduction are not evaluated.",
            "A separate dataset evaluation is not cross-dataset model transfer.",
        ],
    }
    (args.output_dir / "independent_generalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
