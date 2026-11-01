from pathlib import Path

import importlib.util

import pandas as pd
import pytest

from src.datasets.independent_benchmarks import (
    BANKNOTE_FEATURES,
    IONOSPHERE_FEATURES,
    parse_banknote_csv,
    parse_ionosphere_csv,
)


def test_banknote_parser_preserves_binary_labels_and_feature_order(tmp_path):
    path = tmp_path / "banknote.csv"
    pd.DataFrame([
        [3.6216, 8.6661, -2.8073, -0.44699, 0],
        [4.5459, 8.1674, -2.4586, -1.4621, 1],
    ]).to_csv(path, header=False, index=False)

    X, y = parse_banknote_csv(path)

    assert list(X.columns) == BANKNOTE_FEATURES
    assert X.shape == (2, 4)
    assert y.tolist() == [0, 1]


def test_ionosphere_parser_maps_g_and_b_and_checks_schema(tmp_path):
    path = tmp_path / "ionosphere.data"
    pd.DataFrame([
        [0.0] * 34 + ["b"],
        [1.0] * 34 + ["g"],
    ]).to_csv(path, header=False, index=False)

    X, y = parse_ionosphere_csv(path)

    assert list(X.columns) == IONOSPHERE_FEATURES
    assert X.shape == (2, 34)
    assert y.tolist() == [0, 1]


def test_parsers_reject_non_finite_features(tmp_path):
    path = tmp_path / "banknote.csv"
    pd.DataFrame([
        [float("nan"), 1.0, 2.0, 3.0, 0],
        [4.0, 5.0, 6.0, 7.0, 1],
    ]).to_csv(path, header=False, index=False)

    with pytest.raises(ValueError, match="finite"):
        parse_banknote_csv(path)


def test_independent_summary_keeps_datasets_and_models_separate():
    script = Path(__file__).resolve().parents[1] / "experiments" / "run_independent_generalization.py"
    spec = importlib.util.spec_from_file_location("independent_generalization", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def result(dataset_key):
        policies = {
            name: {"test_coverage": 0.75, "observed_error_rate": 0.1}
            for name in ("reliability_driven", "confidence_only", "predictive_entropy", "ensemble_disagreement")
        }
        models = {
            name: {
                "test_accuracy": 0.9,
                "test_error_rate": 0.1,
                "validation_selected_nominal_75pct_coverage": policies,
            }
            for name in ("logistic_regression", "random_forest", "gradient_boosting")
        }
        return {"dataset_key": dataset_key, "primary_model_results": models}

    summary = module.summarize([result("banknote"), result("ionosphere")])

    assert set(summary) == {"banknote", "ionosphere"}
    assert set(summary["banknote"]) == {
        "logistic_regression", "random_forest", "gradient_boosting"
    }
    assert summary["banknote"]["random_forest"]["test_accuracy"]["mean"] == 0.9
