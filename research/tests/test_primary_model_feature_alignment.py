"""Regression tests for primary-model alignment in reliability features."""
import numpy as np
import pandas as pd
import pytest

import src.reliability.feature_extractor as feature_module
from src.reliability.feature_extractor import ReliabilityFeatureExtractor


class FakeAnalyzer:
    def analyze_single_sample(self, sample):
        return {"quality_signal": 0.0}


def make_extractor(primary_model_index=1):
    extractor = ReliabilityFeatureExtractor.__new__(ReliabilityFeatureExtractor)
    extractor.primary_model_index = primary_model_index
    extractor.input_quality_analyzer = FakeAnalyzer()
    extractor.shift_detector = FakeAnalyzer()
    return extractor


def test_uncertainty_features_use_the_configured_primary_model(monkeypatch):
    observed = []
    monkeypatch.setattr(
        feature_module,
        "calculate_uncertainty_features",
        lambda probabilities: observed.append(np.asarray(probabilities).copy()) or {"primary_signal": float(probabilities[0])},
    )
    monkeypatch.setattr(
        feature_module,
        "calculate_disagreement_features",
        lambda probabilities: {"disagreement_signal": 0.0},
    )
    extractor = make_extractor(primary_model_index=1)
    model_probabilities = [
        np.array([0.95, 0.05]),
        np.array([0.62, 0.38]),
        np.array([0.51, 0.49]),
    ]

    features = extractor.extract_features_single_sample(
        pd.Series({"x": 1.0}), model_probabilities
    )

    assert np.array_equal(observed[0], model_probabilities[1])
    assert features["primary_signal"] == pytest.approx(0.62)


def test_rejects_primary_model_index_outside_probability_list(monkeypatch):
    monkeypatch.setattr(
        feature_module,
        "calculate_uncertainty_features",
        lambda probabilities: {},
    )
    extractor = make_extractor(primary_model_index=2)
    with pytest.raises(ValueError, match="outside"):
        extractor.extract_features_single_sample(
            pd.Series({"x": 1.0}),
            [np.array([0.8, 0.2]), np.array([0.7, 0.3])],
        )


def test_batch_rejects_mismatched_sample_and_probability_counts():
    extractor = make_extractor()
    with pytest.raises(ValueError, match="same number of rows"):
        extractor.extract_features_batch(
            pd.DataFrame({"x": [1.0, 2.0]}),
            [[np.array([0.8, 0.2]), np.array([0.7, 0.3])]],
        )
