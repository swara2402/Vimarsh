import numpy as np
import pandas as pd
import pytest

from app.runtime import ReliabilityBundleRuntime


class FakeAnalyzer:
    feature_names = ["x", "y"]


class FakeExtractor:
    input_quality_analyzer = FakeAnalyzer()

    def extract_features_batch(self, samples, batch_probabilities):
        assert len(samples) == len(batch_probabilities) == 1
        return pd.DataFrame([{"score": 0.4}])


class FakeEstimator:
    is_trained = True
    feature_names = ["score"]

    def predict_error_probability(self, features):
        return np.array([0.27])


class FakePreprocessor:
    def transform(self, frame):
        return frame.to_numpy()


class FakeModel:
    classes_ = np.array(["negative", "positive"])

    def predict_proba(self, values):
        return np.array([[0.2, 0.8]])


def runtime():
    instance = ReliabilityBundleRuntime.__new__(ReliabilityBundleRuntime)
    instance.feature_extractor = FakeExtractor()
    instance.reliability_estimator = FakeEstimator()
    instance.base_models = [FakeModel(), FakeModel(), FakeModel()]
    instance.preprocessor = FakePreprocessor()
    instance.primary_model_index = 1
    instance.feature_names = ["x", "y"]
    return instance


def test_runtime_returns_real_model_and_reliability_outputs():
    result = runtime().score_one({"x": 1.0, "y": 2.0})
    assert result["prediction"] == "positive"
    assert result["confidence"] == pytest.approx(0.8)
    assert result["estimated_error_probability"] == pytest.approx(0.27)
    assert result["base_model_count"] == 3


def test_runtime_rejects_missing_or_extra_features():
    with pytest.raises(ValueError, match="schema mismatch"):
        runtime().score_one({"x": 1.0})
    with pytest.raises(ValueError, match="schema mismatch"):
        runtime().score_one({"x": 1.0, "y": 2.0, "label": 1.0})


def test_runtime_rejects_invalid_model_probabilities():
    instance = runtime()
    class InvalidModel:
        classes_ = np.array([0, 1])
        def predict_proba(self, values):
            return np.array([[0.2, 0.2]])
    instance.base_models[0] = InvalidModel()
    with pytest.raises(ValueError, match="sum to one"):
        instance.score_one({"x": 1.0, "y": 2.0})
