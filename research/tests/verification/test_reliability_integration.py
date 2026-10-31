"""Tests for the reliability-to-verification inference bridge."""
import numpy as np
import pandas as pd
import pytest

from src.verification.reliability_integration import ReliabilityVerificationPipeline


class FakeFeatureExtractor:
    def extract_features_batch(self, samples, batch_model_probs):
        assert len(batch_model_probs) == len(samples)
        return pd.DataFrame({"signal": [float(p[0][0]) for p in batch_model_probs]})


class FakeReliabilityEstimator:
    feature_names = ["signal"]

    def predict_error_probability(self, features):
        return features["signal"].to_numpy()


class FakeVerificationEngine:
    def __init__(self):
        self.kwargs = None

    def process_batch(self, **kwargs):
        self.kwargs = kwargs
        return ["evaluation"]


@pytest.fixture
def inputs():
    samples = pd.DataFrame({"x": [1.0, 2.0]})
    probabilities = [
        np.array([[0.8, 0.2], [0.3, 0.7]]),
        np.array([[0.7, 0.3], [0.4, 0.6]]),
    ]
    return samples, probabilities


def make_pipeline():
    return ReliabilityVerificationPipeline(
        FakeFeatureExtractor(), FakeReliabilityEstimator()
    )


def test_scores_use_model_probabilities_and_keep_row_order(inputs):
    samples, probabilities = inputs
    features, scores = make_pipeline().score(samples, probabilities)
    assert features["signal"].tolist() == [0.8, 0.3]
    assert np.allclose(scores, [0.8, 0.3])


def test_run_verification_passes_estimated_error_probabilities(inputs):
    samples, probabilities = inputs
    engine = FakeVerificationEngine()
    result = make_pipeline().run_verification(
        engine=engine,
        samples=samples,
        predictions=[0, 1],
        model_probabilities=probabilities,
        harms=[2.0, 3.0],
    )
    assert engine.kwargs["error_probabilities"] == pytest.approx([0.8, 0.3])
    assert engine.kwargs["confidences"] == pytest.approx([0.8, 0.7])
    assert result.evaluations == ["evaluation"]


def test_rejects_model_probability_row_mismatch(inputs):
    samples, probabilities = inputs
    probabilities[1] = probabilities[1][:1]
    with pytest.raises(ValueError, match="expected 2"):
        make_pipeline().score(samples, probabilities)


def test_rejects_probability_rows_that_do_not_sum_to_one(inputs):
    samples, probabilities = inputs
    probabilities[0][0] = [0.8, 0.8]
    with pytest.raises(ValueError, match="sum to 1"):
        make_pipeline().score(samples, probabilities)


def test_rejects_estimator_output_with_wrong_length(inputs):
    samples, probabilities = inputs
    bad_estimator = type(
        "BadEstimator",
        (),
        {
            "feature_names": ["signal"],
            "predict_error_probability": lambda self, features: np.array([0.2]),
        },
    )()
    pipeline = ReliabilityVerificationPipeline(FakeFeatureExtractor(), bad_estimator)
    with pytest.raises(ValueError, match="different number of scores"):
        pipeline.score(samples, probabilities)


def test_rejects_missing_estimator_features(inputs):
    samples, probabilities = inputs
    bad_estimator = type(
        "MissingFeatureEstimator",
        (),
        {
            "feature_names": ["not_present"],
            "predict_error_probability": lambda self, features: np.array([0.2, 0.3]),
        },
    )()
    pipeline = ReliabilityVerificationPipeline(FakeFeatureExtractor(), bad_estimator)
    with pytest.raises(ValueError, match="missing estimator features"):
        pipeline.score(samples, probabilities)


def test_rejects_nonfinite_estimator_scores(inputs):
    samples, probabilities = inputs
    bad_estimator = type(
        "NonfiniteEstimator",
        (),
        {
            "feature_names": ["signal"],
            "predict_error_probability": lambda self, features: np.array([0.2, np.nan]),
        },
    )()
    pipeline = ReliabilityVerificationPipeline(FakeFeatureExtractor(), bad_estimator)
    with pytest.raises(ValueError, match="non-finite"):
        pipeline.score(samples, probabilities)
