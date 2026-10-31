"""Regression tests for vectorized reliability feature extraction."""
import numpy as np
import pandas as pd
import pytest

from src.reliability.feature_extractor import ReliabilityFeatureExtractor
from src.reliability.input_quality import InputQualityAnalyzer
from src.reliability.distribution_shift import DistributionShiftDetector


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(13)
    training = pd.DataFrame(
        rng.normal(size=(100, 4)),
        columns=["a", "b", "c", "d"],
    )
    samples = training.iloc[:12].copy()
    samples.iloc[0, 0] = np.nan
    samples.iloc[1, 1] = samples["b"].max() + 4
    return training, samples


def test_input_quality_batch_matches_single_sample_results(sample_data):
    training, samples = sample_data
    analyzer = InputQualityAnalyzer(training)
    batch = analyzer.analyze_batch(samples)
    singles = [analyzer.analyze_single_sample(samples.iloc[i]) for i in range(len(samples))]

    assert len(batch) == len(singles)
    for batch_row, single_row in zip(batch, singles):
        assert batch_row.keys() == single_row.keys()
        for key in batch_row:
            assert batch_row[key] == pytest.approx(single_row[key])


def test_distribution_shift_batch_matches_single_sample_results(sample_data):
    training, samples = sample_data
    detector = DistributionShiftDetector(training)
    batch = detector.analyze_batch(samples)
    singles = [detector.analyze_single_sample(samples.iloc[i]) for i in range(len(samples))]

    assert len(batch) == len(singles)
    for batch_row, single_row in zip(batch, singles):
        assert batch_row.keys() == single_row.keys()
        for key in batch_row:
            assert batch_row[key] == pytest.approx(single_row[key], abs=1e-10)


def test_feature_extractor_batch_matches_single_sample_results(sample_data):
    training, samples = sample_data
    extractor = ReliabilityFeatureExtractor(training, primary_model_index=1)
    rng = np.random.default_rng(31)
    batch_probabilities = []
    for _ in range(len(samples)):
        rows = []
        for _model in range(3):
            positive = rng.uniform(0.1, 0.9)
            rows.append(np.array([1 - positive, positive]))
        batch_probabilities.append(rows)

    batch = extractor.extract_features_batch(samples, batch_probabilities)
    singles = pd.DataFrame([
        extractor.extract_features_single_sample(samples.iloc[i], batch_probabilities[i])
        for i in range(len(samples))
    ])

    pd.testing.assert_frame_equal(batch, singles, check_exact=False, rtol=1e-9, atol=1e-9)
