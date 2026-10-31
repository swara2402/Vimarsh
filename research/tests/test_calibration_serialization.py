"""Serialization regression tests for fitted calibration wrappers."""
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.calibration import CalibratedClassifierWrapper
from src.reliability.reliability_model import CalibratedProbabilityModel


def test_base_model_calibration_wrapper_is_joblib_serializable(tmp_path):
    X, y = make_classification(
        n_samples=120, n_features=5, n_informative=3, random_state=7
    )
    base = LogisticRegression(max_iter=500, random_state=7).fit(X, y)
    raw_positive = base.predict_proba(X[:40])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw_positive, y[:40])
    wrapped = CalibratedClassifierWrapper(base, calibrator)

    path = tmp_path / "base_model.joblib"
    joblib.dump(wrapped, path)
    restored = joblib.load(path)

    assert np.allclose(wrapped.predict_proba(X[:10]), restored.predict_proba(X[:10]))
    assert np.array_equal(restored.classes_, np.array([0, 1]))


def test_reliability_calibration_wrapper_is_joblib_serializable(tmp_path):
    X, y = make_classification(
        n_samples=120, n_features=5, n_informative=3, random_state=11
    )
    base = LogisticRegression(max_iter=500, random_state=11).fit(X, y)
    raw_positive = base.predict_proba(X[:40])[:, 1]
    calibrator = LogisticRegression(random_state=11).fit(
        raw_positive.reshape(-1, 1), y[:40]
    )
    wrapped = CalibratedProbabilityModel(base, calibrator)

    path = tmp_path / "reliability_model.joblib"
    joblib.dump(wrapped, path)
    restored = joblib.load(path)

    assert np.allclose(wrapped.predict_proba(X[:10]), restored.predict_proba(X[:10]))
    assert np.array_equal(restored.classes_, np.array([0, 1]))
