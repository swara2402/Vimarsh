"""Trusted local artifact adapter for the trained VIMARSH reliability bundle."""
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd


class ReliabilityBundleRuntime:
    """Load and run a versioned research inference bundle.

    The bundle path must point to an artifact produced by the VIMARSH research
    pipeline and supplied by the operator. Never load user-uploaded joblib files.
    """

    REQUIRED_KEYS = {
        "feature_extractor", "reliability_estimator", "base_models",
        "preprocessor", "primary_model_index", "artifact_version",
    }

    def __init__(self, bundle_path: str):
        path = Path(bundle_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Reliability bundle not found: {path}")
        bundle: Any = joblib.load(path)
        if not isinstance(bundle, dict) or not self.REQUIRED_KEYS.issubset(bundle):
            raise ValueError("Reliability bundle is missing required components")
        if bundle["artifact_version"] != 2:
            raise ValueError("Unsupported reliability bundle version")
        self.feature_extractor = bundle["feature_extractor"]
        self.reliability_estimator = bundle["reliability_estimator"]
        self.base_models = bundle["base_models"]
        self.preprocessor = bundle["preprocessor"]
        self.primary_model_index = int(bundle["primary_model_index"])
        if not self.base_models or not 0 <= self.primary_model_index < len(self.base_models):
            raise ValueError("Reliability bundle has an invalid primary model index")
        if not getattr(self.reliability_estimator, "is_trained", False):
            raise ValueError("Reliability estimator in bundle is not trained")
        self.feature_names = list(self.feature_extractor.input_quality_analyzer.feature_names)

    def score_one(self, sample: dict[str, float]) -> dict[str, Any]:
        if set(sample) != set(self.feature_names):
            missing = sorted(set(self.feature_names) - set(sample))
            extra = sorted(set(sample) - set(self.feature_names))
            raise ValueError(f"Input feature schema mismatch; missing={missing}, extra={extra}")
        frame = pd.DataFrame([[sample[name] for name in self.feature_names]], columns=self.feature_names)
        transformed = self.preprocessor.transform(frame)
        probability_matrices = [
            np.asarray(model.predict_proba(transformed), dtype=float)
            for model in self.base_models
        ]
        for matrix in probability_matrices:
            if matrix.shape[0] != 1 or matrix.ndim != 2 or matrix.shape[1] < 2:
                raise ValueError("A base model returned an invalid probability matrix")
            if not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
                raise ValueError("A base model returned invalid probabilities")
            if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Base-model probabilities must sum to one")

        per_sample = [[matrix[0] for matrix in probability_matrices]]
        features = self.feature_extractor.extract_features_batch(frame, per_sample)
        expected = getattr(self.reliability_estimator, "feature_names", None)
        if expected:
            missing = [name for name in expected if name not in features.columns]
            if missing:
                raise ValueError("Reliability feature mismatch: " + ", ".join(missing))
            features = features.loc[:, expected]
        error_probability = float(
            np.asarray(self.reliability_estimator.predict_error_probability(features)).reshape(-1)[0]
        )
        if not np.isfinite(error_probability) or not 0 <= error_probability <= 1:
            raise ValueError("Reliability estimator returned an invalid error probability")

        primary = self.base_models[self.primary_model_index]
        primary_probabilities = probability_matrices[self.primary_model_index][0]
        classes = getattr(primary, "classes_", np.arange(len(primary_probabilities)))
        prediction = primary.classes_[int(np.argmax(primary_probabilities))]
        return {
            "prediction": str(prediction),
            "confidence": float(np.max(primary_probabilities)),
            "estimated_error_probability": error_probability,
            "base_model_count": len(self.base_models),
            "primary_model_index": self.primary_model_index,
        }
