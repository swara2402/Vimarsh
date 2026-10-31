"""Bridge the trained reliability estimator to verification action selection.

The pipeline consumes model probability outputs and input samples only. Ground-truth
labels are deliberately absent from this inference-time interface.
"""
from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class ReliabilityVerificationResult:
    """Inference-time reliability scores and the resulting action evaluations."""

    reliability_features: pd.DataFrame
    error_probabilities: np.ndarray
    evaluations: List[Any]


class ReliabilityVerificationPipeline:
    """Connect a fitted feature extractor and error estimator to a verification engine.

    model_probabilities must contain one (n_samples, n_classes) probability
    matrix per independently trained model. The first matrix belongs to the
    primary model used for the prediction being evaluated. All models must use
    the same class ordering.
    """

    def __init__(self, feature_extractor: Any, reliability_estimator: Any):
        self.feature_extractor = feature_extractor
        self.reliability_estimator = reliability_estimator

    def score(
        self,
        samples: pd.DataFrame,
        model_probabilities: Sequence[np.ndarray],
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Extract reliability signals and estimate P(original prediction is wrong)."""
        if not isinstance(samples, pd.DataFrame):
            raise TypeError("samples must be a pandas DataFrame")
        if samples.empty:
            raise ValueError("samples must contain at least one row")
        if not model_probabilities:
            raise ValueError("At least one model probability matrix is required")

        normalized_probabilities = []
        expected_rows = len(samples)
        expected_classes = None

        for model_index, probabilities in enumerate(model_probabilities):
            matrix = np.asarray(probabilities, dtype=float)
            if matrix.ndim != 2:
                raise ValueError(
                    f"Model {model_index} probabilities must be a 2D matrix"
                )
            if matrix.shape[0] != expected_rows:
                raise ValueError(
                    f"Model {model_index} returned {matrix.shape[0]} rows; "
                    f"expected {expected_rows}"
                )
            if matrix.shape[1] < 2:
                raise ValueError("Each model must return probabilities for at least 2 classes")
            if expected_classes is None:
                expected_classes = matrix.shape[1]
            elif matrix.shape[1] != expected_classes:
                raise ValueError("All models must return the same number of classes")
            if not np.isfinite(matrix).all():
                raise ValueError(f"Model {model_index} probabilities contain non-finite values")
            if (matrix < 0).any() or (matrix > 1).any():
                raise ValueError(f"Model {model_index} probabilities must be in [0, 1]")
            if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
                raise ValueError(
                    f"Model {model_index} probability rows must sum to 1"
                )
            normalized_probabilities.append(matrix)

        per_sample_probabilities = [
            [model_matrix[row_index] for model_matrix in normalized_probabilities]
            for row_index in range(expected_rows)
        ]
        features = self.feature_extractor.extract_features_batch(
            samples, per_sample_probabilities
        )

        expected_feature_names = getattr(
            self.reliability_estimator, "feature_names", None
        )
        if expected_feature_names is not None:
            missing = [
                name for name in expected_feature_names if name not in features.columns
            ]
            if missing:
                raise ValueError(
                    "Reliability feature extractor is missing estimator features: "
                    + ", ".join(missing)
                )
            features = features.loc[:, expected_feature_names]

        error_probabilities = np.asarray(
            self.reliability_estimator.predict_error_probability(features),
            dtype=float,
        ).reshape(-1)
        if error_probabilities.shape[0] != expected_rows:
            raise ValueError(
                "Reliability estimator returned a different number of scores "
                f"({error_probabilities.shape[0]}) than input samples ({expected_rows})"
            )
        if not np.isfinite(error_probabilities).all():
            raise ValueError("Reliability estimator returned non-finite error probabilities")
        if (error_probabilities < 0).any() or (error_probabilities > 1).any():
            raise ValueError("Reliability estimator scores must be in [0, 1]")

        return features, error_probabilities

    def run_verification(
        self,
        engine: Any,
        samples: pd.DataFrame,
        predictions: Sequence[Any],
        model_probabilities: Sequence[np.ndarray],
        harms: Sequence[float],
    ) -> ReliabilityVerificationResult:
        """Score samples, then pass real reliability estimates to the action engine."""
        if len(predictions) != len(samples):
            raise ValueError("predictions length must match samples")
        if len(harms) != len(samples):
            raise ValueError("harms length must match samples")

        features, error_probabilities = self.score(samples, model_probabilities)
        primary_confidences = np.asarray(model_probabilities[0], dtype=float).max(axis=1)
        evaluations = engine.process_batch(
            samples=samples,
            predictions=list(predictions),
            confidences=primary_confidences.tolist(),
            error_probabilities=error_probabilities.tolist(),
            harms=list(harms),
        )
        return ReliabilityVerificationResult(
            reliability_features=features,
            error_probabilities=error_probabilities,
            evaluations=evaluations,
        )
