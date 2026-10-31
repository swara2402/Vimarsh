"""Combined reliability feature extractor that aggregates all signal types."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from .uncertainty import calculate_uncertainty_features
from .disagreement import calculate_disagreement_features
from .input_quality import InputQualityAnalyzer
from .distribution_shift import DistributionShiftDetector


class ReliabilityFeatureExtractor:
    """Extract reliability signals using an explicitly designated primary model.

    The primary model index must match the model whose prediction-error labels
    train the reliability estimator. Disagreement features still use every model.
    """

    def __init__(self, training_data: pd.DataFrame, primary_model_index: int = 1):
        """Initialize analyzers and record the model whose prediction is evaluated.

        The Wine Quality experiment uses Random Forest at index 1 in the ordered
        list [Logistic Regression, Random Forest, Gradient Boosting].
        """
        if not isinstance(primary_model_index, int) or primary_model_index < 0:
            raise ValueError("primary_model_index must be a non-negative integer")
        self.primary_model_index = primary_model_index
        self.input_quality_analyzer = InputQualityAnalyzer(training_data)
        self.shift_detector = DistributionShiftDetector(training_data)
        self.feature_names = self._get_full_feature_list()

    def _get_full_feature_list(self) -> List[str]:
        """Get all reliability feature names in their stable training order."""
        uncertainty_features = [
            'max_probability', 'predictive_entropy',
            'probability_margin', 'uncertainty_score'
        ]
        disagreement_features = [
            'count_class0', 'count_class1', 'disagreement_fraction',
            'probability_variance', 'probability_std', 'ensemble_entropy'
        ]
        quality_features = [
            'missing_feature_count', 'missing_feature_ratio',
            'invalid_value_count', 'out_of_range_count', 'z_score_anomaly_count',
            'isolation_forest_anomaly', 'isolation_forest_score', 'overall_input_quality_score'
        ]
        shift_features = [
            'mahalanobis_distance', 'mahalanobis_percentile', 'is_extreme_distance'
        ]
        return uncertainty_features + disagreement_features + quality_features + shift_features

    def extract_features_single_sample(
        self,
        sample: pd.Series,
        model_probs_list: List[np.ndarray]
    ) -> Dict[str, float]:
        """Extract uncertainty from the primary model and disagreement from all models."""
        if self.primary_model_index >= len(model_probs_list):
            raise ValueError(
                f"primary_model_index {self.primary_model_index} is outside "
                f"the {len(model_probs_list)} supplied model probability arrays"
            )
        uncertainty_features = calculate_uncertainty_features(
            model_probs_list[self.primary_model_index]
        )
        disagreement_features = calculate_disagreement_features(model_probs_list)
        quality_features = self.input_quality_analyzer.analyze_single_sample(sample)
        shift_features = self.shift_detector.analyze_single_sample(sample)

        all_features = {}
        all_features.update(uncertainty_features)
        all_features.update(disagreement_features)
        all_features.update(quality_features)
        all_features.update(shift_features)
        return all_features

    def extract_features_batch(
        self,
        samples: pd.DataFrame,
        batch_model_probs: List[List[np.ndarray]]
    ) -> pd.DataFrame:
        """Extract reliability features for a batch while preserving row order."""
        if len(samples) != len(batch_model_probs):
            raise ValueError(
                "samples and batch_model_probs must have the same number of rows"
            )
        features_list = []
        for i in range(len(samples)):
            features_list.append(
                self.extract_features_single_sample(samples.iloc[i], batch_model_probs[i])
            )
        return pd.DataFrame(features_list)
