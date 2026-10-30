"""Combined reliability feature extractor that aggregates all signal types."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .uncertainty import calculate_uncertainty_features
from .disagreement import calculate_disagreement_features
from .input_quality import InputQualityAnalyzer
from .distribution_shift import DistributionShiftDetector


class ReliabilityFeatureExtractor:
    """Combined feature extractor that extracts all reliability signals."""
    
    def __init__(self, training_data: pd.DataFrame):
        """
        Initialize with training data to fit all analyzers.
        
        Args:
            training_data: Training data used to fit the original models
        """
        self.input_quality_analyzer = InputQualityAnalyzer(training_data)
        self.shift_detector = DistributionShiftDetector(training_data)
        self.feature_names = self._get_full_feature_list()
    
    def _get_full_feature_list(self) -> List[str]:
        """Get list of all reliability feature names."""
        # Uncertainty features
        uncertainty_features = ['max_probability', 'predictive_entropy', 
                               'probability_margin', 'uncertainty_score']
        # Disagreement features
        disagreement_features = ['count_class0', 'count_class1', 'disagreement_fraction',
                                'probability_variance', 'probability_std', 'ensemble_entropy']
        # Input quality features
        quality_features = ['missing_feature_count', 'missing_feature_ratio',
                           'invalid_value_count', 'out_of_range_count', 'z_score_anomaly_count',
                           'isolation_forest_anomaly', 'isolation_forest_score', 'overall_input_quality_score']
        # Distribution shift features
        shift_features = ['mahalanobis_distance', 'mahalanobis_percentile', 'is_extreme_distance']
        
        return uncertainty_features + disagreement_features + quality_features + shift_features
    
    def extract_features_single_sample(
        self,
        sample: pd.Series,
        model_probs_list: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Extract all reliability features for a single sample.
        
        Args:
            sample: Input sample as pandas Series
            model_probs_list: List of predicted probabilities from each base model
        
        Returns:
            Dictionary of all reliability features
        """
        # Extract uncertainty features from first model (or average if needed)
        # We'll use the main model's probabilities for uncertainty, but could ensemble
        uncertainty_features = calculate_uncertainty_features(model_probs_list[0])
        
        # Extract disagreement features across all models
        disagreement_features = calculate_disagreement_features(model_probs_list)
        
        # Extract input quality features
        quality_features = self.input_quality_analyzer.analyze_single_sample(sample)
        
        # Extract distribution shift features
        shift_features = self.shift_detector.analyze_single_sample(sample)
        
        # Combine all features
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
        """
        Extract reliability features for an entire batch.
        
        Args:
            samples: DataFrame of input samples
            batch_model_probs: List of model probability lists for each sample
        
        Returns:
            DataFrame of reliability features
        """
        features_list = []
        for i in range(len(samples)):
            sample_features = self.extract_features_single_sample(
                samples.iloc[i],
                batch_model_probs[i]
            )
            features_list.append(sample_features)
        
        return pd.DataFrame(features_list)