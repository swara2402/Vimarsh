"""Distribution shift detection for reliability engine."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy.spatial.distance import mahalanobis
from scipy.stats import ks_2samp
from sklearn.covariance import EmpiricalCovariance


class DistributionShiftDetector:
    """Detects distribution shift and sample-level anomalies using multiple methods."""
    
    def __init__(self, training_data: pd.DataFrame):
        """
        Initialize with training data to compute shift detection statistics.
        
        Args:
            training_data: Training data used to fit the original model
        """
        self.training_data = training_data.values
        self.training_df = training_data
        self.feature_names = training_data.columns.tolist()
        
        # Precompute Mahalanobis distance parameters
        self.covariance = EmpiricalCovariance().fit(self.training_data)
        self.inv_cov_matrix = self.covariance.precision_
        self.training_mean = self.covariance.location_
        
        # Precompute ALL training Mahalanobis distances once (not per sample!)
        self.training_mahalanobis_dists = np.array([
            self.calculate_mahalanobis_distance(x) for x in self.training_data
        ])
    
    def calculate_mahalanobis_distance(self, sample: np.ndarray) -> float:
        """Calculate Mahalanobis distance for a single sample from training distribution."""
        return mahalanobis(sample, self.training_mean, self.inv_cov_matrix)
    
    def analyze_single_sample(self, sample: pd.Series) -> Dict[str, float]:
        """
        Analyze distribution shift indicators for a single sample.
        
        Args:
            sample: Single sample as pandas Series
        
        Returns:
            Dictionary of distribution shift features
        """
        sample_values = sample.fillna(0).values
        
        # Mahalanobis distance
        mahalanobis_dist = self.calculate_mahalanobis_distance(sample_values)
        
        # Calculate percentile of Mahalanobis distance in training data using precomputed values
        mahalanobis_percentile = np.sum(self.training_mahalanobis_dists < mahalanobis_dist) / len(self.training_mahalanobis_dists)
        
        # Is this sample in the 95th percentile of distances (unusual)
        is_extreme_distance = 1 if mahalanobis_percentile > 0.95 else 0
        
        return {
            'mahalanobis_distance': float(mahalanobis_dist),
            'mahalanobis_percentile': float(mahalanobis_percentile),
            'is_extreme_distance': int(is_extreme_distance)
        }
    
    def calculate_population_shift(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate population-level distribution shift statistics.
        Compares test data to training data using Kolmogorov-Smirnov tests.
        
        Args:
            test_data: Test data to compare against training
        
        Returns:
            Dictionary of shift statistics per feature
        """
        shift_stats = {}
        significant_shifts = 0
        
        for feature in self.feature_names:
            train_vals = self.training_df[feature].dropna().values
            test_vals = test_data[feature].dropna().values
            
            # KS test for distribution difference
            ks_stat, p_value = ks_2samp(train_vals, test_vals)
            
            shift_stats[feature] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'significant_shift': p_value < 0.05
            }
            
            if p_value < 0.05:
                significant_shifts += 1
        
        shift_stats['overall'] = {
            'features_with_significant_shift': significant_shifts,
            'fraction_features_shifted': significant_shifts / len(self.feature_names)
        }
        
        return shift_stats
    
    def analyze_batch(self, samples: pd.DataFrame) -> List[Dict[str, float]]:
        """Analyze a batch using vectorized Mahalanobis distances and percentiles."""
        if list(samples.columns) != self.feature_names:
            raise ValueError("Batch feature columns must match the fitted training schema")
        values = samples.fillna(0).to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        squared_distances = self.covariance.mahalanobis(values)
        distances = np.sqrt(np.maximum(squared_distances, 0.0))
        sorted_training_distances = np.sort(self.training_mahalanobis_dists)
        percentiles = (
            np.searchsorted(sorted_training_distances, distances, side="left")
            / len(sorted_training_distances)
        )
        extreme = (percentiles > 0.95).astype(int)
        return [
            {
                "mahalanobis_distance": float(distances[i]),
                "mahalanobis_percentile": float(percentiles[i]),
                "is_extreme_distance": int(extreme[i]),
            }
            for i in range(len(samples))
        ]