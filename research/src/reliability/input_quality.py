"""Input quality analysis for reliability engine."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import IsolationForest


class InputQualityAnalyzer:
    """Analyzes input quality and detects anomalies in input features."""
    
    def __init__(self, training_data: pd.DataFrame):
        """
        Initialize with training data to compute feature statistics.
        
        Args:
            training_data: Training data used to fit the original model
        """
        self.training_stats = self._compute_training_stats(training_data)
        self.feature_names = training_data.columns.tolist()
        self.isolation_forest = IsolationForest(random_state=42)
        self.isolation_forest.fit(training_data.values)
    
    def _compute_training_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Compute min, max, mean, std for each feature from training data."""
        stats = {}
        for col in df.columns:
            stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
        return stats
    
    def analyze_single_sample(self, sample: pd.Series) -> Dict[str, float]:
        """
        Analyze quality of a single input sample.
        
        Args:
            sample: Single sample as pandas Series with matching features to training data
        
        Returns:
            Dictionary of input quality features
        """
        # Check for missing values
        missing_count = int(sample.isna().sum())
        missing_ratio = float(missing_count / len(sample))
        
        # Check for invalid numeric values
        invalid_count = 0
        out_of_range_count = 0
        z_score_anomalies = 0
        
        for col, val in sample.items():
            if pd.notna(val):
                # Check for infinite or NaN (already counted)
                if not np.isfinite(val):
                    invalid_count += 1
                else:
                    # Check if value is outside training min/max
                    stats = self.training_stats[col]
                    if val < stats['min'] or val > stats['max']:
                        out_of_range_count += 1
                    
                    # Check z-score > 3 (3 sigma rule)
                    z_score = abs((val - stats['mean']) / stats['std']) if stats['std'] > 0 else 0
                    if z_score > 3:
                        z_score_anomalies += 1
        
        # Isolation Forest anomaly score
        sample_array = sample.fillna(0).values.reshape(1, -1)
        anomaly_score = float(self.isolation_forest.score_samples(sample_array)[0])
        is_anomaly = 1 if self.isolation_forest.predict(sample_array)[0] == -1 else 0
        
        # Overall quality score (0 = perfect, higher = worse)
        overall_quality_score = (
            missing_ratio * 0.3 +
            (out_of_range_count / len(sample)) * 0.3 +
            (z_score_anomalies / len(sample)) * 0.2 +
            (1 - np.clip(anomaly_score + 1, 0, 1)) * 0.2
        )
        
        return {
            'missing_feature_count': missing_count,
            'missing_feature_ratio': missing_ratio,
            'invalid_value_count': invalid_count,
            'out_of_range_count': out_of_range_count,
            'z_score_anomaly_count': z_score_anomalies,
            'isolation_forest_anomaly': is_anomaly,
            'isolation_forest_score': anomaly_score,
            'overall_input_quality_score': float(overall_quality_score)
        }
    
    def analyze_batch(self, samples: pd.DataFrame) -> List[Dict[str, float]]:
        """Analyze quality for a batch of samples."""
        return [self.analyze_single_sample(samples.iloc[i]) for i in range(len(samples))]