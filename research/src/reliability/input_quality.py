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
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "std": df[col].std(),
            }
        return stats

    def analyze_single_sample(self, sample: pd.Series) -> Dict[str, float]:
        """Analyze quality of one sample with the fitted feature schema."""
        if list(sample.index) != self.feature_names:
            raise ValueError("Sample feature columns must match the fitted training schema")

        missing_count = int(sample.isna().sum())
        feature_count = max(1, len(self.feature_names))
        missing_ratio = float(missing_count / feature_count)

        invalid_count = 0
        out_of_range_count = 0
        z_score_anomalies = 0

        for col, val in sample.items():
            if pd.notna(val):
                if not np.isfinite(val):
                    invalid_count += 1
                else:
                    stats = self.training_stats[col]
                    if val < stats["min"] or val > stats["max"]:
                        out_of_range_count += 1
                    z_score = (
                        abs((val - stats["mean"]) / stats["std"])
                        if stats["std"] > 0 else 0
                    )
                    if z_score > 3:
                        z_score_anomalies += 1

        # Match batch behavior by replacing NaN and +/-inf with zero only for
        # Isolation Forest scoring. The original values still drive quality counts.
        values = sample.to_numpy(dtype=float)
        safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        sample_array = safe_values.reshape(1, -1)
        anomaly_score = float(self.isolation_forest.score_samples(sample_array)[0])
        is_anomaly = 1 if self.isolation_forest.predict(sample_array)[0] == -1 else 0

        overall_quality_score = (
            missing_ratio * 0.3
            + (out_of_range_count / feature_count) * 0.3
            + (z_score_anomalies / feature_count) * 0.2
            + (1 - np.clip(anomaly_score + 1, 0, 1)) * 0.2
        )

        return {
            "missing_feature_count": missing_count,
            "missing_feature_ratio": missing_ratio,
            "invalid_value_count": invalid_count,
            "out_of_range_count": out_of_range_count,
            "z_score_anomaly_count": z_score_anomalies,
            "isolation_forest_anomaly": is_anomaly,
            "isolation_forest_score": anomaly_score,
            "overall_input_quality_score": float(overall_quality_score),
        }

    def analyze_batch(self, samples: pd.DataFrame) -> List[Dict[str, float]]:
        """Analyze a batch while scoring Isolation Forest only once for the batch."""
        if list(samples.columns) != self.feature_names:
            raise ValueError("Batch feature columns must match the fitted training schema")
        values = samples.to_numpy(dtype=float)
        missing_mask = np.isnan(values)
        finite_mask = np.isfinite(values)
        missing_counts = missing_mask.sum(axis=1).astype(int)
        invalid_counts = ((~finite_mask) & (~missing_mask)).sum(axis=1).astype(int)

        minima = np.asarray([self.training_stats[name]["min"] for name in self.feature_names])
        maxima = np.asarray([self.training_stats[name]["max"] for name in self.feature_names])
        means = np.asarray([self.training_stats[name]["mean"] for name in self.feature_names])
        stds = np.asarray([self.training_stats[name]["std"] for name in self.feature_names])

        out_of_range = finite_mask & ((values < minima) | (values > maxima))
        out_of_range_counts = out_of_range.sum(axis=1).astype(int)
        safe_stds = np.where(stds > 0, stds, 1.0)
        z_scores = np.abs((values - means) / safe_stds)
        z_anomalies = finite_mask & (stds > 0) & (z_scores > 3)
        z_anomaly_counts = z_anomalies.sum(axis=1).astype(int)

        safe_values = np.where(finite_mask, values, 0.0)
        anomaly_scores = self.isolation_forest.score_samples(safe_values)
        anomaly_flags = (self.isolation_forest.predict(safe_values) == -1).astype(int)
        feature_count = max(1, len(self.feature_names))
        missing_ratios = missing_counts / feature_count
        quality_scores = (
            missing_ratios * 0.3
            + (out_of_range_counts / feature_count) * 0.3
            + (z_anomaly_counts / feature_count) * 0.2
            + (1 - np.clip(anomaly_scores + 1, 0, 1)) * 0.2
        )

        return [
            {
                "missing_feature_count": int(missing_counts[i]),
                "missing_feature_ratio": float(missing_ratios[i]),
                "invalid_value_count": int(invalid_counts[i]),
                "out_of_range_count": int(out_of_range_counts[i]),
                "z_score_anomaly_count": int(z_anomaly_counts[i]),
                "isolation_forest_anomaly": int(anomaly_flags[i]),
                "isolation_forest_score": float(anomaly_scores[i]),
                "overall_input_quality_score": float(quality_scores[i]),
            }
            for i in range(len(samples))
        ]
