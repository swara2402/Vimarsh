"""
Confidence analysis module for investigating prediction confidence vs correctness.
Calculates confidence, correctness, and bucketed accuracy analysis.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

def calculate_prediction_confidence(probabilities: np.ndarray) -> np.ndarray:
    """
    Calculate prediction confidence as max(P(y=0|x), P(y=1|x)).
    """
    return np.max(probabilities, axis=1)

def calculate_prediction_correctness(predictions: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Calculate binary correctness: 1 if prediction matches true label, else 0.
    """
    return (predictions == true_labels).astype(int)

def create_prediction_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    sample_ids: List[int] = None
) -> pd.DataFrame:
    """
    Create comprehensive DataFrame with all prediction metrics.
    Includes actual label, predicted label, probabilities, confidence, correctness.
    """
    confidence = calculate_prediction_confidence(y_prob)
    correctness = calculate_prediction_correctness(y_pred, y_true)
    
    if sample_ids is None:
        sample_ids = list(range(len(y_true)))
    
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'actual_label': y_true,
        'predicted_label': y_pred,
        'probability_class_0': y_prob[:, 0],
        'probability_class_1': y_prob[:, 1],
        'confidence': confidence,
        'correctness': correctness,
        'model_name': model_name
    })
    
    return df

def analyze_confidence_buckets(
    predictions_df: pd.DataFrame,
    buckets: List[Tuple[float, float]]
) -> pd.DataFrame:
    """
    Analyze predictions by confidence buckets.
    For each bucket, calculate count, average confidence, actual accuracy, incorrect predictions.
    """
    bucket_stats = []
    
    for low, high in buckets:
        mask = (predictions_df['confidence'] >= low) & (predictions_df['confidence'] < high)
        bucket_df = predictions_df[mask]
        
        if len(bucket_df) > 0:
            avg_confidence = bucket_df['confidence'].mean()
            accuracy = bucket_df['correctness'].mean()
            incorrect = (1 - bucket_df['correctness']).sum()
            count = len(bucket_df)
        else:
            avg_confidence = np.nan
            accuracy = np.nan
            incorrect = 0
            count = 0
        
        bucket_stats.append({
            'bucket_low': low,
            'bucket_high': high,
            'bucket_label': f"{int(low*100)}-{int(high*100)}%",
            'predictions_count': count,
            'average_confidence': avg_confidence,
            'actual_accuracy': accuracy,
            'incorrect_predictions': incorrect
        })
    
    return pd.DataFrame(bucket_stats)

def analyze_high_confidence_errors(predictions_df: pd.DataFrame, threshold: float = 0.9) -> Dict[str, Any]:
    """
    Analyze highly confident but incorrect predictions.
    """
    high_confidence_mask = predictions_df['confidence'] >= threshold
    high_conf_df = predictions_df[high_confidence_mask]
    incorrect_high_conf = high_conf_df[high_conf_df['correctness'] == 0]
    
    stats = {
        'total_high_confidence': len(high_conf_df),
        'incorrect_high_confidence': len(incorrect_high_conf),
        'error_rate_high_conf': len(incorrect_high_conf) / len(high_conf_df) if len(high_conf_df) > 0 else 0.0,
        'share_of_all_errors': len(incorrect_high_conf) / (predictions_df['correctness'] == 0).sum() if (predictions_df['correctness'] == 0).sum() > 0 else 0.0
    }
    
    return stats

def confidence_vs_accuracy_trend(predictions_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Calculate how accuracy changes with increasing confidence.
    """
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    bin_indices = np.digitize(predictions_df['confidence'], bins) - 1
    
    trend_stats = []
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        bin_df = predictions_df[mask]
        if len(bin_df) > 0:
            trend_stats.append({
                'confidence_bin': bins[bin_idx],
                'avg_confidence': bin_df['confidence'].mean(),
                'accuracy': bin_df['correctness'].mean(),
                'count': len(bin_df)
            })
    
    return pd.DataFrame(trend_stats)

if __name__ == "__main__":
    # Test confidence analysis
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Create test data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model and predict
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Create prediction dataframe
    pred_df = create_prediction_dataframe(y_test, y_pred, y_prob, 'random_forest')
    print(f"Created prediction dataframe: {pred_df.shape}")
    
    # Analyze confidence buckets
    buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    bucket_analysis = analyze_confidence_buckets(pred_df, buckets)
    print("\nConfidence bucket analysis:")
    print(bucket_analysis[['bucket_label', 'predictions_count', 'actual_accuracy']])
    
    # Analyze high confidence errors
    high_conf_errors = analyze_high_confidence_errors(pred_df, threshold=0.9)
    print("\nHigh confidence error analysis:")
    print(high_conf_errors)