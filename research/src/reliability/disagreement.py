"""Model disagreement feature extraction for reliability engine."""
import numpy as np
from typing import Dict, List, Any


def calculate_disagreement_features(model_probs_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Calculate model disagreement features from multiple model predictions.
    
    Args:
        model_probs_list: List of probability arrays from each model, each shape (2,) for binary classification
    
    Returns:
        Dictionary of disagreement features
    """
    n_models = len(model_probs_list)
    if n_models < 2:
        raise ValueError("At least 2 models required to calculate disagreement")
    
    # Extract predictions and probabilities
    predictions = []
    probs_class1 = []
    
    for model_probs in model_probs_list:
        pred = np.argmax(model_probs)
        predictions.append(pred)
        probs_class1.append(model_probs[1])
    
    predictions = np.array(predictions)
    probs_class1 = np.array(probs_class1)
    
    # Count predictions per class
    count_class0 = np.sum(predictions == 0)
    count_class1 = np.sum(predictions == 1)
    
    # Majority vote
    majority = 1 if count_class1 > count_class0 else 0
    disagreeing_models = np.sum(predictions != majority)
    disagreement_fraction = disagreeing_models / n_models
    
    # Variance and standard deviation of predicted probabilities
    prob_variance = float(np.var(probs_class1))
    prob_std = float(np.std(probs_class1))
    
    # Ensemble entropy
    mean_prob_class1 = np.mean(probs_class1)
    mean_probs = np.array([1 - mean_prob_class1, mean_prob_class1])
    eps = 1e-12
    ensemble_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, eps, 1.0)))
    
    return {
        'count_class0': int(count_class0),
        'count_class1': int(count_class1),
        'disagreement_fraction': float(disagreement_fraction),
        'probability_variance': prob_variance,
        'probability_std': prob_std,
        'ensemble_entropy': float(ensemble_entropy)
    }


def calculate_batch_disagreement_features(batch_model_probs: List[List[np.ndarray]]) -> List[Dict[str, float]]:
    """Calculate disagreement features for a batch of samples."""
    return [calculate_disagreement_features(sample_model_probs) for sample_model_probs in batch_model_probs]