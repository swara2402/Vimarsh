"""Uncertainty feature extraction for reliability engine."""
import numpy as np
from typing import Dict, List, Any


def calculate_uncertainty_features(probs: np.ndarray) -> Dict[str, float]:
    """
    Calculate uncertainty features from model probabilities.
    
    Args:
        probs: Model predicted probabilities for binary classification, shape (n_samples, 2)
              or for a single sample, shape (2,)
    
    Returns:
        Dictionary of uncertainty features
    """
    # Handle single sample case
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    
    features = {}
    
    # Maximum predicted class probability (confidence)
    max_probs = np.max(probs, axis=1)
    features['max_probability'] = float(max_probs[0]) if len(max_probs) == 1 else float(np.mean(max_probs))
    
    # Predictive entropy
    eps = 1e-12  # Prevent log(0)
    entropy = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
    features['predictive_entropy'] = float(entropy[0]) if len(entropy) == 1 else float(np.mean(entropy))
    
    # Probability margin between top two classes
    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    features['probability_margin'] = float(margins[0]) if len(margins) == 1 else float(np.mean(margins))
    
    # For binary classification, add additional uncertainty metric
    if probs.shape[1] == 2:
        # Uncertainty score: distance from 0.5 decision boundary
        distance_from_boundary = np.abs(probs[:, 1] - 0.5)
        uncertainty_score = 1.0 - (2 * distance_from_boundary)  # 1 = most uncertain, 0 = most confident
        features['uncertainty_score'] = float(uncertainty_score[0]) if len(uncertainty_score) == 1 else float(np.mean(uncertainty_score))
    
    return features


def calculate_batch_uncertainty_features(probs_batch: np.ndarray) -> List[Dict[str, float]]:
    """Calculate uncertainty features for a batch of samples."""
    return [calculate_uncertainty_features(probs) for probs in probs_batch]