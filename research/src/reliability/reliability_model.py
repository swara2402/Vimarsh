"""Reliability estimator that predicts probability of prediction error."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


class ReliabilityEstimator:
    """Supervised model that estimates probability that an original prediction is incorrect."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize reliability estimator.
        
        Args:
            model_type: One of 'logistic_regression', 'random_forest', 'gradient_boosting'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._initialize_model()
        self.is_trained = False
        self.feature_names = None
    
    def _initialize_model(self) -> Any:
        """Initialize the underlying sklearn model."""
        if self.model_type == 'logistic_regression':
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, n_estimators=100)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """
        Train the reliability estimator. Optionally calibrate on validation data.
        
        Args:
            X_train: Training reliability features
            y_train: Training error labels (1 = incorrect prediction, 0 = correct)
            X_val: Validation features for calibration
            y_val: Validation labels for calibration
        
        Returns:
            Training metrics
        """
        self.feature_names = X_train.columns.tolist()
        
        # Train base model
        self.model.fit(X_train, y_train)
        
        # Calibrate if validation data is provided (manual implementation for scikit-learn < 1.2 compatibility)
        if X_val is not None and y_val is not None:
            # Get model probabilities on validation data for calibration
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            # Fit Platt scaling (sigmoid calibration) manually
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression(random_state=self.random_state)
            calibrator.fit(val_pred_proba.reshape(-1, 1), y_val)
            
            # Create a wrapper that applies calibration
            original_model = self.model
            class CalibratedModelWrapper:
                def __init__(self, base_model, calibrator):
                    self.base_model = base_model
                    self.calibrator = calibrator
                def predict(self, X):
                    return self.base_model.predict(X)
                def predict_proba(self, X):
                    base_probs = self.base_model.predict_proba(X)[:, 1]
                    calibrated_probs_pos = self.calibrator.predict_proba(base_probs.reshape(-1, 1))[:, 1]
                    return np.stack([1 - calibrated_probs_pos, calibrated_probs_pos], axis=1)
            
            self.model = CalibratedModelWrapper(original_model, calibrator)
        
        self.is_trained = True
        
        # Calculate training metrics
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        train_metrics = self._calculate_metrics(y_train, train_pred_proba)
        
        return train_metrics
    
    def predict_error_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of prediction error for samples.
        
        Args:
            X: Reliability features
        
        Returns:
            Array of error probabilities (P(error=1))
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Dynamically find the error class (1) index using model.classes_
        probas = self.model.predict_proba(X)
        if hasattr(self.model, 'classes_'):
            error_class_idx = np.where(self.model.classes_ == 1)[0][0] if 1 in self.model.classes_ else 1
        else:
            error_class_idx = 1  # Fallback to index 1 if classes_ not available
            
        return probas[:, error_class_idx]
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Classification metrics
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
        metrics['pr_auc'] = float(average_precision_score(y_true, y_pred_proba))
        metrics['brier_score'] = float(brier_score_loss(y_true, y_pred_proba))
        metrics['log_loss'] = float(log_loss(y_true, y_pred_proba))
        
        # Calculate ECE (Expected Calibration Error)
        n_bins = 10
        ece = 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for bin_idx in range(n_bins):
            bin_mask = (y_pred_proba >= bin_edges[bin_idx]) & (y_pred_proba < bin_edges[bin_idx + 1])
            bin_count = np.sum(bin_mask)
            
            if bin_count > 0:
                bin_accuracy = np.mean(y_true[bin_mask])  # Fraction of actual errors in bin
                bin_confidence = np.mean(y_pred_proba[bin_mask])
                ece += bin_count / len(y_true) * abs(bin_confidence - bin_accuracy)
        
        metrics['ece'] = float(ece)
        
        return metrics
    
    def evaluate(self, X_test: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        y_pred_proba = self.predict_error_probability(X_test)
        return self._calculate_metrics(y_true, y_pred_proba)
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'random_state': self.random_state,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> 'ReliabilityEstimator':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(model_type=data['model_type'], random_state=data['random_state'])
        instance.model = data['model']
        instance.is_trained = data['is_trained']
        instance.feature_names = data['feature_names']
        return instance