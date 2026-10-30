"""
Training module for baseline ML models.
Implements training for Logistic Regression, Random Forest, and Gradient Boosting.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import yaml

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load baseline configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: pd.Series,
    config: Dict[str, Any]
) -> LogisticRegression:
    """Train a logistic regression model."""
    lr_config = config['models']['logistic_regression']
    model = LogisticRegression(
        max_iter=lr_config['max_iter'],
        C=0.1,  # Add L2 regularization (smaller C = stronger regularization)
        random_state=lr_config['random_state']
    )
    model.fit(X_train, y_train)
    print("Logistic Regression training complete")
    return model

def train_random_forest(
    X_train: np.ndarray,
    y_train: pd.Series,
    config: Dict[str, Any]
) -> RandomForestClassifier:
    """Train a random forest classifier."""
    rf_config = config['models']['random_forest']
    model = RandomForestClassifier(
        n_estimators=rf_config['n_estimators'],
        max_depth=8,  # Add regularization to prevent overfitting
        min_samples_split=10,  # Require at least 10 samples to split
        min_samples_leaf=5,  # Require at least 5 samples in leaf
        random_state=rf_config['random_state']
    )
    model.fit(X_train, y_train)
    print("Random Forest training complete")
    return model

def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: pd.Series,
    config: Dict[str, Any]
) -> GradientBoostingClassifier:
    """Train a gradient boosting classifier."""
    gb_config = config['models']['gradient_boosting']
    model = GradientBoostingClassifier(
        n_estimators=gb_config['n_estimators'],
        learning_rate=gb_config['learning_rate'],
        max_depth=4,  # Add regularization to prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=gb_config['random_state']
    )
    model.fit(X_train, y_train)
    print("Gradient Boosting training complete")
    return model

def save_model(model: Any, model_name: str, save_dir: Path) -> None:
    """Save trained model to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{model_name}.joblib"
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

def load_model(model_path: Path) -> Any:
    """Load trained model from disk."""
    return joblib.load(model_path)

def train_all_models(
    X_train_processed: np.ndarray,
    y_train: pd.Series,
    config: Dict[str, Any],
    models_dir: Path
) -> Dict[str, Any]:
    """Train all enabled baseline models."""
    models = {}
    
    if config['models']['logistic_regression']['enabled']:
        lr_model = train_logistic_regression(X_train_processed, y_train, config)
        models['logistic_regression'] = lr_model
        save_model(lr_model, 'logistic_regression', models_dir)
    
    if config['models']['random_forest']['enabled']:
        rf_model = train_random_forest(X_train_processed, y_train, config)
        models['random_forest'] = rf_model
        save_model(rf_model, 'random_forest', models_dir)
    
    if config['models']['gradient_boosting']['enabled']:
        gb_model = train_gradient_boosting(X_train_processed, y_train, config)
        models['gradient_boosting'] = gb_model
        save_model(gb_model, 'gradient_boosting', models_dir)
    
    return models

if __name__ == "__main__":
    # Test training pipeline
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / "configs" / "baseline_config.yaml"
    config = load_config(config_path)
    
    from data_loader import load_dataset, create_binary_target
    from preprocessing import stratified_data_splitting, create_preprocessing_pipeline, fit_preprocessing
    
    # Load and prepare data
    raw_dir = base_dir / "datasets" / "raw"
    data, stats = load_dataset(str(raw_dir))
    data = create_binary_target(data, threshold=config['quality_threshold'])
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_data_splitting(
        data, 'good_quality',
        train_size=config['train_size'],
        val_size=config['val_size'],
        test_size=config['test_size'],
        random_state=config['random_seed']
    )
    
    # Preprocess data
    preprocessor = create_preprocessing_pipeline(list(X_train.columns))
    preprocessor = fit_preprocessing(preprocessor, X_train)
    X_train_processed = preprocessor.transform(X_train)
    
    # Train models
    models_dir = base_dir / "models"
    models = train_all_models(X_train_processed, y_train, config, models_dir)
    print(f"\nTrained {len(models)} models successfully")