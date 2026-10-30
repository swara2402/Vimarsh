"""
Preprocessing module for data preparation.
Implements stratified train/val/test splitting and preprocessing pipeline.
Ensures no data leakage by fitting preprocessing only on training data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def stratified_data_splitting(
    df: pd.DataFrame,
    target_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into stratified train/validation/test sets.
    Maintains class distribution in all splits.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")
    
    # First split into temp (train+val) and test
    X = df.drop(columns=[target_column, 'quality'])  # Drop both target and original quality column
    y = df[target_column]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Then split temp into train and validation
    val_size_relative = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_relative,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Audit for cross-split duplicates (feature row overlaps across splits)
    print(f"\n=== CROSS-SPLIT DUPLICATE AUDIT ===")
    
    # Convert dataframes to tuples of values for hashing (to compare entire rows)
    train_rows = set(tuple(row) for _, row in X_train.iterrows())
    val_rows = set(tuple(row) for _, row in X_val.iterrows())
    test_rows = set(tuple(row) for _, row in X_test.iterrows())
    
    # Calculate overlaps
    train_val_overlap = train_rows & val_rows
    train_test_overlap = train_rows & test_rows
    val_test_overlap = val_rows & test_rows
    all_splits_overlap = train_rows & val_rows & test_rows
    
    print(f"Total unique feature rows in entire dataset: {len(train_rows | val_rows | test_rows)}")
    print(f"Train-validation cross-split duplicates: {len(train_val_overlap)} ({len(train_val_overlap)/len(X_val):.2%} of val set)")
    print(f"Train-test cross-split duplicates: {len(train_test_overlap)} ({len(train_test_overlap)/len(X_test):.2%} of test set)")
    print(f"Validation-test cross-split duplicates: {len(val_test_overlap)} ({len(val_test_overlap)/len(X_test):.2%} of test set)")
    print(f"Duplicates present in ALL three splits: {len(all_splits_overlap)}")
    
    print(f"\nData splitting complete (stratified, random_state={random_state}):")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(df):.1%})")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(df):.1%})")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(df):.1%})")
    print(f"  Train class distribution: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"  Val class distribution: {y_val.value_counts(normalize=True).to_dict()}")
    print(f"  Test class distribution: {y_test.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def deduplicated_stratified_data_splitting(
    df: pd.DataFrame,
    target_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    DEDUPLICATED VERSION: Split data into stratified train/validation/test sets with no cross-split duplicates.
    First removes all duplicate feature rows to ensure complete isolation between splits.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")
    
    # First deduplicate the entire dataset to eliminate duplicates before splitting
    print("\n=== DEDUPLICATION PREPROCESSING ===")
    # Drop original quality column to keep only features and target
    X_full = df.drop(columns=[target_column, 'quality'])
    y_full = df[target_column]
    
    # Remove duplicate feature rows, keep first occurrence
    deduplicated = X_full.drop_duplicates(keep='first')
    print(f"Original dataset: {len(X_full)} samples")
    print(f"Deduplicated dataset: {len(deduplicated)} samples ({len(X_full)-len(deduplicated)} duplicates removed)")
    
    # Keep only corresponding labels for deduplicated rows
    y_dedup = y_full.loc[deduplicated.index]
    
    # Now split the deduplicated data
    X_temp, X_test, y_temp, y_test = train_test_split(
        deduplicated, y_dedup,
        test_size=test_size,
        stratify=y_dedup,
        random_state=random_state
    )
    
    # Then split temp into train and validation
    val_size_relative = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_relative,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Verify no cross-split duplicates in the deduplicated version
    print(f"\n=== DEDUPLICATED CROSS-SPLIT AUDIT ===")
    train_rows = set(tuple(row) for _, row in X_train.iterrows())
    val_rows = set(tuple(row) for _, row in X_val.iterrows())
    test_rows = set(tuple(row) for _, row in X_test.iterrows())
    
    train_val_overlap = train_rows & val_rows
    train_test_overlap = train_rows & test_rows
    val_test_overlap = val_rows & test_rows
    all_splits_overlap = train_rows & val_rows & test_rows
    
    print(f"Total unique feature rows in deduplicated dataset: {len(train_rows | val_rows | test_rows)}")
    print(f"Train-validation cross-split duplicates: {len(train_val_overlap)} ({len(train_val_overlap)/len(X_val):.2%} of val set)")
    print(f"Train-test cross-split duplicates: {len(train_test_overlap)} ({len(train_test_overlap)/len(X_test):.2%} of test set)")
    print(f"Validation-test cross-split duplicates: {len(val_test_overlap)} ({len(val_test_overlap)/len(X_test):.2%} of test set)")
    print(f"Duplicates present in ALL three splits: {len(all_splits_overlap)}")
    
    print(f"\nDeduplicated data splitting complete (stratified, random_state={random_state}):")
    print(f"  Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(deduplicated):.1%})")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/len(deduplicated):.1%})")
    print(f"  Test: {X_test.shape[0]} samples ({X_test.shape[0]/len(deduplicated):.1%})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_preprocessing_pipeline(feature_names: list) -> ColumnTransformer:
    """
    Create preprocessing pipeline that standardizes numerical features.
    Only fits on training data to prevent leakage.
    """
    # All features are numerical in this dataset
    numerical_features = feature_names.copy()
    
    # Create preprocessing for numerical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ])
    
    return preprocessor

def fit_preprocessing(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessing only on training data."""
    preprocessor.fit(X_train)
    return preprocessor

def save_preprocessor(preprocessor: ColumnTransformer, save_path: Path) -> None:
    """Save fitted preprocessor to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)
    print(f"Preprocessor saved to {save_path}")

def load_preprocessor(load_path: Path) -> ColumnTransformer:
    """Load fitted preprocessor from disk."""
    return joblib.load(load_path)

if __name__ == "__main__":
    # Add project root to Python path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    # Test preprocessing
    from research.src.data_loader import load_dataset, create_binary_target
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "datasets" / "raw"
    
    # Load and prepare data
    data, stats = load_dataset(str(raw_dir))
    data = create_binary_target(data, threshold=6)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_data_splitting(
        data, 'good_quality', random_state=42
    )
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline(list(X_train.columns))
    preprocessor = fit_preprocessing(preprocessor, X_train)
    
    # Transform data
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\nProcessed data shapes:")
    print(f"  X_train: {X_train_processed.shape}")
    print(f"  X_val: {X_val_processed.shape}")
    print(f"  X_test: {X_test_processed.shape}")