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

def grouped_stratified_holdout_indices(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/holdout indices with exact duplicate feature rows grouped."""
    from sklearn.model_selection import StratifiedGroupKFold

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(X) == 0:
        raise ValueError("X must not be empty")

    groups = pd.MultiIndex.from_frame(X).factorize(sort=False)[0]
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough unique feature groups for requested n_splits")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    candidates = []
    prevalence = float(np.mean(y))
    for train_idx, holdout_idx in splitter.split(X, y, groups):
        size_error = abs(len(holdout_idx) / len(X) - test_size)
        class_error = abs(float(np.mean(y.iloc[holdout_idx])) - prevalence)
        candidates.append((size_error + 0.5 * class_error, train_idx, holdout_idx))

    _, train_idx, holdout_idx = min(candidates, key=lambda item: item[0])
    train_groups = set(groups[train_idx])
    holdout_groups = set(groups[holdout_idx])
    if train_groups & holdout_groups:
        raise AssertionError("Identical feature groups leaked across holdout split")
    return train_idx, holdout_idx


def grouped_stratified_kfold_indices(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
):
    """Yield stratified folds with exact duplicate feature rows kept together."""
    from sklearn.model_selection import StratifiedGroupKFold

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    groups = pd.MultiIndex.from_frame(X).factorize(sort=False)[0]
    if len(np.unique(groups)) < n_splits:
        raise ValueError("Not enough unique feature groups for requested n_splits")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    for train_idx, validation_idx in splitter.split(X, y, groups):
        if set(groups[train_idx]) & set(groups[validation_idx]):
            raise AssertionError("Identical feature groups leaked across CV fold")
        yield train_idx, validation_idx


def grouped_stratified_data_splitting(
    df: pd.DataFrame,
    target_column: str,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create stratified splits while keeping identical feature rows in one split.

    Unlike row-wise stratification, this retains repeated observations but groups
    identical feature vectors together. This avoids train/validation/test leakage
    without silently dropping duplicate rows or choosing one label from a
    potentially conflicting duplicate group.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split sizes must sum to 1.0")
    if target_column not in df.columns or "quality" not in df.columns:
        raise ValueError("DataFrame must contain target_column and original 'quality' column")

    X = df.drop(columns=[target_column, "quality"])
    y = df[target_column]
    if X.empty:
        raise ValueError("Feature matrix must not be empty")

    # Factorize exact feature rows. Equal feature vectors receive the same group ID,
    # even when their target labels differ.
    groups = pd.MultiIndex.from_frame(X).factorize(sort=False)[0]
    unique_groups = np.unique(groups)
    if len(unique_groups) < 7:
        raise ValueError("At least 7 unique feature groups are required for grouped splitting")

    def choose_holdout(X_part, y_part, group_part, n_splits, target_fraction, seed):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        candidates = []
        prevalence = float(y_part.mean())
        for _, holdout_idx in splitter.split(X_part, y_part, group_part):
            holdout_fraction = len(holdout_idx) / len(X_part)
            holdout_prevalence = float(y_part.iloc[holdout_idx].mean())
            score = abs(holdout_fraction - target_fraction) + 0.5 * abs(
                holdout_prevalence - prevalence
            )
            candidates.append((score, holdout_idx))
        return min(candidates, key=lambda item: item[0])[1]

    # Seven folds approximate a 1/7 test holdout (~14.3%) while keeping groups intact.
    test_folds = max(2, min(7, len(unique_groups)))
    test_idx = choose_holdout(
        X, y, groups, test_folds, test_size, random_state
    )
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_idx] = True
    remaining_idx = np.flatnonzero(~test_mask)

    X_remaining = X.iloc[remaining_idx]
    y_remaining = y.iloc[remaining_idx]
    groups_remaining = groups[remaining_idx]
    val_relative = val_size / (train_size + val_size)
    remaining_group_count = len(np.unique(groups_remaining))
    val_folds = max(2, min(6, remaining_group_count))
    val_idx_local = choose_holdout(
        X_remaining,
        y_remaining,
        groups_remaining,
        val_folds,
        val_relative,
        random_state + 1,
    )
    val_mask_local = np.zeros(len(X_remaining), dtype=bool)
    val_mask_local[val_idx_local] = True
    train_idx_local = np.flatnonzero(~val_mask_local)

    X_train = X_remaining.iloc[train_idx_local]
    X_val = X_remaining.iloc[val_idx_local]
    X_test = X.iloc[test_idx]
    y_train = y_remaining.iloc[train_idx_local]
    y_val = y_remaining.iloc[val_idx_local]
    y_test = y.iloc[test_idx]

    train_groups = set(groups_remaining[train_idx_local])
    val_groups = set(groups_remaining[val_idx_local])
    test_groups = set(groups[test_idx])
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise AssertionError("Identical feature groups leaked across data splits")

    print("\n=== GROUPED STRATIFIED SPLIT AUDIT ===")
    print(f"Unique feature groups: {len(unique_groups)}")
    print(f"Train: {len(X_train)} ({len(X_train) / len(X):.1%})")
    print(f"Validation: {len(X_val)} ({len(X_val) / len(X):.1%})")
    print(f"Test: {len(X_test)} ({len(X_test) / len(X):.1%})")
    print("Cross-split identical-feature groups: 0")
    print(f"Class prevalence: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

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