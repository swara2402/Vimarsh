"""
Data loader module for downloading and loading the Wine Quality dataset.
Handles download of red and white wine datasets, combines them with wine type feature.
"""
import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

# UCI Wine Quality dataset URLs
REDS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

def download_file(url: str, save_path: Path) -> None:
    """Download a file from a URL and save it to the specified path."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        f.write(response.content)

def load_dataset(raw_data_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Download and load the Wine Quality dataset.
    Returns combined DataFrame and dataset statistics.
    """
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets if not present
    reds_path = raw_dir / "winequality-red.csv"
    whites_path = raw_dir / "winequality-white.csv"
    
    if not reds_path.exists():
        print(f"Downloading red wine dataset to {reds_path}")
        download_file(REDS_URL, reds_path)
    
    if not whites_path.exists():
        print(f"Downloading white wine dataset to {whites_path}")
        download_file(WHITES_URL, whites_path)
    
    # Load datasets
    reds = pd.read_csv(reds_path, sep=';')
    whites = pd.read_csv(whites_path, sep=';')
    
    # Add wine type feature: 0 for red, 1 for white
    reds['wine_type'] = 0
    whites['wine_type'] = 1
    
    # Combine datasets
    combined = pd.concat([reds, whites], axis=0, ignore_index=True)
    
    # Calculate and return dataset statistics
    stats = {
        'reds_shape': reds.shape,
        'whites_shape': whites.shape,
        'combined_shape': combined.shape,
        'reds_quality_dist': reds['quality'].value_counts().sort_index().to_dict(),
        'whites_quality_dist': whites['quality'].value_counts().sort_index().to_dict(),
        'combined_quality_dist': combined['quality'].value_counts().sort_index().to_dict(),
        'missing_values': combined.isnull().sum().to_dict(),
        'duplicates': combined.duplicated().sum(),
        'feature_names': list(combined.columns),
        'data_types': combined.dtypes.astype(str).to_dict(),
        'provenance': {
            'source': 'UCI Machine Learning Repository',
            'dataset_url': 'https://archive.ics.uci.edu/dataset/186/wine+quality',
            'download_date': pd.Timestamp.now().isoformat(),
            'usage_terms': 'This dataset is available for non-commercial research purposes.'
        }
    }
    
    print(f"Dataset loaded successfully:")
    print(f"  Red wines: {reds.shape[0]} samples, {reds.shape[1]-1} features")
    print(f"  White wines: {whites.shape[0]} samples, {whites.shape[1]-1} features")
    print(f"  Combined: {combined.shape[0]} total samples")
    print(f"  Missing values: {combined.isnull().sum().sum()} total")
    print(f"  Duplicate records: {combined.duplicated().sum()}")
    
    return combined, stats

def create_binary_target(df: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    """
    Create binary target: 1 if quality >= threshold (good quality), 0 otherwise.
    Clearly documents that this threshold is an experimental choice.
    """
    df = df.copy()
    # Experimental choice: quality >=6 is "good"
    # This is not an objective measure of wine quality, but an experimental classification threshold
    df['good_quality'] = (df['quality'] >= threshold).astype(int)
    print(f"Binary target created with threshold={threshold}:")
    print(f"  Good quality (1): {df['good_quality'].sum()} samples ({df['good_quality'].mean():.1%})")
    print(f"  Lower quality (0): {len(df) - df['good_quality'].sum()} samples ({1-df['good_quality'].mean():.1%})")
    return df

if __name__ == "__main__":
    # Test the data loader
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "datasets" / "raw"
    data, stats = load_dataset(str(raw_dir))
    data_with_target = create_binary_target(data, threshold=6)