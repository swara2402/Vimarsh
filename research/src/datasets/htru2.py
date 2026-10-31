"""Loader for the UCI HTRU2 pulsar classification benchmark.

Source: https://archive.ics.uci.edu/dataset/372/htru2
Eight continuous profile/statistical features; target 1 denotes a pulsar.
This is an astronomy classification benchmark, not a clinical decision tool.
"""
from pathlib import Path
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
import requests

HTRU2_URL = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
FEATURE_NAMES = [
    "mean_integrated_profile", "std_integrated_profile",
    "excess_kurtosis_integrated_profile", "skewness_integrated_profile",
    "mean_dm_snr_curve", "std_dm_snr_curve",
    "excess_kurtosis_dm_snr_curve", "skewness_dm_snr_curve",
]
TARGET_NAME = "target"


def parse_htru2_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Parse the local UCI CSV and validate numeric features and binary labels."""
    frame = pd.read_csv(path, header=None, names=FEATURE_NAMES + [TARGET_NAME])
    if frame.shape[1] != 9:
        raise ValueError("HTRU2 must contain eight features and one target")
    features = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    target = pd.to_numeric(frame[TARGET_NAME], errors="raise").astype("int64")
    if not set(target.unique()).issubset({0, 1}) or target.nunique() != 2:
        raise ValueError("HTRU2 target must contain both binary labels 0 and 1")
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("HTRU2 features contain missing or non-finite values")
    return features, target


def load_htru2_dataset(raw_data_dir: str | Path):
    """Download and parse the official UCI archive if it is not already cached."""
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "HTRU_2.csv"
    if not csv_path.exists():
        response = requests.get(HTRU2_URL, timeout=60)
        response.raise_for_status()
        with ZipFile(io.BytesIO(response.content)) as archive:
            matches = [name for name in archive.namelist() if name.lower().endswith("htru_2.csv")]
            if not matches:
                raise ValueError("HTRU2 archive did not contain HTRU_2.csv")
            csv_path.write_bytes(archive.read(matches[0]))
    X, y = parse_htru2_csv(csv_path)
    metadata = {
        "name": "HTRU2",
        "source": "UCI Machine Learning Repository",
        "dataset_id": 372,
        "url": "https://archive.ics.uci.edu/dataset/372/htru2",
        "download_url": HTRU2_URL,
        "license": "CC BY 4.0",
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target": "pulsar (1) versus non-pulsar (0)",
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }
    return X, y, metadata
