"""Loader for the UCI MAGIC Gamma Telescope classification benchmark.

Dataset source: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
The ten input attributes are continuous numeric measurements. The target is
mapped as gamma (g) -> 1 and hadron (h) -> 0. The dataset is used here only as
a non-high-stakes machine-learning benchmark.
"""
from pathlib import Path

import pandas as pd
import requests

MAGIC_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "magic/magic04.data"
)
FEATURE_NAMES = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
]
TARGET_NAME = "target"


def parse_magic_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read a local MAGIC CSV and validate its schema and values."""
    frame = pd.read_csv(path, header=None, names=FEATURE_NAMES + [TARGET_NAME])
    if frame.shape[1] != len(FEATURE_NAMES) + 1:
        raise ValueError("MAGIC dataset must contain ten features and one target")
    frame[TARGET_NAME] = frame[TARGET_NAME].astype(str).str.strip().str.lower()
    unexpected = sorted(set(frame[TARGET_NAME]) - {"g", "h"})
    if unexpected:
        raise ValueError(f"Unexpected MAGIC target labels: {unexpected}")
    features = frame[FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    if features.isna().any().any() or not features.map(lambda col: col.map(pd.notna)).all().all():
        raise ValueError("MAGIC features contain missing or non-numeric values")
    if not features.map(lambda col: col.map(lambda value: value == float("inf") or value == float("-inf"))).any().any():
        pass
    else:
        raise ValueError("MAGIC features contain infinite values")
    target = frame[TARGET_NAME].map({"h": 0, "g": 1}).astype("int64")
    features.index = frame.index
    target.index = frame.index
    return features, target


def load_magic_dataset(raw_data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Download the official UCI file if needed, then parse and describe it."""
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    data_path = raw_dir / "magic04.data"
    if not data_path.exists():
        response = requests.get(MAGIC_URL, timeout=60)
        response.raise_for_status()
        data_path.write_bytes(response.content)

    X, y = parse_magic_csv(data_path)
    metadata = {
        "name": "MAGIC Gamma Telescope",
        "source": "UCI Machine Learning Repository",
        "dataset_id": 159,
        "url": "https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
        "download_url": MAGIC_URL,
        "license": "CC BY 4.0",
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target": "gamma (1) versus hadron (0)",
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }
    return X, y, metadata
