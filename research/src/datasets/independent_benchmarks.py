"""Independent UCI benchmark adapters for Milestone 52.

These datasets are evaluated as separate tasks. No model is transferred between
feature spaces and no metric is pooled across datasets.
"""
from pathlib import Path
from zipfile import ZipFile
import io

import numpy as np
import pandas as pd
import requests

BANKNOTE_URL = "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip"
IONOSPHERE_URL = "https://archive.ics.uci.edu/static/public/52/ionosphere.zip"

BANKNOTE_FEATURES = [
    "variance", "skewness", "curtosis", "entropy",
]
IONOSPHERE_FEATURES = [f"feature_{i:02d}" for i in range(1, 35)]


def _download_member(url: str, raw_dir: Path, member_suffix: str, output_name: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / output_name
    if output.exists():
        return output
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with ZipFile(io.BytesIO(response.content)) as archive:
        matches = [
            name for name in archive.namelist()
            if name.lower().endswith(member_suffix.lower())
        ]
        if not matches:
            raise ValueError(f"UCI archive did not contain a *{member_suffix} member")
        output.write_bytes(archive.read(matches[0]))
    return output


def _validate(features: pd.DataFrame, target: pd.Series, name: str):
    values = features.to_numpy(dtype=float)
    if features.empty or not np.isfinite(values).all():
        raise ValueError(f"{name} features must be non-empty and finite")
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError(f"{name} target must contain both binary labels 0 and 1")
    if len(features) != len(target):
        raise ValueError(f"{name} features and labels are misaligned")
    return features.reset_index(drop=True), target.astype("int64").reset_index(drop=True)


def parse_banknote_csv(path: str | Path):
    """Parse UCI Banknote Authentication: four numeric features, binary class."""
    frame = pd.read_csv(path, header=None, names=BANKNOTE_FEATURES + ["target"])
    if frame.shape[1] != 5:
        raise ValueError("Banknote Authentication must contain four features and one target")
    X = frame[BANKNOTE_FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(frame["target"], errors="raise").astype("int64")
    return _validate(X, y, "Banknote Authentication")


def load_banknote_dataset(raw_dir: str | Path):
    path = _download_member(
        BANKNOTE_URL, Path(raw_dir), "data_banknote_authentication.txt",
        "data_banknote_authentication.txt",
    )
    X, y = parse_banknote_csv(path)
    return X, y, {
        "name": "Banknote Authentication",
        "source": "UCI Machine Learning Repository",
        "dataset_id": 267,
        "url": "https://archive.ics.uci.edu/dataset/267/banknote+authentication",
        "license": "CC BY 4.0",
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target": "forged (1) versus authentic (0)",
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }


def parse_ionosphere_csv(path: str | Path):
    """Parse UCI Ionosphere: 34 numeric radar features and g/b labels."""
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] != 35:
        raise ValueError("Ionosphere must contain 34 features and one target")
    X = frame.iloc[:, :34].apply(pd.to_numeric, errors="raise")
    labels = frame.iloc[:, 34].astype(str).str.strip().str.lower()
    unexpected = sorted(set(labels) - {"g", "b"})
    if unexpected:
        raise ValueError(f"Unexpected Ionosphere target labels: {unexpected}")
    y = labels.map({"b": 0, "g": 1}).astype("int64")
    X.columns = IONOSPHERE_FEATURES
    return _validate(X, y, "Ionosphere")


def load_ionosphere_dataset(raw_dir: str | Path):
    path = _download_member(
        IONOSPHERE_URL, Path(raw_dir), "ionosphere.data", "ionosphere.data",
    )
    X, y = parse_ionosphere_csv(path)
    return X, y, {
        "name": "Ionosphere",
        "source": "UCI Machine Learning Repository",
        "dataset_id": 52,
        "url": "https://archive.ics.uci.edu/dataset/52/ionosphere",
        "license": "CC BY 4.0",
        "rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "target": "good (1) versus bad (0) radar return",
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }
