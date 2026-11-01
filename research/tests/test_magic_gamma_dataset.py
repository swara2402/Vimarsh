from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.magic_gamma import FEATURE_NAMES, parse_magic_csv


def test_magic_parser_maps_targets_and_preserves_numeric_features(tmp_path: Path):
    path = tmp_path / "magic04.data"
    row_g = ",".join(["1.0"] * len(FEATURE_NAMES) + ["g"])
    row_h = ",".join(["2.0"] * len(FEATURE_NAMES) + ["h"])
    path.write_text(row_g + "\n" + row_h + "\n", encoding="utf-8")

    X, y = parse_magic_csv(path)

    assert list(X.columns) == FEATURE_NAMES
    assert X.shape == (2, 10)
    assert y.tolist() == [1, 0]
    assert np.isfinite(X.to_numpy()).all()


def test_magic_parser_rejects_unknown_target(tmp_path: Path):
    path = tmp_path / "bad.data"
    path.write_text(",".join(["1.0"] * len(FEATURE_NAMES) + ["x"]), encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected MAGIC target"):
        parse_magic_csv(path)


def test_magic_parser_rejects_nonfinite_features(tmp_path: Path):
    path = tmp_path / "bad.data"
    path.write_text(",".join(["inf"] + ["1.0"] * (len(FEATURE_NAMES) - 1) + ["g"]), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        parse_magic_csv(path)
