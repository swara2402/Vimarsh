"""Tests for duplicate-grouped stratified train/validation/test splitting."""
import numpy as np
import pandas as pd

from src.preprocessing import grouped_stratified_data_splitting


def make_duplicate_dataset():
    rows = []
    for group_id in range(140):
        features = {"feature_a": float(group_id), "feature_b": float(group_id % 13)}
        labels = [group_id % 2, group_id % 2]
        if group_id < 12:
            labels = [0, 1]  # Include conflicting labels for identical feature rows.
        for label in labels:
            rows.append({
                **features,
                "quality": 6 if label else 5,
                "good_quality": label,
            })
    return pd.DataFrame(rows)


def test_grouped_split_retains_rows_and_prevents_feature_overlap():
    data = make_duplicate_dataset()
    splits = grouped_stratified_data_splitting(
        data, "good_quality", random_state=29
    )
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    assert len(X_train) + len(X_val) + len(X_test) == len(data)
    assert len(y_train) + len(y_val) + len(y_test) == len(data)

    def feature_groups(frame):
        return set(map(tuple, frame[["feature_a", "feature_b"]].to_numpy()))

    train_groups = feature_groups(X_train)
    val_groups = feature_groups(X_val)
    test_groups = feature_groups(X_test)

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_val.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_grouped_split_is_reproducible():
    data = make_duplicate_dataset()
    first = grouped_stratified_data_splitting(data, "good_quality", random_state=17)
    second = grouped_stratified_data_splitting(data, "good_quality", random_state=17)

    for first_part, second_part in zip(first, second):
        assert first_part.index.tolist() == second_part.index.tolist()
